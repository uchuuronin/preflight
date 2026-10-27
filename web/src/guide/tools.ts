/**
 * Guide tools: the same actions the person has, offered to Muse Spark as function tools.
 *
 * UI tools run here in the browser through WorkspaceActions and the PredictionProvider,
 * so the guide can never quote a number the UI doesn't show. Server tools (run_probe,
 * propose_patch, run_verify) spend money, so calling one only queues a request that the
 * person must click to approve.
 *
 * Pins are enforced twice: the set_knob schema is rebuilt every turn with pinned knobs
 * removed from its enum, and WorkspaceActions.setKnob refuses a guide move on a pinned knob.
 */
import type { FlightRecord, KnobValue, Objective } from "../workspace/record/types";
import type { PredictionProvider } from "../workspace/provider/PredictionProvider";
import type { WorkspaceActions } from "../workspace/state";
import { trusted } from "../workspace/status";

export interface ToolDef {
	type: "function";
	name: string;
	description: string;
	parameters: Record<string, unknown>;
	strict?: boolean;
}

export const SERVER_TOOLS = new Set(["run_probe", "propose_patch", "run_verify"]);

export function buildTools(record: FlightRecord, pinned: ReadonlySet<string>): ToolDef[] {
	const knobIds = record.knobs.map((k) => k.id);
	const movable = knobIds.filter((id) => !pinned.has(id));
	const nodeIds = record.graph.nodes.map((n) => n.id);
	const knobValue = { anyOf: [{ type: "number" }, { type: "string" }, { type: "boolean" }] };
	const tools: ToolDef[] = [
		{
			type: "function",
			name: "predict",
			description:
				"Readouts (point, interval, status) for a knob setting. Knobs not given keep their current value. Quote only numbers this returns.",
			parameters: {
				type: "object",
				properties: {
					knobs: { type: "object", additionalProperties: knobValue, propertyNames: { enum: knobIds } },
				},
				required: ["knobs"],
				additionalProperties: false,
			},
		},
		{
			type: "function",
			name: "sensitivity",
			description: "Unpinned knobs ranked by how much their best single move improves the objective, never trading into OOM.",
			parameters: {
				type: "object",
				properties: { objective: { type: "string", enum: ["cost", "time", "headroom", "quality"] } },
				required: ["objective"],
				additionalProperties: false,
			},
		},
		{
			type: "function",
			name: "pin",
			description: "Pin or unpin a knob. Only do this when the person asks.",
			parameters: {
				type: "object",
				properties: { knob_id: { type: "string", enum: knobIds }, pinned: { type: "boolean" } },
				required: ["knob_id", "pinned"],
				additionalProperties: false,
			},
		},
		{
			type: "function",
			name: "highlight",
			description: "Light map nodes in the accent colour while you talk about them.",
			parameters: {
				type: "object",
				properties: { nodes: { type: "array", items: { type: "string", enum: nodeIds } } },
				required: ["nodes"],
				additionalProperties: false,
			},
		},
		{
			type: "function",
			name: "snapshot",
			description: "Save the current configuration under a name, to compare later.",
			parameters: {
				type: "object",
				properties: { name: { type: "string" } },
				required: ["name"],
				additionalProperties: false,
			},
		},
		{
			type: "function",
			name: "compare",
			description: "Diff two snapshots target by target.",
			parameters: {
				type: "object",
				properties: { a: { type: "string" }, b: { type: "string" } },
				required: ["a", "b"],
				additionalProperties: false,
			},
		},
		{
			type: "function",
			name: "run_probe",
			description: "Measure a new point on a real GPU. Costs money; the person must approve it.",
			parameters: {
				type: "object",
				properties: { knobs: { type: "object", additionalProperties: knobValue }, reason: { type: "string" } },
				required: ["knobs", "reason"],
				additionalProperties: false,
			},
		},
		{
			type: "function",
			name: "propose_patch",
			description: "Ask Muse Code to write a code change for a knob that needs one. Costs money; the person must approve it.",
			parameters: {
				type: "object",
				properties: { template: { type: "string" }, knob_id: { type: "string", enum: knobIds } },
				required: ["template"],
				additionalProperties: false,
			},
		},
		{
			type: "function",
			name: "run_verify",
			description: "One real run of the chosen plan plus a fixed-seed loss check. Costs money; the person must approve it.",
			parameters: { type: "object", properties: {}, additionalProperties: false },
		},
	];
	// set_knob only exists while at least one knob is movable.
	if (movable.length) {
		tools.unshift({
			type: "function",
			name: "set_knob",
			description:
				"Move one unpinned knob on the live map. The person sees it animate and can apply, revert or pin it. Pinned knobs are not in the enum and cannot be moved.",
			parameters: {
				type: "object",
				properties: { knob_id: { type: "string", enum: movable }, value: knobValue, why: { type: "string" } },
				required: ["knob_id", "value", "why"],
				additionalProperties: false,
			},
		});
	}
	return tools;
}

export interface ToolContext {
	record: FlightRecord;
	provider: PredictionProvider;
	actions: WorkspaceActions;
	knobs: Readonly<Record<string, KnobValue>>;
	pinned: ReadonlySet<string>;
	/** Server tools land here for the person to approve. */
	requestApproval: (name: string, args: Record<string, unknown>) => void;
}

export type ToolResult = { ok: true; result: unknown } | { ok: false; error: string };

function compactPrediction(p: ReturnType<PredictionProvider["predict"]>, ids: string[]) {
	const out: Record<string, unknown> = {};
	for (const id of ids) {
		const e = p[id];
		if (!e) continue;
		out[id] = trusted(e)
			? { point: e.point, lo: e.lo, hi: e.hi, unit: e.unit, status: e.status }
			: { status: "abstain", reason: e.reason };
	}
	return out;
}

const HEADLINE = ["peak_memory", "p_oom", "time_to_finish", "total_cost", "divergence_risk", "tokens_per_sec"];

export function executeTool(name: string, args: Record<string, unknown>, ctx: ToolContext): ToolResult {
	const { record, provider, actions, knobs, pinned } = ctx;
	switch (name) {
		case "predict": {
			const k = (args.knobs ?? {}) as Record<string, KnobValue>;
			return { ok: true, result: compactPrediction(provider.predict(record, { ...knobs, ...k }), HEADLINE) };
		}
		case "sensitivity": {
			const objective = (args.objective as Objective) ?? record.job.objective;
			const ranked = provider.sensitivity(record, objective, { pinned, knobs });
			return { ok: true, result: ranked.slice(0, 6) };
		}
		case "set_knob": {
			const id = String(args.knob_id);
			if (pinned.has(id)) return { ok: false, error: `knob ${id} is pinned by the person; tell them what releasing it would buy instead` };
			const from = knobs[id];
			const to = args.value as KnobValue;
			if (from === undefined) return { ok: false, error: `unknown knob ${id}` };
			if (!actions.setKnob(id, to, "guide")) return { ok: false, error: `could not move ${id}` };
			actions.propose({ knobId: id, from, to, by: "guide" });
			return {
				ok: true,
				result: { moved: id, from, to, readouts: compactPrediction(provider.predict(record, { ...knobs, [id]: to }), HEADLINE) },
			};
		}
		case "pin":
			actions.pin(String(args.knob_id), Boolean(args.pinned));
			return { ok: true, result: { pinned: args.knob_id, value: args.pinned } };
		case "highlight":
			actions.highlight((args.nodes as string[]) ?? []);
			return { ok: true, result: { highlighted: args.nodes } };
		case "snapshot":
			return { ok: true, result: actions.snapshot(String(args.name ?? "")) };
		case "compare":
			return { ok: true, result: actions.compare(String(args.a), String(args.b)) };
		default:
			if (SERVER_TOOLS.has(name)) {
				ctx.requestApproval(name, args);
				return { ok: true, result: { status: "awaiting_person_approval" } };
			}
			return { ok: false, error: `unknown tool ${name}` };
	}
}
