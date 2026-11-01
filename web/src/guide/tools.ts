/**
 * Guide tools: the same actions the person has, offered to Muse Spark as function tools.
 *
 * UI tools run here in the browser through WorkspaceActions and the PredictionProvider, so
 * the guide can never quote a number the UI doesn't show. Server tools (run_probe,
 * propose_patch, run_verify) spend money, so calling one only queues a request the person
 * must click to approve.
 *
 * Pins are enforced twice: set_knob's schema is rebuilt every turn with pinned knobs removed
 * from its enum, and WorkspaceActions.setKnob refuses a guide move on a pinned knob.
 *
 * Every call, whoever makes it (Muse Spark or the rule-based tour), goes through runTool and
 * lands in the trace with its real arguments and result.
 */
import { breaksLimit, type Limits, NO_LIMITS, withinLimits } from "./limits";
import type { PredictionProvider } from "../workspace/provider/PredictionProvider";
import { costStraddlesBudget, SAMPLE_MORE_TASKS, samplesTasks } from "../workspace/record/model";
import type { KnobValue, LoadedRecord, ObjectiveKind, Prediction } from "../workspace/record/types";
import type { WorkspaceActions } from "../workspace/state";
import { changesResults, isFailing } from "./ruleTour";

export interface ToolDef {
	type: "function";
	name: string;
	description: string;
	parameters: Record<string, unknown>;
	strict?: boolean;
}

export const SERVER_TOOLS = new Set(["run_probe", "propose_patch", "run_verify", "sample_more"]);
const OBJECTIVE_KINDS: ObjectiveKind[] = ["min_cost", "min_time", "fit_memory", "min_risk", "max_throughput"];

export function buildTools(record: LoadedRecord, pinned: ReadonlySet<string>): ToolDef[] {
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
				properties: { knobs: { type: "object", additionalProperties: knobValue, propertyNames: { enum: knobIds } } },
				required: ["knobs"],
				additionalProperties: false,
			},
		},
		{
			type: "function",
			name: "sensitivity",
			description: "Unpinned knobs ranked by how much their best single move improves the aim, never trading into a predicted failure.",
			parameters: {
				type: "object",
				properties: { objective: { type: "string", enum: OBJECTIVE_KINDS } },
				required: ["objective"],
				additionalProperties: false,
			},
		},
		{
			type: "function",
			name: "pin",
			description: "Pin or unpin a knob. Only when the person asks.",
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
			parameters: { type: "object", properties: { name: { type: "string" } }, required: ["name"], additionalProperties: false },
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
	if (samplesTasks(record)) {
		tools.push({
			type: "function",
			name: "sample_more",
			description: `Run ${SAMPLE_MORE_TASKS} more of the run's tasks for real to narrow the run cost range. Use it only when that range straddles the budget (low end under it, high end over), because then more tasks change the answer. Costs API or model money; the person must approve it.`,
			parameters: {
				type: "object",
				properties: { reason: { type: "string" } },
				required: ["reason"],
				additionalProperties: false,
			},
		});
	}
	if (movable.length) {
		tools.unshift({
			type: "function",
			name: "set_knob",
			description:
				"Move one unpinned knob on the live map. The person sees it move and can apply, revert or pin it. Pinned knobs are not in the list and cannot be moved.",
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
	record: LoadedRecord;
	provider: PredictionProvider;
	actions: WorkspaceActions;
	knobs: Readonly<Record<string, KnobValue>>;
	pinned: ReadonlySet<string>;
	/** The visitor's limits (guide/limits.ts): hard limits on every move. */
	limits?: Limits;
	/** Money-spending tools land here for the person to approve. */
	requestApproval: (name: string, args: Record<string, unknown>) => void;
}

export type ToolResult = { ok: true; result: unknown } | { ok: false; error: string };

const HEADLINE_READOUTS = ["memory", "time", "cost", "risk"] as const;

function headline(record: LoadedRecord, p: Prediction): Record<string, unknown> {
	const ids = new Set<string>(
		record.predictions.targets.filter((t) => t.readout && HEADLINE_READOUTS.includes(t.readout)).map((t) => t.id),
	);
	if (record.predictions.targets.some((t) => t.id === "throughput.tokens_per_s")) ids.add("throughput.tokens_per_s");
	const out: Record<string, unknown> = {};
	for (const id of ids) {
		const e = p[id];
		if (!e) continue;
		out[id] =
			e.status === "abstain" || e.point === null
				? { status: "abstain", reason: e.reason }
				: { point: e.point, lo: e.lo, hi: e.hi, unit: e.unit, status: e.status };
	}
	return out;
}

export function executeTool(name: string, args: Record<string, unknown>, ctx: ToolContext): ToolResult {
	const { record, provider, actions, knobs, pinned } = ctx;
	switch (name) {
		case "predict": {
			const k = (args.knobs ?? {}) as Record<string, KnobValue>;
			return { ok: true, result: headline(record, provider.predict(record, { ...knobs, ...k })) };
		}
		case "sensitivity": {
			const kind = (args.objective as ObjectiveKind | undefined) ?? record.job.objective?.kind ?? "min_time";
			// Each knob says whether moving it changes what the run produces or measures (seq_len, an
			// eval's task count): those are trade-offs to put to the person, not savings to take.
			return {
				ok: true,
				result: provider
					.sensitivity(record, { kind }, withinLimits(record, provider, ctx.limits ?? NO_LIMITS, knobs, { pinned }))
					.slice(0, 6)
					.map((k) => ({ ...k, changes_results: changesResults(k.knob_id, record) })),
			};
		}
		case "set_knob": {
			const id = String(args.knob_id);
			if (pinned.has(id)) return { ok: false, error: `knob ${id} is pinned by the person; say what releasing it would buy instead` };
			const from = knobs[id];
			const to = args.value as KnobValue;
			if (from === undefined) return { ok: false, error: `unknown knob ${id}` };
			const broken = breaksLimit(record, ctx.limits ?? NO_LIMITS, id, to, provider.predict(record, { ...knobs, [id]: to }), provider.predict(record, knobs), knobs);
			if (broken) return { ok: false, error: `${id} = ${String(to)} breaks the person's limits (${broken}); offer the next lever instead` };
			if (changesResults(id, record) && !isFailing({ record, provider, knobs })) {
				return { ok: false, error: `${id} changes what the run produces or measures; put it to the person as a trade-off instead of moving it` };
			}
			if (!actions.setKnob(id, to, "guide")) return { ok: false, error: `could not move ${id}` };
			actions.propose({ knobId: id, from, to });
			actions.openKnob(id);
			return {
				ok: true,
				result: { moved: id, from, to, readouts: headline(record, provider.predict(record, { ...knobs, [id]: to })) },
			};
		}
		case "pin":
			actions.pin(String(args.knob_id), Boolean(args.pinned));
			return { ok: true, result: { pinned: args.knob_id, value: args.pinned } };
		case "highlight":
			actions.highlight((args.nodes as string[]) ?? []);
			return { ok: true, result: { highlighted: args.nodes } };
		case "snapshot":
			return { ok: true, result: { id: actions.snapshot(String(args.name ?? "")).id } };
		case "compare":
			return { ok: true, result: actions.compare(String(args.a), String(args.b)) };
		case "sample_more": {
			const straddle = costStraddlesBudget(ctx.record, ctx.provider.predict(ctx.record, ctx.knobs));
			if (!straddle) {
				return {
					ok: false,
					error: "the run cost range does not straddle the budget, so more tasks would not change whether it fits",
				};
			}
			ctx.requestApproval(name, { ...args, tasks: SAMPLE_MORE_TASKS, budget: straddle.budget, lo: straddle.lo, hi: straddle.hi });
			return { ok: true, result: { status: "awaiting_person_approval", tasks: SAMPLE_MORE_TASKS } };
		}
		default:
			if (SERVER_TOOLS.has(name)) {
				ctx.requestApproval(name, args);
				return { ok: true, result: { status: "awaiting_person_approval" } };
			}
			return { ok: false, error: `unknown tool ${name}` };
	}
}

let traceSeq = 0;

function short(v: unknown, max = 90): string {
	const s = typeof v === "string" ? v : JSON.stringify(v);
	if (!s) return "";
	return s.length > max ? `${s.slice(0, max - 1)}…` : s;
}

/** Run a tool and put the call and its result in the visible trace. */
export function runTool(name: string, args: Record<string, unknown>, ctx: ToolContext): ToolResult {
	const id = ++traceSeq;
	const res = executeTool(name, args, ctx);
	ctx.actions.trace({
		id,
		tool: name,
		args: short(args),
		result: res.ok ? short(res.result) : res.error,
		ok: res.ok,
	});
	return res;
}
