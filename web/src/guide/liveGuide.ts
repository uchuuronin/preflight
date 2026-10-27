/**
 * The live guide (Muse Spark) as an autonomous tour: sensitivity, one set_knob, the visitor
 * applies or reverts, re-plan. The browser never sends the conversation or tool definitions:
 * only typed facts (the record reference, the cockpit's setting, results of the tools the
 * model asked the browser to run, the visitor's decision on the last move, approvals, and at
 * most one short message). The server keeps the conversation in signed state and builds the
 * tools from the record (api/preflight_api/routes/guide.py).
 *
 * Browser tools run here against the same PredictionProvider the dials use. set_knob is
 * guarded here too: never a pinned knob, never outside the knob's values, never into a
 * setting that is tight, fails or isn't trusted, and at most one per round.
 */
import { fmt } from "../lib/fmt";
import { knobChoices, targetFor } from "../workspace/record/model";
import type { KnobValue, LoadedRecord, Objective, Prediction } from "../workspace/record/types";
import { fitOf } from "../workspace/status";
import type { ToolContext } from "./tools";

/** Same origin: /v1 is served by the web app's own Pages Function (functions/v1). */
const API_BASE = "";
/** Safety stop for browser-tool rounds in one visitor action (the server caps rounds too). */
export const MAX_BROWSER_ROUNDS = 6;

export type Awaiting = "results" | "decision" | "approval" | "visitor";

export interface GuideCall {
	call_id: string;
	name: string;
	arguments: Record<string, unknown>;
	runs_in: "browser" | "approval";
}

export interface RoundOut {
	text: string;
	withheld: string[] | null;
	calls: GuideCall[];
	awaiting: Awaiting;
	state: string;
	provenance: Record<string, unknown>;
}

export interface Setting {
	knobs: Record<string, KnobValue>;
	pinned: string[];
	objective: { kind: NonNullable<Objective["kind"]>; target_id?: string };
}

export type Decision = { knob_id: string; action: "applied" | "reverted" | "pinned" | "changed"; value?: KnobValue };

type Readout = { point: number | null; lo: number | null; hi: number | null; unit: string; status: string; reason?: string };

export type SetKnobRefusal = "pinned" | "not_in_domain" | "would_be_tight" | "would_fail" | "not_trusted" | "one_per_round";

export type ToolOut =
	| { tool: "predict"; call_id: string; readouts: Record<string, Readout> }
	| {
			tool: "sensitivity";
			call_id: string;
			ranked: Array<{
				knob_id: string;
				impact: number;
				direction: "increase" | "decrease" | "change";
				best_value: KnobValue;
				delta: number | null;
				fit: "fits" | "tight";
			}>;
	  }
	| {
			tool: "set_knob";
			call_id: string;
			moved: boolean;
			knob_id: string;
			from_value: KnobValue | null;
			to_value: KnobValue | null;
			readouts: Record<string, Readout>;
			refused: SetKnobRefusal | null;
	  }
	| { tool: "highlight"; call_id: string; nodes: string[] }
	| { tool: "pin"; call_id: string; knob_id: string; pinned: boolean };

export class GuideFallback extends Error {
	constructor(
		message: string,
		readonly reason: string,
	) {
		super(message);
	}
}

export type Poster = (body: unknown) => Promise<Response>;

const defaultPost: Poster = (body) =>
	fetch(`${API_BASE}/v1/guide/turns`, {
		method: "POST",
		credentials: "same-origin",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(body),
	});

const HEADLINE = ["memory", "time", "cost", "risk"] as const;
const CODE = /^[a-z][a-z0-9_.]{0,47}$/;

function readouts(record: LoadedRecord, p: Prediction): Record<string, Readout> {
	const ids = new Set(
		record.predictions.targets.filter((t) => t.readout && (HEADLINE as readonly string[]).includes(t.readout)).map((t) => t.id),
	);
	for (const extra of ["throughput.tokens_per_s", "tokens.total"]) if (record.predictions.targets.some((t) => t.id === extra)) ids.add(extra);
	const out: Record<string, Readout> = {};
	for (const id of ids) {
		const e = p[id];
		if (!e) continue;
		const abstain = e.status === "abstain" || e.point === null;
		out[id] = {
			point: abstain ? null : e.point,
			lo: abstain ? null : (e.lo ?? null),
			hi: abstain ? null : (e.hi ?? null),
			unit: e.unit ?? "",
			status: abstain ? "abstain" : e.status,
			...(e.reason && CODE.test(e.reason) ? { reason: e.reason } : {}),
		};
	}
	return out;
}

/** The browser half of the set_knob guard. Returns the refusal, or null when the move may run. */
export function guardSetKnob(ctx: ToolContext, knobId: string, value: unknown): SetKnobRefusal | null {
	const { record, provider, knobs, pinned } = ctx;
	const knob = record.knobs.find((k) => k.id === knobId);
	if (!knob) return "not_in_domain";
	if (pinned.has(knobId)) return "pinned";
	if (!knobChoices(record, knob).some((v) => v === value)) return "not_in_domain";
	const next = { ...knobs, [knobId]: value as KnobValue };
	const p = provider.predict(record, next);
	const mem = targetFor(record, "memory");
	const memEst = mem ? p[mem.id] : undefined;
	if (memEst && (memEst.status === "abstain" || memEst.point === null)) return "not_trusted";
	const fit = fitOf(record, p, next);
	if (fit === "fails") return "would_fail";
	if (fit === "tight") return "would_be_tight";
	return null;
}

/** Run one browser tool the model asked for and return its typed result. */
export function runBrowserCall(call: GuideCall, ctx: ToolContext, movedThisRound: boolean): ToolOut {
	const { record, provider, actions, knobs, pinned } = ctx;
	const a = call.arguments;
	switch (call.name) {
		case "predict": {
			const raw = (a.knobs ?? {}) as Record<string, KnobValue>;
			const known = Object.fromEntries(Object.entries(raw).filter(([k]) => record.knobs.some((x) => x.id === k)));
			return { tool: "predict", call_id: call.call_id, readouts: readouts(record, provider.predict(record, { ...knobs, ...known })) };
		}
		case "sensitivity": {
			const kind = (a.objective as Setting["objective"]["kind"]) ?? "min_time";
			const moves = provider.moves(record, { kind }, { pinned, knobs, includeTight: true }).slice(0, 8);
			const top = Math.max(...moves.map((m) => Math.abs(m.relative ?? m.delta)), 1e-12);
			return {
				tool: "sensitivity",
				call_id: call.call_id,
				ranked: moves.map((m) => {
					const from = knobs[m.knobId];
					const direction: "increase" | "decrease" | "change" =
						typeof m.to === "number" && typeof from === "number" ? (m.to > from ? "increase" : "decrease") : "change";
					return {
						knob_id: m.knobId,
						impact: Math.min(1, Math.abs(m.relative ?? m.delta) / top),
						direction,
						best_value: m.to,
						delta: Number.isFinite(m.delta) ? m.delta : null,
						fit: m.fit,
					};
				}),
			};
		}
		case "set_knob": {
			const id = String(a.knob_id);
			const from = knobs[id] ?? null;
			const refused: SetKnobRefusal | null = movedThisRound ? "one_per_round" : guardSetKnob(ctx, id, a.value);
			if (refused) {
				return { tool: "set_knob", call_id: call.call_id, moved: false, knob_id: id, from_value: from, to_value: null, readouts: {}, refused };
			}
			const to = a.value as KnobValue;
			actions.setKnob(id, to, "guide");
			if (from !== null) actions.propose({ knobId: id, from, to });
			actions.openKnob(id);
			return {
				tool: "set_knob",
				call_id: call.call_id,
				moved: true,
				knob_id: id,
				from_value: from,
				to_value: to,
				readouts: readouts(record, provider.predict(record, { ...knobs, [id]: to })),
				refused: null,
			};
		}
		case "highlight": {
			const known = new Set(record.graph.nodes.map((n) => n.id));
			const nodes = ((a.nodes as string[]) ?? []).filter((n) => known.has(n)).slice(0, 24);
			actions.highlight(nodes);
			return { tool: "highlight", call_id: call.call_id, nodes };
		}
		case "pin": {
			const id = String(a.knob_id);
			const on = Boolean(a.pinned);
			actions.pin(id, on);
			return { tool: "pin", call_id: call.call_id, knob_id: id, pinned: on };
		}
		default:
			throw new Error(`not a browser tool: ${call.name}`);
	}
}

export function traceText(out: ToolOut): string {
	switch (out.tool) {
		case "set_knob":
			return out.refused ? `refused: ${out.refused}` : `moved ${out.knob_id} ${String(out.from_value)} → ${String(out.to_value)}`;
		case "sensitivity":
			return out.ranked.map((r) => `${r.knob_id}→${String(r.best_value)}${r.fit === "tight" ? " (tight)" : ""}`).join(", ") || "nothing improves the aim";
		case "predict":
			return Object.entries(out.readouts)
				.map(([k, r]) => `${k}=${r.point === null ? "not trusted" : fmt(r.point, r.unit)}`)
				.join(", ");
		case "highlight":
			return out.nodes.join(", ");
		case "pin":
			return `${out.knob_id} ${out.pinned ? "pinned" : "unpinned"}`;
	}
}

export interface LiveView {
	text: string;
	withheld: string[] | null;
	awaiting: Awaiting;
	approvals: GuideCall[];
	provenance: Array<Record<string, unknown>>;
}

export type TourContext = ToolContext & { objective: Setting["objective"]; language?: string };

let traceSeq = 1_000_000;

/**
 * One visitor's live tour on one record. Each public method sends one visitor action and then
 * keeps going on its own (running browser tools and sending their results) until the model
 * waits for the visitor again: a decision on its move, an approval, or a new message.
 */
export class LiveTour {
	private state: string | null = null;
	private view: LiveView = { text: "", withheld: null, awaiting: "visitor", approvals: [], provenance: [] };

	constructor(
		private readonly recordRef: string,
		private readonly ctx: () => TourContext,
		private readonly onUpdate: (v: LiveView) => void,
		private readonly post: Poster = defaultPost,
	) {}

	get current(): LiveView {
		return this.view;
	}

	start(): Promise<LiveView> {
		return this.send({ start: true });
	}

	ask(message: string): Promise<LiveView> {
		return this.send({ message: message.slice(0, 500) });
	}

	decide(decision: Decision): Promise<LiveView> {
		return this.send({ decision });
	}

	approve(callId: string, approved: boolean): Promise<LiveView> {
		return this.send({ approvals: [{ call_id: callId, approved }] });
	}

	private setting(c: TourContext): Setting {
		return { knobs: { ...c.knobs }, pinned: [...c.pinned], objective: c.objective };
	}

	private async send(extra: Record<string, unknown>): Promise<LiveView> {
		let body: Record<string, unknown> = extra;
		const answered = new Set(((extra.approvals as Array<{ call_id: string }> | undefined) ?? []).map((x) => x.call_id));
		for (let round = 0; round <= MAX_BROWSER_ROUNDS; round++) {
			const c = this.ctx();
			const res = await this.post({
				record: this.recordRef,
				setting: this.setting(c),
				...(this.state ? { state: this.state } : {}),
				...(c.language ? { language: c.language } : {}),
				...body,
			});
			if (!res.ok) {
				const err = (await res.json().catch(() => null)) as { error?: { message?: string; fallback?: string; reason?: string; code?: string } } | null;
				if (res.status === 409) this.state = null;
				if (err?.error?.fallback === "rule_based") throw new GuideFallback(err.error.message ?? "", err.error.reason ?? "unknown");
				throw new GuideFallback(err?.error?.message ?? `The guide request failed (${res.status}).`, err?.error?.code ?? "http_error");
			}
			const out = (await res.json()) as RoundOut;
			this.state = out.state;
			this.view = {
				text: out.text || (out.withheld ? "" : this.view.text),
				withheld: out.withheld,
				awaiting: out.awaiting,
				approvals: [...this.view.approvals.filter((a) => !answered.has(a.call_id)), ...out.calls.filter((x) => x.runs_in === "approval")],
				provenance: [...this.view.provenance, out.provenance],
			};
			this.onUpdate(this.view);
			const browser = out.calls.filter((x) => x.runs_in === "browser");
			if (out.awaiting !== "results" || !browser.length) return this.view;
			let moved = false;
			const results: ToolOut[] = [];
			for (const call of browser) {
				const ctx = this.ctx();
				const r = runBrowserCall(call, ctx, moved);
				if (r.tool === "set_knob" && r.moved) moved = true;
				ctx.actions.trace({
					id: ++traceSeq,
					tool: call.name,
					args: JSON.stringify(call.arguments).slice(0, 90),
					result: traceText(r),
					ok: !(r.tool === "set_knob" && r.refused),
				});
				results.push(r);
			}
			body = { results };
		}
		throw new GuideFallback("The live guide went round in circles.", "too_many_rounds");
	}
}
