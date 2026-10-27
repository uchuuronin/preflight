/**
 * Rule-based tour: the guide when Muse Spark is unavailable or the visitor's live turns are
 * spent, labelled "Rule-based tour" on screen. It plans from the provider's ranked moves and
 * re-plans after every Apply, Revert or Pin, so a step never goes stale.
 */
import { hasKey } from "../lib/i18n";
import type { Move, PredictionProvider } from "../workspace/provider/PredictionProvider";
import { failureTarget } from "../workspace/record/model";
import { FAILS_AT, FITS_BELOW } from "../workspace/record/terms";
import type { KnobValue, LoadedRecord, Objective } from "../workspace/record/types";

/**
 * Map nodes a knob acts on, by knob id. Requested as knobs[].affects (docs/requests.md);
 * until A's records carry it, this table and the record's findings decide.
 */
const KNOB_NODES: Record<string, string[]> = {
	eval_batch_size: ["eval"],
	eval_no_grad: ["eval"],
	batch_size: ["data", "model"],
	seq_len: ["model", "lm_head"],
	gradient_checkpointing: ["model"],
	gpu_type: ["model"],
	gpu: ["model"],
	lora_r: ["optimizer"],
	precision: ["model"],
	packing: ["data"],
	num_workers: ["data"],
	grad_accum: ["optimizer"],
};

/**
 * Knobs that change what the model learns, not just how fast or how cheaply. The tour offers
 * them only to clear a failure. Requested as knobs[].changes_results.
 */
const CHANGES_RESULTS = new Set(["seq_len", "lora_r", "precision", "max_new_tokens", "group_size", "packing"]);

export function changesResults(knobId: string): boolean {
	return CHANGES_RESULTS.has(knobId);
}

export function nodesForKnob(record: LoadedRecord, knobId: string): string[] {
	const fromFindings = record.findings.filter((f) => f.knob_id === knobId && f.node_id).map((f) => f.node_id as string);
	const known = new Set(record.graph.nodes.map((n) => n.id));
	return [...new Set([...fromFindings, ...(KNOB_NODES[knobId] ?? [])])].filter((n) => known.has(n));
}

export interface TourStep {
	/**
	 * move: apply a "fits" move. probe_offer: a "tight" setting would save enough to be worth
	 * measuring first (the person's click). blocked_by_pin, nothing_safe, done: as named.
	 */
	kind: "move" | "probe_offer" | "blocked_by_pin" | "nothing_safe" | "done";
	move?: Move;
	pinnedKnob?: string;
	nodes: string[];
	textKey: string;
	whyKey: string;
}

export interface TourContext {
	record: LoadedRecord;
	provider: PredictionProvider;
	knobs: Readonly<Record<string, KnobValue>>;
	pinned: ReadonlySet<string>;
	objective: Objective;
	visited: ReadonlySet<string>;
}

function keyFor(knobId: string, kind: "line" | "why", fixing: boolean): string {
	const specific = `guide.rule.${knobId}.${fixing ? "fix" : "save"}.${kind}`;
	if (hasKey(specific)) return specific;
	return `guide.rule.generic.${fixing ? "fix" : "save"}.${kind}`;
}

/** A "tight" setting is worth a probe when it improves the aim by at least this share. */
export const PROBE_WORTH = 0.1;

function riskNow(ctx: Pick<TourContext, "record" | "provider" | "knobs">): number {
	const fail = failureTarget(ctx.record);
	if (!fail) return 0;
	return ctx.provider.predict(ctx.record, ctx.knobs)[fail]?.point ?? 0;
}

/** Any real chance of failing (not "fits"): the tour deals with it before saving money or time. */
export function isFailing(ctx: Pick<TourContext, "record" | "provider" | "knobs">): boolean {
	return riskNow(ctx) >= FITS_BELOW;
}

/** Likely to fail ("fails"): the knob-specific lines that say the run would stop apply. */
export function likelyFails(ctx: Pick<TourContext, "record" | "provider" | "knobs">): boolean {
	return riskNow(ctx) >= FAILS_AT;
}

function candidates(ctx: TourContext, pinned: ReadonlySet<string>): Move[] {
	const failing = isFailing(ctx);
	return ctx.provider
		.moves(ctx.record, ctx.objective, { pinned, knobs: ctx.knobs })
		.filter((m) => m.fit === "fits")
		.filter((m) => !ctx.visited.has(m.knobId))
		.filter((m) => failing || !changesResults(m.knobId));
}

/** The best "tight" setting worth measuring: beats the best "fits" move by PROBE_WORTH. */
function probeOffer(ctx: TourContext, bestFits: Move | undefined): Move | undefined {
	const failing = isFailing(ctx);
	if (failing) return undefined;
	const offers = ctx.provider
		.moves(ctx.record, ctx.objective, { pinned: ctx.pinned, knobs: ctx.knobs, includeTight: true })
		.filter((m) => m.fit === "tight" && !ctx.visited.has(`probe:${m.knobId}`) && !ctx.visited.has(m.knobId))
		.filter((m) => !changesResults(m.knobId));
	// moves() returns only improvements here, so |relative| is the share the aim improves by.
	const base = Math.abs(bestFits?.relative ?? 0);
	return offers.find((m) => Math.abs(m.relative ?? 0) - base >= PROBE_WORTH);
}

export function nextStep(ctx: TourContext): TourStep {
	const failing = isFailing(ctx);
	const top = candidates(ctx, ctx.pinned)[0];
	const offer = probeOffer(ctx, top);
	if (offer) {
		return {
			kind: "probe_offer",
			move: offer,
			nodes: nodesForKnob(ctx.record, offer.knobId),
			textKey: "guide.probe_offer",
			whyKey: "guide.probe_offer.why",
		};
	}
	if (top) {
		const fixing = top.clearsFailure || failing;
		// Knob-specific fix lines say the run would stop; they fit only when it likely would.
		const specific = !fixing || likelyFails(ctx);
		return {
			kind: "move",
			move: top,
			nodes: nodesForKnob(ctx.record, top.knobId),
			textKey: specific ? keyFor(top.knobId, "line", fixing) : "guide.rule.generic.fix.line",
			whyKey: specific ? keyFor(top.knobId, "why", fixing) : "guide.rule.generic.fix.why",
		};
	}
	// Nothing movable helps. If a pin is in the way, say which one and what releasing it buys.
	if (ctx.pinned.size) {
		const blocker = candidates(ctx, new Set()).find((m) => ctx.pinned.has(m.knobId));
		if (blocker) {
			return {
				kind: "blocked_by_pin",
				move: blocker,
				pinnedKnob: blocker.knobId,
				nodes: nodesForKnob(ctx.record, blocker.knobId),
				textKey: failing ? "guide.pin_blocks_fix" : "guide.pin_blocks_saving",
				whyKey: "guide.pin_blocks.why",
			};
		}
	}
	if (failing) {
		const nodes = ctx.record.findings.filter((f) => f.severity !== "info" && f.node_id).map((f) => f.node_id as string);
		// The setting most worth measuring: the move that lowers the risk most, even if still tight.
		const toMeasure = ctx.provider
			.moves(ctx.record, ctx.objective, { pinned: ctx.pinned, knobs: ctx.knobs, includeTight: true })
			.find((m) => m.fit === "tight");
		return {
			kind: "nothing_safe",
			...(toMeasure ? { move: toMeasure } : {}),
			nodes: [...new Set([...nodes, ...(toMeasure ? nodesForKnob(ctx.record, toMeasure.knobId) : [])])],
			textKey: "guide.nothing_safe",
			whyKey: "guide.nothing_safe.why",
		};
	}
	return { kind: "done", nodes: [], textKey: "guide.done", whyKey: "guide.done.why" };
}

export function plannedSteps(ctx: TourContext): number {
	return candidates(ctx, ctx.pinned).length;
}

export const MAX_TOUR_STEPS = 5;
