/**
 * Rule-based tour: the guide's fallback when Muse Spark is unavailable or the visitor's
 * guide budget is spent, labelled "rule-based tour" on screen. The next step is chosen
 * from sensitivity() after every action, so it never goes stale after an Apply.
 */
import type { FlightRecord, KnobImpact, KnobValue, Objective } from "../workspace/record/types";
import type { PredictionProvider } from "../workspace/provider/PredictionProvider";
import { OOM_LIMIT } from "../workspace/provider/GridProvider";
import { hasKey } from "../lib/i18n";

/**
 * Which map nodes a knob acts on. Requested as knobs[].affects in the schema
 * (docs/requests.md); this table is the fallback until A's records carry it.
 */
const KNOB_NODES: Record<string, string[]> = {
	eval_batch_size: ["eval"],
	batch_size: ["data", "blocks"],
	seq_len: ["blocks", "head"],
	grad_ckpt: ["blocks", "backward"],
	gpu: ["blocks", "backward"],
	lora_rank: ["optimizer"],
	precision: ["blocks"],
	num_workers: ["data"],
	group_size: ["generate", "reward"],
	max_new_tokens: ["generate"],
	concurrency: ["llm"],
};

/**
 * Knobs that change what the model learns, not just how fast or how cheaply. The tour only
 * offers them to clear a failure, never as a saving. Requested as knobs[].changes_results.
 */
const CHANGES_RESULTS = new Set(["seq_len", "lora_rank", "precision", "max_new_tokens", "group_size"]);

export function changesResults(knobId: string): boolean {
	return CHANGES_RESULTS.has(knobId);
}

export function nodesForKnob(knobId: string): string[] {
	return KNOB_NODES[knobId] ?? [];
}

export interface TourStep {
	kind: "move" | "blocked_by_pin" | "done";
	knobId?: string;
	to?: KnobValue;
	impact?: KnobImpact;
	nodes: string[];
	/** i18n key for the one-line explanation, and its "why?" expansion. */
	textKey: string;
	whyKey: string;
	/** The pinned knob in the way, for blocked_by_pin. */
	pinnedKnob?: string;
}

export interface TourContext {
	record: FlightRecord;
	provider: PredictionProvider;
	knobs: Readonly<Record<string, KnobValue>>;
	pinned: ReadonlySet<string>;
	objective: Objective;
	/** Knobs already toured, so a reverted step isn't offered again. */
	visited: ReadonlySet<string>;
}

function keyFor(knobId: string, kind: "line" | "why", failing: boolean): string {
	const specific = `guide.rule.${knobId}.${failing ? "fix" : "save"}.${kind}`;
	if (hasKey(specific)) return specific;
	const any = `guide.rule.${knobId}.${kind}`;
	return hasKey(any) ? any : `guide.rule.generic.${kind}`;
}

export function nextStep(ctx: TourContext): TourStep {
	const { record, provider, knobs, pinned, objective, visited } = ctx;
	const now = provider.predict(record, knobs);
	const failing = (now.p_oom?.point ?? 0) > OOM_LIMIT;
	const ranked = provider
		.sensitivity(record, objective, { pinned, knobs })
		.filter((i) => !visited.has(i.knob_id))
		.filter((i) => failing || objective === "quality" || !changesResults(i.knob_id));
	const top = ranked[0];
	if (top) {
		return {
			kind: "move",
			knobId: top.knob_id,
			to: top.best_value,
			impact: top,
			nodes: nodesForKnob(top.knob_id),
			textKey: keyFor(top.knob_id, "line", failing),
			whyKey: keyFor(top.knob_id, "why", failing),
		};
	}
	// Nothing movable helps. If a pin is in the way, say which one and what releasing it buys.
	if (pinned.size) {
		const unpinned = provider.sensitivity(record, objective, { knobs }).filter((i) => pinned.has(i.knob_id));
		const blocker = unpinned[0];
		if (blocker) {
			return {
				kind: "blocked_by_pin",
				pinnedKnob: blocker.knob_id,
				to: blocker.best_value,
				impact: blocker,
				nodes: nodesForKnob(blocker.knob_id),
				textKey: failing ? "guide.pin_blocks_fix" : "guide.pin_blocks_saving",
				whyKey: "guide.pin_blocks.why",
			};
		}
	}
	return { kind: "done", nodes: [], textKey: failing ? "guide.done.still_failing" : "guide.done", whyKey: "guide.done.why" };
}

/** How many more steps the tour expects, for "Step 2 of 4". */
export function remainingSteps(ctx: TourContext): number {
	const failing = (ctx.provider.predict(ctx.record, ctx.knobs).p_oom?.point ?? 0) > OOM_LIMIT;
	return ctx.provider
		.sensitivity(ctx.record, ctx.objective, { pinned: ctx.pinned, knobs: ctx.knobs })
		.filter((i) => !ctx.visited.has(i.knob_id))
		.filter((i) => failing || ctx.objective === "quality" || !changesResults(i.knob_id)).length;
}

export const MAX_TOUR_STEPS = 5;
