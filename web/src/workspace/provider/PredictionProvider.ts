/**
 * The UI asks a PredictionProvider for every number it shows, never a model directly.
 * Mirrors python/preflight/engine/api.py (predict, build_grids, sensitivity).
 *
 * GridProvider answers instantly from the record's precomputed grids. PyodideProvider
 * (wired in with B) answers exactly, off-grid combinations included, once loaded.
 */
import type { KnobImpact, KnobValue, LoadedRecord, Objective, Prediction } from "../record/types";

export interface SensitivityOptions {
	/** Knob ids the guide must not move. */
	pinned?: ReadonlySet<string>;
	/** Current knob values; defaults to the record's. */
	knobs?: Readonly<Record<string, KnobValue>>;
	/**
	 * Also return moves that land in "tight" (FITS_BELOW <= P(OOM) < FAILS_AT), flagged
	 * `fit: "tight"`. The guide never applies these; it offers to measure them first.
	 */
	includeTight?: boolean;
	/**
	 * The visitor's limits (guide/limits.ts): a value this returns false for is never a move, so
	 * the knob's next-best value inside the limits is ranked instead.
	 */
	allow?: (knobId: string, value: KnobValue, after: Prediction) => boolean;
}

/** One candidate move with its effect, for the guide's wording. C-internal, richer than KnobImpact. */
export interface Move {
	knobId: string;
	to: KnobValue;
	/** Objective target and its change (in the target's unit; negative means lower). */
	targetId: string;
	delta: number;
	/** The move clears a predicted failure (OOM): the run goes from any risk to "fits". */
	clearsFailure: boolean;
	/** Fit after the move: "fits" moves may be applied; "tight" ones need a probe first. */
	fit: "fits" | "tight";
	/** Failure probability after the move, when the record has a failure target. */
	risk: number | null;
	/** Change relative to the current value of the objective (delta / |current|), when known. */
	relative: number | null;
}

export interface PredictionProvider {
	readonly name: "grid" | "pyodide" | "hybrid";
	ready(): boolean;
	predict(record: LoadedRecord, knobs: Readonly<Record<string, KnobValue>>): Prediction;
	/** The contract's sensitivity: knobs ranked by impact on the objective. */
	sensitivity(record: LoadedRecord, objective: Objective, opts?: SensitivityOptions): KnobImpact[];
	/** The best single move per knob, ranked, with real deltas. */
	moves(record: LoadedRecord, objective: Objective, opts?: SensitivityOptions): Move[];
}
