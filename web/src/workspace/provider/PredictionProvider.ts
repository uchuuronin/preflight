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
}

/** One candidate move with its effect, for the guide's wording. C-internal, richer than KnobImpact. */
export interface Move {
	knobId: string;
	to: KnobValue;
	/** Objective target and its change (in the target's unit; negative means lower). */
	targetId: string;
	delta: number;
	/** The move clears a predicted failure (OOM). */
	clearsFailure: boolean;
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
