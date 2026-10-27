/**
 * The UI asks a PredictionProvider for every number it shows, never a model directly.
 * Mirrors python/preflight/engine/api.py (predict, build_grids, sensitivity).
 *
 * GridProvider answers instantly from the record's precomputed grids. PyodideProvider
 * (wired in with B) answers exactly, including off-grid knob combinations, once loaded.
 * HybridProvider asks the grid first and upgrades to exact when available.
 */
import type { FlightRecord, KnobImpact, KnobValue, Objective, Prediction } from "../record/types";

export interface SensitivityOptions {
	/** Knob ids the guide must not move. */
	pinned?: ReadonlySet<string>;
	/** Current knob values; defaults to the record's. */
	knobs?: Readonly<Record<string, KnobValue>>;
}

export interface PredictionProvider {
	readonly name: "grid" | "pyodide" | "hybrid";
	ready(): boolean;
	predict(record: FlightRecord, knobs: Readonly<Record<string, KnobValue>>): Prediction;
	sensitivity(record: FlightRecord, objective: Objective, opts?: SensitivityOptions): KnobImpact[];
}

/** Objective to the target it minimises (or maximises). */
export const OBJECTIVE_TARGET: Record<Objective, string> = {
	cost: "total_cost",
	time: "time_to_finish",
	headroom: "peak_memory",
	quality: "divergence_risk",
};

export function currentKnobs(record: FlightRecord): Record<string, KnobValue> {
	const out: Record<string, KnobValue> = {};
	for (const k of record.knobs) out[k.id] = k.value;
	return out;
}
