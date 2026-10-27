/**
 * Graph nodes and edges name what animates them: a prediction target (`risk.oom`,
 * `time.phase.forward`) or a measured probe value (`probe.data.padding_fraction`,
 * `memory.by_term.logits`). Targets come from the provider at the current knobs; probe values
 * come from a probe, and the UI says which one ("measured at batch 4"), because a measurement
 * at other settings is not a prediction for these.
 */
import type { KnobValue, LoadedRecord, Prediction, Probe } from "../record/types";

export interface MetricValue {
	value: number;
	unit: string;
	kind: "prediction" | "measured_here" | "measured_elsewhere";
	status: string;
	probeId?: string;
	probeKnobs?: Record<string, KnobValue | undefined>;
}

const PROBE_UNITS: Record<string, string> = {
	padding_fraction: "fraction",
	loader_wait_fraction: "fraction",
	tokens_per_s: "tokens/s",
	samples_per_s: "samples/s",
	mfu: "fraction",
	truncation_rate: "fraction",
	zero_spread_fraction: "fraction",
};

function walk(obj: unknown, path: string[]): number | null {
	let cur: unknown = obj;
	for (const p of path) {
		if (cur === null || typeof cur !== "object") return null;
		cur = (cur as Record<string, unknown>)[p];
	}
	return typeof cur === "number" && Number.isFinite(cur) ? cur : null;
}

/** Probe path for a metric key, or null when the key is not a probe metric. */
function probePath(key: string): string[] | null {
	if (key.startsWith("probe.")) return key.slice(6).split(".");
	if (key.startsWith("memory.by_term.")) return key.split(".");
	return null;
}

function distance(record: LoadedRecord, probe: Probe, knobs: Readonly<Record<string, KnobValue>>): number {
	let d = 0;
	for (const k of record.knobs) {
		const a = probe.knobs[k.id] ?? k.current;
		const b = knobs[k.id];
		if (typeof a === "number" && typeof b === "number" && a > 0 && b > 0) d += Math.abs(Math.log(a / b));
		else if (a !== b) d += 3;
	}
	return d;
}

/** The successful probe closest to a knob setting (log distance on numeric knobs). */
export function nearestProbe(record: LoadedRecord, knobs: Readonly<Record<string, KnobValue>>): Probe | null {
	let best: Probe | null = null;
	let bestD = Number.POSITIVE_INFINITY;
	for (const p of record.probes) {
		if (p.status !== "ok" && p.status !== "cached") continue;
		const d = distance(record, p, knobs);
		if (d < bestD) {
			best = p;
			bestD = d;
		}
	}
	return best;
}

export function resolveMetric(
	record: LoadedRecord,
	knobs: Readonly<Record<string, KnobValue>>,
	prediction: Prediction,
	key: string,
): MetricValue | null {
	const est = prediction[key];
	if (est) {
		if (est.point === null || est.status === "abstain") return null;
		return { value: est.point, unit: est.unit, kind: "prediction", status: est.status };
	}
	const path = probePath(key);
	if (!path) return null;
	const probe = nearestProbe(record, knobs);
	if (!probe) return null;
	const value = walk(probe, path);
	if (value === null) return null;
	const here = distance(record, probe, knobs) === 0;
	const leaf = path[path.length - 1];
	return {
		value,
		unit: path[0] === "memory" ? "bytes" : (PROBE_UNITS[leaf] ?? "none"),
		kind: here ? "measured_here" : "measured_elsewhere",
		status: "measured",
		probeId: probe.id,
		probeKnobs: probe.knobs,
	};
}
