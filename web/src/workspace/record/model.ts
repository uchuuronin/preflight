/**
 * Read-only helpers over a loaded record: which target feeds which readout, the knob values
 * a control offers, the run's length, GPU capacity, and whether the numbers are a fixture.
 * No prediction happens here; numbers come from the PredictionProvider.
 */
import type { Knob, KnobValue, LoadedRecord, ObjectiveKind, Readout, Target } from "./types";

export function currentKnobs(record: LoadedRecord): Record<string, KnobValue> {
	const out: Record<string, KnobValue> = {};
	for (const k of record.knobs) out[k.id] = k.current;
	return out;
}

export function targetFor(record: LoadedRecord, readout: Readout): Target | undefined {
	return record.predictions.targets.find((t) => t.readout === readout);
}

export function target(record: LoadedRecord, id: string): Target | undefined {
	return record.predictions.targets.find((t) => t.id === id);
}

/** Objective kind to the target it optimises, and which way is better. */
export function objectiveTarget(
	record: LoadedRecord,
	kind: ObjectiveKind,
	customTarget?: string | null,
): { id: string; better: "lower" | "higher" } | null {
	const byReadout = (r: Readout) => targetFor(record, r)?.id;
	switch (kind) {
		case "min_cost":
			return wrap(byReadout("cost"), "lower");
		case "min_time":
			return wrap(byReadout("time"), "lower");
		case "fit_memory":
			return wrap(byReadout("memory"), "lower");
		case "min_risk":
			return wrap(byReadout("risk"), "lower");
		case "max_throughput":
			return wrap(target(record, "throughput.tokens_per_s")?.id, "higher");
		case "custom":
			return wrap(customTarget ?? undefined, "lower");
		case "max_quality_per_usd":
			return wrap(byReadout("cost"), "lower");
	}
}

function wrap(id: string | undefined, better: "lower" | "higher") {
	return id ? { id, better } : null;
}

/** The OOM-probability target, when the lens has one. */
export function failureTarget(record: LoadedRecord): string | null {
	if (target(record, "risk.oom")) return "risk.oom";
	return null;
}

/** Values a knob's control offers: grid axis values when gridded, else the domain. */
export function knobChoices(record: LoadedRecord, knob: Knob): KnobValue[] {
	const fromGrids = new Set<KnobValue>();
	for (const g of record.predictions.grids) {
		for (const a of g.axes) if (a.knob_id === knob.id) for (const v of a.values) fromGrids.add(v);
	}
	if (knob.kind === "bool") return [false, true];
	const values = knob.domain?.values ?? [];
	if (values.length) return [...values];
	if (fromGrids.size) {
		const all = [...fromGrids];
		if (!all.includes(knob.current)) all.push(knob.current);
		return typeof all[0] === "number" ? (all as number[]).sort((a, b) => a - b) : all;
	}
	return ladder(knob);
}

/** A sensible ladder for a numeric knob with only a min and max: powers of two when log. */
function ladder(knob: Knob): KnobValue[] {
	const min = knob.domain?.min ?? (typeof knob.current === "number" ? knob.current : 0);
	const max = knob.domain?.max ?? (typeof knob.current === "number" ? knob.current : 0);
	const out = new Set<number>();
	if (knob.domain?.log && min > 0) {
		for (let v = 2 ** Math.ceil(Math.log2(min)); v <= max; v *= 2) out.add(v);
	} else {
		const step = knob.domain?.step ?? Math.max(1, (max - min) / 8);
		for (let v = min; v <= max + 1e-9; v += step) out.add(knob.kind === "int" ? Math.round(v) : Number(v.toFixed(6)));
	}
	if (typeof knob.current === "number") out.add(knob.current);
	return [...out].sort((a, b) => a - b);
}

export interface RunPlan {
	totalSteps: number | null;
	evalEvery: number | null;
	checkpointEvery: number | null;
}

/** The run's shape from the script map (A reads it from the code). */
export function runPlan(record: LoadedRecord): RunPlan {
	const t = record.job.script_map?.training;
	return {
		totalSteps: t?.max_steps ?? null,
		evalEvery: t?.eval_every ?? null,
		checkpointEvery: t?.checkpoint_every ?? null,
	};
}

const GIB = 1024 ** 3;

/**
 * Memory per GPU for the capacity line when the record carries no `memory.capacity`
 * target (requested from B). Display only: names follow the spec table's canonical names.
 */
const GPU_MEMORY: Record<string, number> = {
	"A100-80GB": 80 * GIB,
	"A100-40GB": 40 * GIB,
	"H100-SXM-80GB": 80 * GIB,
	"H100-PCIE-80GB": 80 * GIB,
	H100: 80 * GIB,
	H200: 141 * GIB,
	L40S: 48 * GIB,
	L4: 24 * GIB,
	A10G: 24 * GIB,
	T4: 16 * GIB,
	MI300X: 192 * GIB,
	MI250: 128 * GIB,
	"RTX-4090": 24 * GIB,
};

export function capacityBytes(record: LoadedRecord, knobs: Readonly<Record<string, KnobValue>>): number | null {
	const gpu = typeof knobs.gpu === "string" ? knobs.gpu : record.env?.gpu?.name;
	if (!gpu) return null;
	if (gpu === record.env?.gpu?.name && record.env.gpu.memory_bytes) return record.env.gpu.memory_bytes;
	return GPU_MEMORY[gpu] ?? null;
}

/** True when every prediction comes from a fixture model, so the UI says "illustrative". */
export function isFixture(record: LoadedRecord): boolean {
	const ids = new Set<string>();
	for (const g of record.predictions.grids) ids.add(g.model_id);
	for (const e of Object.values(record.predictions.estimates ?? {})) if (e) ids.add(e.model_id);
	if (!ids.size) return false;
	return [...ids].every((id) => /fixture/i.test(id));
}
