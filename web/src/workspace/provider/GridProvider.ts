import type {
	Estimate,
	FlightRecord,
	Grid,
	KnobImpact,
	KnobSpec,
	KnobValue,
	Objective,
	Prediction,
	TargetSpec,
} from "../record/types";
import { decodeFloat16, decodeMask } from "./float16";
import {
	OBJECTIVE_TARGET,
	currentKnobs,
	type PredictionProvider,
	type SensitivityOptions,
} from "./PredictionProvider";

interface DecodedGrid {
	point: Float32Array;
	lo: Float32Array;
	hi: Float32Array;
	abstain: Uint8Array;
	strides: number[];
}

const decoded = new WeakMap<Grid, DecodedGrid>();

/** Above this predicted OOM probability a configuration counts as failing. */
export const OOM_LIMIT = 0.1;

// Ranking offsets for a failing run: clearing the failure beats any objective change,
// and reducing its probability beats any objective change short of clearing it.
const CLEARS = -1e12;
const REDUCES = 1e9;

function decode(grid: Grid): DecodedGrid {
	let d = decoded.get(grid);
	if (d) return d;
	const strides: number[] = new Array(grid.axes.length).fill(1);
	for (let i = grid.axes.length - 2; i >= 0; i--) {
		strides[i] = (strides[i + 1] ?? 1) * (grid.axes[i + 1]?.values.length ?? 1);
	}
	d = {
		point: decodeFloat16(grid.point),
		lo: decodeFloat16(grid.lo),
		hi: decodeFloat16(grid.hi),
		abstain: decodeMask(grid.abstain),
		strides,
	};
	decoded.set(grid, d);
	return d;
}

/** Corner indices and weights along one axis, or null when the value is off the axis. */
function bracket(values: Array<number | string | boolean>, v: KnobValue): Array<[number, number]> | null {
	if (typeof v !== "number" || typeof values[0] !== "number") {
		const i = values.indexOf(v as never);
		return i < 0 ? null : [[i, 1]];
	}
	const nums = values as number[];
	const first = nums[0] ?? 0;
	const last = nums[nums.length - 1] ?? 0;
	if (v < first - 1e-9 || v > last + 1e-9) return null;
	for (let i = 0; i < nums.length - 1; i++) {
		const a = nums[i] ?? 0;
		const b = nums[i + 1] ?? 0;
		if (v >= a && v <= b) {
			if (b === a) return [[i, 1]];
			const w = (v - a) / (b - a);
			if (w < 1e-9) return [[i, 1]];
			if (w > 1 - 1e-9) return [[i + 1, 1]];
			return [
				[i, 1 - w],
				[i + 1, w],
			];
		}
	}
	return [[nums.length - 1, 1]];
}

function sameValue(a: KnobValue | undefined, b: KnobValue | undefined): boolean {
	if (typeof a === "number" && typeof b === "number") return Math.abs(a - b) < 1e-9;
	return a === b;
}

/** A grid answers this knob setting when every knob it holds fixed is at its fixed value. */
function gridMatches(grid: Grid, knobs: Readonly<Record<string, KnobValue>>): boolean {
	for (const [id, v] of Object.entries(grid.fixed)) {
		if (id in knobs && !sameValue(knobs[id], v)) return false;
	}
	return true;
}

function inProbedRange(spec: KnobSpec | undefined, v: KnobValue): boolean {
	const r = spec?.probed_range;
	if (!r) return true;
	if ("min" in r) return typeof v === "number" && v >= r.min - 1e-9 && v <= r.max + 1e-9;
	return r.values.includes(v as never);
}

function abstained(target: TargetSpec, model: string, reason: string): Estimate {
	return {
		point: Number.NaN,
		lo: Number.NaN,
		hi: Number.NaN,
		unit: target.unit,
		status: "abstain",
		model_id: model,
		reason,
		source: "grid",
	};
}

export function interpolate(
	grid: Grid,
	target: TargetSpec,
	knobs: Readonly<Record<string, KnobValue>>,
	specs: ReadonlyMap<string, KnobSpec>,
): Estimate {
	const d = decode(grid);
	const perAxis: Array<Array<[number, number]>> = [];
	let extrapolated = false;
	for (const axis of grid.axes) {
		const v = knobs[axis.knob_id] ?? specs.get(axis.knob_id)?.value;
		if (v === undefined) return abstained(target, grid.model_id, "knob_missing");
		const b = bracket(axis.values, v);
		if (!b) return abstained(target, grid.model_id, "outside_grid");
		if (!inProbedRange(specs.get(axis.knob_id), v)) extrapolated = true;
		perAxis.push(b);
	}
	let point = 0;
	let lo = 0;
	let hi = 0;
	let abstain = false;
	const walk = (axis: number, offset: number, weight: number): void => {
		if (axis === perAxis.length) {
			if (weight <= 0) return;
			point += weight * (d.point[offset] ?? Number.NaN);
			lo += weight * (d.lo[offset] ?? Number.NaN);
			hi += weight * (d.hi[offset] ?? Number.NaN);
			if (d.abstain[offset]) abstain = true;
			return;
		}
		for (const [i, w] of perAxis[axis] ?? []) walk(axis + 1, offset + i * (d.strides[axis] ?? 1), weight * w);
	};
	walk(0, 0, 1);
	const s = grid.scale ?? 1;
	point *= s;
	lo *= s;
	hi *= s;
	return {
		point,
		lo: Math.min(lo, point),
		hi: Math.max(hi, point),
		unit: target.unit,
		status: abstain ? "abstain" : extrapolated ? "extrapolated" : "predicted",
		model_id: grid.model_id,
		...(abstain ? { reason: "outside_trusted_range" } : {}),
		source: "grid",
	};
}

function matchesProbe(record: FlightRecord, knobs: Readonly<Record<string, KnobValue>>): boolean {
	return record.probes.some(
		(p) => p.status === "ok" && Object.entries(p.knobs).every(([k, v]) => sameValue(knobs[k], v)),
	);
}

export class GridProvider implements PredictionProvider {
	readonly name = "grid" as const;

	ready(): boolean {
		return true;
	}

	predict(record: FlightRecord, knobs: Readonly<Record<string, KnobValue>>): Prediction {
		const specs = new Map(record.knobs.map((k) => [k.id, k]));
		const full = { ...currentKnobs(record), ...knobs };
		const measured = matchesProbe(record, full);
		const out: Prediction = {};
		for (const target of record.predictions.targets) {
			const grids = record.predictions.grids.filter((g) => g.target_id === target.id);
			const grid = grids.find((g) => gridMatches(g, full));
			if (!grid) {
				out[target.id] = abstained(target, grids[0]?.model_id ?? "none", "off_grid");
				continue;
			}
			const est = interpolate(grid, target, full, specs);
			out[target.id] = measured && est.status === "predicted" ? { ...est, status: "measured" } : est;
		}
		return out;
	}

	/** Values the grids can answer for one knob, given the others. */
	candidates(record: FlightRecord, knobId: string): KnobValue[] {
		const spec = record.knobs.find((k) => k.id === knobId);
		const fromGrids = new Set<KnobValue>();
		for (const g of record.predictions.grids) {
			for (const a of g.axes) if (a.knob_id === knobId) for (const v of a.values) fromGrids.add(v);
		}
		if (fromGrids.size) return [...fromGrids];
		if (spec && "values" in spec.domain && spec.domain.values) return [...spec.domain.values];
		return [];
	}

	sensitivity(record: FlightRecord, objective: Objective, opts: SensitivityOptions = {}): KnobImpact[] {
		const targetId = OBJECTIVE_TARGET[objective];
		const knobs = { ...currentKnobs(record), ...(opts.knobs ?? {}) };
		const base = this.predict(record, knobs);
		const baseValue = base[targetId]?.point ?? Number.NaN;
		const baseOom = base.p_oom?.point ?? 0;
		const failing = baseOom > OOM_LIMIT;
		const ranked: Array<KnobImpact & { score: number }> = [];
		for (const spec of record.knobs) {
			if (opts.pinned?.has(spec.id)) continue;
			let best: (KnobImpact & { score: number }) | null = null;
			for (const v of this.candidates(record, spec.id)) {
				if (sameValue(v, knobs[spec.id])) continue;
				const p = this.predict(record, { ...knobs, [spec.id]: v });
				const est = p[targetId];
				if (!est || est.status === "abstain" || !Number.isFinite(est.point)) continue;
				const oom = p.p_oom?.point ?? 0;
				// A move that makes the run likely to fail is never an improvement.
				if (oom > OOM_LIMIT && !(failing && oom < baseOom)) continue;
				const delta = est.point - baseValue;
				// When the run is already failing, every move that clears the failure ranks
				// first and counts as equally safe, so the objective picks between them; moves
				// that only reduce the risk come next, ordered by how much.
				const score = !failing ? delta : oom <= OOM_LIMIT ? CLEARS + delta : (oom - baseOom) * REDUCES + delta;
				if (!best || score < best.score) {
					best = { knob_id: spec.id, delta, best_value: v, target_id: targetId, score };
				}
			}
			if (best && (failing ? best.score < 0 : best.delta < 0)) ranked.push(best);
		}
		return ranked.sort((a, b) => a.score - b.score).map(({ score: _score, ...impact }) => impact);
	}
}
