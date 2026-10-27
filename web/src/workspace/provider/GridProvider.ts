import { decodeF16, decodeMask } from "../../../../schema/ts/encoding";
import { currentKnobs, failureTarget, knobChoices, objectiveTarget } from "../record/model";
import { FAILS_AT, FITS_BELOW } from "../record/terms";
import type {
	Estimate,
	Grid,
	Knob,
	KnobImpact,
	KnobValue,
	LoadedRecord,
	Objective,
	Prediction,
	Target,
} from "../record/types";
import type { Move, PredictionProvider, SensitivityOptions } from "./PredictionProvider";

/**
 * The guide applies a move only when it lands in "fits" (P(OOM) < FITS_BELOW); a move into
 * "tight" is offered as "measure it first". Chosen by Monte Carlo over five thresholds
 * (docs/decisions/guide-moves.md): same crash rate as fits-only, a tenth of its extra cost.
 */
export const SAFE_BELOW = FITS_BELOW;

// Ranking offsets for a failing run: clearing the failure beats any objective change, and
// reducing its probability beats any objective change short of clearing it.
const CLEARS = -1e12;
const REDUCES = 1e9;

interface Decoded {
	point: Float64Array;
	lo: Float64Array;
	hi: Float64Array;
	abstain: Uint8Array;
	shape: number[];
}

const cache = new WeakMap<Grid, Decoded>();

function decode(grid: Grid): Decoded {
	const hit = cache.get(grid);
	if (hit) return hit;
	const shape = grid.axes.map((a) => a.values.length);
	const n = shape.reduce((a, b) => a * b, 1);
	const scale = grid.scale ?? 1;
	const d: Decoded = {
		point: decodeF16(grid.point ?? "", n, scale),
		lo: decodeF16(grid.lo ?? "", n, scale),
		hi: decodeF16(grid.hi ?? "", n, scale),
		abstain: grid.abstain ? decodeMask(grid.abstain, n) : new Uint8Array(n),
		shape,
	};
	cache.set(grid, d);
	return d;
}

function same(a: KnobValue | undefined, b: KnobValue | undefined): boolean {
	if (typeof a === "number" && typeof b === "number") return Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(a));
	return a === b;
}

/**
 * A grid answers a setting when every knob it doesn't put on an axis sits where the grid
 * holds it: at its `fixed` value, or, for a knob the grid never mentions, at the record's
 * current value. A knob moved away from either means this grid can't answer.
 */
function applies(grid: Grid, record: LoadedRecord, knobs: Readonly<Record<string, KnobValue>>): boolean {
	const onAxis = new Set(grid.axes.map((a) => a.knob_id));
	const fixed = grid.fixed ?? {};
	for (const k of record.knobs) {
		if (onAxis.has(k.id)) continue;
		const held = k.id in fixed ? fixed[k.id] : k.current;
		if (!same(knobs[k.id], held)) return false;
	}
	return true;
}

/** Corner indices and weights along one axis, or null when the value is off the axis. */
function bracket(
	values: ReadonlyArray<KnobValue>,
	v: KnobValue,
	log: boolean,
): Array<[number, number]> | null {
	if (typeof v !== "number" || typeof values[0] !== "number") {
		const i = values.findIndex((x) => same(x, v));
		return i < 0 ? null : [[i, 1]];
	}
	const nums = values as number[];
	const tx = (x: number) => (log && x > 0 ? Math.log(x) : x);
	const first = nums[0];
	const last = nums[nums.length - 1];
	if (v < first - 1e-9 || v > last + 1e-9) return null;
	for (let i = 0; i < nums.length - 1; i++) {
		const a = nums[i];
		const b = nums[i + 1];
		if (v >= a - 1e-9 && v <= b + 1e-9) {
			if (same(v, a)) return [[i, 1]];
			if (same(v, b)) return [[i + 1, 1]];
			const w = (tx(v) - tx(a)) / (tx(b) - tx(a));
			return [
				[i, 1 - w],
				[i + 1, w],
			];
		}
	}
	return [[nums.length - 1, 1]];
}

export function abstained(target: Target, modelId: string, reason: string): Estimate {
	return { point: null, lo: null, hi: null, unit: target.unit, status: "abstain", model_id: modelId, source: "grid", reason };
}

/** Values of a knob seen in successful probes (a probe holds unlisted knobs at current). */
function probedValues(record: LoadedRecord, knob: Knob): KnobValue[] {
	const out: KnobValue[] = [];
	for (const p of record.probes) {
		if (p.status !== "ok" && p.status !== "cached") continue;
		out.push(p.knobs[knob.id] ?? knob.current);
	}
	return out;
}

function inProbedRange(record: LoadedRecord, knob: Knob | undefined, v: KnobValue): boolean {
	if (!knob) return true;
	if (typeof v === "number" && knob.probed_range) {
		return v >= knob.probed_range.min - 1e-9 && v <= knob.probed_range.max + 1e-9;
	}
	const seen = probedValues(record, knob);
	if (!seen.length) return same(v, knob.current);
	return seen.some((s) => same(s, v));
}

export function interpolate(
	grid: Grid,
	target: Target,
	record: LoadedRecord,
	knobs: Readonly<Record<string, KnobValue>>,
): Estimate {
	const d = decode(grid);
	const specs = new Map(record.knobs.map((k) => [k.id, k]));
	const perAxis: Array<Array<[number, number]>> = [];
	let extrapolated = false;
	for (const axis of grid.axes) {
		const v = knobs[axis.knob_id];
		if (v === undefined) return abstained(target, grid.model_id, "knob_missing");
		const b = bracket(axis.values, v, axis.scale === "log");
		if (!b) return abstained(target, grid.model_id, "outside_grid");
		if (!inProbedRange(record, specs.get(axis.knob_id), v)) extrapolated = true;
		perAxis.push(b);
	}
	const strides = d.shape.map((_, i) => d.shape.slice(i + 1).reduce((a, b) => a * b, 1));
	let point = 0;
	let lo = 0;
	let hi = 0;
	let abstain = false;
	const walk = (axis: number, offset: number, weight: number): void => {
		if (axis === perAxis.length) {
			if (weight <= 0) return;
			point += weight * d.point[offset];
			lo += weight * d.lo[offset];
			hi += weight * d.hi[offset];
			if (d.abstain[offset]) abstain = true;
			return;
		}
		for (const [i, w] of perAxis[axis]) walk(axis + 1, offset + i * strides[axis], weight * w);
	};
	walk(0, 0, 1);
	if (abstain) return abstained(target, grid.model_id, "outside_trusted_range");
	if (![point, lo, hi].every(Number.isFinite)) return abstained(target, grid.model_id, "off_scale");
	return {
		point,
		lo: Math.min(lo, point),
		hi: Math.max(hi, point),
		unit: grid.unit || target.unit,
		status: extrapolated ? "extrapolated" : "predicted",
		model_id: grid.model_id,
		source: "grid",
		level: 0.9,
		...(extrapolated ? { reason: "outside_probed_range" } : {}),
	};
}

function matchesProbe(record: LoadedRecord, knobs: Readonly<Record<string, KnobValue>>): boolean {
	return record.probes.some(
		(p) =>
			(p.status === "ok" || p.status === "cached") &&
			record.knobs.every((k) => same(knobs[k.id], p.knobs[k.id] ?? k.current)),
	);
}

function atCurrent(record: LoadedRecord, knobs: Readonly<Record<string, KnobValue>>): boolean {
	return record.knobs.every((k) => same(knobs[k.id], k.current));
}

export class GridProvider implements PredictionProvider {
	readonly name = "grid" as const;

	ready(): boolean {
		return true;
	}

	predict(record: LoadedRecord, knobs: Readonly<Record<string, KnobValue>>): Prediction {
		const full = { ...currentKnobs(record), ...knobs };
		const measured = matchesProbe(record, full);
		const here = atCurrent(record, full);
		const out: Prediction = {};
		for (const target of record.predictions.targets) {
			// At the record's own settings, B's estimates are authoritative (they may come from
			// an exact model that sees more than any grid does, like the eval pass).
			const atHere = record.predictions.estimates?.[target.id];
			if (here && atHere) {
				out[target.id] = atHere;
				continue;
			}
			const grids = record.predictions.grids.filter((g) => g.target_id === target.id);
			const grid = grids.find((g) => applies(g, record, full));
			if (grid) {
				const est = interpolate(grid, target, record, full);
				out[target.id] = measured && est.status === "predicted" ? { ...est, status: "measured" } : est;
				continue;
			}
			out[target.id] = abstained(target, grids[0]?.model_id ?? atHere?.model_id ?? "none", "off_grid");
		}
		return out;
	}

	moves(record: LoadedRecord, objective: Objective, opts: SensitivityOptions = {}): Move[] {
		const goal = objectiveTarget(record, objective.kind ?? "min_time", objective.target_id);
		if (!goal) return [];
		const fail = failureTarget(record);
		const knobs = { ...currentKnobs(record), ...(opts.knobs ?? {}) };
		const base = this.predict(record, knobs);
		const baseValue = base[goal.id]?.point;
		const baseRisk = fail ? (base[fail]?.point ?? 0) : 0;
		const failing = baseRisk >= SAFE_BELOW;
		const sign = goal.better === "higher" ? -1 : 1;
		const ranked: Array<Move & { score: number }> = [];
		for (const knob of record.knobs) {
			if (opts.pinned?.has(knob.id)) continue;
			let best: (Move & { score: number }) | null = null;
			for (const v of knobChoices(record, knob)) {
				if (same(v, knobs[knob.id])) continue;
				const p = this.predict(record, { ...knobs, [knob.id]: v });
				const est = p[goal.id];
				if (!est || est.status === "abstain" || est.point === null) continue;
				const risk = fail ? (p[fail]?.point ?? 0) : 0;
				if (fail && p[fail]?.status === "abstain") continue;
				// Never into "fails". Into "tight" only when asked, and only as an offer to measure.
				if (risk >= FAILS_AT) continue;
				const tight = risk >= SAFE_BELOW;
				if (tight && !opts.includeTight && !(failing && risk < baseRisk)) continue;
				const delta = baseValue === null || baseValue === undefined ? 0 : est.point - baseValue;
				const clears = failing && !tight;
				const score = !failing ? sign * delta : clears ? CLEARS + sign * delta : (risk - baseRisk) * REDUCES + sign * delta;
				const relative = baseValue ? delta / Math.abs(baseValue) : null;
				const fit = tight ? "tight" : "fits";
				const cand = { knobId: knob.id, to: v, targetId: goal.id, delta, clearsFailure: clears, fit, risk: fail ? risk : null, relative, score } as const;
				// Per knob: the best "fits" move wins over any "tight" one, then the best score.
				if (!best || (best.fit === "tight" && fit === "fits") || (best.fit === fit && score < best.score)) best = { ...cand };
			}
			if (best && (failing ? best.score < 0 : sign * best.delta < 0)) ranked.push(best);
		}
		return ranked.sort((a, b) => a.score - b.score).map(({ score: _s, ...m }) => m);
	}

	sensitivity(record: LoadedRecord, objective: Objective, opts: SensitivityOptions = {}): KnobImpact[] {
		const moves = this.moves(record, objective, opts);
		if (!moves.length) return [];
		const knobs = { ...currentKnobs(record), ...(opts.knobs ?? {}) };
		const top = Math.max(...moves.map((m) => Math.abs(m.delta)), 1e-12);
		return moves.map((m) => {
			const from = knobs[m.knobId];
			const direction =
				typeof m.to === "number" && typeof from === "number" ? (m.to > from ? "increase" : "decrease") : "either";
			return {
				knob_id: m.knobId,
				// Normalised to [0, 1] against the biggest move; clearing a failure counts as the biggest.
				impact: m.clearsFailure ? 1 : Math.min(1, Math.abs(m.delta) / top),
				direction,
				best_value: m.to,
				reason: m.clearsFailure ? "clears_predicted_failure" : "improves_objective",
			};
		});
	}
}
