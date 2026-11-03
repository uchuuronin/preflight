import { decodeF16, decodeMask } from "../../../../schema/ts/encoding";
import { capacityBytes, currentKnobs, failureTarget, knobChoices, objectiveTarget, targetFor } from "../record/model";
import { pOomFromMemory } from "../record/oom";
import { observedAt } from "../record/observed";
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

/**
 * Smallest positive lo and largest hi over every trusted cell of every grid for a target: the
 * full range this record can show, used for the dials' fixed scale (record/scale.ts).
 */
export function gridRange(record: LoadedRecord, targetId: string): { lo: number; hi: number } | null {
	let lo = Number.POSITIVE_INFINITY;
	let hi = 0;
	for (const g of record.predictions.grids) {
		if (g.target_id !== targetId) continue;
		const d = decode(g);
		for (let i = 0; i < d.hi.length; i++) {
			if (d.abstain[i]) continue;
			const h = d.hi[i] as number;
			const l = d.lo[i] as number;
			if (Number.isFinite(h) && h > hi) hi = h;
			if (Number.isFinite(l) && l > 0 && l < lo) lo = l;
		}
	}
	return hi > 0 ? { lo: Number.isFinite(lo) ? lo : hi, hi } : null;
}

/** True when every axis value is one of the grid's own points (no interpolation needed). */
function onGridPoints(grid: Grid, knobs: Readonly<Record<string, KnobValue>>): boolean {
	return grid.axes.every((a) => {
		const v = knobs[a.knob_id];
		return v !== undefined && a.values.some((x) => same(x, v));
	});
}

function same(a: KnobValue | undefined, b: KnobValue | undefined): boolean {
	if (typeof a === "number" && typeof b === "number") return Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(a));
	return a === b;
}

/**
 * The knobs a grid's target doesn't depend on (contract 0.6.4 `Grid.independent_of`: the learning rate,
 * warm-up, weight decay, clipping, LoRA alpha, PPO clip ranges, seed), so the grid answers at any value
 * of them. B emits it only with `engine.carry_free_knobs` on; absent, every knob off the axes must match.
 */
export function freeKnobs(grid: Pick<Grid, "independent_of">): ReadonlySet<string> {
	return new Set(grid.independent_of ?? []);
}

/** A target's family, as B's `independent_of` reads it: memory.*, time.*, cost.*; risk.oom follows memory. */
function familyOf(targetId: string): string {
	return targetId === "risk.oom" ? "memory" : targetId.split(".", 1)[0]!;
}

const familyCache = new WeakMap<LoadedRecord, Map<string, ReadonlySet<string>>>();

/**
 * The knobs free for a target, whether or not it has a grid of its own (time.step has B's estimate at the
 * record's setting but no grid): the knobs every grid of its family names in `independent_of` (none when a
 * grid of the family names none, or the family has no grid). B carries a target's value across these
 * knobs, so the record's own estimate still answers when only they moved (review, Nov 2).
 */
export function freeFor(record: LoadedRecord, targetId: string): ReadonlySet<string> {
	let byFamily = familyCache.get(record);
	if (!byFamily) {
		// Free for the family: every grid of it either names the knob free or has it on an axis (B never
		// lists an axis knob, so a grid over lr mustn't take lr out of the set), and at least one names it.
		const grids = new Map<string, Grid[]>();
		for (const g of record.predictions.grids) {
			const fam = familyOf(g.target_id);
			grids.set(fam, [...(grids.get(fam) ?? []), g]);
		}
		byFamily = new Map();
		for (const [fam, gs] of grids) {
			const named = new Set(gs.flatMap((g) => g.independent_of ?? []));
			const free = [...named].filter((k) => gs.every((g) => freeKnobs(g).has(k) || g.axes.some((a) => a.knob_id === k)));
			byFamily.set(fam, new Set(free));
		}
		familyCache.set(record, byFamily);
	}
	return byFamily.get(familyOf(targetId)) ?? new Set();
}

/**
 * A grid answers a setting when every knob it doesn't put on an axis sits where the grid
 * holds it: at its `fixed` value, or, for a knob the grid never mentions, at the record's
 * current value. A knob moved away from either means this grid can't answer.
 */
function applies(grid: Grid, record: LoadedRecord, knobs: Readonly<Record<string, KnobValue>>): boolean {
	const onAxis = new Set(grid.axes.map((a) => a.knob_id));
	const free = freeKnobs(grid);
	const fixed = grid.fixed ?? {};
	for (const k of record.knobs) {
		if (onAxis.has(k.id) || free.has(k.id)) continue;
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
	/** Every axis's value inside its knob's probed range (numbers only: a categorical has no "between"). */
	const between = () =>
		grid.axes.every((a) => {
			const k = specs.get(a.knob_id);
			const v = knobs[a.knob_id];
			return typeof v === "number" && !!k?.probed_range && inProbedRange(record, k, v);
		});
	for (const axis of grid.axes) {
		const v = knobs[axis.knob_id];
		if (v === undefined) return abstained(target, grid.model_id, "knob_missing");
		const b = bracket(axis.values, v, axis.scale === "log");
		if (!b) return abstained(target, grid.model_id, between() ? "between_measured" : "outside_grid");
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
	// A setting inside every knob's probed range that still has no usable cell is a gap in the
	// grid (B is filling those), not a setting beyond what was measured (PM, Nov 2).
	// Only between grid points (never on one: the engine abstained on that very cell for a reason
	// it doesn't export) and never where a probe ran out of memory (review, Nov 2).
	const onPoint = grid.axes.every((a) => a.values.some((x) => same(x, knobs[a.knob_id])));
	if (abstain) return abstained(target, grid.model_id, !onPoint && between() && !failedAt(record, grid, knobs) ? "between_measured" : "outside_trusted_range");
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

/** A probe ran out of memory at this setting of the grid's knobs. */
function failedAt(record: LoadedRecord, grid: Grid, knobs: Readonly<Record<string, KnobValue>>): boolean {
	return record.probes.some((p) => p.status === "oom" && grid.axes.every((a) => same(p.knobs[a.knob_id] ?? record.knobs.find((k) => k.id === a.knob_id)?.current, knobs[a.knob_id])));
}

/** Every knob moved off the record's setting is a number inside its probed range: the probes
 * bracket this setting, and only the precomputed values are missing (PM, Nov 2). */
function movedWithinProbed(record: LoadedRecord, knobs: Readonly<Record<string, KnobValue>>, free: ReadonlySet<string> = new Set()): boolean {
	const moved = record.knobs.filter((k) => !free.has(k.id) && !same(knobs[k.id], k.current));
	if (!moved.length || !moved.every((k) => typeof knobs[k.id] === "number" && !!k.probed_range && inProbedRange(record, k, knobs[k.id]!))) return false;
	// A probe that ran out of memory at these values measured them: not a gap.
	return !record.probes.some((p) => p.status === "oom" && moved.every((k) => same(p.knobs[k.id], knobs[k.id])));
}

function atCurrent(record: LoadedRecord, knobs: Readonly<Record<string, KnobValue>>, free: ReadonlySet<string> = new Set()): boolean {
	return record.knobs.every((k) => free.has(k.id) || same(knobs[k.id], k.current));
}

export class GridProvider implements PredictionProvider {
	readonly name = "grid" as const;

	ready(): boolean {
		return true;
	}

	predict(record: LoadedRecord, knobs: Readonly<Record<string, KnobValue>>): Prediction {
		const full = { ...currentKnobs(record), ...knobs };
		const out: Prediction = {};
		const fail = failureTarget(record);
		let deriveRisk = false;
		for (const target of record.predictions.targets) {
			// "Measured" only when a probe at exactly these settings observed this target itself;
			// totals and risks are always computed (record/observed.ts).
			const seen = observedAt(record, full, target.id);
			// At the record's own settings, B's estimates are authoritative (they may come from
			// an exact model that sees more than any grid does, like the eval pass), except that
			// a "measured" label needs an observing probe here too.
			const atHere = record.predictions.estimates?.[target.id];
			// "Here" passes over the knobs B carries this target across (0.6.4 independent_of): with only the
			// learning rate moved, the step time is still B's estimate, labelled predicted (no probe ran there).
			const free = freeFor(record, target.id);
			if (atHere && atCurrent(record, full, free)) {
				out[target.id] = atHere.status === "measured" && !seen ? { ...atHere, status: "predicted" } : atHere;
				continue;
			}
			const grids = record.predictions.grids.filter((g) => g.target_id === target.id);
			const grid = grids.find((g) => applies(g, record, full));
			if (grid) {
				const est = interpolate(grid, target, record, full);
				out[target.id] = seen && est.status === "predicted" ? { ...est, status: "measured" } : est;
				if (target.id === fail && !onGridPoints(grid, full)) deriveRisk = true;
				continue;
			}
			out[target.id] = abstained(target, grids[0]?.model_id ?? atHere?.model_id ?? "none", movedWithinProbed(record, full, free) ? "between_measured" : "off_grid");
		}
		// Between grid points, P(OOM) comes from interpolated memory, never from interpolating
		// the probability itself (record/oom.ts).
		if (deriveRisk && fail) this.riskFromMemory(record, full, out, fail);
		return out;
	}

	private riskFromMemory(record: LoadedRecord, knobs: Readonly<Record<string, KnobValue>>, out: Prediction, fail: string): void {
		const memT = targetFor(record, "memory");
		const risk = out[fail];
		if (!memT || memT.unit !== "bytes" || !risk) return;
		const mem = out[memT.id];
		if (!mem || mem.status === "abstain" || mem.point === null) {
			out[fail] = { ...risk, point: null, lo: null, hi: null, status: "abstain", reason: mem?.reason ?? "off_grid" };
			return;
		}
		const p = pOomFromMemory(mem, capacityBytes(record, knobs));
		if (p === null) return;
		out[fail] = {
			...risk,
			point: p,
			lo: p,
			hi: p,
			status: mem.status === "measured" ? "predicted" : mem.status,
			reason: "from_memory",
		};
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
				if (opts.allow && !opts.allow(knob.id, v, p)) continue;
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
