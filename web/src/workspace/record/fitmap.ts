/**
 * The fit map: two knobs against each other, every combination marked fits, tight, fails or
 * not trusted, with the out-of-memory boundary drawn between them.
 *
 * Each cell is a setting the cockpit can fly to, and its status is the same one the dials show
 * there (status.ts fitOf), so the map and the readouts never disagree. The boundary between two
 * neighbouring cells is placed by interpolating memory between them first and solving where
 * P(OOM) derived from it (record/oom.ts) crosses B's cut-offs, never by interpolating the
 * probability itself.
 */
import type { PredictionProvider } from "../provider/PredictionProvider";
import { type Fit, fitOf, trusted } from "../status";
import { capacityBytes, knobChoices, targetFor } from "./model";
import { pOomFromMemory } from "./oom";
import { FAILS_AT, FITS_BELOW } from "./terms";
import type { Knob, KnobValue, LoadedRecord } from "./types";

export type CellStatus = Fit | "abstain";

export interface FitCell {
	x: KnobValue;
	y: KnobValue;
	status: CellStatus;
	/** P(OOM) the dials show here, when trusted. */
	risk: number | null;
	peak: number | null;
	capacity: number | null;
	here: boolean;
}

export interface FitMapData {
	x: Knob;
	y: Knob;
	xs: KnobValue[];
	ys: KnobValue[];
	/** rows[yi][xi] */
	rows: FitCell[][];
	/**
	 * Per row, where along the row the boundary sits, in column units (2.4 = 40% of the way
	 * from column 2 to column 3): `fits` where "fits" ends, `fails` where "fails" begins.
	 * null when the row never crosses it.
	 */
	frontier: { fits: Array<number | null>; fails: Array<number | null> };
}

/** At most this many columns or rows; more would make cells too small to read or tap. */
export const MAX_SIDE = 8;

/** Bisection steps when placing a boundary between two cells: 2^-16 of a cell is plenty. */
const SOLVE_STEPS = 16;

function numeric(k: Knob): boolean {
	return k.kind === "int" || k.kind === "float";
}

/** Knobs the memory grids put on an axis: the only ones whose effect on memory is predicted. */
function memoryAxes(record: LoadedRecord): Set<string> {
	const memT = targetFor(record, "memory");
	const out = new Set<string>();
	for (const g of record.predictions.grids) {
		if (g.target_id !== memT?.id) continue;
		for (const a of g.axes) out.add(a.knob_id);
	}
	return out;
}

/** Candidate knobs for an axis: numeric, on a memory grid axis, with at least two choices. */
export function fitMapKnobs(record: LoadedRecord): Knob[] {
	const axes = memoryAxes(record);
	return record.knobs.filter((k) => numeric(k) && axes.has(k.id) && knobChoices(record, k).length > 1);
}

/**
 * The two knobs that move memory the most from the current setting (largest spread of the
 * predicted peak across each knob's choices), x first. null when fewer than two qualify.
 */
export function defaultAxes(
	record: LoadedRecord,
	provider: PredictionProvider,
	knobs: Readonly<Record<string, KnobValue>>,
): [string, string] | null {
	const memT = targetFor(record, "memory");
	if (!memT) return null;
	const spread = fitMapKnobs(record).map((k) => {
		let lo = Number.POSITIVE_INFINITY;
		let hi = 0;
		for (const v of knobChoices(record, k)) {
			const e = provider.predict(record, { ...knobs, [k.id]: v })[memT.id];
			if (!trusted(e)) continue;
			lo = Math.min(lo, e.point);
			hi = Math.max(hi, e.point);
		}
		return { id: k.id, spread: hi > 0 && Number.isFinite(lo) ? hi / Math.max(lo, 1) : 0 };
	});
	spread.sort((a, b) => b.spread - a.spread);
	const [a, b] = spread;
	return a && b ? [a.id, b.id] : null;
}

/** Choices to show along an axis: all of them, or MAX_SIDE spread evenly and keeping the current one. */
function sample(values: KnobValue[], current: KnobValue | undefined): KnobValue[] {
	if (values.length <= MAX_SIDE) return values;
	const picked = new Set<number>();
	for (let i = 0; i < MAX_SIDE; i++) picked.add(Math.round((i * (values.length - 1)) / (MAX_SIDE - 1)));
	const ci = values.findIndex((v) => v === current);
	if (ci >= 0 && !picked.has(ci)) {
		// swap in the current value for its nearest sampled neighbour
		const nearest = [...picked].sort((p, q) => Math.abs(p - ci) - Math.abs(q - ci))[0];
		if (nearest !== undefined) picked.delete(nearest);
		picked.add(ci);
	}
	return [...picked].sort((p, q) => p - q).map((i) => values[i] as KnobValue);
}

function cellStatus(record: LoadedRecord, provider: PredictionProvider, knobs: Record<string, KnobValue>): Omit<FitCell, "x" | "y" | "here"> {
	const p = provider.predict(record, knobs);
	const memT = targetFor(record, "memory");
	const mem = memT ? p[memT.id] : undefined;
	const capacity = capacityBytes(record, knobs);
	const riskT = record.predictions.targets.find((t) => t.id === "risk.oom");
	const r = riskT ? p[riskT.id] : undefined;
	if (!trusted(mem) && !trusted(r)) return { status: "abstain", risk: null, peak: null, capacity };
	return {
		status: fitOf(record, p, knobs),
		risk: trusted(r) ? r.point : null,
		peak: trusted(mem) ? mem.point : null,
		capacity,
	};
}

/**
 * Where between columns a and a+1 of a row P(OOM), derived from memory interpolated by the
 * provider, crosses `cut`. Numeric axes are walked in log space (the grids' own spacing for
 * batch and sequence length); falls back to the midpoint when the derived risk doesn't bracket
 * the cut (the dials' risk and memory disagree there, as at a record's own setting when its
 * estimate sees more than the grid).
 */
function crossing(
	record: LoadedRecord,
	provider: PredictionProvider,
	base: Record<string, KnobValue>,
	xId: string,
	x0: KnobValue,
	x1: KnobValue,
	cut: number,
): number {
	const memT = targetFor(record, "memory");
	if (!memT || typeof x0 !== "number" || typeof x1 !== "number") return 0.5;
	const log = x0 > 0 && x1 > 0;
	const at = (f: number): number | null => {
		const v = log ? Math.exp(Math.log(x0) + f * (Math.log(x1) - Math.log(x0))) : x0 + f * (x1 - x0);
		const k = { ...base, [xId]: v };
		return pOomFromMemory(provider.predict(record, k)[memT.id], capacityBytes(record, k));
	};
	const p0 = at(0);
	const p1 = at(1);
	if (p0 === null || p1 === null || (p0 - cut) * (p1 - cut) > 0) return 0.5;
	let lo = 0;
	let hi = 1;
	const rising = p1 > p0;
	for (let i = 0; i < SOLVE_STEPS; i++) {
		const mid = (lo + hi) / 2;
		const pm = at(mid);
		if (pm === null) return 0.5;
		if (pm < cut === rising) lo = mid;
		else hi = mid;
	}
	return (lo + hi) / 2;
}

const RANK: Record<CellStatus, number> = { fits: 0, tight: 1, fails: 2, unknown: -1, abstain: -1 };

export function fitMap(
	record: LoadedRecord,
	provider: PredictionProvider,
	knobs: Readonly<Record<string, KnobValue>>,
	xId: string,
	yId: string,
): FitMapData | null {
	const x = record.knobs.find((k) => k.id === xId);
	const y = record.knobs.find((k) => k.id === yId);
	if (!x || !y || xId === yId) return null;
	const xs = sample(knobChoices(record, x), knobs[xId]);
	const ys = sample(knobChoices(record, y), knobs[yId]);
	const rows = ys.map((yv) =>
		xs.map((xv) => ({
			x: xv,
			y: yv,
			here: knobs[xId] === xv && knobs[yId] === yv,
			...cellStatus(record, provider, { ...knobs, [xId]: xv, [yId]: yv }),
		})),
	);
	const fits: Array<number | null> = [];
	const fails: Array<number | null> = [];
	rows.forEach((row, yi) => {
		const base = { ...knobs, [yId]: ys[yi] as KnobValue };
		let fitsAt: number | null = null;
		let failsAt: number | null = null;
		for (let i = 0; i + 1 < row.length; i++) {
			const a = RANK[row[i]!.status];
			const b = RANK[row[i + 1]!.status];
			if (a < 0 || b < 0) continue;
			// "fits" ends where the rank leaves 0; "fails" starts where it reaches 2 (either way along the row).
			if (fitsAt === null && (a === 0) !== (b === 0)) fitsAt = i + crossing(record, provider, base, xId, row[i]!.x, row[i + 1]!.x, FITS_BELOW);
			if (failsAt === null && (a === 2) !== (b === 2)) failsAt = i + crossing(record, provider, base, xId, row[i]!.x, row[i + 1]!.x, FAILS_AT);
		}
		fits.push(fitsAt);
		fails.push(failsAt);
	});
	return { x, y, xs, ys, rows, frontier: { fits, fails } };
}
