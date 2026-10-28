/**
 * Fixed dial scales, one per record and readout, so a needle's position means the same thing
 * all session (no running maximum that rescales as the visitor flies).
 *
 * A limit the visitor set (budget, wall-time cap) or a probability gets a linear scale to that
 * limit. Everything else gets a log scale over the whole range the record's grids can show:
 * chosen by Monte Carlo over five fixed scales (docs/decisions/dial-scale.md), it was the only
 * one with no visit off the scale and none reading as empty, and it moves the needle visibly on
 * the most single-knob steps.
 */
import { gridRange } from "../provider/GridProvider";
import { capacityBytes, currentKnobs, knobChoices, targetFor } from "./model";
import { GPU_KNOBS } from "./terms";
import type { LoadedRecord } from "./types";

export interface DialScale {
	kind: "linear" | "log";
	/** Left end (0 for linear). */
	lo: number;
	/** Right end: full scale. */
	hi: number;
	source: "limit" | "probability" | "grid" | "estimate";
}

/**
 * The log scale spans at most this many times from right end to left end (1,000x, three
 * decades). Source: guess, so one knob step stays visible on the dev record; the Monte Carlo
 * in docs/decisions/dial-scale.md reruns on the first real records from A's job store.
 */
export const LOG_SPAN_MAX = 1000;

/**
 * Without a grid (a record with estimates only), the scale runs from lo / 4 to hi x 4. Source:
 * guess, replaced by the grid range once B ships grids for that lens.
 */
export const ESTIMATE_ONLY_SPAN = 4;

const cache = new WeakMap<LoadedRecord, Map<string, DialScale | null>>();

/** The record's fixed scale for a target. `limit` is a visitor-set ceiling (budget, wall-time cap). */
export function dialScale(record: LoadedRecord, targetId: string, unit: string, limit?: number | null): DialScale | null {
	if (limit && limit > 0) return { kind: "linear", lo: 0, hi: limit, source: "limit" };
	if (unit === "probability" || unit === "fraction") return { kind: "linear", lo: 0, hi: 1, source: "probability" };
	let byTarget = cache.get(record);
	if (!byTarget) {
		byTarget = new Map();
		cache.set(record, byTarget);
	}
	if (byTarget.has(targetId)) return byTarget.get(targetId) ?? null;
	const scale = compute(record, targetId);
	byTarget.set(targetId, scale);
	return scale;
}

function compute(record: LoadedRecord, targetId: string): DialScale | null {
	const range = gridRange(record, targetId);
	const est = record.predictions.estimates?.[targetId];
	const eHi = est && est.status !== "abstain" ? (est.hi ?? est.point) : null;
	const eLo = est && est.status !== "abstain" ? (est.lo ?? est.point) : null;
	if (range) {
		const hi = Math.max(range.hi, eHi ?? 0);
		const lo = Math.max(Math.min(range.lo, eLo && eLo > 0 ? eLo : range.lo), hi / LOG_SPAN_MAX);
		return lo < hi ? { kind: "log", lo, hi, source: "grid" } : { kind: "linear", lo: 0, hi: hi * 2, source: "grid" };
	}
	if (eHi && eHi > 0) {
		const lo = Math.max((eLo && eLo > 0 ? eLo : eHi) / ESTIMATE_ONLY_SPAN, eHi / LOG_SPAN_MAX);
		return { kind: "log", lo, hi: eHi * ESTIMATE_ONLY_SPAN, source: "estimate" };
	}
	return null;
}

/** Position of a value on the arc, 0 to 1, and whether it lies past the right end. */
export function scaleFraction(scale: DialScale, v: number): { frac: number; off: boolean } {
	if (!Number.isFinite(v)) return { frac: 0, off: false };
	let f: number;
	if (scale.kind === "log") f = v <= scale.lo ? 0 : Math.log(v / scale.lo) / Math.log(scale.hi / scale.lo);
	else f = scale.hi > 0 ? (v - scale.lo) / (scale.hi - scale.lo) : 0;
	return { frac: Math.max(0, Math.min(1, f)), off: f > 1 + 1e-9 };
}

/**
 * The memory stack's full width, fixed per record: the largest capacity any GPU this record
 * can choose gives a process, plus a quarter so "over by a little" stays visible. Past that a
 * run fails anyway, so the bar clips and says the peak runs off the scale. The quarter is a
 * guess (the stack's earlier headroom, kept so the bar looks the same at the record's own
 * setting); replaced when D's stranger test reports how visitors read the capacity line.
 */
export const STACK_HEADROOM = 1.25;

const stackCache = new WeakMap<LoadedRecord, number | null>();

export function memoryStackScale(record: LoadedRecord): number | null {
	if (stackCache.has(record)) return stackCache.get(record) ?? null;
	const memT = targetFor(record, "memory");
	let cap: number | null = null;
	if (memT?.unit === "fraction") cap = 1;
	else {
		const base = currentKnobs(record);
		const options: Array<Record<string, unknown>> = [base];
		for (const id of GPU_KNOBS) {
			const knob = record.knobs.find((k) => k.id === id);
			if (knob) for (const v of knobChoices(record, knob)) options.push({ ...base, [id]: v });
		}
		for (const o of options) {
			const c = capacityBytes(record, o as typeof base);
			if (c && (cap === null || c > cap)) cap = c;
		}
	}
	const out = cap ? cap * STACK_HEADROOM : null;
	stackCache.set(record, out);
	return out;
}
