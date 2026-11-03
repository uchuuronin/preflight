/**
 * The live run against its predicted bands, read from the live record alone: the `live` probe's
 * step log (one entry per window of steps, A's recorder/inflight.py) and the predictions the record
 * carries from its parent. The same three checks as A's collector, so the cockpit and the terminal
 * raise the same alerts:
 *
 * - step time: a window's mean above the predicted band, two windows in a row (one slow window is
 *   noise). Faster than the band, two windows in a row, is a note, not an alert (B's `preflight
 *   watch`, Nov 2: a quick run's first steps beat the probe's timed ones; D's shot 9b). Each side
 *   keeps its own streak, so a fast window never hastens or blocks a slow alert.
 * - peak memory: a window's peak above the band's top, at once
 * - total time: time so far plus the remaining steps at the last three windows' pace, above the top
 */
import type { FlightRecord, Probe } from "../record/types";

/** Source: A's collector, inflight.REPEAT_TO_ALERT (guess there): two windows out of band in a row. */
export const REPEAT_TO_ALERT = 2;
/** Source: A's collector: the pace is the last three windows' mean step time. */
const PACE_WINDOWS = 3;

export type LiveAlert =
	| { kind: "step"; step: number; value: number; lo: number; hi: number; side: "above" }
	| { kind: "peak"; step: number; value: number; lo: number; hi: number }
	| { kind: "total"; step: number; value: number; lo: number; hi: number };

/** In the record's favour, said once and not raised: steps faster than the band. */
export type LiveNote = { kind: "step_fast"; step: number; value: number; lo: number; hi: number };

export interface LiveReading {
	status: Probe["status"];
	steps: number;
	elapsedS: number | null;
	stepS: number | null;
	peakBytes: number | null;
	projectedS: number | null;
	totalSteps: number | null;
	bands: { step: [number, number] | null; peak: [number, number] | null; total: [number, number] | null };
	/** Out of band now: the last window's alerts, as the collector would print them. */
	alerts: LiveAlert[];
	/** Out of band in the run's favour (faster steps): said, not alerted. */
	notes: LiveNote[];
}

function band(rec: FlightRecord, id: string): [number, number] | null {
	const e = rec.predictions?.estimates?.[id];
	if (!e || e.status === "abstain" || e.lo == null || e.hi == null) return null;
	return [e.lo, e.hi];
}

/** The run's optimizer steps: the current-settings probe's planned count (A reads p1 too). */
function plannedSteps(rec: FlightRecord): number | null {
	const v = (rec.probes ?? []).find((p) => p.id === "p1")?.extras?.planned_steps;
	return typeof v === "number" && v > 0 ? Math.round(v) : null;
}

export function liveProbe(rec: FlightRecord): Probe | undefined {
	return (rec.probes ?? []).find((p) => p.id === "live");
}

export function readLive(rec: FlightRecord): LiveReading | null {
	const live = liveProbe(rec);
	if (!live) return null;
	const windows = (live.step_log ?? []).filter((w) => Number.isFinite(w.wall_s));
	const bands = { step: band(rec, "time.step"), peak: band(rec, "memory.peak"), total: band(rec, "time.total") };
	const totalSteps = plannedSteps(rec);
	const elapsedS = typeof live.duration_s === "number" ? live.duration_s : null;
	const last = windows.at(-1);
	const recent = windows.slice(-PACE_WINDOWS);
	const pace = recent.length ? recent.reduce((a, w) => a + w.wall_s, 0) / recent.length : null;
	const projectedS = elapsedS !== null && pace !== null && totalSteps ? elapsedS + Math.max(0, totalSteps - live.steps) * pace : null;
	const peakOf = (w: (typeof windows)[number]) => w.peak_reserved_bytes ?? w.peak_allocated_bytes ?? null;
	const peaks = windows.map(peakOf).filter((v): v is number => typeof v === "number");
	const alerts: LiveAlert[] = [];
	const notes: LiveNote[] = [];
	if (last && bands.step) {
		const [lo, hi] = bands.step;
		const tail = windows.slice(-REPEAT_TO_ALERT);
		const full = tail.length === REPEAT_TO_ALERT;
		if (full && tail.every((w) => w.wall_s > hi)) alerts.push({ kind: "step", step: last.index, value: last.wall_s, lo, hi, side: "above" });
		else if (full && tail.every((w) => w.wall_s < lo)) notes.push({ kind: "step_fast", step: last.index, value: last.wall_s, lo, hi });
	}
	const lastPeak = last ? peakOf(last) : null;
	if (last && bands.peak && lastPeak !== null && lastPeak > bands.peak[1]) alerts.push({ kind: "peak", step: last.index, value: lastPeak, lo: bands.peak[0], hi: bands.peak[1] });
	if (last && bands.total && projectedS !== null && projectedS > bands.total[1]) alerts.push({ kind: "total", step: last.index, value: projectedS, lo: bands.total[0], hi: bands.total[1] });
	return {
		status: live.status,
		steps: live.steps,
		elapsedS,
		stepS: last?.wall_s ?? null,
		peakBytes: peaks.length ? Math.max(...peaks) : null,
		projectedS,
		totalSteps,
		bands,
		alerts,
		notes,
	};
}
