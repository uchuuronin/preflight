/**
 * Hypothetical outcome plots (HOPs; Hullman, Resnick and Adar, PLOS ONE 2015): instead of a
 * static band, the readout shows one possible outcome per frame, so how often the run lands
 * past a limit is seen rather than read.
 *
 * The frames come only from the provider's own estimate: its point and its 90% interval, never a
 * number of their own. B's intervals are conformal on log ratios, so the frames are a split
 * log-normal around the point whose 5th and 95th percentiles are the interval's ends (each side
 * fitted on its own, so a lopsided interval stays lopsided). An estimate that abstains has no
 * frames. The risk readout keeps the engine's own probability: frames show where the peak may
 * land, not a second P(OOM).
 */
import { useEffect, useSyncExternalStore } from "react";
import type { Estimate } from "./record/types";
import { bounds, trusted } from "./status";

/** Source: derived, the standard normal's 95th percentile: a 90% two-sided interval's half-width in z. */
export const Z90 = 1.6448536269514722;
/** Source: derived, 20 quantiles at (i + 0.5) / 20: one frame in twenty past each end of a 90% interval,
 * as the interval says, and one cycle in 10 s. */
export const FRAMES = 20;
/** Source: guess, slow enough to read each frame and fast enough to see the spread; revisited with
 * D's accessibility pass (Mon Nov 2) on the phone and desktop builds. */
export const FRAME_MS = 500;

/** Inverse standard normal CDF (Acklam's rational approximation; relative error under 1.2e-9). */
export function normInv(p: number): number {
	if (p <= 0) return Number.NEGATIVE_INFINITY;
	if (p >= 1) return Number.POSITIVE_INFINITY;
	const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
	const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
	const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
	const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
	const lo = 0.02425;
	if (p < lo) {
		const q = Math.sqrt(-2 * Math.log(p));
		return (((((c[0]! * q + c[1]!) * q + c[2]!) * q + c[3]!) * q + c[4]!) * q + c[5]!) / ((((d[0]! * q + d[1]!) * q + d[2]!) * q + d[3]!) * q + 1);
	}
	if (p > 1 - lo) return -normInv(1 - p);
	const q = p - 0.5;
	const r = q * q;
	return ((((((a[0]! * r + a[1]!) * r + a[2]!) * r + a[3]!) * r + a[4]!) * r + a[5]!) * q) / (((((b[0]! * r + b[1]!) * r + b[2]!) * r + b[3]!) * r + b[4]!) * r + 1);
}

/** Frame order: bit-reversed, so neighbouring frames jump across the spread instead of sweeping it. */
function order(n: number): number[] {
	const bits = Math.ceil(Math.log2(Math.max(2, n)));
	const out: number[] = [];
	for (let i = 0; i < 1 << bits; i++) {
		let r = 0;
		for (let b = 0; b < bits; b++) if (i & (1 << b)) r |= 1 << (bits - 1 - b);
		if (r < n) out.push(r);
	}
	return out;
}

/**
 * The possible outcomes for one estimate, in frame order, or [] when it isn't trusted or has no
 * interval. Positive values use the log scale; anything touching zero uses the linear scale,
 * clipped at zero.
 */
export function outcomes(est: Estimate | undefined, n = FRAMES): number[] {
	if (!trusted(est)) return [];
	const [lo, hi] = bounds(est);
	const p = est.point;
	if (!(hi > lo)) return [];
	const log = lo > 0 && p > 0;
	const f = log ? Math.log : (x: number) => x;
	const g = log ? Math.exp : (x: number) => Math.max(0, x);
	const sLo = Math.max(0, (f(p) - f(lo)) / Z90);
	const sHi = Math.max(0, (f(hi) - f(p)) / Z90);
	const sorted = Array.from({ length: n }, (_, i) => {
		const z = normInv((i + 0.5) / n);
		return g(f(p) + z * (z < 0 ? sLo : sHi));
	});
	return order(n).map((i) => sorted[i]!);
}

// One clock for every readout, so all of them show the same frame at once.
let on = false;
let frame = 0;
let timer: ReturnType<typeof setInterval> | null = null;
const listeners = new Set<() => void>();
const emit = () => {
	for (const l of listeners) l();
};

function reduced(): boolean {
	return typeof window !== "undefined" && !!window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
}

export function setHops(next: boolean): void {
	on = next;
	frame = 0;
	if (timer) clearInterval(timer);
	timer = null;
	// With reduced motion nothing moves: every outcome is drawn at once, faintly.
	if (on && !reduced() && typeof window !== "undefined") {
		timer = setInterval(() => {
			frame = (frame + 1) % FRAMES;
			emit();
		}, FRAME_MS);
	}
	emit();
}

export interface HopsState {
	on: boolean;
	/** Index into outcomes(), or null when every outcome shows at once (reduced motion). */
	frame: number | null;
}

let snapshot: HopsState = { on: false, frame: 0 };
function read(): HopsState {
	const f = reduced() ? null : frame;
	if (snapshot.on !== on || snapshot.frame !== f) snapshot = { on, frame: f };
	return snapshot;
}

function subscribe(fn: () => void): () => void {
	listeners.add(fn);
	return () => listeners.delete(fn);
}

export function useHops(): HopsState {
	return useSyncExternalStore(subscribe, read, read);
}

/** The toggle that owns the clock: it turns HOPs off when it leaves the screen. */
export function useHopsOwner(): HopsState {
	useEffect(() => () => setHops(false), []);
	return useHops();
}
