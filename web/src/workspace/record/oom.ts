/**
 * P(out of memory) from a memory estimate, the way B's engine computes it
 * (python/preflight/engine/memory.py: `p_oom`): the peak is log-normal around its point, with
 * the spread read off the estimate's 90% band, and P(OOM) is the chance it lands above the
 * GPU's capacity.
 *
 * Interpolating P(OOM) itself between grid cells is wrong (half way between a cell that fits
 * and one that fails is not "50%"), so between cells the provider interpolates memory first
 * and derives P(OOM) from it here; the fit map draws its boundary the same way.
 */
import type { Estimate } from "./types";

/** Source: cited, the standard normal's 95th percentile; engine.memory.Z90. */
export const Z90 = 1.6448536269514722;

/**
 * Spread of ln(peak) from a band: the wider side, so a lopsided band never looks tighter. Bands
 * are read as 90% (every grid and estimate today; `level` other than 0.9 would need its own
 * quantile here).
 */
export function sigmaLog(e: Pick<Estimate, "point" | "lo" | "hi">): number | null {
	const { point, lo, hi } = e;
	if (point === null || !(point > 0)) return null;
	const up = hi !== null && hi !== undefined && hi > 0 ? Math.log(hi / point) : 0;
	const down = lo !== null && lo !== undefined && lo > 0 ? Math.log(point / lo) : 0;
	return Math.max(up, down, 0) / Z90;
}

/** Abramowitz and Stegun 7.1.26, absolute error under 1.5e-7: plenty for a 1% cut-off. */
function erfc(x: number): number {
	const z = Math.abs(x);
	const t = 1 / (1 + 0.3275911 * z);
	const y = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))) * Math.exp(-z * z);
	return x >= 0 ? y : 2 - y;
}

/** engine.memory.p_oom, line for line. */
export function pOom(pointBytes: number, sigma: number, capacityBytes: number, marginBytes = 0): number {
	const usable = capacityBytes - marginBytes;
	if (usable <= 0) return 1;
	if (pointBytes <= 0) return 0;
	if (sigma <= 0) return pointBytes > usable ? 1 : 0;
	const z = (Math.log(usable) - Math.log(pointBytes)) / sigma;
	return Math.min(1, Math.max(0, 0.5 * erfc(z / Math.SQRT2)));
}

/** P(OOM) for a memory estimate on a GPU with this capacity, or null when either is unknown. */
export function pOomFromMemory(mem: Estimate | undefined, capacityBytes: number | null): number | null {
	if (!mem || mem.status === "abstain" || mem.point === null || !capacityBytes) return null;
	const sigma = sigmaLog(mem);
	if (sigma === null) return null;
	return pOom(mem.point, sigma, capacityBytes);
}
