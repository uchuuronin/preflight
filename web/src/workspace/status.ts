/** Shared status vocabulary: fits / tight / fails, always an icon shape plus a word. */
import { capacityBytes, failureTarget, targetFor } from "./record/model";
import { FAILS_AT, FITS_BELOW, oomStatus } from "./record/terms";
import type { Estimate, KnobValue, LoadedRecord, Prediction } from "./record/types";

export type Fit = "fits" | "tight" | "fails" | "unknown";

/**
 * A number the UI may show. The guard type is branded so the false branch still knows the
 * value is an Estimate (and can read its abstain reason).
 */
export type TrustedEstimate = Estimate & { readonly __trusted: true; point: number; lo: number; hi: number };

export function trusted(e: Estimate | undefined | null): e is TrustedEstimate {
	return !!e && e.status !== "abstain" && e.point !== null && Number.isFinite(e.point);
}

/** Lower and upper bound, falling back to the point when the engine gave none. */
export function bounds(e: TrustedEstimate): [number, number] {
	return [e.lo ?? e.point, e.hi ?? e.point];
}

/**
 * Memory fit. With a failure probability, B's cut-offs decide (fits below FITS_BELOW, fails at
 * FAILS_AT or more, tight between). Without one, the peak against capacity decides, with the
 * upper bound inside 5% of capacity counting as tight.
 */
export function fitOf(record: LoadedRecord, est: Prediction, knobs: Readonly<Record<string, KnobValue>>): Fit {
	const fail = failureTarget(record);
	const risk = fail ? est[fail] : undefined;
	const memT = targetFor(record, "memory");
	const peak = memT ? est[memT.id] : undefined;
	const cap = capacityBytes(record, knobs);
	if (trusted(risk)) return oomStatus(risk.point);
	if (trusted(peak) && cap && memT?.unit === "bytes") {
		const [, hi] = bounds(peak);
		if (peak.point > cap) return "fails";
		if (hi > cap * 0.95) return "tight";
		return "fits";
	}
	return "unknown";
}

/** Likely to fail: the failure target at or past FAILS_AT. */
export function likelyFails(record: LoadedRecord, est: Prediction): boolean {
	const fail = failureTarget(record);
	const risk = fail ? est[fail] : undefined;
	return trusted(risk) && risk.point >= FAILS_AT;
}

/** Any real risk of failing: the failure target at or past FITS_BELOW. */
export function atRisk(record: LoadedRecord, est: Prediction): boolean {
	const fail = failureTarget(record);
	const risk = fail ? est[fail] : undefined;
	return trusted(risk) && risk.point >= FITS_BELOW;
}
