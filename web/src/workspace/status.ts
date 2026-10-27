/** Shared status vocabulary: fits / tight / fails, always an icon shape plus a word. */
import { capacityBytes, failureTarget, targetFor } from "./record/model";
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

/** Memory fit from the failure probability and the headroom under capacity. */
export function fitOf(record: LoadedRecord, est: Prediction, knobs: Readonly<Record<string, KnobValue>>): Fit {
	const fail = failureTarget(record);
	const risk = fail ? est[fail] : undefined;
	const memT = targetFor(record, "memory");
	const peak = memT ? est[memT.id] : undefined;
	const cap = capacityBytes(record, knobs);
	if (trusted(risk)) {
		if (risk.point > 0.5) return "fails";
		if (risk.point > 0.05) return "tight";
	}
	if (trusted(peak) && cap) {
		const [, hi] = bounds(peak);
		if (peak.point > cap) return "fails";
		if (hi > cap * 0.95) return "tight";
		return "fits";
	}
	if (trusted(risk)) return "fits";
	return "unknown";
}
