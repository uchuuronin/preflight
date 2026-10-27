/** Shared status vocabulary: fits / tight / fails, always an icon shape plus a word. */
import type { Estimate, FlightRecord, Prediction } from "./record/types";

export type Fit = "fits" | "tight" | "fails" | "unknown";

/** Memory fit from P(OOM) and headroom. */
export function memoryFit(est: Prediction, record: FlightRecord): Fit {
	const p = est.p_oom;
	const peak = est.peak_memory;
	if (!p || !peak || p.status === "abstain" || peak.status === "abstain") return "unknown";
	if (p.point > 0.5) return "fails";
	const cap = capacityBytes(record, est);
	if (p.point > 0.05 || (cap && peak.hi > cap * 0.95)) return "tight";
	return "fits";
}

export function capacityBytes(record: FlightRecord, _est?: Prediction, gpuName?: string): number | null {
	const name = gpuName ?? record.env.gpu?.name;
	if (!name) return null;
	return GPU_MEMORY[name] ?? record.env.gpu?.memory_bytes ?? null;
}

const GIB = 1024 ** 3;

/**
 * Capacity per GPU type for the capacity line. Display-only lookup; B's spec table
 * replaces this once the record carries per-GPU capacity (requested in docs/requests.md).
 */
export const GPU_MEMORY: Record<string, number> = {
	"A100-80GB": 80 * GIB,
	"A100-40GB": 40 * GIB,
	H100: 80 * GIB,
	H200: 141 * GIB,
	L40S: 48 * GIB,
	L4: 24 * GIB,
	A10G: 24 * GIB,
	T4: 16 * GIB,
	MI300X: 192 * GIB,
	MI250: 128 * GIB,
	"RTX-4090": 24 * GIB,
};

/**
 * A number the UI may show. The guard type is branded so the false branch still knows
 * the value is an Estimate (and can read its abstain reason).
 */
export type TrustedEstimate = Estimate & { readonly __trusted: true };

export function trusted(e: Estimate | undefined | null): e is TrustedEstimate {
	return !!e && e.status !== "abstain" && Number.isFinite(e.point);
}
