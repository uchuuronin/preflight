/**
 * Shared vocabulary between the engine (B) and the UI: memory terms, how they group on the
 * ring and the stack, the OOM cut-offs, and GPU names. One place, so the map, the dials, the
 * bands and the guide never disagree about what "fits" means or what a term is called.
 */
import gpuTable from "./gpus.json";
import type { Estimate, KnobValue, LoadedRecord, Prediction } from "./types";

function usable(e: Estimate | undefined): e is Estimate & { point: number } {
	return !!e && e.status !== "abstain" && e.point !== null && Number.isFinite(e.point);
}

/* ---------- OOM cut-offs (engine.memory.FITS_BELOW / FAILS_AT, engine-settings.md S2b) ---------- */

/** P(OOM) below this is "fits". */
export const FITS_BELOW = 0.01;
/** P(OOM) at or above this is "fails"; in between is "tight". */
export const FAILS_AT = 0.8;

export type OomStatus = "fits" | "tight" | "fails";

export function oomStatus(p: number): OomStatus {
	if (p < FITS_BELOW) return "fits";
	if (p < FAILS_AT) return "tight";
	return "fails";
}

/* ---------- Memory terms ---------- */

/** Every per-GPU term B emits as `memory.<term>`, in stacking order (resident first). */
export const MEMORY_TERMS = [
	"weights",
	"master_weights",
	"frozen_models",
	"context",
	"gradients",
	"optimizer",
	"comm_buffers",
	"activations",
	"logits",
	"kv_cache",
	"generation_engine",
	"eval_spike",
] as const;
export type MemoryTerm = (typeof MEMORY_TERMS)[number];

/** Older ids (`memory.term.<t>`, and `eval` for the spike) still read, so old records open. */
const LEGACY: Partial<Record<MemoryTerm, string[]>> = {
	eval_spike: ["memory.term.eval", "memory.eval"],
};

export function memoryTermId(term: MemoryTerm): string {
	return `memory.${term}`;
}

/** The estimate id a record actually carries for a term, if any. */
export function memoryTermKey(est: Prediction, term: MemoryTerm): string | null {
	for (const id of [memoryTermId(term), `memory.term.${term}`, ...(LEGACY[term] ?? [])]) {
		if (est[id] !== undefined) return id;
	}
	return null;
}

/** Bytes for a term, or 0 when absent or not trusted. The eval spike never goes negative. */
export function memoryTermBytes(est: Prediction, term: MemoryTerm): number {
	const key = memoryTermKey(est, term);
	const e = key ? est[key] : undefined;
	return usable(e) ? Math.max(0, e.point) : 0;
}

export type MemoryGroupKey = "weights" | "training" | "activations" | "logits" | "generation" | "eval" | "untrusted" | "total";

/** Ring and stack groups: colour slot, terms, and the label key. Six slots, D's five plus generation. */
export const MEMORY_GROUPS: Array<{ key: Exclude<MemoryGroupKey, "untrusted" | "total">; slot: 1 | 2 | 3 | 4 | 5 | 6; terms: MemoryTerm[] }> = [
	{ key: "weights", slot: 1, terms: ["weights", "master_weights", "frozen_models", "context"] },
	{ key: "training", slot: 2, terms: ["gradients", "optimizer", "comm_buffers"] },
	{ key: "activations", slot: 3, terms: ["activations"] },
	{ key: "logits", slot: 4, terms: ["logits"] },
	{ key: "generation", slot: 6, terms: ["kv_cache", "generation_engine"] },
	{ key: "eval", slot: 5, terms: ["eval_spike"] },
];

export interface MemorySegment {
	key: MemoryGroupKey;
	bytes: number;
	/** 0 is the hatched "untrusted" fill. */
	slot: 0 | 1 | 2 | 3 | 4 | 5 | 6;
	/** The terms inside this group with their own bytes, for the legend and the detail sheet. */
	parts: Array<{ term: MemoryTerm; bytes: number }>;
}

/**
 * Parts and peak agree when they differ by at most this share of the peak. Source: guess, the
 * float16 grid's rounding (about 0.1% per value) summed over up to 12 terms with headroom;
 * replaced by B's measured residual between summed terms and probe peaks once B reports it.
 */
export const PARTS_AGREE_WITHIN = 0.02;

export interface MemoryStack {
	segments: MemorySegment[];
	breakdown: boolean;
	/** Sum of the trusted parts, at their own sizes. */
	partsBytes: number;
	/** The predicted peak, when trusted. */
	peakBytes: number | null;
	/**
	 * Where parts and peak disagree. "gap": the peak is bigger, the rest is a hatched segment.
	 * "overshoot": the parts add up to more than the peak; nothing is rescaled, the excess past
	 * the peak line is marked untrusted instead.
	 */
	untrusted: { kind: "gap" | "overshoot"; bytes: number } | null;
	/** Terms the record predicts that abstain at these settings (their bytes are unknown). */
	abstaining: MemoryTerm[];
}

/**
 * Memory at the peak, grouped for the ring and the stack, each part at its own predicted size.
 * Terms are reported at their own moments, so the sum can drift from the peak; that drift is
 * shown as untrusted, never hidden by scaling the parts.
 */
export function memorySegments(est: Prediction, peakBytes: number | null): MemoryStack {
	const abstaining: MemoryTerm[] = [];
	for (const term of MEMORY_TERMS) {
		const key = memoryTermKey(est, term);
		if (key && !usable(est[key])) abstaining.push(term);
	}
	const groups: MemorySegment[] = MEMORY_GROUPS.map((g) => {
		const parts = g.terms.map((term) => ({ term, bytes: memoryTermBytes(est, term) })).filter((p) => p.bytes > 0);
		return { key: g.key, slot: g.slot, bytes: parts.reduce((a, p) => a + p.bytes, 0), parts };
	}).filter((s) => s.bytes > 0);
	const breakdown = groups.some((g) => g.key === "weights" || g.key === "activations");
	if (!breakdown) {
		return {
			segments: peakBytes ? [{ key: "total", bytes: peakBytes, slot: 1, parts: [] }] : [],
			breakdown: false,
			partsBytes: 0,
			peakBytes,
			untrusted: null,
			abstaining,
		};
	}
	const sum = groups.reduce((a, g) => a + g.bytes, 0);
	let untrusted: MemoryStack["untrusted"] = null;
	if (peakBytes && sum - peakBytes > peakBytes * PARTS_AGREE_WITHIN) {
		untrusted = { kind: "overshoot", bytes: sum - peakBytes };
	} else if (peakBytes && peakBytes - sum > peakBytes * PARTS_AGREE_WITHIN) {
		untrusted = { kind: "gap", bytes: peakBytes - sum };
		groups.push({ key: "untrusted", bytes: peakBytes - sum, slot: 0, parts: [] });
	}
	return { segments: groups, breakdown: true, partsBytes: sum, peakBytes, untrusted, abstaining };
}

/* ---------- GPUs (generated from engine/gpus.py) ---------- */

interface GpuRow {
	vendor: string;
	arch: string;
	memory_bytes: number;
	confidence: string;
}
const GPUS = gpuTable.gpus as Record<string, GpuRow>;
const ALIASES = gpuTable.aliases as Record<string, string>;

/** Knob ids that choose the GPU: `gpu_type` (contract), `gpu` (older records). */
export const GPU_KNOBS = ["gpu_type", "gpu"] as const;

/** Canonical name for any GPU string the engine would resolve, else null. */
export function resolveGpu(name: string | null | undefined): string | null {
	if (!name) return null;
	if (GPUS[name]) return name;
	const lower = name.trim().toLowerCase();
	return ALIASES[lower] ?? ALIASES[lower.replace(/^nvidia\s+|^amd\s+/, "")] ?? null;
}

export function gpuSpec(name: string | null | undefined): (GpuRow & { name: string }) | null {
	const canonical = resolveGpu(name);
	return canonical ? { name: canonical, ...GPUS[canonical]! } : null;
}

/** The GPU the knobs choose, else the record's own. */
export function chosenGpu(record: LoadedRecord, knobs: Readonly<Record<string, KnobValue>>): string | null {
	for (const id of GPU_KNOBS) {
		const v = knobs[id];
		if (typeof v === "string") return v;
	}
	return record.env?.gpu?.name ?? null;
}

/** Bytes this GPU gives a process: the record's own measurement when it is the same GPU. */
export function gpuMemoryBytes(record: LoadedRecord, gpu: string | null): number | null {
	if (!gpu) return null;
	const own = record.env?.gpu;
	if (own?.memory_bytes && (gpu === own.name || resolveGpu(gpu) === resolveGpu(own.name))) return own.memory_bytes;
	return gpuSpec(gpu)?.memory_bytes ?? null;
}
