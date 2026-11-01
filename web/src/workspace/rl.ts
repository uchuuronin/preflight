/**
 * The RL lens's measured signals, read from the record's probe (A's recorder): shared by the
 * RL views (cockpit/RlBand.tsx) and the guide's tour (guide/ruleTour.ts).
 */
import type { LoadedRecord, Probe } from "./record/types";

/** The probe that carries RL signals, preferring the current-settings one (the first measured). */
export function rlProbe(record: LoadedRecord): Probe | undefined {
	return record.probes.find((p) => p.status === "ok" && p.rl && (p.rl.pass_rate_per_prompt?.length ?? 0) > 0);
}

/** Prompts with no reward spread as a count of the sampled prompts, when the share is a whole count of them. */
export function spreadCount(p: Probe): { k: number; n: number } | null {
	const rates = p.rl?.pass_rate_per_prompt ?? [];
	const share = p.rl?.zero_spread_fraction;
	const n = rates.length;
	if (!n || typeof share !== "number") return null;
	const k = Math.round(share * n);
	// The recorder counts spread over the summed reward; only a share that is k of n prompts is a count.
	return Math.abs(k - share * n) < 1e-6 ? { k, n } : null;
}

/** The completion cap the probe ran at (its own knobs, else what generation reported): what the measured lengths are against. */
export function probeCap(p: Probe): number | null {
	const k = p.knobs?.max_completion_len;
	if (typeof k === "number") return k;
	const g = p.generation?.max_new_tokens;
	return typeof g === "number" ? g : null;
}
