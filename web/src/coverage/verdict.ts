/**
 * The coverage page's status word for one metric, decided by its Wilson 95% interval against the
 * target rate, so a handful of checks can never read as a pass (or as a failure) by luck.
 */

export type Verdict = "unchecked" | "unsettled" | "meets" | "below" | "missed";

export interface MetricSummary {
	target: string;
	n: number;
	hits: number;
	rate: number | null;
	wilson95: [number, number] | number[] | null;
	abstained: number;
}

/**
 * Source: cited, Project Goals: 90% intervals cover at least 88%. Step time and peak memory have
 * a tolerance per check and no stated rate; the page holds them to the same 88% (source: guess,
 * revisited with Chiya).
 */
export const TARGET_RATE: Record<string, number> = { interval: 0.88, step_time: 0.88, peak_memory: 0.88, oom: 1 };

export function verdict(metric: string, s: MetricSummary): Verdict {
	if (!s.n || !s.wilson95) return "unchecked";
	// Project Goals: no missed out-of-memory. One miss is the verdict, whatever the count.
	if (metric === "oom") return s.hits < s.n ? "missed" : s.n >= 10 ? "meets" : "unsettled";
	const target = TARGET_RATE[metric] ?? 0.88;
	const [lo, hi] = s.wilson95;
	if ((hi ?? 1) < target) return "below";
	if ((lo ?? 0) >= target) return "meets";
	return "unsettled";
}
