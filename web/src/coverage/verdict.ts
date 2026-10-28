/**
 * The coverage page's status word for one metric. Step time, peak memory and the interval get a
 * status only when the whole Wilson 95% interval is past the metric's rate bar, so a handful of
 * checks can never read as a pass or a failure by luck. Out of memory follows the Project Goals'
 * rule instead: one miss is the verdict, and none missed so far is said as just that.
 */

export type Verdict = "unchecked" | "unsettled" | "meets" | "below" | "missed" | "no_misses";

export interface MetricSummary {
	tolerance?: number | null;
	/** The share of checks the metric is held to, and where that bar comes from (coverage.py). */
	rate_bar: number;
	rate_bar_source: string;
	n: number;
	hits: number;
	rate: number | null;
	wilson95: number[] | null;
	abstained: number;
	runs?: number;
}

export function verdict(metric: string, s: MetricSummary): Verdict {
	if (!s.n || !s.wilson95) return "unchecked";
	if (metric === "oom") return s.hits < s.n ? "missed" : "no_misses";
	const [lo = 0, hi = 1] = s.wilson95;
	if (hi < s.rate_bar) return "below";
	if (lo >= s.rate_bar) return "meets";
	return "unsettled";
}
