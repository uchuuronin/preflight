/**
 * The learning-rate range test, as the record keeps it (D, Nov 2; A's probes/local.py `with_lr_range`,
 * D's probes/lr_range.py): a probe marked `extras.lr_range`, about 100 steps with the rate rising from
 * 1e-7 to 1, its curve and reading in extras, floats only (no contract change):
 *
 *   lr_range.lr.NNN, lr_range.loss.NNN   the curve, raw loss (rates only when the ramp is checked)
 *   lr_range.suggested_lr                steepest fall of the smoothed loss against log(rate)
 *   lr_range.min_loss_lr                 lowest smoothed loss
 *   lr_range.diverged_lr                 where it blew up (absent if it never did)
 *   lr_range.flat                        the loss never fell enough to read a slope: no suggestion
 *   lr_range.checked                     the updates grew with the rate, so the rates stand
 *   lr_range.not_set / .unverified / .other_optimizers   why the rates were dropped
 *
 * The suggestion is a starting point for the script's own schedule, not a measured optimum: the test
 * runs from the model's first steps, without the script's warm-up or decay (D's docstring).
 */
import type { KnobValue, LoadedRecord } from "../record/types";

type Probe = LoadedRecord["probes"][number];

export interface RangePoint {
	/** The rate at this step, or null when the ramp isn't checked (the losses alone are kept). */
	lr: number | null;
	loss: number;
}

export interface RangeTest {
	probeId: string;
	gpu: string | null;
	/** The rates stand: the optimizer's updates grew with the ramp. */
	checked: boolean;
	/** The loss never fell enough to read a slope: no suggestion. */
	flat: boolean;
	points: RangePoint[];
	suggested: number | null;
	minLoss: number | null;
	diverged: number | null;
	/** Why the rates were dropped, as D's keys: not_set, unverified, other_optimizers. */
	reasons: Array<"not_set" | "unverified" | "other_optimizers">;
}

const num = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
const pos = (v: unknown): number | null => (num(v) && v > 0 ? v : null);

/** The record's range-test probe, read; null when it has none. */
export function rangeTestOf(record: Pick<LoadedRecord, "probes" | "env">): RangeTest | null {
	const probe = [...record.probes].reverse().find((p) => (p.extras ?? {})["lr_range"] === 1);
	if (!probe) return null;
	const own = probe.knobs?.["gpu_type"];
	return readRangeTest(probe, typeof own === "string" ? own : (record.env?.gpu?.name ?? null));
}

export function readRangeTest(probe: Pick<Probe, "id" | "extras" | "knobs">, gpu: string | null = null): RangeTest {
	const x = (probe.extras ?? {}) as Record<string, unknown>;
	const checked = x["lr_range.checked"] === 1;
	const points: RangePoint[] = [];
	for (let j = 0; j < 1000; j++) {
		const key = String(j).padStart(3, "0");
		const loss = x[`lr_range.loss.${key}`];
		if (loss === undefined) break;
		if (!num(loss)) continue;
		const lr = checked ? pos(x[`lr_range.lr.${key}`]) : null;
		if (checked && lr === null) continue; // a checked curve with a missing rate: the point can't be placed
		points.push({ lr, loss });
	}
	const reasons: RangeTest["reasons"] = [];
	if (x["lr_range.not_set"] === 1) reasons.push("not_set");
	if (x["lr_range.unverified"] === 1) reasons.push("unverified");
	if (num(x["lr_range.other_optimizers"]) && (x["lr_range.other_optimizers"] as number) >= 1) reasons.push("other_optimizers");
	return {
		probeId: probe.id,
		gpu,
		checked,
		flat: x["lr_range.flat"] === 1,
		points,
		// Rates stand only on a checked ramp (A drops them otherwise; read defensively all the same).
		suggested: checked && x["lr_range.flat"] !== 1 ? pos(x["lr_range.suggested_lr"]) : null,
		minLoss: checked ? pos(x["lr_range.min_loss_lr"]) : null,
		diverged: checked ? pos(x["lr_range.diverged_lr"]) : null,
		reasons,
	};
}

/** Source: D's probes/lr_range.py BETA (0.9), the smoothing the suggestion is read from. */
const BETA = 0.9;

function ema(xs: readonly number[]): number[] {
	const out: number[] = [];
	let avg = 0;
	xs.forEach((v, i) => {
		avg = BETA * avg + (1 - BETA) * v;
		out.push(avg / (1 - BETA ** (i + 1))); // bias-corrected
	});
	return out;
}

/** D's smoothing (`_smooth`): an EMA forward, then backward, so the curve's features keep their rates. */
export function smoothed(losses: readonly number[]): number[] {
	return ema(ema(losses).reverse()).reverse();
}

/** Is this probe the range test (so it isn't one of the run's own measurements)? */
export function isRangeTest(probe: Pick<Probe, "extras">): boolean {
	return (probe.extras ?? {})["lr_range"] === 1;
}

/** The knob's choice nearest this rate, on a log scale (rates are compared by ratio). */
export function nearestRate(values: readonly KnobValue[], rate: number): number | null {
	let best: number | null = null;
	let gap = Infinity;
	for (const v of values) {
		if (!num(v) || v <= 0) continue;
		const d = Math.abs(Math.log10(v) - Math.log10(rate));
		if (d < gap) {
			gap = d;
			best = v;
		}
	}
	return best;
}

/** Where a rate sits against the test's reading, for the words under the curve. */
export type RateVerdict = "past_divergence" | "past_lowest" | "far_below" | "near" | "unknown";

/** Source: guess, a tenth of the steepest fall is "well below" (a decade, on the curve's log axis). */
const FAR_BELOW = 10;

export function rateVerdict(test: RangeTest, rate: number | null): RateVerdict {
	if (!test.checked || rate === null || !(rate > 0)) return "unknown";
	if (test.diverged !== null && rate >= test.diverged) return "past_divergence";
	if (test.minLoss !== null && rate > test.minLoss) return "past_lowest";
	if (test.suggested !== null && rate < test.suggested / FAR_BELOW) return "far_below";
	if (test.suggested !== null) return "near";
	return "unknown";
}

/** A rate as text, as the knob prints rates (GuidePanel valueText): two significant figures, "0.000079". */
const RATE = new Intl.NumberFormat("en-GB", { maximumSignificantDigits: 2 });
export function rateText(v: number): string {
	return RATE.format(v);
}
