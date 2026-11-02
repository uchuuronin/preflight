/**
 * The one plain-language line above the readouts (competitive review item 8). Filled by a
 * template from the readouts and their intervals, never by a model: every number in it is a
 * number a dial shows, so it can't say anything the cockpit doesn't.
 */
import { fmt, fmtRange } from "../../lib/fmt";
import { t } from "../../lib/i18n";
import { capacityBytes, failureTarget, targetFor } from "../record/model";
import { chosenGpu } from "../record/terms";
import type { KnobValue, LoadedRecord, Prediction } from "../record/types";
import { bounds, fitOf, type TrustedEstimate, trusted } from "../status";

export interface StatusLine {
	tone: "fits" | "tight" | "fails" | "unknown";
	text: string;
}

/** "4 h 13 min (3 h 42 min to 5 h)" or just the point when the band is that narrow. */
function withRange(e: TrustedEstimate, unit: string): string {
	const [lo, hi] = bounds(e);
	const point = fmt(e.point, unit);
	const range = fmtRange(lo, hi, unit);
	return fmt(lo, unit) === fmt(hi, unit) ? point : t("status.with_range", { point, range });
}

export function statusLine(record: LoadedRecord, est: Prediction, knobs: Readonly<Record<string, KnobValue>>): StatusLine {
	const memT = targetFor(record, "memory");
	const timeT = targetFor(record, "time");
	const costT = targetFor(record, "cost");
	const fail = failureTarget(record);
	const mem = memT ? est[memT.id] : undefined;
	const time = timeT ? est[timeT.id] : undefined;
	const cost = costT ? est[costT.id] : undefined;
	const risk = fail ? est[fail] : undefined;
	const gpu = chosenGpu(record, knobs) ?? t("status.this_gpu");
	const cap = capacityBytes(record, knobs);
	const fit = memT ? fitOf(record, est, knobs) : "unknown";

	const run = [
		trusted(time) ? t("status.finishes", { time: withRange(time, timeT?.unit ?? "s") }) : null,
		trusted(cost) ? t("status.costs", { cost: withRange(cost, costT?.unit ?? "usd") }) : null,
	].filter(Boolean) as string[];
	const budget = record.job.constraints?.budget_usd;
	let budgetNote = "";
	if (budget && trusted(cost)) {
		const [lo, hi] = bounds(cost);
		if (lo > budget) budgetNote = t("status.over_budget", { budget: fmt(budget, "usd") });
		else if (hi > budget) budgetNote = t("status.may_exceed", { budget: fmt(budget, "usd") });
		else budgetNote = t("status.within_budget", { budget: fmt(budget, "usd") });
	}
	const tail = run.length ? ` ${t("status.run", { parts: run.join(t("status.and")) })}${budgetNote ? ` ${budgetNote}` : ""}` : "";

	if (fit === "fails") {
		const head =
			// The peak form only when the peak itself is over the card: a run can fail on a part the
			// training peak doesn't cover (an eval loop that keeps gradients), and "peaks at 9 GiB
			// of 22 GiB" would read as a contradiction.
			trusted(mem) && cap && memT?.unit === "bytes" && mem.point > cap
				? t("status.fails_peak", { gpu, peak: fmt(mem.point, "bytes"), cap: fmt(cap, "bytes") })
				: t("status.fails", { gpu, risk: trusted(risk) ? fmt(risk.point, "probability") : "" });
		return { tone: "fails", text: head };
	}
	if (fit === "tight") {
		return { tone: "tight", text: `${t("status.tight", { gpu, risk: trusted(risk) ? fmt(risk.point, "probability") : "" })}${tail}` };
	}
	if (fit === "fits") return { tone: "fits", text: `${t("status.fits", { gpu })}${tail}` };
	if (!run.length) {
		// Inside the probed range with no precomputed value is a gap, not "too far" (PM, Nov 2).
		const gap = [mem, time, cost, risk].some((e) => e?.status === "abstain" && e.reason === "between_measured");
		return { tone: "unknown", text: t(gap ? "status.between_measured" : "status.untrusted") };
	}
	return { tone: "unknown", text: tail.trim() };
}
