/**
 * D1 · The beginner's line: the readouts restated in plain words, above the cockpit's own status
 * line, at the "New to this" level only (Chiya, Nov 3). Built by a template from the readouts
 * and their ranges, never by a model: every number in it is one a dial or the memory ring shows,
 * and it adds no advice and no verdict of its own. A readout with no number is said plainly
 * ("time: not measured here").
 */
import { fmt, fmtSpread } from "../../lib/fmt";
import { t } from "../../lib/i18n";
import { capacityBytes, failureTarget, targetFor } from "../../workspace/record/model";
import { chosenGpu, FAILS_AT } from "../../workspace/record/terms";
import type { Estimate, KnobValue, LoadedRecord, Prediction } from "../../workspace/record/types";
import { bounds, fitOf, type TrustedEstimate, trusted } from "../../workspace/status";

/** The engine abstains where a probe ran out of memory: that was measured (as the dials read it). */
const RAN_OUT = /\bran out of memory\b/i;
/** The engine knows the step but not the run's length (as the dials read it). */
const LENGTH_UNKNOWN = /run.s length isn.t known/i;

/** "Context window filled" to "context window filled", keeping "KV cache" as it is. */
function lowerFirst(s: string): string {
	return /^[A-Z][a-z]/.test(s) ? s.charAt(0).toLowerCase() + s.slice(1) : s;
}

/** "2 h 6 min (1 h 42 min to 2 h 30 min)", or the point alone when the range's ends read the same. */
function about(e: TrustedEstimate, unit: string, withRange: boolean): string {
	const point = fmt(e.point, unit);
	const [lo, hi] = bounds(e);
	const range = withRange ? fmtSpread(lo, hi, unit) : null;
	return range ? t("addon.expertise.plain.about_range", { point, range }) : t("addon.expertise.plain.about", { point });
}

function memoryPart(record: LoadedRecord, est: Prediction, knobs: Readonly<Record<string, KnobValue>>): string | null {
	const memT = targetFor(record, "memory");
	if (!memT) return null;
	const mem = est[memT.id];
	// Not GPU memory (an agent's context window, a share of something): the readout's own label.
	if (memT.unit !== "bytes") {
		const what = lowerFirst(t(memT.label_key));
		return trusted(mem) ? t("addon.expertise.plain.readout", { what, value: about(mem, memT.unit, false) }) : t("addon.expertise.plain.risk_abstain", { what });
	}
	const gpu = chosenGpu(record, knobs) ?? t("status.this_gpu");
	if (!trusted(mem)) {
		return t(RAN_OUT.test(mem?.reason ?? "") ? "addon.expertise.plain.memory_ran_out" : "addon.expertise.plain.memory_abstain", { gpu });
	}
	const fit = fitOf(record, est, knobs);
	const cap = capacityBytes(record, knobs);
	const peak = fmt(mem.point, memT.unit);
	if (fit === "fails") {
		return cap && mem.point > cap
			? t("addon.expertise.plain.fails_peak", { gpu, peak, cap: fmt(cap, "bytes") })
			: t("addon.expertise.plain.fails", { gpu });
	}
	const spare = cap ? cap - mem.point : null;
	if (spare === null || spare <= 0) return t(fit === "tight" ? "addon.expertise.plain.tight" : "addon.expertise.plain.peak", { gpu, peak });
	return t(fit === "tight" ? "addon.expertise.plain.tight_spare" : "addon.expertise.plain.fits_spare", { gpu, spare: fmt(spare, "bytes") });
}

/** The time or cost readout in words, or why it has no number. */
function runPart(est: Estimate | undefined, unit: string, which: "time" | "cost", withRange: boolean): string {
	if (trusted(est)) return about(est, unit, withRange);
	if (LENGTH_UNKNOWN.test(est?.reason ?? "")) return t(`addon.expertise.plain.${which}_unknown`);
	return t(`addon.expertise.plain.${which}_abstain`);
}

/** The line's parts, in reading order; the first starts with a capital. */
export function plainParts(record: LoadedRecord, est: Prediction, knobs: Readonly<Record<string, KnobValue>>): string[] {
	const parts: string[] = [];
	const mem = memoryPart(record, est, knobs);
	if (mem) parts.push(mem);
	const gpuMemory = targetFor(record, "memory")?.unit === "bytes";

	const timeT = targetFor(record, "time");
	const costT = targetFor(record, "cost");
	// A run likely to run out of memory: its whole-run time and cost are what it would take if it
	// didn't stop, as the dials say under their numbers ("If it didn't run out of memory").
	const fail = failureTarget(record);
	const oom = fail ? est[fail] : undefined;
	const stops = fail === "risk.oom" && trusted(oom) && oom.point >= FAILS_AT;
	const time = timeT ? runPart(est[timeT.id], timeT.unit, "time", true) : null;
	const cost = costT ? runPart(est[costT.id], costT.unit, "cost", false) : null;
	const timeOk = !!timeT && trusted(est[timeT.id]);
	const costOk = !!costT && trusted(est[costT.id]);
	if (stops && time && cost && timeOk && costOk) {
		// Both are what it would take to the end, so the condition goes once, in front of both.
		parts.push(t("addon.expertise.plain.if_it_ran_both", { time, cost }));
	} else {
		if (time) parts.push(stops && timeOk ? t("addon.expertise.plain.if_it_ran", { time }) : time);
		if (cost) parts.push(stops && costOk ? t("addon.expertise.plain.if_it_ran", { time: cost }) : cost);
	}

	// No GPU memory readout (an agent or API run): its risk dial in words, as the dial labels it.
	if (!gpuMemory) {
		const riskT = targetFor(record, "risk");
		const risk = riskT ? est[riskT.id] : undefined;
		if (riskT) {
			const what = t(riskT.label_key);
			parts.push(trusted(risk) ? t("addon.expertise.plain.risk", { what: lowerFirst(what), p: fmt(risk.point, "probability") }) : t("addon.expertise.plain.risk_abstain", { what: lowerFirst(what) }));
		}
	}
	if (parts.length) parts[0] = parts[0]!.charAt(0).toLocaleUpperCase() + parts[0]!.slice(1);
	return parts;
}

export function plainLine(record: LoadedRecord, est: Prediction, knobs: Readonly<Record<string, KnobValue>>): string {
	return plainParts(record, est, knobs).join(" · ");
}
