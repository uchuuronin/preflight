/**
 * fmt(value, unit): every number on screen goes through here.
 *
 * Handles units, locale and currency. D's local-currency add-on calls setCurrency()
 * with a rate; nothing else changes. Units are the record's unit strings.
 */
import { language } from "./i18n";
import { t } from "./i18n";

export type Unit =
	| "bytes"
	| "s"
	/** a duration for tight spaces: "10.3 h" instead of "10 h 19 min" */
	| "s_short"
	/** dollars where amounts can be fractions of a cent (cost per task, a rate per token): "US$0.0042" */
	| "usd_small"
	| "ms"
	| "h"
	| "usd"
	| "tokens"
	| "tokens/s"
	| "requests/s"
	| "ratio"
	| "prob"
	| "probability"
	| "fraction"
	| "percent"
	| "samples"
	| "samples/s"
	| "count"
	| "flops"
	| "steps"
	| "gpu_h"
	/** the engine's name for GPU-hours (cost.gpu_hours, job_cost.gpu_hours): formatted as gpu_h */
	| "gpu_hours"
	| "none";

interface CurrencyOptions {
	code: string;
	/** Multiply USD by this. */
	rate: number;
}

let currency: CurrencyOptions = { code: "USD", rate: 1 };
let localeOverride: string | null = null;

export function setCurrency(code: string, rate: number): void {
	currency = { code, rate };
}

export function currencyCode(): string {
	return currency.code;
}

export function setLocale(locale: string | null): void {
	localeOverride = locale;
}

function locale(): string {
	return localeOverride ?? (language() === "en" ? "en-GB" : language());
}

function num(value: number, maxFrac: number, minFrac = 0): string {
	return new Intl.NumberFormat(locale(), {
		maximumFractionDigits: maxFrac,
		minimumFractionDigits: minFrac,
	}).format(value);
}

/** Significant-figure rounding that keeps small numbers readable. */
function sig(value: number): number {
	const a = Math.abs(value);
	if (a >= 100) return 0;
	if (a >= 10) return 1;
	return 2;
}

const BYTE_STEPS: Array<[number, string]> = [
	[1024 ** 4, "unit.tib"],
	[1024 ** 3, "unit.gib"],
	[1024 ** 2, "unit.mib"],
	[1024, "unit.kib"],
];

function duration(seconds: number): string {
	const s = Math.abs(seconds);
	if (s < 1) return t("unit.ms", { n: num(seconds * 1000, sig(seconds * 1000)) });
	if (s < 60) return t("unit.s", { n: num(seconds, sig(seconds)) });
	if (s < 3600) {
		const m = Math.floor(s / 60);
		const r = Math.round(s - m * 60);
		return r ? t("unit.min_s", { m, s: r }) : t("unit.min", { n: m });
	}
	if (s < 86400 * 2) {
		const h = Math.floor(s / 3600);
		const m = Math.round((s - h * 3600) / 60);
		return m ? t("unit.h_min", { h, m }) : t("unit.h", { n: h });
	}
	return t("unit.days", { n: num(s / 86400, 1) });
}

function compact(value: number): string {
	return new Intl.NumberFormat(locale(), { notation: "compact", maximumFractionDigits: 1 }).format(value);
}

export function fmt(value: number | null | undefined, unit: Unit | string = "none"): string {
	if (value === null || value === undefined || Number.isNaN(value)) return t("value.none");
	if (!Number.isFinite(value)) return t("value.unbounded");
	switch (unit as Unit) {
		case "bytes": {
			for (const [size, key] of BYTE_STEPS) {
				if (Math.abs(value) >= size) return t(key, { n: num(value / size, sig(value / size)) });
			}
			return t("unit.b", { n: num(value, 0) });
		}
		case "s":
			return duration(value);
		case "s_short":
			// For tight spaces (a dial face): past 10 hours the minutes are noise, "10.3 h".
			return Math.abs(value) >= 36_000 && Math.abs(value) < 86_400 * 2 ? t("unit.h", { n: num(value / 3600, 1) }) : duration(value);
		case "ms":
			return duration(value / 1000);
		case "h":
			return duration(value * 3600);
		case "usd": {
			const v = value * currency.rate;
			return new Intl.NumberFormat(locale(), {
				style: "currency",
				currency: currency.code,
				maximumFractionDigits: Math.abs(v) < 10 ? 2 : 0,
			}).format(v);
		}
		case "usd_small": {
			const v = value * currency.rate;
			// Under ten cents, two significant figures (a cent's fraction still reads: "US$0.0042"); from
			// there, cents ("US$0.80"), as "usd" does from a dollar.
			const tiny = Math.abs(v) > 0 && Math.abs(v) < 0.1;
			return new Intl.NumberFormat(locale(), {
				style: "currency",
				currency: currency.code,
				...(tiny ? { maximumSignificantDigits: 2 } : { minimumFractionDigits: Math.abs(v) < 10 ? 2 : 0, maximumFractionDigits: Math.abs(v) < 10 ? 2 : 0 }),
			}).format(v);
		}
		case "tokens":
			return t("unit.tokens", { n: compact(value) });
		case "tokens/s":
			return t("unit.tokens_per_s", { n: compact(value) });
		case "samples":
			return t("unit.samples", { n: num(value, 0) });
		case "samples/s":
			return t("unit.samples_per_s", { n: num(value, sig(value)) });
		case "requests/s":
			return t("unit.requests_per_s", { n: num(value, sig(value)) });
		case "prob":
		case "probability":
		case "fraction":
		case "ratio":
		case "percent": {
			const p = unit === "percent" ? value / 100 : value;
			return new Intl.NumberFormat(locale(), {
				style: "percent",
				maximumFractionDigits: Math.abs(p) < 0.1 ? 1 : 0,
			}).format(p);
		}
		case "gpu_h":
		case "gpu_hours":
			return t("unit.gpu_h", { n: num(value, sig(value)) });
		case "steps":
			return t("unit.steps", { n: compact(value) });
		case "flops":
			return t("unit.flops", { n: compact(value) });
		case "count":
			return num(value, 0);
		default:
			return num(value, sig(value));
	}
}

/** An interval "lo – hi" in one unit, shortened when both ends share a unit word. */
export function fmtRange(lo: number, hi: number, unit: Unit | string): string {
	return t("value.range", { lo: fmt(lo, unit), hi: fmt(hi, unit) });
}

/** A signed change for delta badges: "+2.1 GiB", "−14 min". */
export function fmtDelta(delta: number, unit: Unit | string): string {
	if (delta === 0) return t("value.no_change");
	const sign = delta > 0 ? "+" : "−";
	return `${sign}${fmt(Math.abs(delta), unit)}`;
}
