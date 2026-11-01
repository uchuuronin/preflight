/**
 * Combined moves: when no single knob clears a predicted failure, a setting that changes two or
 * three knobs together may (the live LoRA record, Nov 1: the eval fix alone isn't enough at
 * seq_len 1,024). Only settings the record's own predictions cover are offered: B's pair grids
 * (`.clears`, engine.api.clearing_pairs: two knobs, every other knob held at its value when the
 * record was filled), read through the provider like any other setting. Nothing here adds up
 * single-knob effects.
 */
import { fmt, fmtDelta } from "../lib/fmt";
import { hasKey, t } from "../lib/i18n";
import type { PredictionProvider, SensitivityOptions } from "../workspace/provider/PredictionProvider";
import { failureTarget, objectiveTarget, target, targetFor } from "../workspace/record/model";
import { FITS_BELOW } from "../workspace/record/terms";
import type { KnobValue, LoadedRecord, Objective, Prediction } from "../workspace/record/types";

/** The most knobs one combined move changes: the server's `Combined` takes 2 to 4. */
export const MAX_COMBO_KNOBS = 4;
/**
 * Source: B's engine.api.CLEARS_P_OOM (oom.UNCERTAIN[0], the "unlikely" end of the shown P(OOM)
 * range): a pair grid (`.clears`) clears at or under this. Between FITS_BELOW and this the setting
 * is "tight" in the UI's terms, so it's offered to measure first, as a tight single move is.
 */
export const CLEARS_P_OOM = 0.05;
/** Source: B's engine.api.CLEARING_GRID_SUFFIX: a pair grid's id ends with this. */
export const CLEARING_GRID_SUFFIX = ".clears";

export interface ComboMove {
	/** Every knob the setting changes, in the grid's axis order, then the ones it holds fixed. */
	changes: Array<{ knobId: string; to: KnobValue }>;
	/** The target the move is ranked on (the aim's, or memory when the aim abstains) and its change. */
	targetId: string;
	delta: number | null;
	/** Failure probability after the move. */
	risk: number;
	/** "fits" may be applied; "tight" (at most CLEARS_P_OOM) is offered to measure first. */
	fit: "fits" | "tight";
	/** Peak memory after the move, when predicted. */
	peak: number | null;
	/** A knob of the move whose change on its own was predicted not to clear the failure, if one was. */
	alone: string | null;
}

export interface ComboOptions {
	pinned?: ReadonlySet<string>;
	/** The visitor's limits (guide/limits.ts withinLimits): false for a value they excluded or one that breaks a limit. */
	/** A tight combination is offered only if it beats this (the best single move's risk, tight ones included). */
	below?: number;
	allow?: (knobId: string, value: KnobValue, after: Prediction) => boolean;
	/** Visited markers: a combination already offered or turned down isn't offered again. */
	visited?: ReadonlySet<string>;
}

export const comboVisit = (c: Pick<ComboMove, "changes">) =>
	`combo:${[...c.changes]
		.map((x) => `${x.knobId}=${String(x.to)}`)
		.sort()
		.join("+")}`;

function same(a: KnobValue | undefined, b: KnobValue | undefined): boolean {
	if (typeof a === "number" && typeof b === "number") return Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(a));
	return a === b;
}

/** Every cell of a grid's axes (the product of their values). */
function* cells(axes: ReadonlyArray<{ knob_id: string; values: ReadonlyArray<KnobValue> }>): Generator<Record<string, KnobValue>> {
	if (axes.some((a) => !a.values.length)) return;
	const idx = axes.map(() => 0);
	for (;;) {
		yield Object.fromEntries(axes.map((a, i) => [a.knob_id, a.values[idx[i]!]!]));
		let i = axes.length - 1;
		while (i >= 0 && ++idx[i]! >= axes[i]!.values.length) idx[i--] = 0;
		if (i < 0) return;
	}
}

/**
 * Whether one knob alone clears the failure, for the aim or for memory: a combined move is offered
 * only when neither does. (The aim's own moves alone aren't enough: where the aim's number
 * abstains, as for a run whose length isn't known, the provider ranks nothing for it.)
 */
export function singleClears(record: LoadedRecord, provider: PredictionProvider, objective: Objective, opts: SensitivityOptions): boolean {
	const fits = (kind: Objective["kind"]) => provider.moves(record, { ...objective, kind }, opts).some((m) => m.fit === "fits");
	return fits(objective.kind ?? "min_time") || fits("fit_memory");
}

/** Knobs B leaves out of the pair search (engine.api._NOT_A_CLEARING_KNOB): the GPU is a separate decision, offered on its own. */
const NOT_A_CLEARING_KNOB = new Set(["gpu_type", "gpu", "gpus"]);

/**
 * The lowest failure risk any single move on the job's own settings reaches, tight moves included:
 * the bar a tight combination must beat. A GPU swap isn't counted, as in B's search: a bigger
 * card is its own choice, not a fix to the run.
 */
export function bestSingleRisk(record: LoadedRecord, provider: PredictionProvider, objective: Objective, opts: SensitivityOptions): number | undefined {
	const risks = [objective.kind ?? "min_time", "fit_memory" as const]
		.flatMap((kind) => provider.moves(record, { ...objective, kind }, { ...opts, includeTight: true }))
		.filter((m) => !NOT_A_CLEARING_KNOB.has(m.knobId))
		.map((m) => m.risk)
		.filter((r): r is number => typeof r === "number");
	return risks.length ? Math.min(...risks) : undefined;
}

/** `below` for bestCombo from bestSingleRisk, when there is one. */
export const beat = (risk: number | undefined): { below?: number } => (risk === undefined ? {} : { below: risk });

interface Cell {
	changes: ComboMove["changes"];
	after: Prediction;
	risk: number;
	/** Steps from the current setting along the grid's axes (B ranks pairs the same way). */
	steps: number;
}

const known = (id: string, p: Prediction) => !!p[id] && p[id]!.status !== "abstain" && typeof p[id]!.point === "number";

/**
 * The best combined setting from B's pair grids: one that fits before one that's only tight (at
 * most CLEARS_P_OOM, and better than `below`), then the fewest knobs, then the nearest to the
 * current setting, then the aim (or memory when the aim isn't predicted for every candidate: one
 * unit for the comparison). Null when the record has no pair grid or none of its cells clears.
 */
export function bestCombo(
	record: LoadedRecord,
	provider: PredictionProvider,
	knobs: Readonly<Record<string, KnobValue>>,
	objective: Objective,
	opts: ComboOptions = {},
): ComboMove | null {
	const fail = failureTarget(record);
	if (!fail) return null;
	const ids = new Set(record.knobs.map((k) => k.id));
	const seen = new Set<string>();
	const found: Cell[] = [];
	for (const grid of record.predictions.grids) {
		// Only B's pair grids (`.clears`: "no single move clears; this pair does somewhere"), not a
		// record's own multi-axis grids over its gridded knobs, which say nothing about clearing.
		if (grid.target_id !== fail || grid.axes.length < 2 || !grid.id.endsWith(CLEARING_GRID_SUFFIX)) continue;
		const fixed = (grid.fixed ?? {}) as Record<string, KnobValue>;
		// B holds every other knob where it was when the record was filled; once the visitor has
		// moved one, the grid no longer answers for this setting (and its move would undo theirs).
		if (Object.entries(fixed).some(([k, v]) => !grid.axes.some((a) => a.knob_id === k) && k in knobs && !same(knobs[k], v))) continue;
		// Axis order first, then the held knobs, so the words read as the grid does.
		const order = [...new Set([...grid.axes.map((a) => a.knob_id), ...Object.keys(fixed)])];
		const axisOf = new Map(grid.axes.map((a) => [a.knob_id, a.values] as const));
		const stepsFor = (knobId: string, to: KnobValue) => {
			const values = axisOf.get(knobId);
			if (!values) return 1;
			const from = values.findIndex((v) => same(v, knobs[knobId]));
			const at = values.findIndex((v) => same(v, to));
			return from < 0 || at < 0 ? 1 : Math.abs(at - from);
		};
		for (const cell of cells(grid.axes)) {
			const setting: Record<string, KnobValue> = { ...knobs, ...fixed, ...cell };
			const changes = order.filter((k) => ids.has(k) && !same(knobs[k], setting[k])).map((knobId) => ({ knobId, to: setting[knobId]! }));
			// One knob is a single move, which the tour ranks; more than the server takes isn't offered.
			if (changes.length < 2 || changes.length > MAX_COMBO_KNOBS) continue;
			if (changes.some((c) => opts.pinned?.has(c.knobId))) continue;
			const key = comboVisit({ changes });
			if (seen.has(key) || opts.visited?.has(key)) continue;
			seen.add(key);
			const after = provider.predict(record, setting);
			if (opts.allow && changes.some((c) => !opts.allow!(c.knobId, c.to, after))) continue;
			const risk = after[fail];
			if (!risk || risk.status === "abstain" || risk.point === null || risk.point > CLEARS_P_OOM) continue;
			// Tight is offered only where it beats the best single move, tight ones included.
			if (risk.point >= FITS_BELOW && opts.below !== undefined && risk.point >= opts.below) continue;
			found.push({ changes, after, risk: risk.point, steps: changes.reduce((n, c) => n + stepsFor(c.knobId, c.to), 0) });
		}
	}
	if (!found.length) return null;
	// A setting that fits beats one that's only tight; then the fewest knobs; then the nearest to
	// the current setting (the smallest change that clears, as B ranks pairs); then the aim.
	const fitting = found.filter((c) => c.risk < FITS_BELOW);
	const pool0 = fitting.length ? fitting : found;
	const fewest = Math.min(...pool0.map((c) => c.changes.length));
	const pool1 = pool0.filter((c) => c.changes.length === fewest);
	const nearest = Math.min(...pool1.map((c) => c.steps));
	const pool = pool1.filter((c) => c.steps === nearest);
	const base = provider.predict(record, knobs);
	const goal = objectiveTarget(record, objective.kind ?? "min_time", objective.target_id);
	const mem = targetFor(record, "memory")?.id ?? null;
	const onAim = !!goal && known(goal.id, base) && pool.every((c) => known(goal.id, c.after));
	const onMem = !!mem && known(mem, base) && pool.every((c) => known(mem, c.after));
	const targetId = onAim ? goal!.id : onMem ? mem! : fail;
	const sign = onAim && goal!.better === "higher" ? -1 : 1;
	const value = (p: Prediction) => (p[targetId]?.point as number | null) ?? Number.POSITIVE_INFINITY;
	const best = pool.reduce((a, b) => (sign * value(b.after) < sign * value(a.after) ? b : a));
	const now = base[targetId]?.point;
	const then = best.after[targetId]?.point;
	const peak = mem ? best.after[mem] : undefined;
	// The "alone isn't enough" line names a knob only where its own change was predicted not to clear.
	const alone =
		best.changes.find((c) => {
			const p = provider.predict(record, { ...knobs, [c.knobId]: c.to })[fail];
			return !!p && p.status !== "abstain" && typeof p.point === "number" && p.point > CLEARS_P_OOM;
		})?.knobId ?? null;
	return {
		changes: best.changes,
		targetId,
		delta: typeof now === "number" && typeof then === "number" ? then - now : null,
		risk: best.risk,
		fit: best.risk < FITS_BELOW ? "fits" : "tight",
		peak: peak && peak.status !== "abstain" ? peak.point : null,
		alone,
	};
}

/** As GuidePanel's valueText, which this module can't import (GuidePanel imports it). */
function valueWords(record: LoadedRecord, knobId: string, v: KnobValue | undefined): string {
	if (v === undefined || v === null) return "";
	if (typeof v === "boolean") return t(v ? "value.on" : "value.off").toLowerCase(); // mid-sentence
	if (typeof v === "string") return hasKey(`knobvalue.${v}`) ? t(`knobvalue.${v}`) : v;
	if (record.knobs.find((x) => x.id === knobId)?.unit === "tokens") return new Intl.NumberFormat("en-GB").format(v);
	if (!Number.isInteger(v) && Math.abs(v) < 1) return new Intl.NumberFormat("en-GB", { maximumSignificantDigits: 2 }).format(v);
	return fmt(v, "count");
}

/** The line and why keys for a combined move: without a peak, or without a knob checked alone, the words that need them aren't used. */
export function comboKeys(c: ComboMove): { textKey: string; whyKey: string } {
	const tight = c.fit === "tight" ? "_tight" : "";
	return { textKey: `${c.peak !== null ? "guide.combo.line" : "guide.combo.line_nopeak"}${tight}`, whyKey: c.alone ? "guide.combo.why" : "guide.combo.why_plain" };
}

/** The words for a combined move: each knob from and to, the peak after, the chance of running out. */
export function comboWords(record: LoadedRecord, current: Readonly<Record<string, KnobValue>>, c: ComboMove): Record<string, string> {
	const label = (id: string) => {
		const k = record.knobs.find((x) => x.id === id);
		return k ? t(k.label_key) : id;
	};
	const parts = c.changes.map((x) =>
		t("guide.combo.part", { knob: label(x.knobId).toLowerCase(), from: valueWords(record, x.knobId, current[x.knobId]), to: valueWords(record, x.knobId, x.to) }),
	);
	const tgt = target(record, c.targetId);
	const alone = c.alone ? label(c.alone) : "";
	return {
		first: alone ? alone[0]!.toUpperCase() + alone.slice(1) : "",
		changes: parts.length > 1 ? `${parts.slice(0, -1).join(", ")} ${t("guide.combo.and")} ${parts.at(-1)}` : (parts[0] ?? ""),
		combo_peak: c.peak !== null ? fmt(c.peak, "bytes") : "",
		combo_risk: fmt(c.risk, "probability"),
		combo_target: tgt ? t(tgt.label_key).toLowerCase() : "",
		ids: c.changes.map((x) => `${x.knobId}=${String(x.to)}`).join(", "),
		combo_delta: c.delta !== null && tgt ? fmtDelta(c.delta, tgt.unit) : "",
		n: String(c.changes.length),
	};
}
