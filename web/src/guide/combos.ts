/**
 * Combined moves: when no single knob clears a predicted failure, a setting that changes two or
 * three knobs together may (the live LoRA record, Nov 1: the eval fix alone isn't enough at
 * seq_len 1,024). Only settings the record's own predictions cover are offered: B's multi-axis
 * grids (contract Grid, up to 4 axes, with `fixed` for the knobs held elsewhere), read through
 * the provider like any other setting. Nothing here adds up single-knob effects.
 */
import { fmt, fmtDelta } from "../lib/fmt";
import { hasKey, t } from "../lib/i18n";
import type { PredictionProvider } from "../workspace/provider/PredictionProvider";
import {
	failureTarget,
	objectiveTarget,
	target,
	targetFor,
} from "../workspace/record/model";
import { FITS_BELOW } from "../workspace/record/terms";
import type {
	KnobValue,
	LoadedRecord,
	Objective,
	Prediction,
} from "../workspace/record/types";

export interface ComboMove {
	/** Every knob the setting changes, in the grid's axis order, then the ones it holds fixed. */
	changes: Array<{ knobId: string; to: KnobValue }>;
	/** The target the move is ranked on (the aim's, or memory when the aim abstains) and its change. */
	targetId: string;
	delta: number | null;
	/** Failure probability after the move. */
	risk: number;
	/** Peak memory after the move, when predicted. */
	peak: number | null;
}

export interface ComboOptions {
	pinned?: ReadonlySet<string>;
	/** The visitor's limits (guide/limits.ts withinLimits): false for a value they excluded or one that breaks a limit. */
	allow?: (knobId: string, value: KnobValue, after: Prediction) => boolean;
	/** Visited markers: a combination already offered isn't offered again. */
	visited?: ReadonlySet<string>;
}

export const comboVisit = (c: Pick<ComboMove, "changes">) =>
	`combo:${c.changes.map((x) => `${x.knobId}=${String(x.to)}`).join("+")}`;

function same(a: KnobValue | undefined, b: KnobValue | undefined): boolean {
	if (typeof a === "number" && typeof b === "number")
		return Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(a));
	return a === b;
}

/** Every cell of a grid's axes (the product of their values). */
function* cells(
	axes: ReadonlyArray<{ knob_id: string; values: ReadonlyArray<KnobValue> }>,
): Generator<Record<string, KnobValue>> {
	const idx = axes.map(() => 0);
	if (axes.some((a) => !a.values.length)) return;
	for (;;) {
		yield Object.fromEntries(
			axes.map((a, i) => [a.knob_id, a.values[idx[i]!]!]),
		);
		let i = axes.length - 1;
		while (i >= 0 && ++idx[i]! >= axes[i]!.values.length) idx[i--] = 0;
		if (i < 0) return;
	}
}

/**
 * The best combined setting that clears the failure, from the record's multi-axis grids: the
 * fewest knobs changed, then the best for the aim (memory when the aim isn't predicted there).
 * Null when the record has no such grid or no cell of one fits.
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
	const mem = targetFor(record, "memory")?.id ?? null;
	const goal = objectiveTarget(
		record,
		objective.kind ?? "min_time",
		objective.target_id,
	);
	const base = provider.predict(record, knobs);
	const known = new Set(record.knobs.map((k) => k.id));
	const seen = new Set<string>();
	let best: (ComboMove & { score: number }) | null = null;
	for (const grid of record.predictions.grids) {
		if (grid.target_id !== fail || grid.axes.length < 2) continue;
		for (const cell of cells(grid.axes)) {
			const setting: Record<string, KnobValue> = {
				...knobs,
				...((grid.fixed ?? {}) as Record<string, KnobValue>),
				...cell,
			};
			const changes = Object.entries(setting)
				.filter(([k, v]) => known.has(k) && !same(knobs[k], v))
				.map(([knobId, to]) => ({ knobId, to }));
			// One knob is a single move, which the tour already ranks; nothing changed is no move.
			if (changes.length < 2) continue;
			if (changes.some((c) => opts.pinned?.has(c.knobId))) continue;
			const key = comboVisit({ changes });
			if (seen.has(key) || opts.visited?.has(key)) continue;
			seen.add(key);
			const after = provider.predict(record, setting);
			if (
				opts.allow &&
				changes.some((c) => !opts.allow!(c.knobId, c.to, after))
			)
				continue;
			const risk = after[fail];
			if (
				!risk ||
				risk.status === "abstain" ||
				risk.point === null ||
				risk.point >= FITS_BELOW
			)
				continue;
			const aimEst = goal ? after[goal.id] : undefined;
			const onAim =
				!!goal &&
				!!aimEst &&
				aimEst.status !== "abstain" &&
				aimEst.point !== null &&
				base[goal.id]?.point != null;
			const targetId = onAim ? goal!.id : (mem ?? fail);
			const now = base[targetId]?.point;
			const then = after[targetId]?.point;
			const delta =
				typeof now === "number" && typeof then === "number"
					? then - now
					: null;
			const better =
				onAim && goal!.better === "higher"
					? -(delta ?? 0)
					: (delta ?? 0);
			const score = changes.length * 1e18 + better;
			if (!best || score < best.score) {
				const peak = mem ? after[mem] : undefined;
				best = {
					changes,
					targetId,
					delta,
					risk: risk.point,
					peak: peak && peak.status !== "abstain" ? peak.point : null,
					score,
				};
			}
		}
	}
	if (!best) return null;
	const { score: _score, ...move } = best;
	return move;
}

/** As GuidePanel's valueText, which this module can't import (GuidePanel imports it). */
function valueWords(
	record: LoadedRecord,
	knobId: string,
	v: KnobValue | undefined,
): string {
	if (v === undefined || v === null) return "";
	if (typeof v === "boolean") return t(v ? "value.on" : "value.off");
	if (typeof v === "string")
		return hasKey(`knobvalue.${v}`) ? t(`knobvalue.${v}`) : v;
	if (record.knobs.find((x) => x.id === knobId)?.unit === "tokens")
		return new Intl.NumberFormat("en-GB").format(v);
	if (!Number.isInteger(v) && Math.abs(v) < 1)
		return new Intl.NumberFormat("en-GB", {
			maximumSignificantDigits: 2,
		}).format(v);
	return fmt(v, "count");
}

/** The words for a combined move: each knob from and to, the peak after, the chance of running out. */
export function comboWords(
	record: LoadedRecord,
	current: Readonly<Record<string, KnobValue>>,
	c: ComboMove,
): Record<string, string> {
	const label = (id: string) => {
		const k = record.knobs.find((x) => x.id === id);
		return k ? t(k.label_key) : id;
	};
	const parts = c.changes.map((x) =>
		t("guide.combo.part", {
			knob: label(x.knobId).toLowerCase(),
			from: valueWords(record, x.knobId, current[x.knobId]),
			to: valueWords(record, x.knobId, x.to),
		}),
	);
	const tgt = target(record, c.targetId);
	const first = c.changes[0] ? label(c.changes[0].knobId) : "";
	return {
		first: first ? first[0]!.toUpperCase() + first.slice(1) : "",
		changes:
			parts.length > 1
				? `${parts.slice(0, -1).join(", ")} ${t("guide.combo.and")} ${parts.at(-1)}`
				: (parts[0] ?? ""),
		combo_peak: c.peak !== null ? fmt(c.peak, "bytes") : t("value.none"),
		combo_risk: fmt(c.risk, "probability"),
		combo_target: tgt ? t(tgt.label_key).toLowerCase() : "",
		combo_delta: c.delta !== null && tgt ? fmtDelta(c.delta, tgt.unit) : "",
		n: String(c.changes.length),
	};
}
