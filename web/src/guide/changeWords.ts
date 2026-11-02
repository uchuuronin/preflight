/**
 * How a guide step words what a move does to the aim: "changes run cost by -1.7%", or, when the
 * aim's range before and after the move overlap, "costs about the same". A signed percentage
 * inside the noise reads as a saving that may not be there (B, Nov 2: the B200 offer).
 */
import { fmtDelta } from "../lib/fmt";
import { t } from "../lib/i18n";
import type { PredictionProvider } from "../workspace/provider/PredictionProvider";
import type { KnobValue, LoadedRecord, Prediction, Target } from "../workspace/record/types";
import { bounds, type TrustedEstimate, trusted } from "../workspace/status";

function overlap(a: TrustedEstimate, b: TrustedEstimate): boolean {
	const [alo, ahi] = bounds(a);
	const [blo, bhi] = bounds(b);
	return alo <= bhi && blo <= ahi;
}

export function changeWords(
	ctx: { record: LoadedRecord; provider: Pick<PredictionProvider, "predict">; knobs: Readonly<Record<string, KnobValue>>; estimates: Prediction },
	move: { knobId: string; to: KnobValue; delta: number } | undefined,
	from: KnobValue | undefined,
	goal: Pick<Target, "id" | "unit" | "label_key" | "readout"> | undefined,
): Record<"change" | "target_change" | "would_change", string> {
	if (!move || !goal) return { change: "", target_change: "", would_change: "" };
	const vars = { target: t(goal.label_key).toLowerCase(), delta: fmtDelta(move.delta, goal.unit) };
	const before = from === undefined ? ctx.estimates[goal.id] : ctx.provider.predict(ctx.record, { ...ctx.knobs, [move.knobId]: from })[goal.id];
	const after = ctx.provider.predict(ctx.record, { ...ctx.knobs, [move.knobId]: move.to })[goal.id];
	const same = trusted(before) && trusted(after) && overlap(before, after);
	const kind = !same ? "by" : goal.readout === "cost" ? "same_cost" : "same";
	return {
		change: t(`guide.change.${kind}`, vars),
		target_change: t(`guide.target_change.${kind}`, vars),
		would_change: t(`guide.would_change.${kind}`, vars),
	};
}
