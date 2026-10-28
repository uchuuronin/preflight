/**
 * The knob dock: one capsule per knob (label, value, a dot when pinned, a glow when the guide
 * is on it). Pressing one opens its control above the dock: a slider over the values the
 * model can answer, with fits / tight / fails / not trusted drawn on each tick, or a choice
 * row; plus pin and, for code knobs, what changing it involves.
 */
import { memo, useId, useMemo, useState } from "react";
import { hasKey, t } from "../../lib/i18n";
import { applyKnobViews, type KnobViewContext, useKnobDecorations, useKnobViews } from "../../registry";
import { valueText } from "../../guide/GuidePanel";
import { knobChoices, targetFor } from "../record/model";
import type { Knob, KnobValue } from "../record/types";
import { guidedKnobs, useRanked, useWorkspace } from "../state";
import { fitOf, type Fit } from "../status";
import { FitMark } from "../components/StatusMark";
import { AddonBoundary } from "../../components/AddonBoundary";

type ValueStatus = Fit | "abstain";

function useValueStatuses(knob: Knob, values: KnobValue[]): ValueStatus[] {
	const { record, provider, state } = useWorkspace();
	return useMemo(() => {
		const memT = targetFor(record, "memory");
		return values.map((v) => {
			const knobs = { ...state.knobs, [knob.id]: v };
			const p = provider.predict(record, knobs);
			const head = memT ? p[memT.id] : undefined;
			if (!head || head.status === "abstain") return "abstain";
			return fitOf(record, p, knobs);
		});
	}, [record, provider, state.knobs, knob.id, values]);
}

function Control({ knob, labelKey, ranked }: { knob: Knob; labelKey: string; ranked: readonly string[] }) {
	const ws = useWorkspace();
	const { record, state, actions, estimates, provider } = ws;
	const id = useId();
	const values = useMemo(() => knobChoices(record, knob), [record, knob]);
	const statuses = useValueStatuses(knob, values);
	const value = state.knobs[knob.id] ?? knob.current;
	const index = Math.max(0, values.findIndex((v) => v === value));
	const pinned = state.pinned.has(knob.id);
	const decorations = useKnobDecorations(knob.id);
	const label = t(labelKey);
	const status = statuses[index] ?? "unknown";
	const proposed = state.proposal?.knobId === knob.id;
	const numeric = knob.kind === "int" || knob.kind === "float";

	return (
		<div className="glass pf-control" role="group" aria-labelledby={`${id}-title`}>
			<div className="pf-control__head">
				<b id={`${id}-title`}>{label}</b>
				<span>{proposed ? t("knob.proposed") : pinned ? t("knob.pinned_hint") : t(`knob.hint.${knob.group ?? "other"}`)}</span>
			</div>
			{pinned ? (
				<div className="pf-choice">
					<button type="button" onClick={() => actions.pin(knob.id, false)}>
						{t("knob.unpin_so_it_moves")}
					</button>
				</div>
			) : numeric && values.length > 1 ? (
				<>
					<input
						className="pf-range"
						type="range"
						min={0}
						max={values.length - 1}
						step={1}
						value={index}
						aria-label={label}
						aria-valuetext={`${valueText(knob, value)}, ${status === "abstain" ? t("provenance.abstain") : t(`fit.${status}`)}`}
						onChange={(e) => {
							const v = values[Number(e.currentTarget.value)];
							if (v !== undefined) actions.setKnob(knob.id, v, "user");
						}}
					/>
					<div className="pf-ticks" aria-hidden="true">
						{values.map((v, i) => (
							<span
								key={String(v)}
								className={`pf-tick pf-tick--${statuses[i]}${i === index ? " is-on" : ""}`}
								style={{ left: `calc(11px + (100% - 22px) * ${values.length > 1 ? i / (values.length - 1) : 0})` }}
							>
								<i />
								{valueText(knob, v)}
							</span>
						))}
					</div>
				</>
			) : (
				<div className="pf-choice" role="radiogroup" aria-label={label}>
					{values.map((v, i) => (
						<button
							key={String(v)}
							type="button"
							role="radio"
							aria-checked={v === value}
							className={`pf-choice__opt pf-choice__opt--${statuses[i]}`}
							onClick={() => actions.setKnob(knob.id, v, "user")}
						>
							{valueText(knob, v)}
							{statuses[i] === "fails" ? <FitMark fit="fails" label="" /> : null}
						</button>
					))}
				</div>
			)}
			<div className="pf-control__foot">
				{status === "abstain" ? <span>{t("knob.not_trusted_here")}</span> : null}
				{knob.needs_code ? <span>{t("knob.needs_code.help")}</span> : null}
				{!pinned ? (
					<button type="button" className="pf-link" onClick={() => actions.pin(knob.id, true)}>
						{t("knob.pin", { knob: label.toLowerCase() })}
					</button>
				) : null}
			</div>
			{decorations.map((d) => (
				<AddonBoundary key={d.id} id={d.id}>
					<d.component knob={knob} value={value} record={record} knobs={state.knobs} estimates={estimates} provider={provider} ranked={ranked} />
				</AddonBoundary>
			))}
		</div>
	);
}

const Capsule = memo(function Capsule({
	knob,
	labelKey,
	value,
	active,
	pinned,
	guideOn,
	onPick,
}: {
	knob: Knob;
	labelKey: string;
	value: KnobValue;
	active: boolean;
	pinned: boolean;
	guideOn: boolean;
	onPick: () => void;
}) {
	return (
		<button type="button" className={`pf-knob${guideOn ? " is-guide" : ""}`} aria-pressed={active} onClick={onPick}>
			{pinned ? <span className="pf-knob__pin" aria-label={t("knob.pinned")} /> : null}
			{t(labelKey)} <b>{valueText(knob, value)}</b>
			{knob.needs_code ? <span className="pf-knob__code">{t("knob.needs_code")}</span> : null}
		</button>
	);
});

export function Dock() {
	const { record, state, actions } = useWorkspace();
	const views = useKnobViews();
	const ranked = useRanked(views.length > 0 || state.activeKnob !== null);
	const [showAll, setShowAll] = useState(false);
	const active = record.knobs.find((k) => k.id === state.activeKnob) ?? null;
	// Knobs the model can answer for come first; code knobs last.
	const ordered = useMemo(
		() =>
			[...record.knobs].sort(
				(a, b) => Number(b.gridded ?? false) - Number(a.gridded ?? false) || Number(a.needs_code ?? false) - Number(b.needs_code ?? false),
			),
		[record.knobs],
	);
	const guided = guidedKnobs(state);
	const ctx: KnobViewContext = {
		record,
		lens: record.job.lens,
		knobs: state.knobs,
		pinned: state.pinned,
		highlight: guided,
		ranked,
		active: state.activeKnob,
	};
	const view = applyKnobViews(ordered, ctx, views, hasKey);
	const shown = showAll ? ordered : view.shown;
	const labelOf = (k: Knob) => view.labelKeys[k.id] ?? k.label_key;
	return (
		<div className="pf-dock">
			{active ? <Control key={active.id} knob={active} labelKey={labelOf(active)} ranked={ranked} /> : null}
			<div className="glass pf-dock__row" role="group" aria-label={t("rail.knobs")}>
				{shown.map((k) => (
					<Capsule
						key={k.id}
						knob={k}
						labelKey={labelOf(k)}
						value={state.knobs[k.id] ?? k.current}
						active={state.activeKnob === k.id}
						pinned={state.pinned.has(k.id)}
						guideOn={guided.has(k.id)}
						onPick={() => actions.openKnob(state.activeKnob === k.id ? null : k.id)}
					/>
				))}
				{view.hidden.length ? (
					<button type="button" className="pf-knob pf-knob--more" aria-expanded={showAll} onClick={() => setShowAll(!showAll)}>
						{showAll
							? t("dock.fewer")
							: t(view.moreKey && hasKey(view.moreKey) ? view.moreKey : "dock.more", { n: view.hidden.length })}
					</button>
				) : null}
			</div>
			<p className="pf-dock__key">{t("dock.key")}</p>
		</div>
	);
}
