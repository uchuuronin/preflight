/**
 * The knob dock: one capsule per knob (label, value, a dot when pinned, a glow when the guide
 * is on it). Pressing one opens its control above the dock: a slider over the values the
 * model can answer, with fits / tight / fails / not measured here drawn on each tick, or a choice
 * row; plus pin and, for code knobs, what changing it involves.
 */
import { memo, useEffect, useId, useMemo, useRef, useState } from "react";
import { hasKey, t } from "../../lib/i18n";
import { applyKnobViews, type KnobViewContext, useKnobDecorations, useKnobViews } from "../../registry";
import { valueText } from "../../guide/GuidePanel";
import { knobChoices } from "../record/model";
import type { Knob, KnobValue } from "../record/types";
import { guidedKnobs, useRanked, useWorkspace } from "../state";
import { fitOf, type Fit } from "../status";
import { flag } from "../../lib/flags";
import { rangeTestOf } from "./lrRange";
import { RangeTestCurve } from "./RangeTest";
import { Info } from "../components/Info";
import { FitMark } from "../components/StatusMark";
import { AddonBoundary } from "../../components/AddonBoundary";
import { WakeStrip } from "./Dials";

type ValueStatus = Fit | "abstain";

function useValueStatuses(knob: Knob, values: KnobValue[]): ValueStatus[] {
	const { record, provider, state } = useWorkspace();
	return useMemo(() => {
		// The fit first: a measured out-of-memory risk is a verdict even when the peak itself has no number
		// (the LoRA sample at batch 4: the dial said 100%, the slider said "not predicted"; Chiya, Nov 2).
		return values.map((v) => {
			const knobs = { ...state.knobs, [knob.id]: v };
			const p = provider.predict(record, knobs);
			const f = fitOf(record, p, knobs);
			return f === "unknown" ? "abstain" : f;
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
	// The learning-rate range test's curve, when the record has one (D, Nov 2; behind probes.lr_range).
	const rangeTest = useMemo(() => (knob.id === "lr" && flag("probes.lr_range") ? rangeTestOf(record) : null), [knob.id, record]);
	// The editor opens above the rail, and on a laptop the rail sits at the fold: without this only its
	// title showed (Chiya, Nov 2). Scrolled the least that brings it into view, once, when it opens.
	const box = useRef<HTMLDivElement>(null);
	const openedByGuide = useRef(state.activeKnobBy === "guide");
	useEffect(() => {
		if (openedByGuide.current) return; // the guide's own move: the page stays where the person is reading
		const still = typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
		box.current?.scrollIntoView?.({ block: "nearest", behavior: still ? "auto" : "smooth" });
	}, []);

	return (
		<div ref={box} className="glass pf-control" role="group" aria-labelledby={`${id}-title`}>
			<div className="pf-control__head">
				<b id={`${id}-title`}>{label}</b>
				<span>
					{proposed
						? t("knob.proposed")
						: state.limits.frozen.includes(knob.id)
							? t("dock.frozen_hint")
							: pinned
								? t("knob.pinned_hint", { value: valueText(knob, value) })
								: t(`knob.hint.${knob.group ?? "other"}`)}
				</span>
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
							{/* The same marks as the slider's ticks: a dot fits, a triangle is tight, a cross fails; hatching means not predicted. */}
							{statuses[i] === "fails" || statuses[i] === "tight" || statuses[i] === "fits" ? (
								<>
									<FitMark fit={statuses[i] as Fit} label="" />
									<span className="pf-visually-hidden">{t(`fit.${statuses[i]}`)}</span>
								</>
							) : null}
						</button>
					))}
				</div>
			)}
			{rangeTest ? <RangeTestCurve test={rangeTest} value={value} choices={values} onPick={pinned ? null : (v) => actions.setKnob(knob.id, v, "user")} /> : null}
			<WakeStrip />
			<div className="pf-control__foot">
				{status === "abstain" ? <span>{t("knob.not_trusted_here")}</span> : null}
				{knob.needs_code ? <span>{t("knob.needs_code.help")}</span> : null}
				{!pinned ? (
					<button type="button" className="pf-link" onClick={() => actions.pin(knob.id, true)}>
						{t("knob.pin", { value: valueText(knob, value) })}
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
						onPick={() => actions.openKnob(state.activeKnob === k.id ? null : k.id, "user")}
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
			<span className="pf-dock__key">
				<Info label={t("rail.knobs")}>{t("dock.key")}</Info>
			</span>
		</div>
	);
}
