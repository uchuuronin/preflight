import { memo, useEffect, useId, useMemo, useRef } from "react";
import { fmt, fmtDelta } from "../../lib/fmt";
import { hasKey, t } from "../../lib/i18n";
import { useKnobDecorations } from "../../registry";
import { OBJECTIVE_TARGET } from "../provider/PredictionProvider";
import type { FlightRecord, KnobSpec, KnobValue, Prediction } from "../record/types";
import { useWorkspace } from "../state";
import { memoryFit, trusted, type Fit } from "../status";
import { FitMark } from "./StatusMark";

export function KnobRail() {
	const { record, state, estimates } = useWorkspace();
	return (
		<aside className="pf-rail" aria-label={t("rail.label")}>
			<Constraints record={record} />
			<h2 className="pf-rail__heading">{t("rail.knobs")}</h2>
			<ul className="pf-knobs">
				{record.knobs.map((k) => (
					<li key={k.id}>
						<Knob
							spec={k}
							value={state.knobs[k.id] ?? k.value}
							pinned={state.pinned.has(k.id)}
							proposed={state.proposal?.knobId === k.id}
							accent={state.highlight.includes(`knob:${k.id}`) || state.proposal?.knobId === k.id}
							trace={state.lastChange?.knobId === k.id ? traceFor(record, estimates, state.previous) : null}
						/>
					</li>
				))}
			</ul>
		</aside>
	);
}

function traceFor(record: FlightRecord, now: Prediction, before: Prediction | null): string | null {
	if (!before) return null;
	const id = OBJECTIVE_TARGET[record.job.objective];
	const a = before[id];
	const b = now[id];
	const target = record.predictions.targets.find((x) => x.id === id);
	if (!trusted(a) || !trusted(b) || !target) return null;
	return t("rail.trace", { target: t(target.label_key).toLowerCase(), delta: fmtDelta(b.point - a.point, target.unit) });
}

function Constraints({ record }: { record: FlightRecord }) {
	const c = record.job.constraints;
	const rows: Array<[string, string]> = [];
	if (c.budget_usd !== undefined) rows.push([t("constraint.budget"), fmt(c.budget_usd, "usd")]);
	if (c.deadline) {
		rows.push([
			t("constraint.deadline"),
			new Intl.DateTimeFormat(undefined, { weekday: "short", hour: "numeric", minute: "2-digit" }).format(
				new Date(c.deadline),
			),
		]);
	}
	if (c.gpus?.length) rows.push([t("constraint.gpus"), c.gpus.join(", ")]);
	if (c.wall_time_cap_s) rows.push([t("constraint.wall_time"), fmt(c.wall_time_cap_s, "s")]);
	if (c.quota_bytes) rows.push([t("constraint.quota"), fmt(c.quota_bytes, "bytes")]);
	if (!rows.length) return null;
	return (
		<dl className="pf-constraints">
			{rows.map(([k, v]) => (
				<div key={k} className="pf-constraints__row">
					<dt>{k}</dt>
					<dd>{v}</dd>
				</div>
			))}
		</dl>
	);
}

function knobValues(spec: KnobSpec): KnobValue[] {
	if ("values" in spec.domain && spec.domain.values) return spec.domain.values as KnobValue[];
	const d = spec.domain as { min: number; max: number; step?: number };
	const step = d.step ?? (spec.kind === "int" ? 1 : (d.max - d.min) / 20);
	const out: number[] = [];
	for (let v = d.min; v <= d.max + 1e-9; v += step) out.push(Number(v.toFixed(6)));
	return out;
}

type ValueStatus = Fit | "abstain";

function valueLabel(spec: KnobSpec, v: KnobValue): string {
	if (typeof v === "boolean") return t(v ? "value.on" : "value.off");
	if (typeof v === "string") return hasKey(`knobvalue.${v}`) ? t(`knobvalue.${v}`) : v;
	return spec.unit === "tokens" ? new Intl.NumberFormat().format(v) : fmt(v, spec.unit === "none" ? "count" : spec.unit);
}

const Knob = memo(function Knob({
	spec,
	value,
	pinned,
	proposed,
	accent,
	trace,
}: {
	spec: KnobSpec;
	value: KnobValue;
	pinned: boolean;
	proposed: boolean;
	accent: boolean;
	trace: string | null;
}) {
	const ws = useWorkspace();
	const { record, provider, state, actions } = ws;
	const id = useId();
	const values = knobValues(spec);
	const index = Math.max(
		0,
		values.findIndex((v) => v === value),
	);
	const decorations = useKnobDecorations(spec.id);

	// Status of every value of this knob with the others held where they are: the OOM
	// frontier and the abstention region drawn right on the control.
	const statuses = useMemo<ValueStatus[]>(
		() =>
			values.map((v) => {
				const p = provider.predict(record, { ...state.knobs, [spec.id]: v });
				const headline = p.peak_memory ?? p[OBJECTIVE_TARGET[record.job.objective]];
				if (!headline || headline.status === "abstain") return "abstain";
				return memoryFit(p, record);
			}),
		// eslint-disable-next-line react-hooks/exhaustive-deps
		[provider, record, spec.id, state.knobs],
	);

	// Bring the knob the guide is talking about into view, without stealing focus.
	const rootRef = useRef<HTMLDivElement>(null);
	useEffect(() => {
		if (!accent) return;
		const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
		rootRef.current?.scrollIntoView({ block: "nearest", behavior: reduce ? "auto" : "smooth" });
	}, [accent]);

	const label = t(spec.label_key);
	const onPick = (v: KnobValue) => actions.setKnob(spec.id, v, "user");

	return (
		<div ref={rootRef} className={`pf-knob${accent ? " is-accent" : ""}${pinned ? " is-pinned" : ""}`} data-knob={spec.id}>
			<div className="pf-knob__head">
				<label className="pf-knob__label" htmlFor={id}>
					{label}
				</label>
				{spec.needs_code ? (
					<span className="pf-badge pf-badge--code" title={t("knob.needs_code.help")}>
						{t("knob.needs_code")}
					</span>
				) : null}
				<output className="pf-knob__value" htmlFor={id}>
					{valueLabel(spec, value)}
				</output>
				<button
					type="button"
					className={`pf-pin${pinned ? " is-on" : ""}`}
					aria-pressed={pinned}
					onClick={() => actions.pin(spec.id, !pinned)}
					title={pinned ? t("knob.unpin.help") : t("knob.pin.help")}
				>
					<svg width="14" height="14" viewBox="0 0 14 14" aria-hidden="true">
						<path d="M5 1.5h4l-.7 4 2.2 2H3.5l2.2-2z M7 7.5v5" />
					</svg>
					<span className="pf-visually-hidden">{pinned ? t("knob.unpin", { knob: label }) : t("knob.pin", { knob: label })}</span>
				</button>
			</div>

			{spec.kind === "enum" || spec.kind === "bool" ? (
				<div className="pf-segmented" role="radiogroup" aria-labelledby={id} id={id}>
					{values.map((v, i) => (
						<button
							key={String(v)}
							type="button"
							role="radio"
							aria-checked={v === value}
							className={`pf-segmented__opt pf-vs--${statuses[i]}${v === value ? " is-on" : ""}`}
							onClick={() => onPick(v)}
						>
							<span>{valueLabel(spec, v)}</span>
							{statuses[i] === "fails" ? <FitMark fit="fails" compact /> : null}
						</button>
					))}
				</div>
			) : (
				<div className="pf-slider">
					<input
						id={id}
						type="range"
						min={0}
						max={values.length - 1}
						step={1}
						value={index}
						aria-valuetext={`${valueLabel(spec, value)}, ${t(`fit.${statuses[index] ?? "unknown"}`)}`}
						onChange={(e) => {
							const v = values[Number(e.currentTarget.value)];
							if (v !== undefined) onPick(v);
						}}
					/>
					<div className="pf-slider__ticks" aria-hidden="true">
						{values.map((v, i) => (
							<span
								key={String(v)}
								className={`pf-tick pf-vs--${statuses[i]}${i === index ? " is-on" : ""}`}
								style={{ left: `${values.length > 1 ? (i / (values.length - 1)) * 100 : 50}%` }}
							/>
						))}
					</div>
					<div className="pf-slider__scale" aria-hidden="true">
						<span>{valueLabel(spec, values[0] ?? "")}</span>
						<span>{valueLabel(spec, values[values.length - 1] ?? "")}</span>
					</div>
				</div>
			)}

			{proposed ? <p className="pf-knob__note is-accent">{t("knob.proposed")}</p> : null}
			{!proposed && trace ? <p className="pf-knob__note">{trace}</p> : null}
			{decorations.map((d) => (
				<d.component key={d.id} knob={spec} value={value} record={record} knobs={state.knobs} estimates={ws.estimates} />
			))}
		</div>
	);
});
