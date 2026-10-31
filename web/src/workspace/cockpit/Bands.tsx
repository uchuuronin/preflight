/**
 * "The whole run, step by step": the flight plan along the run's steps, and what fills the
 * GPU at its peak. Both read the record and the provider; neither computes a prediction.
 */
import { useRef } from "react";
import { fmt, fmtRange } from "../../lib/fmt";
import { t } from "../../lib/i18n";
import { useOrbitLayout } from "../map/Orbit";
import { runPlan, targetFor } from "../record/model";
import { memoryStackScale } from "../record/scale";
import { memoryTermBytes, type MemorySegment } from "../record/terms";
import type { LoadedRecord, Prediction } from "../record/types";
import { useWorkspace } from "../state";
import { Info } from "../components/Info";
import { Provenance } from "../components/StatusMark";
import { bounds, likelyFails, trusted } from "../status";
import { outcomes, useHops } from "../hops";

const MAX_MARKS = 24;

interface PlanMark {
	kind: "eval" | "checkpoint";
	step: number;
	at: number;
}

/** Where the run is predicted to stop, if it is: the first eval when the eval spike drives it. */
function predictedStop(record: LoadedRecord, est: Prediction, evalEvery: number | null): number | null {
	if (!likelyFails(record, est)) return null;
	const evalDriven =
		memoryTermBytes(est, "eval_spike") > 0 ||
		record.findings.some((f) => f.node_id === "eval" && (f.severity === "error" || f.severity === "critical"));
	return evalDriven && evalEvery ? evalEvery : 1;
}

export function FlightPlanBand() {
	const { record, estimates, state, actions } = useWorkspace();
	const track = useRef<HTMLDivElement>(null);
	const plan = runPlan(record, state.knobs);
	const total = plan.totalSteps;
	const timeT = targetFor(record, "time");
	const costT = targetFor(record, "cost");
	const time = timeT ? estimates[timeT.id] : undefined;
	const cost = costT ? estimates[costT.id] : undefined;
	const stop = predictedStop(record, estimates, plan.evalEvery);

	if (!total) {
		return (
			<div className="glass pf-band">
				<div className="pf-band__head">
					<b>{t("plan.title")}</b>
					<span>{t("plan.no_length")}</span>
				</div>
			</div>
		);
	}

	const marks: PlanMark[] = [];
	const push = (kind: PlanMark["kind"], every: number | null) => {
		if (!every) return;
		const n = Math.floor(total / every);
		const stride = Math.max(1, Math.ceil(n / MAX_MARKS));
		for (let i = 1; i <= n; i += stride) marks.push({ kind, step: i * every, at: (i * every) / total });
	};
	push("eval", plan.evalEvery);
	push("checkpoint", plan.checkpointEvery);
	const stopAt = stop ? stop / total : null;
	const atPointer = (x: number) => {
		const r = track.current?.getBoundingClientRect();
		return r ? (x - r.left) / r.width : 0;
	};
	const playStep = Math.round(state.playhead * total);
	const note = stop
		? t("plan.stops_at", { step: fmt(stop, "count") })
		: t("plan.finishes", { steps: fmt(total, "count"), evals: fmt(plan.evalEvery ? Math.floor(total / plan.evalEvery) : 0, "count"), saves: fmt(plan.checkpointEvery ? Math.floor(total / plan.checkpointEvery) : 0, "count") });

	return (
		<div className="glass pf-band" aria-label={t("plan.label")}>
			<div className="pf-band__head">
				<b>{t("plan.title")}</b>
				<span className={stop ? "is-bad" : ""}>{note}</span>
				<span>
					{trusted(cost) && trusted(time)
						? t("plan.cost_over_time", { cost: fmtRange(...bounds(cost), "usd"), time: fmtRange(...bounds(time), "s") })
						: t("plan.cost_unknown")}
				</span>
				<Info label={t("plan.title")} align="end">
					{[
						t("plan.drag_hint"),
						plan.source === "epochs"
							? t("plan.counted", {
									epochs: fmt(plan.epochs ?? 0, "count"),
									examples: fmt(plan.examples ?? 0, "count"),
									per: fmt(plan.samplesPerStep ?? 0, "count"),
									steps: fmt(total, "count"),
								})
							: null,
					]
						.filter(Boolean)
						.join(" ")}
				</Info>
			</div>
			<div
				className="pf-ptrack"
				ref={track}
				onPointerDown={(e) => {
					e.currentTarget.setPointerCapture(e.pointerId);
					actions.setPlayhead(atPointer(e.clientX));
				}}
				onPointerMove={(e) => {
					if (e.buttons & 1) actions.setPlayhead(atPointer(e.clientX));
				}}
			>
				<div className="pf-pline" />
				{marks.map((m, i) => (
					<button
						type="button"
						key={`${m.kind}-${m.step}-${i}`}
						className={`pf-pmark pf-pmark--${m.kind}${stopAt !== null && m.at > stopAt ? " is-ghost" : ""}${stopAt !== null && m.kind === "eval" && Math.abs(m.at - stopAt) < 1e-9 ? " is-fail" : ""}`}
						style={{ left: `${m.at * 100}%` }}
						aria-label={t(`plan.event.${m.kind}`, { step: fmt(m.step, "count") })}
						onPointerDown={(e) => e.stopPropagation()}
						onClick={() => {
							actions.setPlayhead(m.at);
							actions.focus(m.kind === "eval" ? "eval" : (record.graph.nodes.find((n) => n.kind === "io")?.id ?? null));
						}}
					/>
				))}
				{stopAt !== null ? (
					<span className="pf-pflag" style={{ left: `${stopAt * 100}%` }}>
						{t("plan.flag_stop", { step: fmt(stop ?? 0, "count") })}
					</span>
				) : null}
				<div
					className="pf-playhead"
					style={{ left: `${state.playhead * 100}%` }}
					role="slider"
					tabIndex={0}
					aria-label={t("plan.playhead")}
					aria-valuemin={0}
					aria-valuemax={total}
					aria-valuenow={playStep}
					aria-valuetext={t("plan.playhead.value", { step: fmt(playStep, "count") })}
					onKeyDown={(e) => {
						const d = e.shiftKey ? 0.1 : 0.01;
						if (e.key === "ArrowRight") actions.setPlayhead(state.playhead + d);
						if (e.key === "ArrowLeft") actions.setPlayhead(state.playhead - d);
						if (e.key === "Home") actions.setPlayhead(0);
						if (e.key === "End") actions.setPlayhead(1);
					}}
				/>
			</div>
			<div className="pf-axis" aria-hidden="true">
				{[0, 0.25, 0.5, 0.75, 1].map((f) => (
					<span key={f}>{f === 0 ? t("plan.step0") : fmt(Math.round(total * f), "count")}</span>
				))}
			</div>
			{state.playhead > 0 && trusted(time) ? (
				<p className="pf-band__foot pf-num">
					{t("plan.at", { step: fmt(playStep, "count"), time: fmt(time.point * state.playhead, "s"), cost: trusted(cost) ? fmt(cost.point * state.playhead, "usd") : "" })}
				</p>
			) : null}
		</div>
	);
}

function groupLabel(s: MemorySegment): string {
	return t(`term.${s.key}`);
}

/** The terms inside a group, when it holds more than one ("Weights 13 GiB · CUDA context 1 GiB"). */
function partsLabel(s: MemorySegment): string {
	if (s.parts.length < 2) return "";
	return s.parts.map((p) => `${t(`target.memory.${p.term}`)} ${fmt(p.bytes, "bytes")}`).join(" · ");
}

/**
 * What fills the GPU at its peak: each predicted part at its own size on a scale fixed for the
 * record (record/scale.ts). Where parts and peak disagree, the difference is hatched and named
 * untrusted, never squeezed away; terms that abstain here are listed by name.
 */
export function MemoryBand() {
	const { record, estimates } = useWorkspace();
	const hops = useHops();
	const layout = useOrbitLayout();
	const { segments, total, capacity, breakdown, untrusted, abstaining, partsBytes } = layout.memory;
	const memT = targetFor(record, "memory");
	const peak = memT ? estimates[memT.id] : undefined;
	const scaleMax = memoryStackScale(record) ?? (capacity ? capacity * 1.25 : null);
	if (!segments.length || !capacity || !scaleMax) {
		return (
			<div className="glass pf-band">
				<div className="pf-band__head">
					<b>{t("membar.title")}</b>
					<span>{peak?.status === "abstain" ? t(`abstain.${peak.reason ?? "off_grid"}`) : t("membar.unknown")}</span>
				</div>
			</div>
		);
	}
	const pct = (b: number) => `${Math.min(100, (b / scaleMax) * 100).toFixed(2)}%`;
	const offScale = Math.max(total ?? 0, partsBytes) > scaleMax;
	const status = peak?.status ?? "abstain";
	const unit = memT?.unit === "fraction" ? "fraction" : "bytes";
	const hopDraws = hops.on && unit === "bytes" ? outcomes(peak) : [];
	const shownHops = hops.frame === null ? hopDraws : hopDraws.length ? [hopDraws[hops.frame % hopDraws.length]!] : [];
	const frameOver = hops.frame !== null && shownHops.length === 1 && shownHops[0]! > capacity;
	return (
		<div className="glass pf-band" aria-label={t("membar.title")}>
			<div className="pf-band__head">
				<b>{t("membar.title")}</b>
				<span>{total ? t("membar.line", { total: fmt(total, unit), cap: fmt(capacity, unit) }) : ""}</span>
				<Provenance status={status} />
				{!breakdown ? <span className="pf-dim">{t("membar.no_breakdown")}</span> : null}
				{frameOver ? <span className="pf-hop-note" aria-hidden="true">{t("hops.oom")}</span> : null}
			</div>
			<div className="pf-stack" role="img" aria-label={`${t("membar.aria", { total: fmt(total ?? 0, unit), cap: fmt(capacity, unit) })}${offScale ? `. ${t("membar.off_scale")}` : ""}`}>
				{segments.map((s) => (
					<div
						key={s.key}
						className={`pf-mseg pf-mseg--${s.slot}`}
						style={{ width: pct(s.bytes) }}
						title={s.key === "untrusted" ? t("membar.untrusted.gap.help", { gap: fmt(s.bytes, unit) }) : `${groupLabel(s)}: ${fmt(s.bytes, unit)}${partsLabel(s) ? ` (${partsLabel(s)})` : ""}`}
					/>
				))}
				{untrusted?.kind === "overshoot" && total ? (
					<div
						className="pf-mseg-over"
						style={{ left: pct(total), width: pct(untrusted.bytes) }}
						title={t("membar.untrusted.overshoot.help", { over: fmt(untrusted.bytes, unit) })}
					/>
				) : null}
				{total && breakdown ? (
					<div className="pf-peak" style={{ left: pct(total) }} title={t("membar.peak", { total: fmt(total, unit) })} />
				) : null}
				{shownHops.map((v, i) => (
					<div
						key={hops.frame === null ? i : "hop"}
						className={`pf-hop${v > capacity ? " is-over" : ""}${hops.frame === null ? " is-all" : ""}`}
						style={{ left: pct(v) }}
						aria-hidden="true"
					/>
				))}
				<div className="pf-cap" style={{ left: pct(capacity) }}>
					<span>{t("membar.capacity", { cap: fmt(capacity, unit) })}</span>
				</div>
				{offScale ? <span className="pf-stack__off" aria-hidden="true">›</span> : null}
			</div>
			<ul className="pf-legend">
				{segments.map((s) => (
					<li key={s.key} title={partsLabel(s) || undefined}>
						<i className={`pf-mseg--${s.slot}`} />
						{groupLabel(s)} <span className="pf-dim">{fmt(s.bytes, unit)}</span>
					</li>
				))}
				{untrusted?.kind === "overshoot" ? (
					<li>
						<i className="pf-mseg-over" />
						{t("membar.untrusted.overshoot")} <span className="pf-dim">{fmt(untrusted.bytes, unit)}</span>
					</li>
				) : null}
				{abstaining.length ? (
					<li className="pf-dim">{t("membar.untrusted.abstain", { terms: abstaining.map((x) => t(`target.memory.${x}`)).join(", ") })}</li>
				) : null}
			</ul>
		</div>
	);
}
