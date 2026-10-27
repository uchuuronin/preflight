/**
 * "The whole run, step by step": the flight plan along the run's steps, and what fills the
 * GPU at its peak. Both read the record and the provider; neither computes a prediction.
 */
import { useRef } from "react";
import { fmt, fmtRange } from "../../lib/fmt";
import { t } from "../../lib/i18n";
import { useOrbitLayout } from "../map/Orbit";
import { failureTarget, runPlan, targetFor } from "../record/model";
import type { LoadedRecord, Prediction } from "../record/types";
import { useWorkspace } from "../state";
import { bounds, trusted } from "../status";

const MAX_MARKS = 24;

interface PlanMark {
	kind: "eval" | "checkpoint";
	step: number;
	at: number;
}

/** Where the run is predicted to stop, if it is: the first eval when the eval spike drives it. */
function predictedStop(record: LoadedRecord, est: Prediction, evalEvery: number | null): number | null {
	const fail = failureTarget(record);
	const risk = fail ? est[fail] : undefined;
	if (!trusted(risk) || risk.point <= 0.5) return null;
	const evalDriven =
		(trusted(est["memory.term.eval"]) && (est["memory.term.eval"]?.point ?? 0) > 0) ||
		record.findings.some((f) => f.node_id === "eval" && (f.severity === "error" || f.severity === "critical"));
	return evalDriven && evalEvery ? evalEvery : 1;
}

export function FlightPlanBand() {
	const { record, estimates, state, actions } = useWorkspace();
	const track = useRef<HTMLDivElement>(null);
	const plan = runPlan(record);
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
			<p className="pf-band__foot">
				{state.playhead > 0 && trusted(time)
					? t("plan.at", { step: fmt(playStep, "count"), time: fmt(time.point * state.playhead, "s"), cost: trusted(cost) ? fmt(cost.point * state.playhead, "usd") : "" })
					: t("plan.drag_hint")}
			</p>
		</div>
	);
}

const TERM_KEYS: Record<string, string> = {
	weights: "term.weights",
	training: "term.training",
	activations: "term.activations",
	logits: "term.logits",
	eval: "term.eval_spike",
	total: "term.total",
};

export function MemoryBand() {
	const { record, estimates } = useWorkspace();
	const layout = useOrbitLayout();
	const { segments, total, capacity, breakdown } = layout.memory;
	const memT = targetFor(record, "memory");
	const peak = memT ? estimates[memT.id] : undefined;
	if (!segments.length || !capacity) {
		return (
			<div className="glass pf-band">
				<div className="pf-band__head">
					<b>{t("membar.title")}</b>
					<span>{peak?.status === "abstain" ? t(`abstain.${peak.reason ?? "off_grid"}`) : t("membar.unknown")}</span>
				</div>
			</div>
		);
	}
	const scaleMax = Math.max(capacity * 1.25, total ?? 0);
	return (
		<div className="glass pf-band" aria-label={t("membar.title")}>
			<div className="pf-band__head">
				<b>{t("membar.title")}</b>
				<span>{total ? t("membar.line", { total: fmt(total, "bytes"), cap: fmt(capacity, "bytes") }) : ""}</span>
				{!breakdown ? <span className="pf-dim">{t("membar.no_breakdown")}</span> : null}
			</div>
			<div className="pf-stack" role="img" aria-label={t("membar.aria", { total: fmt(total ?? 0, "bytes"), cap: fmt(capacity, "bytes") })}>
				{segments.map((s) => (
					<div key={s.key} className={`pf-mseg pf-mseg--${s.slot}`} style={{ width: `${((s.bytes / scaleMax) * 100).toFixed(2)}%` }} title={`${t(TERM_KEYS[s.key] ?? s.key)}: ${fmt(s.bytes, "bytes")}`} />
				))}
				<div className="pf-cap" style={{ left: `${((capacity / scaleMax) * 100).toFixed(2)}%` }}>
					<span>{t("membar.capacity", { cap: fmt(capacity, "bytes") })}</span>
				</div>
			</div>
			<ul className="pf-legend">
				{segments.map((s) => (
					<li key={s.key}>
						<i className={`pf-mseg--${s.slot}`} />
						{t(TERM_KEYS[s.key] ?? s.key)} <span className="pf-dim">{fmt(s.bytes, "bytes")}</span>
					</li>
				))}
			</ul>
		</div>
	);
}
