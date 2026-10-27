/**
 * "The whole run, step by step": the flight plan along the run's steps, and what fills the
 * GPU at its peak. Both read the record and the provider; neither computes a prediction.
 */
import { useRef } from "react";
import { errorCopy } from "../../lib/errors";
import { fmt, fmtRange } from "../../lib/fmt";
import { t } from "../../lib/i18n";
import { useOrbitLayout } from "../map/Orbit";
import { runPlan, targetFor } from "../record/model";
import { memoryTermBytes, type MemorySegment } from "../record/terms";
import type { KnobValue, LoadedRecord, Prediction } from "../record/types";
import { useWorkspace } from "../state";
import { valueText } from "../../guide/GuidePanel";
import { FitMark } from "../components/StatusMark";
import { bounds, type Fit, likelyFails, trusted } from "../status";

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
			{plan.source === "epochs" ? (
				<p className="pf-band__foot">
					{t("plan.counted", {
						epochs: fmt(plan.epochs ?? 0, "count"),
						examples: fmt(plan.examples ?? 0, "count"),
						per: fmt(plan.samplesPerStep ?? 0, "count"),
						steps: fmt(total, "count"),
					})}
				</p>
			) : null}
			<p className="pf-band__foot">
				{state.playhead > 0 && trusted(time)
					? t("plan.at", { step: fmt(playStep, "count"), time: fmt(time.point * state.playhead, "s"), cost: trusted(cost) ? fmt(cost.point * state.playhead, "usd") : "" })
					: t("plan.drag_hint")}
			</p>
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
					<div key={s.key} className={`pf-mseg pf-mseg--${s.slot}`} style={{ width: `${((s.bytes / scaleMax) * 100).toFixed(2)}%` }} title={`${groupLabel(s)}: ${fmt(s.bytes, "bytes")}${partsLabel(s) ? ` (${partsLabel(s)})` : ""}`} />
				))}
				<div className="pf-cap" style={{ left: `${((capacity / scaleMax) * 100).toFixed(2)}%` }}>
					<span>{t("membar.capacity", { cap: fmt(capacity, "bytes") })}</span>
				</div>
			</div>
			<ul className="pf-legend">
				{segments.map((s) => (
					<li key={s.key}>
						<i className={`pf-mseg--${s.slot}`} />
						{groupLabel(s)} <span className="pf-dim">{fmt(s.bytes, "bytes")}</span>
						{partsLabel(s) ? <small className="pf-dim">{partsLabel(s)}</small> : null}
					</li>
				))}
			</ul>
		</div>
	);
}

const PROBE_FIT: Record<string, Fit> = {
	ok: "fits",
	cached: "fits",
	oom: "fails",
	error: "fails",
	timeout: "fails",
	cancelled: "unknown",
	queued: "unknown",
	running: "unknown",
};

/**
 * "What was measured": every probe in the record with the settings it ran at, what it cost,
 * its peak memory, and for a failed probe what went wrong and what to try (A's error codes).
 * "Fly here" moves the knobs to that probe's settings, where the numbers are measured.
 */
export function ProbesBand() {
	const { record, state, actions } = useWorkspace();
	if (!record.probes.length) return null;
	const knobs = new Map(record.knobs.map((k) => [k.id, k]));
	const total = record.probes.reduce((a, p) => a + (p.cost?.usd ?? 0), 0);
	return (
		<div className="glass pf-band" aria-label={t("probes.title")}>
			<div className="pf-band__head">
				<b>{t("probes.title")}</b>
				<span>{t("probes.summary", { n: fmt(record.probes.length, "count"), cost: fmt(total, "usd") })}</span>
			</div>
			<ul className="pf-probes">
				{record.probes.map((p) => {
					const fit = PROBE_FIT[p.status] ?? "unknown";
					const failed = fit === "fails";
					const err = failed ? errorCopy(p.error_code ?? (p.status === "oom" ? "oom" : p.status === "timeout" ? "timeout" : null), p.error) : null;
					const settings = Object.entries(p.knobs ?? {}).filter((e): e is [string, KnobValue] => e[1] !== undefined && e[1] !== null);
					const here = settings.length > 0 && settings.every(([k, v]) => state.knobs[k] === v);
					const peak = p.memory?.peak_allocated_bytes ?? p.memory?.peak_reserved_bytes ?? null;
					return (
						<li key={p.id} className={`pf-probe${here ? " is-here" : ""}`}>
							<div className="pf-probe__head">
								<FitMark fit={fit} label={t(`probes.status.${p.status}`)} />
								<span className="pf-probe__knobs">
									{settings.length
										? settings.map(([k, v]) => `${t(knobs.get(k)?.label_key ?? k)} ${valueText(knobs.get(k), v)}`).join(" · ")
										: t("probes.as_written")}
								</span>
							</div>
							<div className="pf-probe__meta pf-dim">
								{[
									p.steps ? t("probes.steps", { n: fmt(p.steps, "count") }) : null,
									peak ? t("probes.peak", { bytes: fmt(peak, "bytes") }) : null,
									p.cost?.gpu_seconds ? t("probes.gpu_time", { s: fmt(p.cost.gpu_seconds, "s") }) : null,
									p.cost?.usd ? fmt(p.cost.usd, "usd") : null,
								]
									.filter(Boolean)
									.join(" · ")}
							</div>
							{err ? (
								<div className="pf-probe__error">
									<p>{err.message}</p>
									{err.hint ? <p className="pf-dim">{err.hint}</p> : null}
								</div>
							) : null}
							{settings.length && !here ? (
								<button
									type="button"
									className="pf-link"
									onClick={() => {
										for (const [k, v] of settings) if (knobs.has(k)) actions.setKnob(k, v, "user");
									}}
								>
									{t("probes.fly_here")}
								</button>
							) : null}
						</li>
					);
				})}
			</ul>
		</div>
	);
}
