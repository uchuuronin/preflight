import { useRef } from "react";
import { fmt, fmtRange } from "../../lib/fmt";
import { t } from "../../lib/i18n";
import { useWorkspace } from "../state";
import { trusted } from "../status";
import { runEvents, totalSteps, type RunEvent } from "../timeline";

const MARK: Record<RunEvent["kind"], string> = {
	eval: "M0 -5 L5 0 L0 5 L-5 0 Z",
	checkpoint: "M-4 -4 H4 V4 H-4 Z",
	predicted_oom: "M-5 -5 L5 5 M5 -5 L-5 5",
	end: "M0 -5 V5",
};

export function FlightPlan() {
	const { record, estimates, state, actions } = useWorkspace();
	const trackRef = useRef<HTMLDivElement>(null);
	const events = runEvents(record, estimates);
	const time = estimates.time_to_finish;
	const cost = estimates.total_cost;
	const total = totalSteps(record);
	const firstTrouble = events.find((e) => e.kind === "predicted_oom");

	const atFromPointer = (clientX: number) => {
		const r = trackRef.current?.getBoundingClientRect();
		if (!r) return 0;
		return (clientX - r.left) / r.width;
	};

	const playStep = Math.round(state.playhead * total);
	const playTime = trusted(time) ? time.point * state.playhead : null;
	const playCost = trusted(cost) ? cost.point * state.playhead : null;

	return (
		<section className="pf-plan" aria-label={t("plan.label")}>
			<header className="pf-plan__head">
				<h2 className="pf-plan__title">{t("plan.title")}</h2>
				{firstTrouble ? (
					<p className="pf-plan__trouble">
						{t("plan.first_trouble", { step: fmt(firstTrouble.step, "count") })}
					</p>
				) : (
					<p className="pf-plan__ok">{t("plan.no_trouble")}</p>
				)}
				<p className="pf-plan__total">
					{trusted(time) ? t("plan.total", { time: fmtRange(time.lo, time.hi, "s"), steps: fmt(total, "count") }) : null}
				</p>
			</header>
			<div
				className="pf-plan__track"
				ref={trackRef}
				onPointerDown={(e) => {
					e.currentTarget.setPointerCapture(e.pointerId);
					actions.setPlayhead(atFromPointer(e.clientX));
				}}
				onPointerMove={(e) => {
					if (e.buttons & 1) actions.setPlayhead(atFromPointer(e.clientX));
				}}
			>
				{/* Running cost grows linearly with time in this lens; the line shows where money goes. */}
				<svg className="pf-plan__cost" viewBox="0 0 100 20" preserveAspectRatio="none" aria-hidden="true">
					<path d="M0 20 L100 2" />
				</svg>
				<div className="pf-plan__axis" aria-hidden="true" />
				{events.map((ev, i) => (
					<button
						type="button"
						key={`${ev.kind}-${ev.step}-${i}`}
						className={`pf-plan__event pf-plan__event--${ev.kind}${ev.nodes.some((n) => state.highlight.includes(n)) ? " is-accent" : ""}`}
						style={{ left: `${ev.at * 100}%` }}
						onPointerDown={(e) => e.stopPropagation()}
						onClick={() => {
							actions.setPlayhead(ev.at);
							actions.highlight(ev.nodes);
						}}
						aria-label={t(`plan.event.${ev.kind}`, { step: fmt(ev.step, "count") })}
					>
						<svg width="12" height="12" viewBox="-6 -6 12 12" aria-hidden="true">
							<path d={MARK[ev.kind]} />
						</svg>
					</button>
				))}
				<div
					className="pf-plan__playhead"
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
			<footer className="pf-plan__foot">
				{state.playhead > 0 ? (
					<>
						<span>{t("plan.at_playhead", { step: fmt(playStep, "count") })}</span>
						{playTime !== null ? <span>{t("plan.elapsed", { time: fmt(playTime, "s") })}</span> : null}
						{playCost !== null ? <span>{t("plan.spent", { cost: fmt(playCost, "usd") })}</span> : null}
					</>
				) : (
					<span className="pf-muted">{t("plan.drag_hint")}</span>
				)}
				<span className="pf-plan__keys">
					<span><i className="pf-key pf-key--eval" /> {t("plan.key.eval")}</span>
					<span><i className="pf-key pf-key--checkpoint" /> {t("plan.key.checkpoint")}</span>
					<span><i className="pf-key pf-key--oom" /> {t("plan.key.oom")}</span>
				</span>
			</footer>
		</section>
	);
}
