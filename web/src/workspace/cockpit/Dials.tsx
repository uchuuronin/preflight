/**
 * The readouts as dials. Time and cost are buttons: pressing one makes it the aim (pressing
 * the current aim again switches to "most headroom"). Throughput and risk are meters.
 * Every dial shows its interval as a band behind the needle arc; an abstaining estimate shows
 * a hatched track and says it isn't trusted, never a number.
 */
import { useRef } from "react";
import { fmt } from "../../lib/fmt";
import { t } from "../../lib/i18n";
import { useReadouts } from "../../registry";
import { failureTarget, target, targetFor } from "../record/model";
import type { Estimate, ObjectiveKind, Readout } from "../record/types";
import { useWorkspace } from "../state";
import { FAILS_AT, FITS_BELOW } from "../record/terms";
import { bounds, trusted } from "../status";

const ARC = 188.5; // 270 degrees of a radius-40 circle
const FULL = 251.3;

interface DialSpec {
	id: string;
	label: string;
	unit: string;
	est: Estimate | undefined;
	scaleMax: number | null;
	goal?: ObjectiveKind;
	format: (v: number) => string;
	note?: string;
	/** fails: red note and needle; tight: amber note. */
	noteRisk?: boolean;
	noteTight?: boolean;
}

function Dial({ spec, onGoal, goalOn }: { spec: DialSpec; onGoal?: () => void; goalOn: boolean }) {
	const est = spec.est;
	const ok = trusted(est);
	const max = spec.scaleMax ?? 1;
	const point = ok ? est.point : 0;
	const frac = ok ? Math.max(0.004, Math.min(1, point / max)) : 0;
	const [lo, hi] = ok ? bounds(est) : [0, 0];
	const bandStart = Math.min(1, Math.max(0, lo / max)) * ARC;
	const bandLen = Math.max(0, Math.min(1, hi / max) - Math.min(1, Math.max(0, lo / max))) * ARC;
	const value = ok ? spec.format(point) : t("readout.abstain_short");
	const aria = ok
		? t("dial.aria", { label: spec.label, value, lo: spec.format(lo), hi: spec.format(hi) })
		: t("dial.aria_abstain", { label: spec.label });
	const face = (
		<>
			<svg viewBox="0 0 100 100" aria-hidden="true">
				<circle className={`pf-dial__track${ok ? "" : " is-abstain"}`} cx="50" cy="50" r="40" strokeDasharray={`${ARC} ${FULL}`} transform="rotate(135 50 50)" />
				{ok ? (
					<>
						<circle className="pf-dial__band" cx="50" cy="50" r="40" strokeDasharray={`0 ${bandStart.toFixed(1)} ${bandLen.toFixed(1)} 400`} transform="rotate(135 50 50)" />
						<circle className={`pf-dial__val${spec.noteRisk ? " is-risk" : ""}`} cx="50" cy="50" r="40" strokeDasharray={`${(frac * ARC).toFixed(1)} ${FULL}`} transform="rotate(135 50 50)" />
					</>
				) : null}
			</svg>
			<span className="pf-dial__text">
				<b>{value}</b>
				<small>{spec.unit}</small>
			</span>
		</>
	);
	return (
		<div className="pf-dial-wrap">
			{onGoal ? (
				<button type="button" className={`glass pf-dial pf-dial--button${goalOn ? " is-goal" : ""}`} onClick={onGoal} aria-pressed={goalOn} aria-label={`${aria}. ${goalOn ? t("dial.is_goal") : t("dial.make_goal")}`}>
					{face}
				</button>
			) : (
				<div className="glass pf-dial" role="img" aria-label={aria}>
					{face}
				</div>
			)}
			<span className="pf-dial__label">{spec.label}</span>
			{spec.note ? <span className={`pf-dial__note${spec.noteRisk ? " is-risk" : spec.noteTight ? " is-tight" : ""}`}>{spec.note}</span> : null}
			{goalOn ? <span className="pf-dial__goal">{t("dial.your_goal")}</span> : null}
		</div>
	);
}

export function Dials() {
	const { record, estimates, state, actions } = useWorkspace();
	const extra = useReadouts(record.job.lens);
	const maxSeen = useRef<Record<string, number>>({});

	const est = (r: Readout) => {
		const tg = targetFor(record, r);
		return tg ? estimates[tg.id] : undefined;
	};
	const remember = (id: string, e: Estimate | undefined): number | null => {
		if (!trusted(e)) return maxSeen.current[id] ?? null;
		const [, hi] = bounds(e);
		maxSeen.current[id] = Math.max(maxSeen.current[id] ?? 0, hi);
		return maxSeen.current[id] * 1.1;
	};

	const c = record.job.constraints ?? {};
	const timeEst = est("time");
	const costEst = est("cost");
	const deadlineWindow = c.deadline ? (Date.parse(c.deadline) - Date.parse(record.created_at ?? new Date().toISOString())) / 1000 : null;
	const timeMax = c.wall_time_cap_s ?? (deadlineWindow && deadlineWindow > 0 ? deadlineWindow : null) ?? remember("time", timeEst);
	const costMax = c.budget_usd ?? remember("cost", costEst);
	const tpsT = target(record, "throughput.tokens_per_s");
	const tpsEst = tpsT ? estimates[tpsT.id] : undefined;
	const fail = failureTarget(record);
	const riskT = targetFor(record, "risk") ?? (fail ? target(record, fail) : undefined);
	const riskEst = riskT ? estimates[riskT.id] : undefined;
	const goal = state.objective.kind;

	const specs: DialSpec[] = [
		{
			id: "time",
			label: t("dial.time"),
			unit: t("dial.time.unit"),
			est: timeEst,
			scaleMax: timeMax,
			goal: "min_time",
			format: (v) => fmt(v, "s"),
		},
		{
			id: "cost",
			label: t("dial.cost"),
			unit: costMax && c.budget_usd ? t("dial.cost.of_budget", { budget: fmt(c.budget_usd, "usd") }) : t("dial.cost.unit"),
			est: costEst,
			scaleMax: costMax,
			goal: "min_cost",
			format: (v) => fmt(v, "usd"),
		},
		{
			id: "tps",
			label: t("dial.tps"),
			unit: t("dial.tps.unit"),
			est: tpsEst,
			scaleMax: remember("tps", tpsEst),
			format: (v) => fmt(v, "tokens/s").replace(/ tokens\/s$/, ""),
		},
		{
			id: "risk",
			label: riskT ? t(riskT.label_key) : t("dial.risk"),
			unit: t("dial.risk.unit"),
			est: riskEst,
			scaleMax: 1,
			format: (v) => fmt(v, "probability"),
			noteRisk: trusted(riskEst) && riskEst.point >= FAILS_AT,
			noteTight: trusted(riskEst) && riskEst.point >= FITS_BELOW && riskEst.point < FAILS_AT,
			...(trusted(riskEst) && riskEst.point >= FAILS_AT
				? { note: t("dial.risk.likely") }
				: trusted(riskEst) && riskEst.point >= FITS_BELOW
					? { note: t("dial.risk.some") }
					: {}),
		},
	];

	return (
		<div className="pf-dials" aria-label={t("readouts.label")}>
			{specs
				.filter((s) => s.est !== undefined)
				.map((s) => (
					<Dial
						key={s.id}
						spec={s}
						goalOn={!!s.goal && goal === s.goal}
						{...(s.goal
							? { onGoal: () => actions.setObjective(goal === s.goal ? "fit_memory" : (s.goal as ObjectiveKind)) }
							: {})}
					/>
				))}
			{extra.map((r) => {
				const tg = target(record, r.targetId);
				if (!tg) return null;
				return (
					<Dial
						key={r.id}
						goalOn={false}
						spec={{ id: r.id, label: t(r.titleKey), unit: tg.unit, est: estimates[tg.id], scaleMax: remember(r.id, estimates[tg.id]), format: (v) => fmt(v, tg.unit) }}
					/>
				);
			})}
		</div>
	);
}
