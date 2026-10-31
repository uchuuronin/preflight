/**
 * The readouts as dials. Time and cost are buttons: pressing one makes it the aim (pressing
 * the current aim again switches to "most headroom"). Throughput and risk are meters.
 * Every dial shows its interval as a band behind the needle arc and says in a word where the
 * number comes from (measured, predicted, extrapolated, not trusted); an abstaining estimate
 * shows a hatched track, never a number. Scales are fixed per record (record/scale.ts).
 */
import { fmt } from "../../lib/fmt";
import { t } from "../../lib/i18n";
import { useReadouts } from "../../registry";
import { failureTarget, isFixture, target, targetFor } from "../record/model";
import type { Estimate, ObjectiveKind, Readout } from "../record/types";
import { Provenance, provenanceKey } from "../components/StatusMark";
import { useWorkspace } from "../state";
import { type DialScale, dialScale, scaleFraction } from "../record/scale";
import { FAILS_AT, FITS_BELOW, oomStatus } from "../record/terms";
import { bounds, trusted } from "../status";

const ARC = 188.5; // 270 degrees of a radius-40 circle
const FULL = 251.3;

interface DialSpec {
	id: string;
	label: string;
	unit: string;
	est: Estimate | undefined;
	scale: DialScale | null;
	goal?: ObjectiveKind;
	format: (v: number) => string;
	note?: string;
	/** The engine's sentence about this number (e.g. the runaway margin in an agent run's cost). */
	reason?: string;
	/** fails: red note and needle; tight: amber note; fits: calm note. */
	noteRisk?: boolean;
	noteTight?: boolean;
	noteFits?: boolean;
}

/** The one word under every dial: where this number comes from. */
function statusWord(est: Estimate | undefined, illustrative: boolean): string {
	return t(provenanceKey(est?.status ?? "abstain", illustrative));
}

function Dial({ spec, onGoal, goalOn }: { spec: DialSpec; onGoal?: () => void; goalOn: boolean }) {
	const est = spec.est;
	const illustrative = isFixture(useWorkspace().record);
	const ok = trusted(est) && !!spec.scale;
	const scale = spec.scale;
	const point = ok ? est.point : 0;
	const at = ok && scale ? scaleFraction(scale, point) : { frac: 0, off: false };
	const frac = ok ? Math.max(0.004, at.frac) : 0;
	const [lo, hi] = ok ? bounds(est) : [0, 0];
	const fLo = ok && scale ? scaleFraction(scale, lo).frac : 0;
	const fHi = ok && scale ? scaleFraction(scale, hi).frac : 0;
	const bandStart = fLo * ARC;
	const bandLen = Math.max(0, fHi - fLo) * ARC;
	const scaleNote = ok && scale ? scaleText(scale, spec.format) : "";
	const title = [spec.reason, scaleNote].filter(Boolean).join("\n");
	const value = ok ? spec.format(point) : t("readout.abstain_short");
	const aria = ok
		? `${t("dial.aria", { label: spec.label, value, lo: spec.format(lo), hi: spec.format(hi) })}. ${statusWord(est, illustrative)}${at.off ? `. ${t("dial.off_scale")}` : ""}`
		: t("dial.aria_abstain", { label: spec.label });
	const face = (
		<>
			<svg viewBox="0 0 100 100" aria-hidden="true">
				<circle className={`pf-dial__track${ok ? "" : " is-abstain"}`} cx="50" cy="50" r="40" strokeDasharray={`${ARC} ${FULL}`} transform="rotate(135 50 50)" />
				{ok ? (
					<>
						<circle className="pf-dial__band" cx="50" cy="50" r="40" strokeDasharray={`0 ${bandStart.toFixed(1)} ${bandLen.toFixed(1)} 400`} transform="rotate(135 50 50)" />
						<circle className={`pf-dial__val${spec.noteRisk ? " is-risk" : ""}${at.off ? " is-off" : ""}`} cx="50" cy="50" r="40" strokeDasharray={`${(frac * ARC).toFixed(1)} ${FULL}`} transform="rotate(135 50 50)" />
					</>
				) : null}
			</svg>
			<span className="pf-dial__text">
				<b className={value.length > 10 ? "is-long" : undefined}>{value}</b>
				<small>{spec.unit}</small>
			</span>
		</>
	);
	return (
		<div className="pf-dial-wrap" title={title || undefined}>
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
			<span className="pf-dial__status">
				<Provenance status={est?.status ?? "abstain"} />
				{at.off ? <span className="pf-dial__off">{t("dial.off_scale_short")}</span> : null}
			</span>
			{spec.note ? <span className={`pf-dial__note${spec.noteRisk ? " is-risk" : spec.noteTight ? " is-tight" : spec.noteFits ? " is-fits" : ""}`}>{spec.note}</span> : null}
			{goalOn ? <span className="pf-dial__goal">{t("dial.your_goal")}</span> : null}
		</div>
	);
}

const RISK_NOTE = { fits: "dial.risk.fits", tight: "dial.risk.some", fails: "dial.risk.likely" } as const;

function scaleText(scale: DialScale, format: (v: number) => string): string {
	if (scale.source === "limit") return t("dial.scale.limit", { hi: format(scale.hi) });
	if (scale.kind === "log") return t("dial.scale.log", { lo: format(scale.lo), hi: format(scale.hi) });
	return "";
}

export function Dials() {
	const { record, estimates, state, actions } = useWorkspace();
	const extra = useReadouts(record.job.lens);

	const est = (r: Readout) => {
		const tg = targetFor(record, r);
		return tg ? estimates[tg.id] : undefined;
	};
	const scaleFor = (r: Readout, limit?: number | null) => {
		const tg = targetFor(record, r);
		return tg ? dialScale(record, tg.id, tg.unit, limit) : null;
	};

	const c = record.job.constraints ?? {};
	const timeEst = est("time");
	const costEst = est("cost");
	const deadlineWindow = c.deadline ? (Date.parse(c.deadline) - Date.parse(record.created_at ?? new Date().toISOString())) / 1000 : null;
	const timeScale = scaleFor("time", c.wall_time_cap_s ?? (deadlineWindow && deadlineWindow > 0 ? deadlineWindow : null));
	const costScale = scaleFor("cost", c.budget_usd);
	const tpsT = target(record, "throughput.tokens_per_s");
	const tpsEst = tpsT ? estimates[tpsT.id] : undefined;
	const tokensT = target(record, "tokens.total");
	const tokensEst = tokensT ? estimates[tokensT.id] : undefined;
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
			scale: timeScale,
			goal: "min_time",
			format: (v) => fmt(v, "s_short"),
		},
		{
			id: "cost",
			label: t("dial.cost"),
			unit: costScale?.source === "limit" && c.budget_usd ? t("dial.cost.of_budget", { budget: fmt(c.budget_usd, "usd") }) : t("dial.cost.unit"),
			est: costEst,
			scale: costScale,
			goal: "min_cost",
			format: (v) => fmt(v, "usd"),
			...(costEst?.reason && costEst.status !== "abstain" ? { reason: costEst.reason } : {}),
		},
		tpsEst || !tokensEst
			? {
					id: "tps",
					label: t("dial.tps"),
					unit: t("dial.tps.unit"),
					est: tpsEst,
					scale: tpsT ? dialScale(record, tpsT.id, tpsT.unit) : null,
					format: (v) => fmt(v, "tokens/s").replace(/ tokens\/s$/, ""),
				}
			: {
					// Agent and API runs: the run's tokens instead of a training speed.
					id: "tokens",
					label: t("dial.tokens"),
					unit: t("dial.tokens.unit"),
					est: tokensEst,
					scale: tokensT ? dialScale(record, tokensT.id, tokensT.unit) : null,
					format: (v) => fmt(v, "tokens").replace(/ tokens$/, ""),
				},
		{
			id: "risk",
			label: riskT ? t(riskT.label_key) : t("dial.risk"),
			unit: t("dial.risk.unit"),
			est: riskEst,
			scale: { kind: "linear", lo: 0, hi: 1, source: "probability" },
			format: (v) => fmt(v, "probability"),
			noteRisk: trusted(riskEst) && riskEst.point >= FAILS_AT,
			noteTight: trusted(riskEst) && riskEst.point >= FITS_BELOW && riskEst.point < FAILS_AT,
			noteFits: trusted(riskEst) && riskEst.point < FITS_BELOW,
			// The fit word, always: fits, tight (measure first) or likely to fail.
			...(trusted(riskEst) ? { note: t(RISK_NOTE[oomStatus(riskEst.point)]) } : {}),
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
						spec={{ id: r.id, label: t(r.titleKey), unit: tg.unit, est: estimates[tg.id], scale: dialScale(record, tg.id, tg.unit), format: (v) => fmt(v, tg.unit) }}
					/>
				);
			})}
		</div>
	);
}
