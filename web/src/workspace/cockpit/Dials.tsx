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
import { flag } from "../../lib/flags";
import { outcomes, setHops, useHops, useHopsOwner } from "../hops";
import { useEffect, useRef, useSyncExternalStore } from "react";
import { durMs, useTweenedText } from "../../lib/tween";
import { duskTokens } from "../../theme/tokens";

const ARC = 188.5; // 270 degrees of a radius-40 circle
const FULL = 251.3;

interface DialSpec {
	id: string;
	/** Possible outcomes for the HOPs toggle; off for probabilities, which the engine states itself. */
	hops?: boolean;
	label: string;
	unit: string;
	est: Estimate | undefined;
	scale: DialScale | null;
	goal?: ObjectiveKind;
	format: (v: number) => string;
	/** The value with its unit, where the dial's unit line isn't beside it (the wake strip). */
	formatFull?: (v: number) => string;
	/** A change in this readout, when it reads differently from a value (a chance moves in points). */
	formatDelta?: (v: number) => string;
	/** A one-word name, for the strip of wakes on a phone. */
	short?: string;
	note?: string;
	/** The engine's sentence about this number (e.g. the runaway margin in an agent run's cost). */
	reason?: string;
	/** Which way is better, for the wake after a move: lower (time, cost, risk) or higher (speed). */
	better?: "lower" | "higher";
	/** fails: red note and needle; tight: amber note; fits: calm note. */
	noteRisk?: boolean;
	noteTight?: boolean;
	noteFits?: boolean;
}

/** The one word under every dial: where this number comes from. */
function statusWord(est: Estimate | undefined, illustrative: boolean): string {
	return t(provenanceKey(est?.status ?? "abstain", illustrative));
}

/** What a knob move did to one readout, shown in the dial's open gap for WAKE_MS. */
export interface Wake {
	key: number;
	text: string;
	/** With the unit, for the strip away from the dial. */
	full: string;
	/** null: no better side (a readout without a direction). */
	better: boolean | null;
}

/**
 * Source: tokens.ts `dur-wake` (a guess: the 600 ms sweep plus about two seconds to read a short
 * delta, gone before the visitor's next look). Revisit with the stranger test's interaction log.
 */
export const WAKE_MS = durMs(duskTokens["dur-wake"]);
/** Source: guess, below this share of the old value (or a point of probability) a change isn't worth a mark. */
const WAKE_MIN = 0.01;

/** The change between two values of a readout as signed text, e.g. "+1 h 22 min"; null when it rounds to nothing. */
export function deltaText(from: number, to: number, format: (v: number) => string): string | null {
	const mag = format(Math.abs(to - from));
	if (mag === format(0)) return null;
	return `${to > from ? "+" : "\u2212"}${mag}`;
}

/**
 * True when a change is big enough to mark: more than 1% of the old value, or for a probability
 * more than one point (a share of a chance near zero means nothing).
 */
export function worthMarking(from: number, to: number, scale: DialScale | null): boolean {
	const d = Math.abs(to - from);
	if (!(d > 0)) return false;
	if (scale?.source === "probability") return d > WAKE_MIN;
	return from === 0 || d > WAKE_MIN * Math.abs(from);
}

/**
 * The wakes, one set per cockpit, published here by the dials and read wherever they show (the
 * dials, and the open knob control on a phone): one source, so a view that mounts with the move
 * (the guide opening the knob it just moved) still shows it.
 */
let wakesNow: Record<string, Wake> = {};
const wakeListeners = new Set<() => void>();
function publishWakes(w: Record<string, Wake>): void {
	if (w === wakesNow || (!Object.keys(w).length && !Object.keys(wakesNow).length)) return;
	wakesNow = w;
	for (const l of wakeListeners) l();
}
function subscribeWakes(l: () => void): () => void {
	wakeListeners.add(l);
	return () => wakeListeners.delete(l);
}
export function useWakeList(): Record<string, Wake> {
	return useSyncExternalStore(subscribeWakes, () => wakesNow, () => wakesNow);
}

/**
 * After each knob move (the visitor's or the guide's), mark the readouts it changed. Nothing on
 * first draw or when a readout stops being trusted; a move that changes nothing clears the last
 * move's marks rather than leave them describing a setting that's gone. (A new record mounts a
 * new cockpit: CockpitScreen keys the workspace by record.)
 */
function useWakeSource(specs: DialSpec[], knobs: object): void {
	const last = useRef<{ knobs: object; points: Record<string, number | null> } | null>(null);
	const seq = useRef(0);
	const timer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
	const points: Record<string, number | null> = {};
	for (const s of specs) points[s.id] = trusted(s.est) && s.scale ? s.est.point : null;
	const sig = JSON.stringify(points);

	useEffect(() => {
		const prev = last.current;
		last.current = { knobs, points };
		if (!prev || prev.knobs === knobs) return;
		const next: Record<string, Wake> = {};
		for (const s of specs) {
			const a = prev.points[s.id];
			const b = points[s.id];
			if (a == null || b == null || !worthMarking(a, b, s.scale)) continue;
			const text = deltaText(a, b, s.formatDelta ?? s.format);
			if (!text) continue;
			next[s.id] = { key: ++seq.current, text, full: deltaText(a, b, s.formatDelta ?? s.formatFull ?? s.format) ?? text, better: s.better ? (s.better === "lower" ? b < a : b > a) : null };
		}
		clearTimeout(timer.current);
		publishWakes(next);
		if (Object.keys(next).length) timer.current = setTimeout(() => publishWakes({}), WAKE_MS);
		// specs and points are read through `sig` and `knobs`: a new array each render, the same values.
	}, [knobs, sig]);

	useEffect(
		() => () => {
			clearTimeout(timer.current);
			publishWakes({});
		},
		[],
	);
}

/** The dial's size class for a value's text (its length): applied by React and by the tween per frame. */
function sizeClass(text: string): string | undefined {
	return text.length > 9 ? "is-longer" : text.length > 6 ? "is-long" : undefined;
}

function Dial({ spec, onGoal, goalOn, wake }: { spec: DialSpec; onGoal?: () => void; goalOn: boolean; wake?: Wake | undefined }) {
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
	const hops = useHops();
	const draws = hops.on && ok && scale && spec.hops ? outcomes(est) : [];
	const shown = hops.frame === null ? draws : draws.length ? [draws[hops.frame % draws.length]!] : [];
	const title = [spec.reason, scaleNote].filter(Boolean).join("\n");
	const value = ok ? spec.format(point) : t("readout.abstain_short");
	const valueRef = useRef<HTMLElement>(null);
	useTweenedText(valueRef, ok ? point : null, spec.format, { classFor: sizeClass });
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
						{scale
							? shown.map((v, i) => {
									const f = Math.min(1, scaleFraction(scale, v).frac);
									return (
										<circle
											key={hops.frame === null ? i : "hop"}
											className={`pf-dial__hop${hops.frame === null ? " is-all" : ""}`}
											cx="50"
											cy="50"
											r="40"
											strokeDasharray={`0 ${Math.max(0, f * ARC - 2).toFixed(1)} 4 400`}
											transform="rotate(135 50 50)"
										/>
									);
								})
							: null}
					</>
				) : null}
			</svg>
			<span className="pf-dial__text">
				<b ref={valueRef} className={sizeClass(value)}>
					{value}
				</b>
				<small>{spec.unit}</small>
			</span>
			{wake ? (
				<span key={wake.key} className={`pf-dial__wake${wake.better === true ? " is-better" : wake.better === false ? " is-worse" : ""}`} aria-hidden="true">
					{wake.text}
				</span>
			) : null}
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

/** The four main readouts as dial specs (time, cost, speed or tokens, risk). */
function useDialSpecs(): DialSpec[] {
	const { record, estimates } = useWorkspace();

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

	return [
		{
			id: "time",
			label: t("dial.time"),
			short: t("dial.short.time"),
			unit: t("dial.time.unit"),
			est: timeEst,
			scale: timeScale,
			goal: "min_time",
			better: "lower",
			hops: true,
			format: (v) => fmt(v, "s_short"),
		},
		{
			id: "cost",
			label: t("dial.cost"),
			short: t("dial.short.cost"),
			unit: costScale?.source === "limit" && c.budget_usd ? t("dial.cost.of_budget", { budget: fmt(c.budget_usd, "usd") }) : t("dial.cost.unit"),
			est: costEst,
			scale: costScale,
			goal: "min_cost",
			better: "lower",
			hops: true,
			format: (v) => fmt(v, "usd"),
			...(costEst?.reason && costEst.status !== "abstain" ? { reason: costEst.reason } : {}),
		},
		tpsEst || !tokensEst
			? {
					id: "tps",
					label: t("dial.tps"),
					short: t("dial.tps"),
					unit: t("dial.tps.unit"),
					est: tpsEst,
					scale: tpsT ? dialScale(record, tpsT.id, tpsT.unit) : null,
					better: "higher",
					format: (v) => fmt(v, "tokens/s").replace(/ tokens\/s$/, ""),
					formatFull: (v) => fmt(v, "tokens/s"),
				}
			: {
					// Agent and API runs: the run's tokens instead of a training speed.
					id: "tokens",
					label: t("dial.tokens"),
					short: t("dial.tokens"),
					unit: t("dial.tokens.unit"),
					est: tokensEst,
					scale: tokensT ? dialScale(record, tokensT.id, tokensT.unit) : null,
					better: "lower",
					format: (v) => fmt(v, "tokens").replace(/ tokens$/, ""),
					formatFull: (v) => fmt(v, "tokens"),
				},
		{
			id: "risk",
			label: riskT ? t(riskT.label_key) : t("dial.risk"),
			short: t("dial.short.risk"),
			formatDelta: (v) => {
				const n = Math.round(v * 100);
				return t(n === 1 ? "dial.delta_point" : "dial.delta_points", { n });
			},
			unit: t("dial.risk.unit"),
			est: riskEst,
			scale: { kind: "linear", lo: 0, hi: 1, source: "probability" },
			better: "lower",
			format: (v) => fmt(v, "probability"),
			noteRisk: trusted(riskEst) && riskEst.point >= FAILS_AT,
			noteTight: trusted(riskEst) && riskEst.point >= FITS_BELOW && riskEst.point < FAILS_AT,
			noteFits: trusted(riskEst) && riskEst.point < FITS_BELOW,
			// The fit word, always: fits, tight (measure first) or likely to fail.
			...(trusted(riskEst) ? { note: t(RISK_NOTE[oomStatus(riskEst.point)]) } : {}),
		},
	];
}

/**
 * The wakes as a strip, for where the dials are out of sight (the open knob control on a phone,
 * where the dials sit a screen above): the effect of a move appears under the finger that made it.
 */
export function WakeStrip() {
	const specs = useDialSpecs();
	const wakes = useWakeList();
	const shown = specs.filter((s) => wakes[s.id]);
	return (
		<div className="pf-wakes" aria-hidden="true">
			{shown.length ? (
				shown.map((s) => {
					const w = wakes[s.id]!;
					return (
						<span key={`${s.id}-${w.key}`} className={`pf-wakes__item${w.better === true ? " is-better" : w.better === false ? " is-worse" : ""}`}>
							<span className="pf-wakes__label">{s.short ?? s.label}</span> {w.full}
						</span>
					);
				})
			) : (
				<span className="pf-wakes__empty">{t("wake.empty")}</span>
			)}
		</div>
	);
}

export function Dials() {
	const { record, estimates, state, actions } = useWorkspace();
	const extra = useReadouts(record.job.lens);
	const specs = useDialSpecs();
	const goal = state.objective.kind;
	useWakeSource(specs, state.knobs);
	const wakes = useWakeList();

	return (
		<div className="pf-dials" aria-label={t("readouts.label")}>
			{specs
				.filter((s) => s.est !== undefined)
				.map((s) => (
					<Dial
						key={s.id}
						spec={s}
						wake={wakes[s.id]}
						goalOn={!!s.goal && goal === s.goal}
						{...(s.goal
							? { onGoal: () => actions.setObjective(goal === s.goal ? "fit_memory" : (s.goal as ObjectiveKind)) }
							: {})}
					/>
				))}
			{flag("c.hops") ? <HopsToggle /> : null}
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

/** "Possible outcomes": the time and cost dials and the memory band show one outcome per frame. */
export function HopsToggle() {
	const hops = useHopsOwner();
	return (
		<div className="pf-hops">
			<button type="button" className={`pf-tag pf-tag--button${hops.on ? " is-on" : ""}`} aria-pressed={hops.on} onClick={() => setHops(!hops.on)}>
				{t("hops.toggle")}
			</button>
			{hops.on ? <span className="pf-hops__help">{t(hops.frame === null ? "hops.help_still" : "hops.help")}</span> : null}
		</div>
	);
}
