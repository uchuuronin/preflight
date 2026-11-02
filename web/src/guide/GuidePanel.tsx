import { useEffect, useMemo, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { finishBand, planFromRecord } from "../workspace/campaign/plan";
import { currentKnobs } from "../workspace/record/model";
import { estimateFactor } from "../workspace/rate";
import { fmt, fmtDelta, fmtRange } from "../lib/fmt";
import { wilson } from "../addons/_stats/wilson";
import { probeCap, rlProbe, spreadCount } from "../workspace/rl";
import { flag } from "../lib/flags";
import { hasKey, t } from "../lib/i18n";
import { usePanels } from "../registry";
import { capacityBytes, costStraddlesBudget, failureTarget, objectiveTarget, SAMPLE_MORE_TASKS, target, targetFor } from "../workspace/record/model";
import type { Finding, Knob, KnobValue } from "../workspace/record/types";
import { useWorkspace } from "../workspace/state";
import { fitOf, trusted, type Fit } from "../workspace/status";
import { changeWords } from "./changeWords";
import { runApprovedTool, type SpendTool, type ToolRun } from "./approvals";
import { LiveGuideSection } from "./LiveGuideSection";
import { comboVisit, comboWords } from "./combos";
import { campaignParts, campaignVisit, rlParts, rlVisit, findingVisit, MAX_TOUR_STEPS, nextStep, pendingFindings, plannedSteps, SAMPLE_VISIT, type TourStep } from "./ruleTour";
import { runTool, type ToolContext } from "./tools";
import { AddonBoundary } from "../components/AddonBoundary";
import { FixRace } from "../fix/FixRace";
import { limitsForServer } from "./limits";
import { LimitsPanel } from "./LimitsPanel";
import { MoveChoices } from "./MoveChoices";

/** Muse Code writes a finding's fix only once the fix loop is on; until then the tour moves on. */
const canFix = (f: Finding) => !!f.fix_template && flag("c.fix_loop");

interface Active {
	step: TourStep;
	from?: KnobValue;
	n: number;
	of: number;
}

export function valueText(knob: Knob | undefined, v: KnobValue | undefined): string {
	if (v === undefined) return "";
	if (typeof v === "boolean") return t(v ? "value.on" : "value.off");
	if (typeof v === "string") return hasKey(`knobvalue.${v}`) ? t(`knobvalue.${v}`) : v;
	if (knob?.unit === "tokens") return new Intl.NumberFormat("en-GB").format(v);
	if (!Number.isInteger(v) && Math.abs(v) < 1) return new Intl.NumberFormat("en-GB", { maximumSignificantDigits: 2 }).format(v);
	return fmt(v, "count");
}

/** The view each band step lights up (below the cockpit), and where the step speaks inside it. */
const BANDS = { campaign: "pf-campaign", rl: "pf-rl" } as const;
type BandKind = keyof typeof BANDS;
const isBand = (k: string | undefined): k is BandKind => k === "campaign" || k === "rl";

/** Light up the view the tour is on (none: all off), and bring it into view. */
function guideBand(on: BandKind | null): void {
	if (typeof document === "undefined") return;
	for (const [kind, id] of Object.entries(BANDS)) {
		const el = document.getElementById(id);
		if (!el) continue;
		el.classList.toggle("is-guided", kind === on);
		if (kind === on) el.scrollIntoView?.({ behavior: window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
	}
}

export function GuidePanel() {
	const ws = useWorkspace();
	const { record, recordRef, provider, state, actions, estimates } = ws;
	const footer = usePanels("guide-footer", record.job.lens);
	const [active, setActive] = useState<Active | null>(null);
	const [visited, setVisited] = useState<ReadonlySet<string>>(new Set());
	const [taken, setTaken] = useState(0);
	const [why, setWhy] = useState(false);
	const [fixing, setFixing] = useState<string | null>(null);
	const [fallback, setFallback] = useState<string | null>(null);
	const [approvals, setApprovals] = useState<Array<{ id: number; name: string; args: Record<string, unknown>; run?: ToolRun; running?: boolean }>>([]);
	const [cliff, setCliff] = useState<{ knobId: string; from: KnobValue } | null>(null);
	const [liveStatus, setLiveStatus] = useState("ready");
	// Whether the live guide has answered at least once, and whether a failed first press has already
	// started the rule-based tour (it's started once, not again on a later failure).
	const liveAnswered = useRef(false);
	const autoStarted = useRef(false);
	const live = flag("c.guide.live") && !fallback;
	const knobs = new Map(record.knobs.map((k) => [k.id, k]));
	// The campaign plan the cockpit draws (same flag, same record), so the tour walks what's on screen.
	// B's plan at the table's rates it was made at: the budget cut is judged there, whatever rate the visitor types.
	const campaignTable = useMemo(() => (flag("c.campaign_calendar") && record.campaign ? planFromRecord(record.campaign, record.models) : null), [record]);
	const campaign = useMemo(
		() => (flag("c.campaign_calendar") && record.campaign ? planFromRecord(record.campaign, record.models, (k, e) => estimateFactor(record, { ...currentKnobs(record), ...k }, e, state.rate)) : null),
		// B's plan is priced at the record's own setting, so only the visitor's rate changes its dollars.
		[record, state.rate],
	);

	const latest = useRef(ws);
	latest.current = ws;

	const ctx = (): ToolContext => {
		const { state: s } = latest.current;
		return {
			record,
			provider,
			actions,
			knobs: s.knobs,
			pinned: s.pinned,
			limits: s.limits,
			requestApproval: (name, args) => setApprovals((a) => [...a, { id: Date.now() + a.length, name, args }]),
		};
	};

	const begin = (nextVisited: ReadonlySet<string>, nextTaken: number) => {
		const { state: s } = latest.current;
		const tc = { record, provider, knobs: s.knobs, pinned: s.pinned, objective: s.objective, visited: nextVisited, limits: s.limits, campaign };
		const step = nextStep(tc);
		const n = nextTaken + 1;
		// Steps that walk a view (the campaign plan, the RL signals) come on top of the tour's moves:
		// the five-move cap counts moves, not these. Review, Nov 1.
		const bands = [...nextVisited].filter((v) => v.startsWith("campaign:") || v.startsWith("rl:")).length;
		const cap = MAX_TOUR_STEPS + bands;
		guideBand(isBand(step.kind) ? step.kind : null);
		if (isBand(step.kind)) {
			setWhy(false);
			actions.highlight([]);
			actions.propose(null);
			const left = rlParts(tc).length + campaignParts(tc).length - 1;
			setActive({ step, n, of: Math.max(n, nextTaken + 1 + left + pendingFindings(tc) + Math.min(MAX_TOUR_STEPS, plannedSteps(tc))) });
			return;
		}
		if (step.kind === "probe_offer" && step.move) {
			if (step.nodes.length) runTool("highlight", { nodes: step.nodes }, ctx());
			runTool("predict", { knobs: { [step.move.knobId]: step.move.to } }, ctx());
			actions.propose(null);
			actions.openKnob(step.move.knobId);
			setWhy(false);
			setActive({ step, n, of: Math.min(cap, Math.max(n, nextTaken + plannedSteps(tc))) });
			return;
		}
		setWhy(false);
		if (step.nodes.length) runTool("highlight", { nodes: step.nodes }, ctx());
		else actions.highlight([]);
		if (step.kind === "move" && step.move && n <= cap) {
			const from = s.knobs[step.move.knobId];
			runTool("predict", { knobs: { [step.move.knobId]: step.move.to } }, ctx());
			runTool("set_knob", { knob_id: step.move.knobId, value: step.move.to, why: step.textKey }, ctx());
			setActive({ step, ...(from !== undefined ? { from } : {}), n, of: Math.min(cap, Math.max(n, nextTaken + pendingFindings(tc) + plannedSteps(tc))) });
			return;
		}
		actions.propose(null);
		if (step.kind === "combo" && step.combo) {
			// Shown with its prediction; nothing moves until the visitor applies it.
			runTool("predict", { knobs: Object.fromEntries(step.combo.changes.map((c) => [c.knobId, c.to])) }, ctx());
			setActive({ step, n, of: Math.max(n, Math.min(cap, nextTaken + pendingFindings(tc) + plannedSteps(tc) + 1)) });
			return;
		}
		if (step.kind === "finding") {
			// A finding is a step of the tour: counted with the findings and moves still to come.
			setActive({ step, n, of: Math.max(n, Math.min(cap, nextTaken + pendingFindings(tc) + plannedSteps(tc))) });
			return;
		}
		// With limits set or a suggestion turned down, "done" says what it means: nothing else
		// inside the visitor's limits helps.
		const l = s.limits;
		const limited = l.declined.length > 0 || l.frozen.length > 0 || Object.keys(l.excluded).length > 0 || Object.keys(l.allowed).length > 0 || l.budgetUsd !== null || l.deadlineS !== null || l.gpuHours !== null;
		const doneKey = limited ? "guide.done.limits" : "guide.done";
		const shown =
			step.kind === "move" || step.kind === "done" ? { ...step, kind: "done" as const, textKey: doneKey, whyKey: limited ? "guide.done.limits.why" : "guide.done.why" } : step;
		setActive({ step: shown, n, of: n });
	};

	/** Move past a "measure it first" or "sample more" offer without spending. */
	const skipOffer = () => {
		if (!active) return;
		const v = new Set(visited);
		let next = taken;
		if (active.step.kind === "sample_offer") v.add(SAMPLE_VISIT);
		else if ((active.step.kind === "campaign" && active.step.campaignPart) || (active.step.kind === "rl" && active.step.rlPart)) {
			v.add(active.step.campaignPart ? campaignVisit(active.step.campaignPart) : rlVisit(active.step.rlPart!));
			next = taken + 1; // a part of the plan shown is a step taken
			setTaken(next);
		} else if (active.step.kind === "finding" && active.step.finding) {
			v.add(findingVisit(active.step.finding));
			next = taken + 1; // a finding shown is a step taken, so the counter moves on
			setTaken(next);
		} else if (active.step.move) v.add(`probe:${active.step.move.knobId}`);
		else return;
		setVisited(v);
		requestAnimationFrame(() => begin(v, next));
	};

	const measure = (move: { knobId: string; to: KnobValue } | undefined, reason: string) =>
		runTool("run_probe", { knobs: move ? { [move.knobId]: move.to } : {}, reason }, ctx());

	/**
	 * apply: keep the suggestion. revert: put it back. pin: keep the suggestion and pin it there.
	 * keep: put it back and pin the visitor's own value. decline ("Not this"): put it back, exclude
	 * that value for the session, and move on to another lever.
	 */
	/** A combined move: apply sets every knob it names (the visitor's own click); "Not this" moves on. */
	const finishCombo = (apply: boolean) => {
		const combo = active?.step.combo;
		if (!combo) return;
		if (apply) for (const c of combo.changes) actions.setKnob(c.knobId, c.to, "user");
		const v = new Set(visited);
		v.add(comboVisit(combo));
		setVisited(v);
		setTaken(taken + 1);
		requestAnimationFrame(() => begin(v, taken + 1));
	};

	const finish = (outcome: "apply" | "revert" | "pin" | "keep" | "decline") => {
		if (!active?.step.move) return;
		const { move } = active.step;
		const from = active.from;
		// Clear the suggestion first, so putting the knob back isn't read as the visitor's own move.
		actions.propose(null);
		if (outcome !== "apply" && outcome !== "pin" && from !== undefined) actions.setKnob(move.knobId, from, "user", { putBack: true });
		if (outcome === "pin" || outcome === "keep") actions.pin(move.knobId, true);
		if (outcome === "decline") actions.decline(move.knobId, move.to);
		const v = new Set(visited);
		v.add(move.knobId);
		setVisited(v);
		setTaken(taken + 1);
		requestAnimationFrame(() => begin(v, taken + 1));
	};

	// The visitor's own move always wins: the pending suggestion is set aside (the reducer already
	// put any other knob back), and the tour offers to carry on from their setting.
	const [setAside, setSetAside] = useState<string | null>(null);
	useEffect(() => {
		const p = state.setAside;
		if (!p || active?.step.kind !== "move" || active.step.move?.knobId !== p.knobId) return;
		const v = new Set(visited);
		v.add(p.knobId);
		setVisited(v);
		setTaken(taken + 1);
		setActive(null);
		setSetAside(p.knobId);
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [state.setAside]);
	const carryOn = () => {
		setSetAside(null);
		begin(visited, taken);
	};

	const restart = () => {
		setSetAside(null);
		// A fresh tour may offer knobs set aside last time; values ruled out stay ruled out.
		const l = latest.current.state.limits;
		const v = new Set<string>();
		setVisited(v);
		setTaken(0);
		if (!l.declined.length) return begin(v, 0);
		actions.setLimits({ ...l, declined: [] });
		requestAnimationFrame(() => begin(v, 0)); // after the cleared limits are committed
	};

	// Restart the tour when the aim changes (a dial pressed, or the person's choice).
	const objectiveKind = state.objective.kind;
	const firstRun = useRef(true);
	useEffect(() => {
		if (firstRun.current) {
			firstRun.current = false;
			return;
		}
		if (active) restart();
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [objectiveKind]);

	// Hand over the controls: when the person crosses the memory cliff themselves, speak up once.
	const prevFit = useRef<Fit>(fitOf(record, estimates, state.knobs));
	const prevKnobs = useRef(state.knobs);
	useEffect(() => {
		const fit = fitOf(record, estimates, state.knobs);
		const change = state.lastChange;
		// Putting a knob back after a suggestion (Not this, Keep, a guide revert) is not crossing the limit: no nudge.
		if (change?.by === "user" && !change.putBack && fit === "fails" && prevFit.current !== "fails") {
			const from = prevKnobs.current[change.knobId];
			if (from !== undefined) setCliff({ knobId: change.knobId, from });
		} else if (fit !== "fails") setCliff(null);
		prevFit.current = fit;
		prevKnobs.current = state.knobs;
	}, [estimates, record, state.knobs, state.lastChange]);

	/** The visitor approved a spending tool from the rule-based tour: run it on the server. */
	const approve = async (id: number) => {
		const a = approvals.find((x) => x.id === id);
		if (!a) return;
		setApprovals((all) => all.map((x) => (x.id === id ? { ...x, running: true } : x)));
		const args = a.args as { knobs?: Record<string, KnobValue>; knob_id?: string; reason?: string };
		const reason = typeof args.reason === "string" && /^[a-z][a-z0-9_.]{0,47}$/.test(args.reason) ? args.reason : undefined;
		const run = await runApprovedTool(recordRef, a.name as SpendTool, { ...(args.knobs ? { knobs: args.knobs } : {}), ...(args.knob_id ? { knob_id: args.knob_id } : {}), ...(reason ? { reason } : {}) }, latest.current.state.knobs, fetch, limitsForServer(latest.current.state.limits));
		setApprovals((all) => all.map((x) => (x.id === id ? { ...x, running: false, run } : x)));
	};

	const vars = stepVars(active);

	/** The RL steps' numbers: the probe's measurements (A's) and B's estimates, as the RL view shows them. */
	function rlVars(): Record<string, string> {
		const p = rlProbe(record);
		const c = p ? spreadCount(p) : null;
		if (!p?.rl) return {};
		const w = c ? wilson(c.k, c.n) : null;
		const rates = p.rl.pass_rate_per_prompt ?? [];
		const len = p.rl.response_length;
		const cap = probeCap(p);
		const e = (id: string) => {
			const x = estimates[id];
			return x && trusted(x) ? `${fmt(x.point, "probability")} (${fmtRange(x.lo ?? x.point, x.hi ?? x.point, "probability")})` : t("value.none");
		};
		return {
			k: w ? fmt(w.k, "count") : "",
			n: w ? fmt(w.n, "count") : "",
			share: w ? fmt(w.p, "probability") : "",
			lo: w ? fmt(w.lo, "probability") : "",
			hi: w ? fmt(w.hi, "probability") : "",
			never: fmt(rates.filter((r) => r <= 0).length, "count"),
			g: fmt(Number(state.knobs.group_size ?? p.rl.group_size ?? 0), "count"),
			pred: e("rl.zero_spread"),
			trunc: e("rl.truncation"),
			cap: typeof cap === "number" ? fmt(cap, "count") : t("value.none"),
			p95: len?.p95 !== undefined && len?.p95 !== null ? fmt(len.p95, "count") : "",
			longest: len?.max !== undefined && len?.max !== null ? fmt(len.max, "count") : "",
			trunc_line:
				typeof p.rl.truncation_rate === "number"
					? t("guide.rl.trunc_measured", { share: fmt(p.rl.truncation_rate, "probability") })
					: t("guide.rl.trunc_estimated", { trunc: e("rl.truncation") }),
		};
	}

	/** The campaign steps' numbers, every one from the plan (B's), formatted as the calendar formats them. */
	function campaignVars(): Record<string, string> {
		const c = campaign;
		if (!c) return {};
		const usd = (e: { point: number | null; lo?: number | null; hi?: number | null; status: string } | null | undefined, k: "point" | "lo" | "hi") =>
			e && e.status !== "abstain" && typeof e[k] === "number" ? fmt(e[k] as number, "usd") : t("value.none");
		const placed = c.cells.filter((x) => !x.cut).length;
		const day = (iso: string | undefined) =>
			iso ? new Intl.DateTimeFormat("en-GB", { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit", timeZone: "UTC" }).format(new Date(iso)) : "";
		const budget = record.job.constraints?.budget_usd;
		return {
			runs: fmt(c.cells.length, "count"),
			placed: fmt(placed, "count"),
			cut: fmt(c.cut_list.length, "count"),
			lanes: fmt(c.lanes.length, "count"),
			people: c.lanes.join(", "),
			usd: usd(c.usd, "point"),
			lo: usd(c.usd, "lo"),
			hi: usd(c.usd, "hi"),
			grid_hi: usd(campaignTable?.grid_usd, "hi"),
			cut_hi: usd(campaignTable?.usd, "hi"),
			budget: typeof budget === "number" ? fmt(budget, "usd") : t("value.none"),
			p10: day(c.completion.p10),
			p50: day(c.completion.p50),
			p90: day(c.completion.p90),
			finish_why: c.finish_s?.reason ?? (finishBand(c) ? "" : t("value.none")),
		};
	}

	function stepVars(a: Active | null): Record<string, string> {
		const step = a?.step;
		const knobId = step?.move?.knobId ?? step?.pinnedKnob;
		const knob = knobId ? knobs.get(knobId) : undefined;
		const goal = objectiveTarget(record, state.objective.kind ?? "min_time", state.objective.target_id);
		const goalTarget = step?.move ? target(record, step.move.targetId) : goal ? target(record, goal.id) : undefined;
		const peakT = targetFor(record, "memory");
		const peak = peakT ? estimates[peakT.id] : undefined;
		const cap = capacityBytes(record, state.knobs);
		const fail = failureTarget(record);
		const risk = fail ? estimates[fail] : undefined;
		return {
			knob: knob ? t(knob.label_key) : "",
			from: valueText(knob, a?.from),
			to: valueText(knob, step?.move?.to),
			delta: step?.move && goalTarget ? fmtDelta(step.move.delta, goalTarget.unit) : "",
			target: goalTarget ? t(goalTarget.label_key).toLowerCase() : "",
			...changeWords({ record, provider, knobs: state.knobs, estimates }, step?.move, a?.from, goalTarget),
			peak: trusted(peak) ? fmt(peak.point, "bytes") : t("value.none"),
			peakTrusted: trusted(peak) ? "1" : "",
			cap: cap ? fmt(cap, "bytes") : "",
			risk: trusted(risk) ? fmt(risk.point, "probability") : "",
			chance: step?.move?.risk !== null && step?.move?.risk !== undefined ? fmt(step.move.risk, "probability") : "",
			...(step?.combo ? comboWords(record, state.knobs, step.combo) : {}),
			title: step?.finding?.title ?? "",
			// A's detail can end without a full stop; the line reads it as a sentence
			detail: sentence(step?.finding?.detail ?? ""),
			...campaignVars(),
			...rlVars(),
			...((): Record<string, string> => {
				const st = costStraddlesBudget(record, estimates);
				return st
					? {
							budget: fmt(st.budget, "usd"),
							lo: fmt(st.lo, "usd"),
							hi: fmt(st.hi, "usd"),
							tasks: String(SAMPLE_MORE_TASKS),
							sampled: st.sampled !== null ? fmt(st.sampled, "count") : "",
						}
					: {};
			})(),
		};
	}

	const primary = (() => {
		if (!active) return setAside ? { label: t("guide.carry_on"), run: carryOn } : { label: t("guide.start"), run: restart };
		const s = active.step;
		if (s.kind === "move") return { label: t("guide.apply_value", { knob: vars.knob, to: vars.to }), run: () => finish("apply") };
		if (s.kind === "combo" && s.combo?.fit === "tight")
			return { label: t("guide.combo.measure"), run: () => runTool("run_probe", { knobs: Object.fromEntries(s.combo!.changes.map((c) => [c.knobId, c.to])), reason: "combined_tight" }, ctx()) };
		if (s.kind === "combo") return { label: t("guide.combo.apply", { n: vars.n ?? "" }), run: () => finishCombo(true) };
		if (s.kind === "blocked_by_pin" && s.pinnedKnob)
			return {
				label: t("guide.unpin_and_continue", { knob: vars.knob }),
				run: () => {
					if (s.pinnedKnob) actions.pin(s.pinnedKnob, false);
					requestAnimationFrame(restart);
				},
			};
		if (s.kind === "probe_offer") return { label: t("guide.measure_fix"), run: () => measure(s.move, "tight") };
		if (s.kind === "finding" && s.finding) {
			const f = s.finding;
			if (canFix(f)) return { label: t("guide.fix_it"), run: () => setFixing(f.id) };
			return { label: t("guide.next"), run: skipOffer };
		}
		if (isBand(s.kind)) return { label: t("guide.next"), run: skipOffer };
		if (s.kind === "sample_offer")
			return { label: t("guide.sample_more", { tasks: SAMPLE_MORE_TASKS }), run: () => runTool("sample_more", { reason: "budget_straddle" }, ctx()) };
		if (s.kind === "nothing_safe") {
			const f = record.findings.find((x) => x.knob_id && x.severity !== "info");
			const move = s.move ?? (f?.knob_id ? { knobId: f.knob_id, to: knobs.get(f.knob_id)?.current ?? null } : undefined);
			return { label: t("guide.measure_fix"), run: () => measure(move as { knobId: string; to: KnobValue } | undefined, f?.code ?? "no_fits_setting") };
		}
		return { label: t("guide.restart"), run: restart };
	})();

	// While the tour is on the calendar (below the cockpit), its step is repeated there, so the
	// visitor reads it beside what it's about and moves on without scrolling back up.
	const [calSlot, setCalSlot] = useState<HTMLElement | null>(null);
	const onBand = !live && isBand(active?.step.kind) ? (active!.step.kind as BandKind) : null;
	useEffect(() => {
		setCalSlot(onBand ? document.getElementById(`${BANDS[onBand]}-guide`) : null);
	}, [onBand]);

	return (
		<section className="glass pf-guide" aria-label={t("guide.label")}>
			{calSlot && active
				? createPortal(
						<div className="pf-cal__guide">
							<span className="pf-model pf-model--quiet">{t("guide.step_of", { n: active.n, of: active.of })}</span>
							<p>{t(active.step.textKey, vars)}</p>
							<button type="button" className="pf-btn pf-btn--primary pf-btn--small" onClick={primary.run}>
								{primary.label}
							</button>
						</div>,
						calSlot,
					)
				: null}
			<div className="pf-guide__head">
				<span className={live ? "pf-model" : "pf-model pf-model--quiet"}>{live ? t("guide.model.muse_spark") : t("guide.model.rule_based")}</span>
				<span className="pf-guide__step">
					{live
						? t(`guide.live.status.${liveStatus}`)
						: active?.step.kind === "move" || active?.step.kind === "probe_offer" || active?.step.kind === "sample_offer" || active?.step.kind === "finding" || isBand(active?.step.kind)
							? t("guide.step_of", { n: active.n, of: active.of })
							: active
								? t("guide.waiting")
								: t("guide.ready")}
				</span>
			</div>

			<LimitsPanel />

			{fallback ? <p className="pf-guide__notice" role="status">{fallback}</p> : null}

			{cliff ? (
				<div className="pf-guide__cliff" role="status">
					<p>
						{/* The peak can be untrusted where the setting fails (past what was measured): say the limit alone then. */}
						{t(vars.peakTrusted ? "guide.cliff" : "guide.cliff_no_peak", {
							knob: t(knobs.get(cliff.knobId)?.label_key ?? cliff.knobId),
							value: valueText(knobs.get(cliff.knobId), state.knobs[cliff.knobId]),
							peak: vars.peak,
							// Not vars.cap: an RL record's step vars reuse "cap" for the probe's token cap.
							cap: fmt(capacityBytes(record, state.knobs) ?? 0, "bytes"),
						})}
					</p>
					<button type="button" className="pf-btn pf-btn--small" onClick={() => actions.setKnob(cliff.knobId, cliff.from, "user")}>
						{t("guide.cliff.undo", { value: valueText(knobs.get(cliff.knobId), cliff.from) })}
					</button>
				</div>
			) : null}

			{live ? (
				<LiveGuideSection
					onFallback={(message) => {
						setFallback(message);
						// The live guide failed before it ever answered (the visitor's first press): the
						// rule-based tour starts in its place at once, rather than leaving a notice and a second
						// button to press (the deploy's stranger gate, Nov 1: with the live guide on and no API,
						// no move was offered). Only once, and not after the live guide has been talking: then
						// the visitor's own moves and any suggestion on screen stay theirs to decide (review).
						if (liveAnswered.current || autoStarted.current) return;
						autoStarted.current = true;
						restart();
						// The pressed button is gone: focus the tour's first move, so a screen reader hears it.
						requestAnimationFrame(() =>
							requestAnimationFrame(() => {
								const panel = document.querySelector<HTMLElement>(".pf-guide");
								const next = [...(panel?.querySelectorAll<HTMLButtonElement>("button") ?? [])].find((b) => b.textContent?.startsWith("Apply"));
								(next ?? panel?.querySelector<HTMLButtonElement>("button"))?.focus();
							}),
						);
					}}
					onStatus={(s) => {
						if (s !== "ready" && s !== "thinking") liveAnswered.current = true;
						setLiveStatus(s);
					}}
				/>
			) : (
				<>
					<p className="pf-guide__say">
						{active
							? t(active.step.textKey, vars)
							: setAside
								? t("guide.set_aside", { knob: t(knobs.get(setAside)?.label_key ?? setAside) })
								: t("guide.intro", { aim: t(`objective.${state.objective.kind ?? "min_time"}`).toLowerCase() })}
					</p>
					{active?.step.move && knobs.get(active.step.move.knobId)?.needs_code ? <p className="pf-guide__code">{t("guide.needs_code")}</p> : null}
					<div className="pf-guide__actions">
						<button type="button" className="pf-btn pf-btn--primary pf-btn--small" onClick={primary.run}>
							{primary.label}
						</button>
						{active?.step.kind === "probe_offer" || active?.step.kind === "sample_offer" || (active?.step.kind === "finding" && !!active.step.finding && canFix(active.step.finding)) ? (
							<button type="button" className="pf-btn pf-btn--small" onClick={skipOffer}>
								{t("guide.skip")}
							</button>
						) : null}
						{active?.step.kind === "combo" && active.step.combo?.fit === "tight" ? (
							<button type="button" className="pf-btn pf-btn--small" onClick={() => finishCombo(true)}>
								{t("guide.combo.apply", vars)}
							</button>
						) : null}
						{active?.step.kind === "combo" ? (
							<button type="button" className="pf-btn pf-btn--small" onClick={() => finishCombo(false)}>
								{t("guide.not_this")}
							</button>
						) : null}
						{active?.step.kind === "move" && active.step.move ? (
							<MoveChoices
								knob={knobs.get(active.step.move.knobId)}
								from={active.from}
								to={active.step.move.to}
								onDecline={() => finish("decline")}
								onKeep={() => finish("keep")}
								onPin={() => finish("pin")}
							/>
						) : null}
						{active ? (
							<button type="button" className="pf-btn pf-btn--small pf-btn--quiet" aria-expanded={why} onClick={() => setWhy(!why)}>
								{t("guide.why")}
							</button>
						) : null}
					</div>
					{why && active ? <p className="pf-guide__why">{t(active.step.whyKey, vars)}</p> : null}
					{active?.step.kind === "finding" && active.step.finding && fixing === active.step.finding.id ? <FixRace finding={active.step.finding} /> : null}
				</>
			)}

			{approvals.map((a) => (
				<div key={a.id} className="pf-guide__approval">
					<p>{a.run ? t(`guide.run.${a.run.status}`, { job: a.run.job_id ?? "", position: a.run.queue_position ?? "", message: a.run.message ?? "" }) : t(`guide.approval.${a.name}`)}</p>
					{a.run?.status === "submitted" && a.run.job_id ? (
						<a className="pf-link" href={`#/boarding/${encodeURIComponent(a.run.job_id)}`}>
							{t("guide.run.follow")}
						</a>
					) : null}
					<div className="pf-guide__actions">
						{!a.run ? (
							<button type="button" className="pf-btn pf-btn--primary pf-btn--small" disabled={a.running} onClick={() => void approve(a.id)}>
								{a.running ? t("guide.run.running") : t(`guide.approval.${a.name}.go`)}
							</button>
						) : null}
						<button type="button" className="pf-btn pf-btn--small" onClick={() => setApprovals((all) => all.filter((x) => x.id !== a.id))}>
							{a.run ? t("guide.run.close") : t("guide.approval.dismiss")}
						</button>
					</div>
				</div>
			))}

			{state.trace.length ? (
				<ol className="pf-trace" aria-label={t("guide.trace")}>
					{state.trace.slice(-4).map((e) => (
						<li key={e.id} className={e.ok ? "" : "is-error"}>
							<span className="pf-trace__call">
								{e.tool}({e.args})
							</span>
							<span className="pf-trace__result">→ {e.result}</span>
						</li>
					))}
				</ol>
			) : null}

			{footer.map((p) => (
				<AddonBoundary key={p.id} id={p.id}>
					<p.component record={record} knobs={state.knobs} estimates={estimates} />
				</AddonBoundary>
			))}
		</section>
	);
}

function sentence(text: string): string {
	const s = text.trim();
	return !s || /[.!?)"'’”]$/.test(s) ? s : `${s}.`;
}
