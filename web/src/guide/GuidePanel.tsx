import { useEffect, useRef, useState } from "react";
import { fmt, fmtDelta } from "../lib/fmt";
import { flag } from "../lib/flags";
import { hasKey, t } from "../lib/i18n";
import { usePanels } from "../registry";
import { capacityBytes, costStraddlesBudget, failureTarget, objectiveTarget, SAMPLE_MORE_TASKS, target, targetFor } from "../workspace/record/model";
import type { Finding, Knob, KnobValue } from "../workspace/record/types";
import { useWorkspace } from "../workspace/state";
import { fitOf, trusted, type Fit } from "../workspace/status";
import { runApprovedTool, type SpendTool, type ToolRun } from "./approvals";
import { LiveGuideSection } from "./LiveGuideSection";
import { findingVisit, MAX_TOUR_STEPS, nextStep, pendingFindings, plannedSteps, SAMPLE_VISIT, type TourStep } from "./ruleTour";
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
	const live = flag("c.guide.live") && !fallback;
	const knobs = new Map(record.knobs.map((k) => [k.id, k]));

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
		const tc = { record, provider, knobs: s.knobs, pinned: s.pinned, objective: s.objective, visited: nextVisited, limits: s.limits };
		const step = nextStep(tc);
		const n = nextTaken + 1;
		if (step.kind === "probe_offer" && step.move) {
			if (step.nodes.length) runTool("highlight", { nodes: step.nodes }, ctx());
			runTool("predict", { knobs: { [step.move.knobId]: step.move.to } }, ctx());
			actions.propose(null);
			actions.openKnob(step.move.knobId);
			setWhy(false);
			setActive({ step, n, of: Math.min(MAX_TOUR_STEPS, Math.max(n, nextTaken + plannedSteps(tc))) });
			return;
		}
		setWhy(false);
		if (step.nodes.length) runTool("highlight", { nodes: step.nodes }, ctx());
		else actions.highlight([]);
		if (step.kind === "move" && step.move && n <= MAX_TOUR_STEPS) {
			const from = s.knobs[step.move.knobId];
			runTool("predict", { knobs: { [step.move.knobId]: step.move.to } }, ctx());
			runTool("set_knob", { knob_id: step.move.knobId, value: step.move.to, why: step.textKey }, ctx());
			setActive({ step, ...(from !== undefined ? { from } : {}), n, of: Math.min(MAX_TOUR_STEPS, Math.max(n, nextTaken + pendingFindings(tc) + plannedSteps(tc))) });
			return;
		}
		actions.propose(null);
		if (step.kind === "finding") {
			// A finding is a step of the tour: counted with the findings and moves still to come.
			setActive({ step, n, of: Math.max(n, Math.min(MAX_TOUR_STEPS, nextTaken + pendingFindings(tc) + plannedSteps(tc))) });
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
		else if (active.step.kind === "finding" && active.step.finding) {
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
	const finish = (outcome: "apply" | "revert" | "pin" | "keep" | "decline") => {
		if (!active?.step.move) return;
		const { move } = active.step;
		const from = active.from;
		// Clear the suggestion first, so putting the knob back isn't read as the visitor's own move.
		actions.propose(null);
		if (outcome !== "apply" && outcome !== "pin" && from !== undefined) actions.setKnob(move.knobId, from, "user");
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
		if (change?.by === "user" && fit === "fails" && prevFit.current !== "fails") {
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
			peak: trusted(peak) ? fmt(peak.point, "bytes") : t("value.none"),
			cap: cap ? fmt(cap, "bytes") : "",
			risk: trusted(risk) ? fmt(risk.point, "probability") : "",
			chance: step?.move?.risk !== null && step?.move?.risk !== undefined ? fmt(step.move.risk, "probability") : "",
			title: step?.finding?.title ?? "",
			detail: step?.finding?.detail ?? "",
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
		if (s.kind === "sample_offer")
			return { label: t("guide.sample_more", { tasks: SAMPLE_MORE_TASKS }), run: () => runTool("sample_more", { reason: "budget_straddle" }, ctx()) };
		if (s.kind === "nothing_safe") {
			const f = record.findings.find((x) => x.knob_id && x.severity !== "info");
			const move = s.move ?? (f?.knob_id ? { knobId: f.knob_id, to: knobs.get(f.knob_id)?.current ?? null } : undefined);
			return { label: t("guide.measure_fix"), run: () => measure(move as { knobId: string; to: KnobValue } | undefined, f?.code ?? "no_fits_setting") };
		}
		return { label: t("guide.restart"), run: restart };
	})();

	return (
		<section className="glass pf-guide" aria-label={t("guide.label")}>
			<div className="pf-guide__head">
				<span className={live ? "pf-model" : "pf-model pf-model--quiet"}>{live ? t("guide.model.muse_spark") : t("guide.model.rule_based")}</span>
				<span className="pf-guide__step">
					{live
						? t(`guide.live.status.${liveStatus}`)
						: active?.step.kind === "move" || active?.step.kind === "probe_offer" || active?.step.kind === "sample_offer" || active?.step.kind === "finding"
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
						{t("guide.cliff", {
							knob: t(knobs.get(cliff.knobId)?.label_key ?? cliff.knobId),
							value: valueText(knobs.get(cliff.knobId), state.knobs[cliff.knobId]),
							peak: vars.peak,
							cap: vars.cap,
						})}
					</p>
					<button type="button" className="pf-btn pf-btn--small" onClick={() => actions.setKnob(cliff.knobId, cliff.from, "user")}>
						{t("guide.cliff.undo", { value: valueText(knobs.get(cliff.knobId), cliff.from) })}
					</button>
				</div>
			) : null}

			{live ? (
				<LiveGuideSection onFallback={setFallback} onStatus={setLiveStatus} />
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
					<p>{a.run ? t(`guide.run.${a.run.status}`, { job: a.run.job_id ?? "", position: a.run.queue_position ?? "" }) : t(`guide.approval.${a.name}`)}</p>
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
