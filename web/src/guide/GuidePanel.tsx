import { useEffect, useRef, useState } from "react";
import { fmt, fmtDelta } from "../lib/fmt";
import { flag } from "../lib/flags";
import { hasKey, t } from "../lib/i18n";
import { usePanels } from "../registry";
import type { KnobSpec, KnobValue, Objective } from "../workspace/record/types";
import { useWorkspace } from "../workspace/state";
import { memoryFit, trusted, type Fit } from "../workspace/status";
import { GuideFallback, runLiveTurn, type InputItem } from "./liveGuide";
import { MAX_TOUR_STEPS, nextStep, remainingSteps, type TourStep } from "./ruleTour";

interface ActiveStep {
	step: TourStep;
	from?: KnobValue;
	n: number;
	of: number;
}

const OBJECTIVES: Objective[] = ["cost", "time", "headroom", "quality"];

export function valueText(spec: KnobSpec | undefined, v: KnobValue | undefined): string {
	if (v === undefined) return "";
	if (typeof v === "boolean") return t(v ? "value.on" : "value.off");
	if (typeof v === "string") return hasKey(`knobvalue.${v}`) ? t(`knobvalue.${v}`) : v;
	if (spec?.unit === "tokens") return new Intl.NumberFormat().format(v);
	return fmt(v, "count");
}

export function GuidePanel() {
	const ws = useWorkspace();
	const { record, provider, state, actions, estimates } = ws;
	const footer = usePanels("guide-footer", record.job.lens);
	const [objective, setObjective] = useState<Objective>(record.job.objective);
	const [active, setActive] = useState<ActiveStep | null>(null);
	const [visited, setVisited] = useState<ReadonlySet<string>>(new Set());
	const [taken, setTaken] = useState(0);
	const [why, setWhy] = useState(false);
	const [cliff, setCliff] = useState<{ knobId: string; from: KnobValue; fit: Fit } | null>(null);
	const [fallback, setFallback] = useState<string | null>(null);
	const live = flag("c.guide.live") && !fallback;
	const [history, setHistory] = useState<InputItem[]>([]);
	const [liveText, setLiveText] = useState("");
	const [ask, setAsk] = useState("");
	const [busy, setBusy] = useState(false);
	const [approvals, setApprovals] = useState<Array<{ name: string; args: Record<string, unknown> }>>([]);
	const specs = new Map(record.knobs.map((k) => [k.id, k]));

	// Read fresh values inside handlers without re-binding them every render.
	const latest = useRef(ws);
	latest.current = ws;

	const begin = (nextVisited: ReadonlySet<string>, nextTaken: number, obj = objective) => {
		const { state: s } = latest.current;
		const ctx = { record, provider, knobs: s.knobs, pinned: s.pinned, objective: obj, visited: nextVisited };
		const step = nextStep(ctx);
		const n = nextTaken + 1;
		const of = Math.min(MAX_TOUR_STEPS, nextTaken + remainingSteps(ctx));
		setWhy(false);
		if (step.kind === "move" && step.knobId !== undefined && step.to !== undefined && n <= MAX_TOUR_STEPS) {
			const from = s.knobs[step.knobId];
			actions.highlight([...step.nodes, `knob:${step.knobId}`]);
			actions.setKnob(step.knobId, step.to, "guide");
			if (from !== undefined) actions.propose({ knobId: step.knobId, from, to: step.to, by: "guide" });
			setActive({ step, ...(from !== undefined ? { from } : {}), n, of: Math.max(of, n) });
			return;
		}
		actions.highlight(step.kind === "blocked_by_pin" ? [...step.nodes, `knob:${step.pinnedKnob}`] : []);
		actions.propose(null);
		setActive({ step: step.kind === "move" ? { ...step, kind: "done", textKey: "guide.done", whyKey: "guide.done.why" } : step, n, of: n });
	};

	const finishStep = (outcome: "apply" | "revert" | "pin") => {
		if (!active) return;
		const { step, from } = active;
		if (step.kind === "move" && step.knobId) {
			if (outcome !== "apply" && from !== undefined) actions.setKnob(step.knobId, from, "guide");
			if (outcome === "pin") actions.pin(step.knobId, true);
		}
		actions.propose(null);
		const v = new Set(visited);
		if (step.knobId) v.add(step.knobId);
		setVisited(v);
		setTaken(taken + 1);
		// Let the knob change land before ranking the next step.
		requestAnimationFrame(() => begin(v, taken + 1));
	};

	const sendLive = async (text: string) => {
		setBusy(true);
		try {
			const out = await runLiveTurn(
				history,
				text,
				() => {
					const { state: s } = latest.current;
					return {
						record,
						provider,
						actions,
						knobs: s.knobs,
						pinned: s.pinned,
						requestApproval: (name, args) => setApprovals((a) => [...a, { name, args }]),
					};
				},
				{ recordRef: record.record_id, onText: setLiveText },
			);
			setHistory(out.history);
		} catch (e) {
			if (e instanceof GuideFallback) setFallback(e.message || t("guide.fallback"));
			else setFallback(t("guide.fallback"));
		} finally {
			setBusy(false);
		}
	};

	const restart = (obj = objective) => {
		const v = new Set<string>();
		setVisited(v);
		setTaken(0);
		begin(v, 0, obj);
	};

	// Hand over the controls: when the person crosses a cliff themselves, speak up once.
	const prevFit = useRef<Fit>(memoryFit(estimates, record));
	const prevKnobs = useRef(state.knobs);
	useEffect(() => {
		const fit = memoryFit(estimates, record);
		const change = state.lastChange;
		if (change?.by === "user" && fit === "fails" && prevFit.current !== "fails") {
			const from = prevKnobs.current[change.knobId];
			if (from !== undefined) setCliff({ knobId: change.knobId, from, fit });
		} else if (fit !== "fails") {
			setCliff(null);
		}
		prevFit.current = fit;
		prevKnobs.current = state.knobs;
	}, [estimates, record, state.knobs, state.lastChange]);

	const objTarget = active?.step.impact?.target_id;
	const target = record.predictions.targets.find((x) => x.id === objTarget);
	const vars = (() => {
		const step = active?.step;
		const knobId = step?.knobId ?? step?.pinnedKnob;
		const spec = knobId ? specs.get(knobId) : undefined;
		const peak = estimates.peak_memory;
		return {
			knob: spec ? t(spec.label_key) : "",
			from: valueText(spec, active?.from),
			to: valueText(spec, step?.to),
			delta: step?.impact && target ? fmtDelta(step.impact.delta, target.unit) : "",
			target: target ? t(target.label_key).toLowerCase() : "",
			peak: trusted(peak) ? fmt(peak.point, "bytes") : "",
		};
	})();

	return (
		<aside className="pf-guide" aria-label={t("guide.label")}>
			<header className="pf-guide__head">
				<h2 className="pf-guide__title">{t("guide.title")}</h2>
				<span className="pf-guide__model">{live ? t("guide.model.muse_spark") : t("guide.model.rule_based")}</span>
			</header>

			<div className="pf-guide__objective">
				<label htmlFor="pf-objective">{t("guide.objective")}</label>
				<select
					id="pf-objective"
					value={objective}
					onChange={(e) => {
						const o = e.currentTarget.value as Objective;
						setObjective(o);
						if (active) restart(o);
					}}
				>
					{OBJECTIVES.map((o) => (
						<option key={o} value={o}>
							{t(`objective.${o}`)}
						</option>
					))}
				</select>
				{state.pinned.size ? (
					<p className="pf-muted">
						{t("guide.keeping", {
							knobs: [...state.pinned].map((id) => t(specs.get(id)?.label_key ?? id).toLowerCase()).join(", "),
						})}
					</p>
				) : null}
			</div>

			{fallback ? <p className="pf-guide__notice" role="status">{fallback}</p> : null}

			{live ? (
				<form
					className="pf-guide__ask"
					onSubmit={(e) => {
						e.preventDefault();
						if (ask.trim() && !busy) {
							void sendLive(ask.trim());
							setAsk("");
						}
					}}
				>
					<label htmlFor="pf-guide-ask">{t("guide.ask.label")}</label>
					<textarea
						id="pf-guide-ask"
						rows={3}
						value={ask}
						placeholder={t("guide.ask.placeholder")}
						onChange={(e) => setAsk(e.currentTarget.value)}
					/>
					<button type="submit" className="pf-btn pf-btn--primary" disabled={busy || !ask.trim()}>
						{busy ? t("guide.ask.thinking") : t("guide.ask.send")}
					</button>
					{liveText ? <p className="pf-guide__line">{liveText}</p> : null}
				</form>
			) : null}

			{approvals.map((a, i) => (
				<div key={`${a.name}-${i}`} className="pf-guide__approval">
					<p>{t(`guide.approval.${a.name}`)}</p>
					<button type="button" className="pf-btn" onClick={() => setApprovals((all) => all.filter((_, j) => j !== i))}>
						{t("guide.approval.dismiss")}
					</button>
				</div>
			))}

			{cliff ? (
				<div className="pf-guide__cliff" role="status">
					<p>
						{t("guide.cliff", {
							knob: t(specs.get(cliff.knobId)?.label_key ?? cliff.knobId),
							value: valueText(specs.get(cliff.knobId), state.knobs[cliff.knobId]),
							peak: vars.peak,
						})}
					</p>
					<button type="button" className="pf-btn" onClick={() => actions.setKnob(cliff.knobId, cliff.from, "user")}>
						{t("guide.cliff.undo", { value: valueText(specs.get(cliff.knobId), cliff.from) })}
					</button>
				</div>
			) : null}

			{live ? null : !active ? (
				<div className="pf-guide__intro">
					<p>{t("guide.intro")}</p>
					<button type="button" className="pf-btn pf-btn--primary" onClick={() => restart()}>
						{t("guide.start")}
					</button>
				</div>
			) : (
				<div className="pf-guide__step">
					{active.step.kind === "move" ? (
						<p className="pf-guide__count">{t("guide.step_of", { n: active.n, of: active.of })}</p>
					) : null}
					<p className="pf-guide__line">{t(active.step.textKey, vars)}</p>
					{active.step.kind === "move" && active.step.knobId && specs.get(active.step.knobId)?.needs_code ? (
						<p className="pf-guide__code">{t("guide.needs_code")}</p>
					) : null}
					{why ? <p className="pf-guide__why">{t(active.step.whyKey, vars)}</p> : null}
					<div className="pf-guide__actions">
						{active.step.kind === "move" ? (
							<>
								<button type="button" className="pf-btn pf-btn--primary" onClick={() => finishStep("apply")}>
									{t("guide.apply")}
								</button>
								<button type="button" className="pf-btn" onClick={() => finishStep("revert")}>
									{t("guide.revert")}
								</button>
								<button type="button" className="pf-btn" onClick={() => finishStep("pin")} title={t("guide.pin.help")}>
									{t("guide.pin")}
								</button>
							</>
						) : active.step.kind === "blocked_by_pin" && active.step.pinnedKnob ? (
							<button
								type="button"
								className="pf-btn pf-btn--primary"
								onClick={() => {
									if (active.step.pinnedKnob) actions.pin(active.step.pinnedKnob, false);
									restart();
								}}
							>
								{t("guide.unpin_and_continue", { knob: vars.knob })}
							</button>
						) : (
							<button type="button" className="pf-btn" onClick={() => restart()}>
								{t("guide.restart")}
							</button>
						)}
						<button type="button" className="pf-btn pf-btn--quiet" aria-expanded={why} onClick={() => setWhy(!why)}>
							{t("guide.why")}
						</button>
					</div>
				</div>
			)}

			<footer className="pf-guide__foot">
				{footer.map((p) => (
					<p.component key={p.id} record={record} knobs={state.knobs} estimates={estimates} />
				))}
			</footer>
		</aside>
	);
}
