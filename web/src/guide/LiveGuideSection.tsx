/**
 * The live guide's part of the guide panel: start the tour, read what Muse Spark says, decide
 * on its move (Apply, Not this, Keep the visitor's value, Pin the suggestion), approve or decline the spending tools it asks for (the
 * approval runs the tool on the server and the guide hears the result), or ask it something.
 * Any failure hands over to the rule-based tour through `onFallback`.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { t } from "../lib/i18n";
import type { KnobValue } from "../workspace/record/types";
import { useWorkspace } from "../workspace/state";
import { comboKeys, comboVisit, comboWords } from "./combos";
import { valueText } from "./GuidePanel";
import { NO_LIMITS } from "./limits";
import { isFailing } from "./ruleTour";
import { GuideFallback, guideCombo, LiveTour, TourCancelled, type LiveView, type TourContext } from "./liveGuide";
import { MoveChoices } from "./MoveChoices";

export function LiveGuideSection({
	onFallback,
	onStatus,
	language,
}: {
	/** `moved`: the live guide moved a knob before it failed (its suggestion stays the visitor's to decide). */
	onFallback: (message: string, moved: boolean) => void;
	onStatus?: (status: string) => void;
	language?: string;
}) {
	const ws = useWorkspace();
	const { record, recordRef, provider, state, actions } = ws;
	const latest = useRef(ws);
	latest.current = ws;
	const [view, setView] = useState<LiveView | null>(null);
	const [busy, setBusy] = useState(false);
	// What the visitor wrote on New flight ("What are you trying to do? The guide reads this."), saved as the
	// record's objective.description: it waits in the guide's own box, so Muse Spark reads it when they
	// press Ask the guide (PM, Nov 3). Theirs to edit or clear first.
	const [ask, setAsk] = useState(() => (record.job.objective?.description ?? "").trim().slice(0, 500));
	const [unpinAsk, setUnpinAsk] = useState<string | null>(null);
	// Combined moves turned down on the card: the guide's sensitivity doesn't offer them again.
	const [turnedDown, setTurnedDown] = useState<ReadonlySet<string>>(new Set());
	const declinedRef = useRef(turnedDown);
	declinedRef.current = turnedDown;
	const knobs = new Map(record.knobs.map((k) => [k.id, k]));

	const tourRef = useRef<LiveTour | null>(null);
	if (!tourRef.current) {
		const ctx = (): TourContext => {
			const { state: s } = latest.current;
			return {
				record,
				provider,
				actions,
				knobs: s.knobs,
				pinned: s.pinned,
				limits: s.limits,
				objective: { kind: s.objective.kind ?? "min_time", ...(s.objective.target_id ? { target_id: s.objective.target_id } : {}) },
				requestApproval: () => {},
				requestUnpin: (id) => setUnpinAsk(id),
				combosDeclined: declinedRef.current,
				...(language ? { language } : {}),
			};
		};
		tourRef.current = new LiveTour(recordRef, ctx, setView);
	}
	const tour = tourRef.current;
	// Closed (the visitor took the quick read, the guide fell back, another record): the tour stops, so a
	// late answer can't move a knob behind the rule-based tour.
	useEffect(() => {
		tour.reopen(); // React's development double mount closes it once on the way in
		return () => tour.cancel();
	}, [tour]);

	useEffect(() => {
		// "working": busy after a round came back (Muse Spark has spoken or moved a knob and goes on).
		onStatus?.(busy ? (view ? "working" : "thinking") : view ? view.awaiting : "ready");
	}, [busy, view, onStatus]);

	const run = async (step: (tour: LiveTour) => Promise<LiveView>) => {
		setBusy(true);
		try {
			await step(tour);
		} catch (e) {
			if (e instanceof TourCancelled) return;
			onFallback(e instanceof GuideFallback && e.message ? e.message : t("guide.fallback"), tour.moved);
		} finally {
			setBusy(false);
		}
	};

	// The guide's move waits for the visitor. Their own knob move always wins: the reducer sets the
	// suggestion aside (putting another knob back if they moved a different one), and the guide hears it.
	const proposal = state.proposal;
	const setAside = state.setAside;
	const told = useRef<unknown>(null);
	useEffect(() => {
		if (!setAside || view?.awaiting !== "decision" || busy || told.current === setAside) return;
		told.current = setAside;
		const mine = state.lastChange?.knobId === setAside.knobId;
		void run((g) =>
			g.decide(
				mine
					? { knob_id: setAside.knobId, action: "changed", value: state.knobs[setAside.knobId] as KnobValue }
					: { knob_id: setAside.knobId, action: "reverted", value: setAside.from },
			),
		);
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [setAside, view?.awaiting]);

	/** applied, declined ("Not this"), kept (back to the visitor's value, pinned), pinned (at the suggestion). */
	const decide = (action: "applied" | "declined" | "kept" | "pinned") => {
		if (!proposal) return;
		const { knobId, from, to } = proposal;
		// Clear the suggestion first, so putting the knob back isn't read as the visitor's own move.
		actions.propose(null);
		if (action === "declined" || action === "kept") actions.setKnob(knobId, from, "user", { putBack: true });
		if (action === "kept" || action === "pinned") actions.pin(knobId, true);
		if (action === "declined") actions.decline(knobId, to);
		const value = action === "applied" || action === "pinned" || action === "declined" ? to : from;
		// Sent on the next frame, once the put-back knob, the new pin or exclusion are committed, so
		// the setting the server checks the next move against is the one on screen.
		requestAnimationFrame(() => void run((g) => g.decide({ knob_id: knobId, action, value })));
	};

	const waitingOnMove = view?.awaiting === "decision" && proposal;

	// No single knob clears the failure, but a combined setting the record's predictions cover does:
	// the guide is told (sensitivity's "combined") and the visitor applies it here, in one click.
	const combo = useMemo(() => {
		// Never beside a move of the guide's still waiting: applying would put that knob back.
		if (!view || proposal || !isFailing({ record, provider, knobs: state.knobs })) return null;
		// The same rule as the guide's sensitivity tool, so the card is there whenever the guide names it.
		return guideCombo(record, provider, state.knobs, state.objective.kind ?? "min_time", state.limits ?? NO_LIMITS, state.pinned, turnedDown);
	}, [view, proposal, record, provider, state.knobs, state.limits, state.pinned, state.objective.kind, turnedDown]);
	const words = combo ? comboWords(record, state.knobs, combo) : null;
	const knob = proposal ? knobs.get(proposal.knobId) : undefined;

	return (
		<>
			{view?.text ? <p className="pf-guide__say">{view.text}</p> : null}
			{view?.withheld ? (
				<p className="pf-guide__notice" role="status">
					{t("guide.withheld")}
				</p>
			) : null}
			{!view ? (
				<>
					<p className="pf-guide__say pf-guide__intro">{t("guide.live.intro")}</p>
					<div className="pf-guide__actions">
						<button type="button" className="pf-btn pf-btn--primary pf-btn--small" disabled={busy} onClick={() => void run((g) => g.start())}>
							{busy ? t("guide.ask.thinking") : t("guide.start")}
						</button>
					</div>
				</>
			) : null}
			{waitingOnMove && proposal ? (
				<div className="pf-guide__actions">
					<button type="button" className="pf-btn pf-btn--primary pf-btn--small" disabled={busy} onClick={() => decide("applied")}>
						{t("guide.apply_value", { knob: knob ? t(knob.label_key) : proposal.knobId, to: valueText(knob, proposal.to) })}
					</button>
					<MoveChoices
						knob={knob}
						from={proposal.from}
						to={proposal.to}
						disabled={busy}
						onDecline={() => decide("declined")}
						onKeep={() => decide("kept")}
						onPin={() => decide("pinned")}
					/>
				</div>
			) : null}
			{combo && words && !busy ? (
				<div className="pf-guide__approval" role="group" aria-label={t("guide.combo.label")}>
					<p>{t(comboKeys(combo).textKey, words)}</p>
					{words.rows_cut ? <p className="pf-guide__why">{words.rows_cut}</p> : null}
					<div className="pf-guide__actions">
						{combo.fit === "tight" ? (
							<button type="button" className="pf-btn pf-btn--primary pf-btn--small" onClick={() => void run((g) => g.ask(t("guide.combo.measure_ask", words)))}>
								{t("guide.combo.measure")}
							</button>
						) : null}
						<button
							type="button"
							className={`pf-btn pf-btn--small${combo.fit === "fits" ? " pf-btn--primary" : ""}`}
							onClick={() => {
								for (const c of combo.changes) actions.setKnob(c.knobId, c.to, "user");
							}}
						>
							{t("guide.combo.apply", words)}
						</button>
						<button type="button" className="pf-btn pf-btn--small" onClick={() => setTurnedDown(new Set([...turnedDown, comboVisit(combo)]))}>
							{t("guide.not_this")}
						</button>
					</div>
				</div>
			) : null}
			{unpinAsk ? (
				<div className="pf-guide__approval" role="group" aria-label={t("guide.unpin_ask.label")}>
					<p>
						{t("guide.unpin_ask", {
							knob: knobs.get(unpinAsk) ? t(knobs.get(unpinAsk)!.label_key) : unpinAsk,
							value: valueText(knobs.get(unpinAsk), state.knobs[unpinAsk]),
						})}
					</p>
					<div className="pf-guide__actions">
						<button
							type="button"
							className="pf-btn pf-btn--small"
							onClick={() => {
								actions.pin(unpinAsk, false);
								setUnpinAsk(null);
							}}
						>
							{t("guide.unpin_ask.release")}
						</button>
						<button type="button" className="pf-btn pf-btn--small pf-btn--quiet" onClick={() => setUnpinAsk(null)}>
							{t("guide.unpin_ask.keep", { value: valueText(knobs.get(unpinAsk), state.knobs[unpinAsk]) })}
						</button>
					</div>
				</div>
			) : null}
			{view?.approvals.map((a) => (
				<div key={a.call_id} className="pf-guide__approval">
					<p>{t(`guide.approval.${a.name}`)}</p>
					<div className="pf-guide__actions">
						<button type="button" className="pf-btn pf-btn--primary pf-btn--small" disabled={busy} onClick={() => void run((g) => g.approve(a.call_id, true))}>
							{t(`guide.approval.${a.name}.go`)}
						</button>
						<button type="button" className="pf-btn pf-btn--small" disabled={busy} onClick={() => void run((g) => g.approve(a.call_id, false))}>
							{t("guide.approval.dismiss")}
						</button>
					</div>
				</div>
			))}
			{view ? (
				<form
					className="pf-guide__ask"
					onSubmit={(e) => {
						e.preventDefault();
						const text = ask.trim();
						if (text && !busy) {
							setAsk("");
							void run((g) => g.ask(text));
						}
					}}
				>
					<label htmlFor="pf-guide-ask">{t("guide.ask.label")}</label>
					<textarea id="pf-guide-ask" rows={2} maxLength={500} value={ask} placeholder={t("guide.ask.placeholder")} onChange={(e) => setAsk(e.currentTarget.value)} />
					<p className="pf-guide__fine">{t("guide.ask.datause")}</p>
					<button type="submit" className="pf-btn pf-btn--small" disabled={busy || !ask.trim()}>
						{busy ? t("guide.ask.thinking") : t("guide.ask.send")}
					</button>
				</form>
			) : null}
		</>
	);
}
