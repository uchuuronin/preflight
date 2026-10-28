/**
 * The live guide's part of the guide panel: start the tour, read what Muse Spark says, decide
 * on its move (Apply, Revert, Pin), approve or decline the spending tools it asks for (the
 * approval runs the tool on the server and the guide hears the result), or ask it something.
 * Any failure hands over to the rule-based tour through `onFallback`.
 */
import { useEffect, useRef, useState } from "react";
import { t } from "../lib/i18n";
import type { KnobValue } from "../workspace/record/types";
import { useWorkspace } from "../workspace/state";
import { valueText } from "./GuidePanel";
import { type Decision, GuideFallback, LiveTour, type LiveView, type TourContext } from "./liveGuide";

export function LiveGuideSection({
	onFallback,
	onStatus,
	language,
}: {
	onFallback: (message: string) => void;
	onStatus?: (status: string) => void;
	language?: string;
}) {
	const ws = useWorkspace();
	const { record, recordRef, provider, state, actions } = ws;
	const latest = useRef(ws);
	latest.current = ws;
	const [view, setView] = useState<LiveView | null>(null);
	const [busy, setBusy] = useState(false);
	const [ask, setAsk] = useState("");
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
				objective: { kind: s.objective.kind ?? "min_time", ...(s.objective.target_id ? { target_id: s.objective.target_id } : {}) },
				requestApproval: () => {},
				...(language ? { language } : {}),
			};
		};
		tourRef.current = new LiveTour(recordRef, ctx, setView);
	}
	const tour = tourRef.current;

	useEffect(() => {
		onStatus?.(busy ? "thinking" : view ? view.awaiting : "ready");
	}, [busy, view, onStatus]);

	const run = async (step: (tour: LiveTour) => Promise<LiveView>) => {
		setBusy(true);
		try {
			await step(tour);
		} catch (e) {
			onFallback(e instanceof GuideFallback && e.message ? e.message : t("guide.fallback"));
		} finally {
			setBusy(false);
		}
	};

	// The guide's move waits for the visitor. If they change that knob themselves, the guide hears it.
	const proposal = state.proposal;
	const lastChange = state.lastChange;
	const told = useRef<string | null>(null);
	useEffect(() => {
		if (!proposal || view?.awaiting !== "decision" || busy) return;
		if (lastChange?.by === "user" && lastChange.knobId === proposal.knobId) {
			const stamp = `${proposal.knobId}:${String(state.knobs[proposal.knobId])}`;
			if (told.current === stamp) return;
			told.current = stamp;
			actions.propose(null);
			void run((g) => g.decide({ knob_id: proposal.knobId, action: "changed", value: state.knobs[proposal.knobId] as KnobValue }));
		}
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [lastChange, proposal, view?.awaiting]);

	const decide = (action: Decision["action"]) => {
		if (!proposal) return;
		const { knobId, from, to } = proposal;
		if (action === "reverted" || action === "pinned") actions.setKnob(knobId, from, "user");
		if (action === "pinned") actions.pin(knobId, true);
		actions.propose(null);
		void run((g) => g.decide({ knob_id: knobId, action, value: action === "applied" ? to : from }));
	};

	const waitingOnMove = view?.awaiting === "decision" && proposal;
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
					<p className="pf-guide__say">{t("guide.live.intro")}</p>
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
					<button type="button" className="pf-btn pf-btn--small" disabled={busy} onClick={() => decide("reverted")}>
						{t("guide.revert")}
					</button>
					<button type="button" className="pf-btn pf-btn--small" disabled={busy} onClick={() => decide("pinned")} title={t("guide.pin.help")}>
						{t("guide.pin")}
					</button>
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
					<button type="submit" className="pf-btn pf-btn--small" disabled={busy || !ask.trim()}>
						{busy ? t("guide.ask.thinking") : t("guide.ask.send")}
					</button>
				</form>
			) : null}
		</>
	);
}
