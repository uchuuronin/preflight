/**
 * Boarding: probes running. Follows the job's server-sent events (A emits, C relays) and
 * moves along the route as stages complete. The record's share token arrives with the map
 * skeleton; when the job is done the cockpit opens on it (boarding.ts).
 */
import { useEffect, useState } from "react";
import { t } from "../../lib/i18n";
import { type BoardingEvent, boardingState, type ModelCallSeen, modelsRan, STAGES } from "../boarding";
import { go, href } from "../route";

/** Same origin: /v1 is served by the web app's own Pages Function (functions/v1). */
const API_BASE = "";

export function Boarding({ jobId }: { jobId: string | null }) {
	const [events, setEvents] = useState<BoardingEvent[]>([]);
	const [lost, setLost] = useState(false);

	useEffect(() => {
		setEvents([]);
		setLost(false);
		if (!jobId || typeof EventSource === "undefined") return;
		const es = new EventSource(`${API_BASE}/v1/jobs/${encodeURIComponent(jobId)}/events`, { withCredentials: true });
		es.onmessage = (m) => {
			try {
				const ev = JSON.parse(m.data as string) as BoardingEvent;
				setEvents((all) => [...all, ev]);
				// The server ends the stream after these; closing here stops EventSource reconnecting.
				if (ev.stage === "done" || ev.stage === "failed" || ev.stage === "cancelled") es.close();
			} catch {
				// a malformed event is skipped, not fatal
			}
		};
		es.onerror = () => setLost(true);
		return () => es.close();
	}, [jobId]);

	const { last, done, failed, cancelled, open, error, at: idx } = boardingState(events);

	// A model's tag shows only once the record says it was called (its provenance), never just
	// because the job reached that leg. Re-read when the record changes (record_rev).
	const rev = events.reduce((n, e) => Math.max(n, e.record_rev ?? 0), 0);
	const [calls, setCalls] = useState<ModelCallSeen[]>([]);
	// A new job starts with no tags: the last job's model calls aren't this one's (D's audit 12).
	useEffect(() => setCalls([]), [jobId]);
	useEffect(() => {
		if (!open) return;
		let gone = false;
		fetch(`${API_BASE}/v1/records/${encodeURIComponent(open)}`, { credentials: "same-origin" })
			.then((r) => (r.ok ? r.json() : null))
			.then((rec: { provenance?: ModelCallSeen[] } | null) => {
				if (!gone && rec?.provenance) setCalls(rec.provenance);
			})
			.catch(() => {});
		return () => {
			gone = true;
		};
	}, [open, rev, done]);
	const ran = modelsRan(calls);
	const progress = done ? 100 : Math.max(0, idx) * 25;

	return (
		<div className="pf-scroll">
			<div className="pf-inner pf-boarding">
				<span className="pf-eyebrow">{t("boarding.eyebrow")}</span>
				<h1 className="pf-display pf-display--md">{!jobId ? t("boarding.idle") : done ? t("boarding.ready") : cancelled ? t("boarding.cancelled") : failed ? t("boarding.failed") : t("boarding.measuring")}</h1>
				{last?.queue_position ? <p className="pf-lede pf-lede--sm">{t("boarding.queue", { n: last.queue_position })}</p> : null}
				<div className="pf-route">
					<svg viewBox="0 0 900 260" preserveAspectRatio="none" aria-hidden="true">
						<path className="pf-route__base" d="M30 230 C 260 -40, 640 -40, 870 230" />
						<path className={`pf-route__done${failed ? " is-stopped" : ""}`} d="M30 230 C 260 -40, 640 -40, 870 230" pathLength={100} style={{ strokeDasharray: 100, strokeDashoffset: 100 - progress }} />
					</svg>
					<span className="pf-route__end pf-route__end--from" style={{ left: "3.3%" }}>
						{t("boarding.from")}
					</span>
					<span className="pf-route__end pf-route__end--to" style={{ left: "96.7%" }}>
						{t("boarding.to")}
					</span>
				</div>
				<ol className="pf-stages">
					{STAGES.map((s, i) => (
						<li
							key={s.key}
							className={`glass pf-stage${i < idx || done ? " is-done" : i === idx ? (failed ? " is-stopped" : " is-current") : ""}`}
							aria-current={i === idx && !done ? "step" : undefined}
						>
							<span className="pf-stage__dot" aria-hidden="true" />
							{t(s.key)}
							{ran[i]?.map((m) => (
								<span key={m} className="pf-model">
									{m}
								</span>
							))}
							{i === idx && failed ? <span className="pf-stage__stopped">{t(cancelled ? "boarding.stage.cancelled_here" : "boarding.stage.stopped_here")}</span> : null}
						</li>
					))}
				</ol>
				{error ? (
					<div className="pf-error" role="alert">
						<p>{error.message}</p>
						{error.hint ? <p className="pf-dim">{error.hint}</p> : null}
					</div>
				) : null}
				{!jobId ? <p className="pf-hint">{t("boarding.no_job")}</p> : null}
				{lost && !done ? <p className="pf-hint">{t("boarding.lost")}</p> : null}
				<div className="pf-cta">
					<button type="button" className="pf-btn pf-btn--primary" disabled={!done || !open} onClick={() => open && go("cockpit", open)}>
						{t("boarding.open")}
					</button>
					<a className="pf-btn" href={href("new")}>
						{t("boarding.back")}
					</a>
				</div>
			</div>
		</div>
	);
}
