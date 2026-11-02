/**
 * Boarding: probes running. Follows the job's server-sent events (A emits, C relays) and
 * moves along the route as stages complete. The record's share token arrives with the map
 * skeleton; when the job is done the cockpit opens on it (boarding.ts).
 */
import { useEffect, useState } from "react";
import { t } from "../../lib/i18n";
import { type BoardingEvent, boardingState, eventFromStatus, type ModelCallSeen, modelsRan, STAGES } from "../boarding";
import { go, href } from "../route";
import { CheckinHelp } from "../CheckinHelp";
import { type ProbeSeen, troubleFromFailure, troubleFromProbes } from "../checkinTrouble";

/** Same origin: /v1 is served by the web app's own Pages Function (functions/v1). */
const API_BASE = "";

/** The job's events so far, plus one more, unless the job has already ended: the stream and the
 * job's own status can both report the end, and nothing after it may reopen a finished job. */
function withEvent(all: BoardingEvent[], ev: BoardingEvent): BoardingEvent[] {
	return all.some((e) => e.stage === "done" || e.stage === "failed" || e.stage === "cancelled") ? all : [...all, ev];
}

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
				setEvents((all) => withEvent(all, ev));
				// The server ends the stream after these; closing here stops EventSource reconnecting.
				if (ev.stage === "done" || ev.stage === "failed" || ev.stage === "cancelled") es.close();
			} catch {
				// a malformed event is skipped, not fatal
			}
		};
		es.onerror = () => setLost(true);
		es.onopen = () => setLost(false); // back: the stream speaks for the job again
		return () => es.close();
	}, [jobId]);

	const { last, done, failed, cancelled, open, error, at: idx, note } = boardingState(events);
	const ended = done || failed;

	// The stream lost: ask the job itself every 10 s until it has ended, so the page still learns it's
	// done (or why it stopped) if the stream doesn't come back.
	useEffect(() => {
		if (!lost || ended || !jobId) return;
		let gone = false;
		const ask = () =>
			fetch(`${API_BASE}/v1/jobs/${encodeURIComponent(jobId)}`, { credentials: "same-origin" })
				.then((r) => {
					// Not this visitor's job, or no such job: asking again won't change that.
					if (r.status >= 400 && r.status < 500 && r.status !== 429) {
						gone = true;
						clearInterval(id);
					}
					return r.ok ? r.json() : null;
				})
				.then((st) => {
					const ev = eventFromStatus(st);
					if (!gone && ev) setEvents((all) => withEvent(all, ev));
				})
				.catch(() => {});
		const id = setInterval(ask, 10_000);
		void ask();
		return () => {
			gone = true;
			clearInterval(id);
		};
	}, [lost, ended, jobId]);

	// A model's tag shows only once the record says it was called (its provenance), never just
	// because the job reached that leg. Re-read when the record changes (record_rev).
	const rev = events.reduce((n, e) => Math.max(n, e.record_rev ?? 0), 0);
	const [calls, setCalls] = useState<ModelCallSeen[]>([]);
	const [probes, setProbes] = useState<ProbeSeen[]>([]);
	// The packages Preflight installs, by import name (GET /v1/packages, A's list), to fill the box.
	const [known, setKnown] = useState<{ known: Record<string, string>; noWheels: string[] }>({ known: {}, noWheels: [] });
	// A new job, or a new record, starts with no tags: the last one's model calls aren't this one's
	// (D's audit 12, with D's fix: clear on the record too, which the same-page case needs).
	useEffect(() => {
		setCalls([]);
		setProbes([]);
	}, [jobId, open]);
	useEffect(() => {
		if (!open) return;
		let gone = false;
		fetch(`${API_BASE}/v1/records/${encodeURIComponent(open)}`, { credentials: "same-origin" })
			.then((r) => (r.ok ? r.json() : null))
			.then((rec: { provenance?: ModelCallSeen[]; probes?: ProbeSeen[] } | null) => {
				if (gone || !rec) return;
				if (rec.provenance) setCalls(rec.provenance);
				if (done && rec.probes) setProbes(rec.probes);
			})
			.catch(() => {});
		return () => {
			gone = true;
		};
	}, [open, rev, done]);
	const ran = modelsRan(calls);
	const trouble = failed ? troubleFromFailure(last?.error_code, last?.message) : done ? troubleFromProbes(probes, known.known, known.noWheels) : null;
	const installStop = trouble?.step === "install";
	useEffect(() => {
		if (!installStop) return;
		let gone = false;
		fetch(`${API_BASE}/v1/packages`, { credentials: "same-origin" })
			.then((r) => (r.ok ? r.json() : null))
			.then((j: { known?: Record<string, string>; no_wheels?: string[] } | null) => {
				if (!gone && j?.known) setKnown({ known: j.known, noWheels: j.no_wheels ?? [] });
			})
			.catch(() => {});
		return () => {
			gone = true;
		};
	}, [installStop]);
	// A finished job whose probes all stopped at an import measured nothing: the route stops at the probe leg.
	const probeLeg = STAGES.findIndex((s) => s.key === "boarding.stage.probe");
	const stopAt = installStop ? probeLeg : -1;
	const progress = stopAt >= 0 ? stopAt * 25 : done ? 100 : Math.max(0, idx) * 25;

	return (
		<div className="pf-scroll">
			<div className="pf-inner pf-boarding">
				<span className="pf-eyebrow">{t("boarding.eyebrow")}</span>
				<h1 className="pf-display pf-display--md">{!jobId ? t("boarding.idle") : done ? t(installStop ? "boarding.stopped_install" : "boarding.ready") : cancelled ? t("boarding.cancelled") : failed ? t("boarding.failed") : t("boarding.measuring")}</h1>
				{last?.queue_position ? <p className="pf-lede pf-lede--sm">{t("boarding.queue", { n: last.queue_position })}</p> : null}
				<div className="pf-route">
					<svg viewBox="0 0 900 260" preserveAspectRatio="none" aria-hidden="true">
						<path className="pf-route__base" d="M30 230 C 260 -40, 640 -40, 870 230" />
						<path className={`pf-route__done${failed || stopAt >= 0 ? " is-stopped" : ""}`} d="M30 230 C 260 -40, 640 -40, 870 230" pathLength={100} style={{ strokeDasharray: 100, strokeDashoffset: 100 - progress }} />
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
							className={`glass pf-stage${stopAt >= 0 ? (i < stopAt ? " is-done" : i === stopAt ? " is-stopped" : "") : i < idx || done ? " is-done" : i === idx ? (failed ? " is-stopped" : " is-current") : ""}`}
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
				{/* The runner's own words on the leg the job is on (a GPU wait), said politely as they change. */}
				<p className="pf-hint pf-stage__note" role="status" aria-live="polite">
					{!failed && !done && note ? note : ""}
				</p>
				{error ? (
					<div className="pf-error" role="alert">
						<p>{error.message}</p>
						{error.hint ? <p className="pf-dim">{error.hint}</p> : null}
					</div>
				) : null}
				{trouble ? <CheckinHelp key={jobId ?? ""} trouble={trouble} jobId={jobId} /> : null}
				{!jobId ? <p className="pf-hint">{t("boarding.no_job")}</p> : null}
				{lost && !ended ? <p className="pf-hint">{t("boarding.lost")}</p> : null}
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
