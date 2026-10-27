/**
 * Boarding: probes running. Follows the job's server-sent events (A emits, C relays) and
 * moves along the route as stages complete. The map skeleton arrives within a minute; when
 * the record is ready the cockpit opens on it.
 */
import { useEffect, useState } from "react";
import { eventText } from "../../lib/errors";
import { t } from "../../lib/i18n";
import type { JobEvent } from "../../workspace/record/types";
import { go, href } from "../route";

/** Same origin: /v1 is served by the web app's own Pages Function (functions/v1). */
const API_BASE = "";

const STAGES: Array<{ stage: JobEvent["stage"][]; key: string; model?: string }> = [
	{ stage: ["queued", "admitted"], key: "boarding.stage.queue" },
	{ stage: ["fetching", "mapping", "skeleton"], key: "boarding.stage.read", model: "Muse Spark" },
	{ stage: ["harnessing", "probing"], key: "boarding.stage.probe" },
	{ stage: ["triage"], key: "boarding.stage.triage", model: "Muse Spark Open" },
	{ stage: ["predicting", "done"], key: "boarding.stage.fit" },
];

export function Boarding({ jobId }: { jobId: string | null }) {
	const [events, setEvents] = useState<JobEvent[]>([]);
	const [lost, setLost] = useState(false);

	useEffect(() => {
		setEvents([]);
		setLost(false);
		if (!jobId || typeof EventSource === "undefined") return;
		const es = new EventSource(`${API_BASE}/v1/jobs/${encodeURIComponent(jobId)}/events`, { withCredentials: true });
		es.onmessage = (m) => {
			try {
				const ev = JSON.parse(m.data as string) as JobEvent;
				setEvents((all) => [...all, ev]);
				if (ev.stage === "done" && ev.record_id) es.close();
			} catch {
				// a malformed event is skipped, not fatal
			}
		};
		es.onerror = () => setLost(true);
		return () => es.close();
	}, [jobId]);

	const last = events[events.length - 1];
	const idx = last ? STAGES.findIndex((s) => s.stage.includes(last.stage)) : -1;
	const failed = last?.stage === "failed" || last?.stage === "cancelled";
	const done = last?.stage === "done";
	const progress = done ? 100 : Math.max(0, idx) * 25;

	return (
		<div className="pf-scroll">
			<div className="pf-inner pf-boarding">
				<span className="pf-eyebrow">{t("boarding.eyebrow")}</span>
				<h1 className="pf-display pf-display--md">{!jobId ? t("boarding.idle") : done ? t("boarding.ready") : failed ? t("boarding.failed") : t("boarding.measuring")}</h1>
				{last?.queue_position ? <p className="pf-lede pf-lede--sm">{t("boarding.queue", { n: last.queue_position })}</p> : null}
				<div className="pf-route">
					<svg viewBox="0 0 900 260" preserveAspectRatio="none" aria-hidden="true">
						<path className="pf-route__base" d="M30 230 C 260 -40, 640 -40, 870 230" />
						<path className="pf-route__done" d="M30 230 C 260 -40, 640 -40, 870 230" pathLength={100} style={{ strokeDasharray: 100, strokeDashoffset: 100 - progress }} />
					</svg>
					<span className="pf-route__end" style={{ left: "3.3%" }}>
						{t("boarding.from")}
					</span>
					<span className="pf-route__end" style={{ left: "96.7%" }}>
						{t("boarding.to")}
					</span>
				</div>
				<ol className="pf-stages">
					{STAGES.map((s, i) => (
						<li key={s.key} className={`glass pf-stage${i < idx || done ? " is-done" : i === idx && !failed ? " is-current" : ""}`}>
							<span className="pf-stage__dot" aria-hidden="true" />
							{t(s.key)}
							{s.model ? <span className="pf-model">{s.model}</span> : null}
						</li>
					))}
				</ol>
				{failed && last?.message
					? (() => {
							const e = eventText(last.message);
							return e ? (
								<div className="pf-error" role="alert">
									<p>{e.text}</p>
									{e.hint ? <p className="pf-dim">{e.hint}</p> : null}
								</div>
							) : null;
						})()
					: null}
				{!jobId ? <p className="pf-hint">{t("boarding.no_job")}</p> : null}
				{lost && !done ? <p className="pf-hint">{t("boarding.lost")}</p> : null}
				<div className="pf-cta">
					<button type="button" className="pf-btn pf-btn--primary" disabled={!done || !last?.record_id} onClick={() => last?.record_id && go("cockpit", last.record_id)}>
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
