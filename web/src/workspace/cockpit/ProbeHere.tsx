/**
 * "Run a probe here": when a readout at the current setting is too far from anything measured to
 * give a number (it abstains, hatched on the dials and the fit map), the visitor can measure that
 * setting. One probe of this record's own script at these knob values, through /v1/jobs (mode
 * "probe", a record source); the click is the approval, quotas are A's as for any job, and the
 * boarding screen follows it. Never on a sample: a sample has no script to run.
 */
import { useRef, useState } from "react";
import { go } from "../../app/route";
import { flag } from "../../lib/flags";
import { t } from "../../lib/i18n";
import { isFixture } from "../record/model";
import { useWorkspace } from "../state";

const API_BASE = "";
/** Readouts whose abstention means "measure it", in the order the dials show them. */
const READOUTS = ["memory", "time", "cost", "risk"] as const;
/** Source: derived, the API's share-token pattern (routes/jobs.py RECORD_REF). */
const SHARE_TOKEN = /^[A-Za-z0-9_-]{16,64}$/;

function newKey(): string {
	return `probe-${crypto.randomUUID?.() ?? Math.random().toString(36).slice(2)}`;
}

export function ProbeHere() {
	const { record, recordRef, estimates, state } = useWorkspace();
	const [busy, setBusy] = useState(false);
	const [error, setError] = useState<string | null>(null);
	const idem = useRef<{ key: string; body: string | null }>({ key: newKey(), body: null });
	if (!flag("runner.queue") || isFixture(record) || !SHARE_TOKEN.test(recordRef)) return null;
	const abstaining = READOUTS.filter((r) => {
		const target = record.predictions.targets.find((x) => x.readout === r);
		return target && estimates[target.id]?.status === "abstain";
	});
	if (!abstaining.length) return null;

	const run = async () => {
		setError(null);
		setBusy(true);
		const sent = JSON.stringify({ source: { kind: "record", record_ref: recordRef }, mode: "probe", knobs: state.knobs });
		// The same setting asked twice reuses its key, so a double click runs one probe.
		if (idem.current.body !== sent) idem.current = { key: newKey(), body: sent };
		try {
			const res = await fetch(`${API_BASE}/v1/jobs`, {
				method: "POST",
				credentials: "same-origin",
				headers: { "Content-Type": "application/json", "Idempotency-Key": idem.current.key },
				body: sent,
			});
			const json = (await res.json().catch(() => null)) as { job_id?: string; error?: { message?: string } } | null;
			if (!res.ok || !json?.job_id) {
				const retry = res.headers.get("Retry-After");
				setError(json?.error?.message ?? (res.status === 429 && retry ? t("newflight.error.busy", { s: retry }) : t("newflight.error.generic")));
				return;
			}
			go("boarding", json.job_id);
		} catch {
			setError(t("newflight.error.offline"));
		} finally {
			setBusy(false);
		}
	};

	return (
		<div className="pf-probehere" role="group" aria-label={t("probehere.label")}>
			<p className="pf-probehere__why">{t("probehere.why", { what: abstaining.map((r) => t(`probehere.readout.${r}`)).join(", ") })}</p>
			<button type="button" className="pf-btn pf-btn--primary pf-btn--small" disabled={busy} onClick={run}>
				{busy ? t("probehere.sending") : t("probehere.go")}
			</button>
			<p className="pf-probehere__cost">{t("probehere.cost")}</p>
			{error ? (
				<p className="pf-probehere__error" role="alert">
					{error}
				</p>
			) : null}
		</div>
	);
}
