/**
 * The real run beside its prediction (in-flight mode, behind `c.inflight`): shown on a live record
 * from `preflight watch`, under the status line. It says where the run is, how it compares with
 * the predicted band, and raises the collector's alerts when the run leaves it. A live record opened
 * from a file on this machine follows the file as the run rewrites it (liveFile.ts).
 */
import { useEffect, useRef, useState } from "react";
import { fmt } from "../../lib/fmt";
import { flag } from "../../lib/flags";
import { t } from "../../lib/i18n";
import type { FlightRecord } from "../record/types";
import { useWorkspace } from "../state";
import { LIVE_REF, stopFollowing, useLive } from "./liveFile";
import { type LiveAlert, readLive } from "./liveCheck";

const bytes = (v: number) => fmt(v, "bytes");
const secs = (v: number) => fmt(v, "s");

export function alertText(a: LiveAlert): string {
	if (a.kind === "step") return t("live.alert.step_above", { step: fmt(a.step, "count"), value: secs(a.value), lo: secs(a.lo), hi: secs(a.hi) });
	if (a.kind === "peak") return t("live.alert.peak", { step: fmt(a.step, "count"), value: bytes(a.value), lo: bytes(a.lo), hi: bytes(a.hi) });
	return t("live.alert.total", { step: fmt(a.step, "count"), value: secs(a.value), lo: secs(a.lo), hi: secs(a.hi) });
}

function useNow(ms: number): number {
	const [now, setNow] = useState(() => Date.now());
	useEffect(() => {
		const id = setInterval(() => setNow(Date.now()), ms);
		return () => clearInterval(id);
	}, [ms]);
	return now;
}

const STATUS_KEY: Partial<Record<string, string>> = {
	running: "live.status.running",
	ok: "live.status.ok",
	oom: "live.status.oom",
	error: "live.status.error",
	timeout: "live.status.timeout",
	cancelled: "live.status.stopped",
};

/** Says an alert once when it starts, as the collector prints it once until the run is back in
 * band; the numbers on screen keep moving without being read out again. Always mounted. */
function Announce({ kinds, text }: { kinds: string; text: string }) {
	const last = useRef("");
	const [said, setSaid] = useState("");
	useEffect(() => {
		const fresh = kinds.split(",").filter((k) => k && !last.current.split(",").includes(k));
		last.current = kinds;
		if (fresh.length) setSaid(text);
	}, [kinds, text]);
	return (
		<p className="pf-visually-hidden" role="status" aria-live="polite">
			{said}
		</p>
	);
}

export function LivePanel() {
	const { record, recordRef } = useWorkspace();
	const live = useLive();
	const now = useNow(5_000);
	if (!flag("c.inflight") || record.job.source?.kind !== "inflight") return null;
	// Opened from a file here: the file's latest copy. By link: the record as stored.
	const fromFile = recordRef === LIVE_REF && live.record !== null;
	const rec: FlightRecord = fromFile ? live.record! : record;
	const r = readLive(rec);
	if (!r) return null;
	const statusKey = STATUS_KEY[r.status] ?? "live.status.stopped";
	const ago = fromFile && live.modified ? Math.max(0, Math.round((now - live.modified) / 1000)) : null;
	return (
		<section className="pf-live" aria-label={t("live.title")}>
			<p className="pf-live__head">
				<b>{t("live.title")}</b> <span>{t(statusKey, { step: fmt(r.steps, "count") })}</span>
				{ago !== null ? <span className="pf-dim"> {t("live.updated", { s: fmt(ago, "count") })}</span> : null}
			</p>
			<ul className="pf-live__rows">
				{r.stepS !== null ? <li>{r.bands.step ? t("live.step", { value: secs(r.stepS), lo: secs(r.bands.step[0]), hi: secs(r.bands.step[1]) }) : t("live.step_alone", { value: secs(r.stepS) })}</li> : null}
				{r.peakBytes !== null ? <li>{r.bands.peak ? t("live.peak", { value: bytes(r.peakBytes), hi: bytes(r.bands.peak[1]) }) : t("live.peak_alone", { value: bytes(r.peakBytes) })}</li> : null}
				{r.projectedS !== null && r.bands.total ? <li>{t("live.total", { value: secs(r.projectedS), lo: secs(r.bands.total[0]), hi: secs(r.bands.total[1]) })}</li> : null}
			</ul>
			{r.alerts.length ? (
				<div className="pf-live__alerts">
					{r.alerts.map((a) => (
						<p key={a.kind} className="pf-live__alert">
							{alertText(a)}
						</p>
					))}
				</div>
			) : null}
			<Announce kinds={r.alerts.map((a) => a.kind).join(",")} text={r.alerts.map(alertText).join(" ")} />
			{r.notes.map((n) => (
				<p key={n.kind} className="pf-live__note">
					{t("live.note.step_fast", { step: fmt(n.step, "count"), value: secs(n.value), lo: secs(n.lo), hi: secs(n.hi) })}
				</p>
			))}
			{!r.alerts.length && !r.notes.length && r.stepS !== null ? <p className="pf-live__ok">{t("live.in_band")}</p> : null}
			{recordRef === LIVE_REF ? (
				<p className="pf-hint">
					{live.following ? t("live.following", { name: live.name ?? "" }) : t("live.read_once", { name: live.name ?? "" })}{" "}
					{live.following ? (
						<button type="button" className="pf-link" onClick={stopFollowing}>
							{t("live.stop")}
						</button>
					) : null}
					{live.error ? <span className="pf-error"> {live.error}</span> : null}
				</p>
			) : null}
		</section>
	);
}
