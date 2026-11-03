/**
 * On Boarding, when a check-in stopped while reading the code or installing what it imports: the
 * step, what's missing, and the way past it (checkinTrouble.ts). With `c.checkin_help` on (A's runner
 * takes `extra_packages` and `source.scope`), the way past is a box that sends the check-in again;
 * without it, the words say what to change by hand.
 */
import { useEffect, useId, useRef, useState } from "react";
import { flag } from "../lib/flags";
import { errorCopy } from "../lib/errors";
import { t } from "../lib/i18n";
import { fromFile, MAX_EXTRA_PACKAGES, parsePackages, rememberRequest, requestFor, type Trouble, withPackages } from "./checkinTrouble";
import { go } from "./route";

const API_BASE = "";

function newKey(): string {
	return typeof crypto !== "undefined" && "randomUUID" in crypto ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`;
}

/** "a", "a and b", "a, b and c" */
function listOf(names: readonly string[]): string {
	return names.length <= 1 ? (names[0] ?? "") : `${names.slice(0, -1).join(", ")} ${t("checkin.and")} ${names[names.length - 1]}`;
}

export function CheckinHelp({ trouble, jobId }: { trouble: Trouble; jobId: string | null }) {
	const ids = { box: useId(), hint: useId(), error: useId() };
	const on = flag("c.checkin_help");
	const body = requestFor(jobId);
	const startFile = body && typeof (body.source as { entrypoint?: unknown }).entrypoint === "string" ? String((body.source as { entrypoint?: string }).entrypoint) : "";
	// Only names Preflight installs go in the box: a module's own name isn't always its package's
	// (cv2 is opencv-python-headless), and a module off the list may be a file of the visitor's own.
	const suggested = trouble.step === "install" ? trouble.missing.flatMap((m) => (m.pkg ? [m.pkg] : [])) : [];
	const fill = trouble.step === "install" ? suggested.join(", ") : startFile;
	const [text, setText] = useState(fill);
	const [busy, setBusy] = useState(false);
	const [error, setError] = useState<string | null>(null);
	// a new stop, or the package list arriving, fills the box afresh
	useEffect(() => setText(fill), [jobId, fill]);
	// One key per distinct request, as New flight does: a retry of the same body can't make a second job.
	const idem = useRef<{ key: string; body: string } | null>(null);

	const send = async (next: unknown) => {
		setBusy(true);
		setError(null);
		// With A's runner.free_resend on (contract 0.6.5), the resend names the job it follows, so a job
		// that stopped at an import before measuring anything doesn't cost a second of the day's
		// check-ins. The runner decides whether it's free; the field only names the earlier job. Off, the
		// API's schema (extra fields refused) would turn the field away, so it isn't sent.
		const sent = JSON.stringify(flag("runner.free_resend") && jobId ? { ...(next as object), resend_of: jobId } : next);
		if (idem.current?.body !== sent) idem.current = { key: newKey(), body: sent };
		try {
			const res = await fetch(`${API_BASE}/v1/jobs`, {
				method: "POST",
				credentials: "same-origin",
				headers: { "Content-Type": "application/json", "Idempotency-Key": idem.current.key },
				body: sent,
			});
			const json = (await res.json().catch(() => null)) as { job_id?: string; error?: { message?: string } } | null;
			if (!res.ok || !json?.job_id) {
				setError(json?.error?.message ?? t("checkin.again_failed"));
				return;
			}
			rememberRequest(json.job_id, next as never);
			go("boarding", json.job_id);
		} catch {
			setError(t("newflight.error.offline"));
		} finally {
			setBusy(false);
		}
	};

	if (trouble.step === "read") {
		const path = text.trim();
		const okPath = /^[A-Za-z0-9_./-]+\.py$/.test(path) && !path.startsWith("/") && !path.split("/").includes("..");
		return (
			<section className="glass pf-checkin-help" aria-labelledby={ids.hint}>
				<b id={ids.hint}>{t("checkin.stop.read")}</b>
				<p>{t(trouble.limit === "map" ? "checkin.read.map" : "checkin.read.repo")}</p>
				{on && body ? (
					<>
						<p>{t("checkin.read.way")}</p>
						<div className="pf-field">
							<label htmlFor={ids.box}>{t("checkin.read.file_label")}</label>
							<input id={ids.box} value={text} onChange={(e) => setText(e.target.value)} placeholder="train.py" aria-invalid={(path !== "" && !okPath) || undefined} aria-describedby={`${ids.box}-hint${error ? ` ${ids.error}` : ""}`} spellCheck={false} autoComplete="off" />
							<p id={`${ids.box}-hint`} className={path !== "" && !okPath ? "pf-error" : "pf-dim"}>
								{t("checkin.read.file_hint")}
							</p>
						</div>
						<button type="button" className="pf-btn pf-btn--primary" disabled={busy || !okPath} onClick={() => void send(fromFile(body, path))}>
							{t("checkin.read.go")}
						</button>
					</>
				) : (
					<p className="pf-dim">
						{t("checkin.read.way_hand")}
					</p>
				)}
				{error ? (
					<p id={ids.error} className="pf-error" role="alert">
						{error}
					</p>
				) : null}
			</section>
		);
	}

	if (trouble.step === "measure") {
		const why = errorCopy(trouble.code);
		return (
			<section className="glass pf-checkin-help" aria-labelledby={ids.hint}>
				<b id={ids.hint}>{t("checkin.stop.measure")}</b>
				<p>{trouble.code ? why.message : t("checkin.measure.unknown")}</p>
				{trouble.code && why.hint ? <p className="pf-dim">{why.hint}</p> : null}
				{body ? (
					<button type="button" className="pf-btn pf-btn--primary" disabled={busy} onClick={() => void send(body)}>
						{t("checkin.measure.again")}
					</button>
				) : (
					<p className="pf-dim">{t("checkin.measure.again_hand")}</p>
				)}
				{error ? (
					<p id={ids.error} className="pf-error" role="alert">
						{error}
					</p>
				) : null}
			</section>
		);
	}

	const names = trouble.missing.map((m) => m.module);
	const noWheels = trouble.missing.filter((m) => m.noWheels).map((m) => m.module);
	const offList = trouble.missing.filter((m) => m.pkg === null && !m.noWheels).map((m) => m.module);
	const parsed = parsePackages(text);
	const asked = text.split(/[\s,]+/).filter(Boolean).length;
	const notes = (
		<>
			{offList.length ? <p className="pf-dim">{t(offList.length > 1 ? "checkin.install.off_list_many" : "checkin.install.off_list", { names: listOf(offList) })}</p> : null}
			{noWheels.length ? <p className="pf-dim">{t("checkin.install.no_wheels", { names: listOf(noWheels) })}</p> : null}
		</>
	);
	return (
		<section className="glass pf-checkin-help" aria-labelledby={ids.hint}>
			<b id={ids.hint}>{t("checkin.stop.install")}</b>
			<p>{t(names.length > 1 ? "checkin.install.many" : "checkin.install.one", { names: listOf(names) })}</p>
			{on && body ? (
				<>
					<p>{t("checkin.install.way")}</p>
					<div className="pf-field">
						<label htmlFor={ids.box}>{t("checkin.install.label")}</label>
						<input
							id={ids.box}
							value={text}
							onChange={(e) => setText(e.target.value)}
							aria-invalid={parsed.bad.length > 0 || undefined}
							aria-describedby={`${ids.box}-hint${parsed.bad.length || asked > MAX_EXTRA_PACKAGES ? ` ${ids.box}-bad` : ""}${error ? ` ${ids.error}` : ""}`}
							spellCheck={false}
							autoComplete="off"
						/>
						<p id={`${ids.box}-hint`} className="pf-dim">
							{t("checkin.install.hint", { n: String(MAX_EXTRA_PACKAGES) })}
						</p>
						{parsed.bad.length ? (
							<p id={`${ids.box}-bad`} className="pf-error">
								{t("checkin.bad_names", { names: listOf(parsed.bad) })}
							</p>
						) : asked > MAX_EXTRA_PACKAGES ? (
							<p id={`${ids.box}-bad`} className="pf-error">
								{t("checkin.too_many", { n: String(MAX_EXTRA_PACKAGES) })}
							</p>
						) : null}
					</div>
					{notes}
					<button type="button" className="pf-btn pf-btn--primary" disabled={busy || !parsed.names.length || parsed.bad.length > 0 || asked > MAX_EXTRA_PACKAGES} onClick={() => void send(withPackages(body, parsed.names))}>
						{t("checkin.install.go")}
					</button>
				</>
			) : (
				<>
					{suggested.length ? <p className="pf-dim">{t("checkin.install.way_hand", { names: listOf(suggested) })}</p> : null}
					{notes}
				</>
			)}
			{error ? (
				<p id={ids.error} className="pf-error" role="alert">
					{error}
				</p>
			) : null}
		</section>
	);
}
