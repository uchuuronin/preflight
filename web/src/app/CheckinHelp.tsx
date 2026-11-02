/**
 * On Boarding, when a check-in stopped while reading the code or installing what it imports: the
 * step, what's missing, and the way past it (checkinHelp.ts). With `c.checkin_help` on (A's runner
 * takes `extra_packages` and `source.scope`), the way past is a box that sends the check-in again;
 * without it, the words say what to change by hand.
 */
import { useEffect, useId, useState } from "react";
import { flag } from "../lib/flags";
import { t } from "../lib/i18n";
import { fromFile, MAX_EXTRA_PACKAGES, parsePackages, rememberRequest, requestFor, type Trouble, withPackages } from "./checkinHelp";
import { go, href } from "./route";

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
	const suggested = trouble.step === "install" ? trouble.missing.map((m) => m.pkg ?? m.module) : [];
	const [text, setText] = useState(trouble.step === "install" ? suggested.join(", ") : startFile);
	const [busy, setBusy] = useState(false);
	const [error, setError] = useState<string | null>(null);
	// a new stop (another job) starts the box afresh
	useEffect(() => setText(trouble.step === "install" ? suggested.join(", ") : startFile), [jobId, trouble.step]); // eslint-disable-line react-hooks/exhaustive-deps

	const send = async (next: unknown) => {
		setBusy(true);
		setError(null);
		try {
			const res = await fetch(`${API_BASE}/v1/jobs`, {
				method: "POST",
				credentials: "same-origin",
				headers: { "Content-Type": "application/json", "Idempotency-Key": newKey() },
				body: JSON.stringify(next),
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
							<input id={ids.box} value={text} onChange={(e) => setText(e.target.value)} placeholder="train.py" aria-describedby={error ? ids.error : undefined} spellCheck={false} autoComplete="off" />
						</div>
						<button type="button" className="pf-btn pf-btn--primary" disabled={busy || !okPath} onClick={() => void send(fromFile(body, path))}>
							{t("checkin.read.go")}
						</button>
					</>
				) : (
					<p className="pf-dim">
						{t("checkin.read.way_hand")} <a href={href("new")}>{t("boarding.back")}</a>
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

	const names = trouble.missing.map((m) => m.module);
	const offList = trouble.missing.filter((m) => m.pkg === null).map((m) => m.module);
	const parsed = parsePackages(text);
	return (
		<section className="glass pf-checkin-help" aria-labelledby={ids.hint}>
			<b id={ids.hint}>{t("checkin.stop.install")}</b>
			<p>{t(names.length > 1 ? "checkin.install.many" : "checkin.install.one", { names: listOf(names) })}</p>
			{on && body ? (
				<>
					<p>{t("checkin.install.way")}</p>
					<div className="pf-field">
						<label htmlFor={ids.box}>{t("checkin.install.label")}</label>
						<input id={ids.box} value={text} onChange={(e) => setText(e.target.value)} aria-describedby={`${ids.box}-hint${error ? ` ${ids.error}` : ""}`} spellCheck={false} autoComplete="off" />
						<p id={`${ids.box}-hint`} className="pf-dim">
							{t("checkin.install.hint", { n: String(MAX_EXTRA_PACKAGES) })}
						</p>
					</div>
					{parsed.bad.length ? <p className="pf-error">{t("checkin.bad_names", { names: listOf(parsed.bad) })}</p> : null}
					{offList.length ? <p className="pf-dim">{t("checkin.install.off_list", { names: listOf(offList) })}</p> : null}
					<button type="button" className="pf-btn pf-btn--primary" disabled={busy || !parsed.names.length || parsed.bad.length > 0} onClick={() => void send(withPackages(body, parsed.names))}>
						{t("checkin.install.go")}
					</button>
				</>
			) : (
				<p className="pf-dim">
					{t("checkin.install.way_hand", { names: listOf(suggested) })} <a href={href("new")}>{t("boarding.back")}</a>
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
