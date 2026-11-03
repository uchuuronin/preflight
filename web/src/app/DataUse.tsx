/**
 * What happens to a visitor's code and words, said before they check in, and only what is
 * true of this build: every line is switched by the flag that turns its behaviour on. The
 * visitor ticks it once per browser (kept in localStorage as a convenience; the check-in
 * button stays off until it is ticked in this visit or a past one).
 */
import { useId, useState } from "react";
import { flag } from "../lib/flags";
import { t } from "../lib/i18n";

const SEEN_KEY = "pf.data_use.v1";
/** The "Got it" box, so the check-in button's hint can take a visitor to it. */
export const DATA_USE_OK_ID = "pf-datause-ok";

/**
 * How long code and records are kept: only what A's runner does today (python/preflight/runner,
 * retention.py; Chiya, Oct 30; the runner deployed Nov 1 18:04 UTC runs it once a day).
 * - A repo's clone is deleted as soon as it is read, before any probe runs (sources.py); the
 *   record keeps only the repo link and commit.
 * - Uploaded files (a notebook as sent, outputs included) are deleted 30 days after the newest job
 *   that used them, and never while a kept record or a queued or running job still names them.
 * - Records are deleted about 30 days after they are made (records.py RECORD_TTL_S; PM decision,
 *   Oct 28).
 * - Probe results, error messages included (which can quote the visitor's code), stay in the
 *   probe cache for 30 days after they are cached, so a repeat run reuses them.
 */
export const RETENTION: { repoDeletedAtProbeEnd: boolean; uploadDays: number | null; recordDays: number | null; probeDays: number | null } = {
	repoDeletedAtProbeEnd: true,
	uploadDays: 30,
	recordDays: 30,
	probeDays: 30,
};

function readSeen(): boolean {
	try {
		return localStorage.getItem(SEEN_KEY) === "1";
	} catch {
		return false;
	}
}

function writeSeen(): void {
	try {
		localStorage.setItem(SEEN_KEY, "1");
	} catch {
		// private window or blocked storage: the tick still counts for this visit
	}
}

/** The lines that apply to this build, in the order a visitor's data travels. */
export function dataUseLines(retention: typeof RETENTION = RETENTION): string[] {
	const lines = [t("datause.sent"), t("datause.probe")];
	if (flag("harness.spark_script_map")) lines.push(t("datause.script_map"));
	if (flag("harness.spark_open_triage")) lines.push(t("datause.triage"));
	// Private triage on the check-in desk (NewFlight.tsx) is offered only with both flags on.
	if (flag("harness.spark_open_triage") && flag("harness.glimmer_private")) lines.push(t(flag("harness.spark_script_map") ? "datause.glimmer_map" : "datause.glimmer"));
	// Muse Spark's diagnosis of a finished record (A's harness/diagnosis.py, behind its flag).
	if (flag("harness.spark_diagnosis")) lines.push(t("datause.diagnosis"));
	// Spark Open's check of each diagnosis claim: its own flag, and the diagnosis flag too, since A's
	// runner runs the check under harness.spark_diagnosis today (pipeline._diagnose). Never quieter
	// than the runner: api/tests/test_spark_flag_shared.py holds the two together.
	if (flag("harness.spark_claim_check") || flag("harness.spark_diagnosis")) lines.push(t("datause.claim_check"));
	if (flag("harness.spark_search")) lines.push(t("datause.search"));
	if (flag("c.guide.live")) lines.push(t("datause.guide"));
	if (flag("c.fix_loop")) lines.push(t("datause.fix"));
	lines.push(t("datause.record"));
	if (retention.repoDeletedAtProbeEnd) lines.push(t("datause.retention.repo"));
	lines.push(retention.uploadDays ? t("datause.retention.uploads", { days: String(retention.uploadDays) }) : t("datause.retention.uploads_kept"));
	lines.push(retention.recordDays ? t("datause.retention.records", { days: String(retention.recordDays) }) : t("datause.retention.unset"));
	lines.push(retention.probeDays ? t("datause.retention.probe_errors_days", { days: String(retention.probeDays) }) : t("datause.retention.probe_errors"));
	lines.push(t("datause.secrets"));
	return lines;
}

/** Who reads the code or its logs in this build, by the flags that send it to them. */
export function readersOf(): string[] {
	const who: string[] = [];
	if (flag("harness.spark_script_map") || flag("harness.spark_diagnosis") || flag("harness.spark_search") || flag("c.guide.live")) who.push(t("datause.who.spark"));
	if (flag("harness.spark_open_triage") || flag("harness.spark_claim_check") || flag("harness.spark_diagnosis")) who.push(t("datause.who.spark_open"));
	if (flag("harness.spark_open_triage") && flag("harness.glimmer_private")) who.push(t("datause.who.glimmer"));
	if (flag("c.fix_loop")) who.push(t("datause.who.code"));
	return who;
}

/**
 * The two lines a visitor reads before ticking "Got it", with every detail folded under them (D's
 * walkthrough, Nov 2: fifteen bullets and six names, and the box sat a screen below Check in). It sits
 * on the check-in desk, right above the button.
 */
export function DataUseNotice({ onChange }: { onChange: (accepted: boolean) => void }) {
	const id = useId();
	const [accepted, setAccepted] = useState(readSeen);
	const who = readersOf();
	return (
		<section className="pf-datause" aria-labelledby={`${id}-title`}>
			<b id={`${id}-title`} className="pf-datause__title">
				{t("datause.title")}
			</b>
			<p className="pf-hint">{t("datause.summary.run")}</p>
			{who.length ? <p className="pf-hint">{t("datause.summary.read", { who: listWords(who) })}</p> : null}
			<p className="pf-hint">
				{RETENTION.recordDays ? t("datause.summary.kept", { days: String(RETENTION.recordDays) }) : t("datause.retention.unset")}
			</p>
			<details className="pf-datause__more">
				<summary>{t("datause.more")}</summary>
				<ul className="pf-datause__list">
					{dataUseLines().map((line) => (
						<li key={line}>{line}</li>
					))}
				</ul>
			</details>
			<label className="pf-datause__ok">
				<input
					id={DATA_USE_OK_ID}
					type="checkbox"
					checked={accepted}
					onChange={(e) => {
						const on = e.currentTarget.checked;
						setAccepted(on);
						if (on) writeSeen();
						onChange(on);
					}}
				/>
				{t("datause.accept")}
			</label>
		</section>
	);
}

/** "a", "a and b", "a, b and c" */
function listWords(names: readonly string[]): string {
	return names.length <= 1 ? (names[0] ?? "") : `${names.slice(0, -1).join(", ")} ${t("checkin.and")} ${names[names.length - 1]}`;
}

export function dataUseAccepted(): boolean {
	return readSeen();
}
