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

/**
 * How long records and code are kept. null until Chiya decides and deletion is built; the
 * notice then says plainly that nothing is deleted automatically yet, instead of a promise
 * the system doesn't keep.
 */
export const RETENTION: { codeHours: number; recordDays: number } | null = null;

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
export function dataUseLines(): string[] {
	const lines = [t("datause.sent"), t("datause.probe")];
	if (flag("harness.spark_script_map")) lines.push(t("datause.script_map"));
	if (flag("c.guide.live")) lines.push(t("datause.guide"));
	lines.push(t("datause.record"));
	lines.push(
		RETENTION
			? t("datause.retention", { hours: String(RETENTION.codeHours), days: String(RETENTION.recordDays) })
			: t("datause.retention.unset"),
	);
	lines.push(t("datause.secrets"));
	return lines;
}

export function DataUseNotice({ onChange }: { onChange: (accepted: boolean) => void }) {
	const id = useId();
	const [accepted, setAccepted] = useState(readSeen);
	return (
		<section className="glass pf-datause" aria-labelledby={`${id}-title`}>
			<b id={`${id}-title`} className="pf-datause__title">
				{t("datause.title")}
			</b>
			<ul className="pf-datause__list">
				{dataUseLines().map((line) => (
					<li key={line}>{line}</li>
				))}
			</ul>
			<label className="pf-datause__ok">
				<input
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

export function dataUseAccepted(): boolean {
	return readSeen();
}
