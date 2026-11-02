/**
 * "Watch the real run" in "Take it with you" (behind `c.inflight`): the record as a file, for
 * `preflight watch` on the visitor's machine, and the command. The same markup as the other rows.
 */
import { useState } from "react";
import { t } from "../../lib/i18n";
import type { FlightRecord } from "../record/types";

/** The file name the command below uses; the share token stays out of file names. */
export const WATCH_FILE = "preflight-record.json";

/** One shell word, quoted only when it needs to be (POSIX single quotes). */
function word(s: string): string {
	return /^[\w./=:,+-]+$/.test(s) ? s : `'${s.replace(/'/g, `'\\''`)}'`;
}

/** The command, with the arguments the record was predicted for, so the collector doesn't warn
 * that its bands may not apply (inflight.py compares them). */
export function watchCommand(entry: string | null | undefined, args: readonly string[] = []): string {
	const script = entry && /^[\w.][\w./-]*$/.test(entry) ? entry : "train.py";
	const tail = args.length ? ` -- ${args.map(word).join(" ")}` : "";
	return `preflight watch ${WATCH_FILE} ${script}${tail}`;
}

export function WatchExport({ record }: { record: FlightRecord }) {
	const [said, setSaid] = useState("");
	const name = t("exports.watch.title");
	const command = watchCommand(record.job.source?.entrypoint ?? record.job.script_map?.entrypoint, record.job.source?.args ?? []);
	const say = (m: string) => {
		setSaid("");
		setTimeout(() => setSaid(m), 60);
	};
	const download = () => {
		if (typeof URL.createObjectURL !== "function") return say(t("exports.wandb.download_failed"));
		const url = URL.createObjectURL(new Blob([JSON.stringify(record, null, 1)], { type: "application/json" }));
		const a = document.createElement("a");
		a.href = url;
		a.download = WATCH_FILE;
		document.body.append(a);
		a.click();
		a.remove();
		setTimeout(() => URL.revokeObjectURL(url), 1000);
	};
	const copy = async () => {
		try {
			await navigator.clipboard.writeText(command);
			say(t("exports.wandb.copied"));
		} catch {
			say(t("exports.wandb.copy_failed"));
		}
	};
	return (
		<>
			<ul className="pf-addon-exports__items pf-exports__watch">
				<li className="pf-addon-exports__item">
					<details>
						<summary>
							<span className="pf-addon-exports__name">{name}</span>
							<span className="pf-addon-exports__say">{t("exports.watch.body")}</span>
						</summary>
						<pre className="pf-addon-exports__text" tabIndex={0} aria-label={name}>
							{command}
						</pre>
					</details>
					<div className="pf-addon-exports__actions">
						<button type="button" className="pf-btn pf-btn--small" onClick={() => void copy()} aria-label={t("exports.watch.copy")}>
							{t("exports.wandb.copy")}
						</button>
						<button type="button" className="pf-btn pf-btn--small pf-btn--quiet" onClick={download} aria-label={t("exports.watch.download", { file: WATCH_FILE })}>
							{t("exports.wandb.download_short")}
						</button>
					</div>
				</li>
			</ul>
			<p className="pf-visually-hidden" role="status" aria-live="polite">
				{said}
			</p>
		</>
	);
}
