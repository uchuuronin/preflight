/**
 * "Watch a real run" (behind `c.inflight`): how to start `preflight watch` on your machine, and
 * the button that opens the live record it writes. On the logbook, beside the flights you opened.
 */
import { useId, useState } from "react";
import { go } from "../../app/route";
import { flag } from "../../lib/flags";
import { t } from "../../lib/i18n";
import { LIVE_REF, openLiveFile, pickAndFollow } from "./liveFile";

export function WatchRun() {
	const [error, setError] = useState<string | null>(null);
	const fileId = useId();
	if (!flag("c.inflight")) return null;
	const opened = (ok: boolean) => {
		if (ok) go("cockpit", LIVE_REF);
	};
	const choose = async () => {
		setError(null);
		const r = await pickAndFollow();
		if (r === null) document.getElementById(fileId)?.click(); // no picker with a handle: a plain file input
		else opened(r);
	};
	return (
		<section className="glass pf-watch" aria-labelledby={`${fileId}-title`}>
			<h2 id={`${fileId}-title`} className="pf-h2">
				{t("live.screen.title")}
			</h2>
			<p className="pf-hint">{t("live.screen.lede")}</p>
			<ol className="pf-watch__steps">
				<li>{t("live.screen.step1")}</li>
				<li>
					{t("live.screen.step2")} <code>preflight watch preflight-record.json train.py</code>
				</li>
				<li>{t("live.screen.step3")}</li>
			</ol>
			<button type="button" className="pf-btn pf-btn--primary pf-btn--small" onClick={() => void choose()}>
				{t("live.choose")}
			</button>
			<input
				id={fileId}
				type="file"
				accept=".json,application/json"
				hidden
				onChange={async (e) => {
					const input = e.currentTarget;
					const file = input.files?.[0];
					input.value = "";
					if (!file) return;
					const ok = await openLiveFile(file);
					if (!ok) setError(t("live.error.not_live"));
					opened(ok);
				}}
			/>
			{/* Always mounted, so the message is read out when it appears. */}
			<div role="status" aria-live="polite">
				{error ? <p className="pf-error">{error}</p> : null}
			</div>
		</section>
	);
}
