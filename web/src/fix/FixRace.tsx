/**
 * The fix loop on screen: Muse Code writes up to three fixes for one finding side by side, each
 * in its own lane, and each comes back as a change the visitor can read. The click on "Write the
 * fixes" is the approval: the cost cap is shown on the button's line, from what the API reserves.
 * Nothing here calls a fix verified: verify runs through A's job store once its runner applies a
 * patch, and until then every lane says the fix hasn't been run.
 */
import { useEffect, useRef, useState } from "react";
import { fmt } from "../lib/fmt";
import { hasKey, t } from "../lib/i18n";
import type { Finding } from "../workspace/record/types";
import { useWorkspace } from "../workspace/state";
import { candidatesFor, template, USD_PER_CANDIDATE } from "./candidates";
import { cancelRace, getRace, POLL_MS, type RaceCandidate, type RaceView, startRace } from "./race";

function Lane({ c, running, elapsed }: { c: { template_id: string; value?: unknown } & Partial<RaceCandidate>; running: boolean; elapsed: number }) {
	const title = template(c.template_id)?.title ?? c.template_id;
	const status = running ? "writing" : (c.status ?? "failed");
	return (
		<li className={`pf-fix__lane is-${status}`}>
			<div className="pf-fix__head">
				<b>{title}</b>
				{c.value !== undefined && c.value !== null ? <span className="pf-dim"> · {String(c.value)}</span> : null}
			</div>
			<div className="pf-fix__status" aria-live="polite">
				{running ? (
					<>
						<i className="pf-fix__bar" aria-hidden="true" />
						{t("fix.writing", { s: fmt(elapsed, "count") })}
					</>
				) : (
					<>
						{hasKey(`fix.status.${status}`) ? t(`fix.status.${status}`) : t("fix.status.failed")}
						{c.seconds ? <span className="pf-dim"> · {t("fix.took", { s: fmt(Math.round(c.seconds), "count") })}</span> : null}
					</>
				)}
			</div>
			{!running && c.status === "patched" ? (
				<>
					{c.files?.length ? <p className="pf-dim">{t("fix.files", { files: c.files.join(", ") })}</p> : null}
					{c.final ? <p className="pf-fix__said">{c.final}</p> : null}
					<details className="pf-fix__diff">
						<summary>{t("fix.diff")}</summary>
						<pre>{c.diff}</pre>
					</details>
					<p className="pf-fix__unverified">{t("fix.not_verified")}</p>
				</>
			) : null}
		</li>
	);
}

export function FixRace({ finding }: { finding: Finding }) {
	const { record, recordRef, provider, state } = useWorkspace();
	const [race, setRace] = useState<RaceView | null>(null);
	const [error, setError] = useState<{ code: string; message: string } | null>(null);
	const [busy, setBusy] = useState(false);
	const [elapsed, setElapsed] = useState(0);
	const started = useRef(0);
	const cands = candidatesFor(finding, { record, provider, knobs: state.knobs, pinned: state.pinned, objective: state.objective });
	const running = race?.status === "running";

	useEffect(() => {
		if (!race || race.status !== "running") return;
		const tick = setInterval(() => setElapsed(Math.round((Date.now() - started.current) / 1000)), 1000);
		const poll = setInterval(async () => {
			const next = await getRace(race.patch_id);
			if (next && next.status !== "running") setRace((r) => ({ ...r, ...next }));
		}, POLL_MS);
		return () => {
			clearInterval(tick);
			clearInterval(poll);
		};
	}, [race]);

	if (!cands.length) return <p className="pf-dim">{t("fix.none")}</p>;

	const begin = async () => {
		setBusy(true);
		setError(null);
		const out = await startRace(recordRef, finding.id, cands);
		setBusy(false);
		if (!out.ok) {
			setError({ code: out.code, message: out.message });
			return;
		}
		started.current = Date.now();
		setElapsed(0);
		setRace(out.race);
	};

	const lanes = race && race.status !== "running" && race.candidates.length ? race.candidates : (race?.candidates_asked ?? cands);
	return (
		<section className="pf-fix" aria-label={t("fix.title")}>
			{!race ? (
				<>
					<p>{cands.length > 1 ? t("fix.offer", { n: fmt(cands.length, "count") }) : t("fix.offer.one")}</p>
					<ul className="pf-fix__plan">
						{cands.map((c) => (
							<li key={c.template_id}>
								{template(c.template_id)?.title}
								{c.value !== undefined ? <span className="pf-dim"> · {String(c.value)}</span> : null}
							</li>
						))}
					</ul>
					<button type="button" className="pf-btn pf-btn--primary pf-btn--small" onClick={begin} disabled={busy}>
						{cands.length > 1 ? t("fix.start") : t("fix.start.one")}
					</button>
					<p className="pf-dim">
						{t("fix.by")} · {t("fix.cost", { usd: fmt(Math.round(USD_PER_CANDIDATE * cands.length * 100) / 100, "usd") })}
					</p>
					{error ? (
						<p className="pf-fix__error" role="alert">
							{hasKey(`fix.error.${error.code}`) ? t(`fix.error.${error.code}`) : t("fix.error", { message: error.message || error.code })}
						</p>
					) : null}
				</>
			) : (
				<>
					<p className="pf-dim">
						{t("fix.by")}
						{race.cap_usd ? ` · ${t("fix.cap", { usd: fmt(race.cap_usd, "usd") })}` : ""}
					</p>
					<ol className="pf-fix__lanes">
						{lanes.map((c, i) => (
							<Lane key={`${c.template_id}-${i}`} c={c as RaceCandidate} running={running} elapsed={elapsed} />
						))}
					</ol>
					{running ? (
						<button
							type="button"
							className="pf-btn pf-btn--small"
							onClick={async () => {
								const next = await cancelRace(race.patch_id);
								if (next) setRace((r) => ({ ...r, ...next }));
							}}
						>
							{t("fix.cancel")}
						</button>
					) : (
						<p className="pf-dim">
							{race.error ? (hasKey(`fix.error.${race.error}`) ? t(`fix.error.${race.error}`) : t("fix.error", { message: race.error })) : null}
							{typeof race.spent_usd === "number" ? ` ${t("fix.spent", { usd: fmt(race.spent_usd, "usd") })}` : ""}
						</p>
					)}
				</>
			)}
		</section>
	);
}
