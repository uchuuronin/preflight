/**
 * The fix loop on screen: Muse Code writes up to three fixes for one finding side by side, each
 * in its own lane, and each comes back as a change the visitor can read. The click on "Write the
 * fixes" is the approval: the cost cap is shown on the button's line, from what the API reserves.
 * Nothing here calls a fix verified: verify runs through A's job store once its runner applies a
 * patch, and until then every lane says the fix hasn't been run.
 */
import { useEffect, useState } from "react";
import { fmt } from "../lib/fmt";
import { hasKey, t } from "../lib/i18n";
import type { Finding } from "../workspace/record/types";
import { useWorkspace } from "../workspace/state";
import { saveLimits } from "../guide/limits";
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
					{c.final ? <p className="pf-fix__said">{t("fix.said", { text: c.final })}</p> : null}
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

// Source: derived, the API calls a race lost after 13 minutes (routes/patches.py RACE_LOST_S); a
// minute more and the tab stops asking.
const GIVE_UP_MS = 14 * 60 * 1000;

/** Races started in this tab, by record and finding: closing the panel and opening it again
 * picks the race up where it is, instead of losing it (and its cost) behind a new click. */
const started = new Map<string, { race: RaceView; at: number }>();

export function FixRace({ finding }: { finding: Finding }) {
	const { record, recordRef, provider, state } = useWorkspace();
	const slot = `${recordRef}|${finding.id}`;
	const [race, setRaceState] = useState<RaceView | null>(() => started.get(slot)?.race ?? null);
	const setRace = (next: RaceView | null | ((r: RaceView | null) => RaceView | null)) =>
		setRaceState((r) => {
			const v = typeof next === "function" ? next(r) : next;
			if (v) started.set(slot, { race: v, at: started.get(slot)?.at ?? Date.now() });
			return v;
		});
	const [error, setError] = useState<{ code: string; message: string } | null>(null);
	const [busy, setBusy] = useState(false);
	const [elapsed, setElapsed] = useState(0);
	const cands = candidatesFor(finding, { record, provider, knobs: state.knobs, pinned: state.pinned, objective: state.objective, limits: state.limits });
	const running = race?.status === "running";

	useEffect(() => {
		if (!race || race.status !== "running") return;
		const at = started.get(slot)?.at ?? Date.now();
		const tick = setInterval(() => setElapsed(Math.round((Date.now() - at) / 1000)), 1000);
		const poll = setInterval(async () => {
			if (Date.now() - at > GIVE_UP_MS) {
				setRace((r) => (r ? { ...r, status: "failed", error: "lost" } : r));
				return;
			}
			const next = await getRace(race.patch_id);
			if (next && next.status !== "running") setRace((r) => (r ? { ...r, ...next } : next));
		}, POLL_MS);
		return () => {
			clearInterval(tick);
			clearInterval(poll);
		};
	}, [race, slot]);

	if (!cands.length) return <p className="pf-dim">{t("fix.none")}</p>;

	const begin = async () => {
		setBusy(true);
		setError(null);
		// The server judges a fix by its own copy of the limits: make sure it has the latest.
		await saveLimits(recordRef, state.limits);
		const out = await startRace(recordRef, finding.id, cands);
		setBusy(false);
		if (!out.ok) {
			setError({ code: out.code, message: out.message });
			return;
		}
		started.set(slot, { race: out.race, at: Date.now() });
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
