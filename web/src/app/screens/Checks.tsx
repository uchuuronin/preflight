/**
 * Checks: how B's engine holds up against real probe measurements it never saw (review item 2,
 * the coverage page). The numbers come from api/scripts/coverage.py over A's real runs and ship
 * with the build; the page shows labels, numbers, intervals and status words, with the method
 * behind an "i". A separate screen, never part of the cockpit.
 */
import data from "../../coverage/coverage.json";
import { type MetricSummary, verdict, type Verdict } from "../../coverage/verdict";
import { fmt, fmtRange } from "../../lib/fmt";
import { t } from "../../lib/i18n";
import { Info } from "../../workspace/components/Info";
import { FitMark } from "../../workspace/components/StatusMark";
import type { Fit } from "../../workspace/status";

const METRICS = ["oom", "step_time", "peak_memory", "interval"] as const;
const TONE: Record<Verdict, Fit> = { meets: "fits", unsettled: "unknown", unchecked: "unknown", below: "fails", missed: "fails" };

interface Check {
	run: string;
	probe: string;
	gpu: string | null;
	knobs: Record<string, unknown>;
	metric: string;
	of?: string;
	measured: number | string;
	predicted: { status?: string; point?: number; lo?: number; hi?: number } | null;
	error?: number;
	hit?: boolean;
	verdict?: string;
}

const UNIT: Record<string, string> = { step_time: "s", "time.step": "s", peak_memory: "bytes", "memory.peak": "bytes" };

function settings(knobs: Record<string, unknown>): string {
	return Object.entries(knobs)
		.map(([k, v]) => `${t(`knob.${k}`)} ${String(v)}`)
		.join(" · ");
}

function checkFit(c: Check): Fit {
	if (c.metric === "oom") return c.verdict === "caught" ? "fits" : c.verdict === "missed" ? "fails" : "unknown";
	if (c.hit === undefined) return "unknown";
	return c.hit ? "fits" : "fails";
}

export function Checks() {
	const summary = data.summary as Record<string, MetricSummary>;
	const checks = data.checks as Check[];
	const date = new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short" }).format(new Date(data.generated_at));
	return (
		<div className="pf-scroll">
			<div className="pf-inner pf-checks">
				<header className="pf-checks__head">
					<span className="pf-eyebrow">{t("checks.eyebrow")}</span>
					<h1 className="pf-title">
						{t("checks.title")}{" "}
						<Info label={t("checks.title")} align="start">
							{t("checks.method")}
						</Info>
					</h1>
					<p className="pf-dim">
						{t("checks.meta", { runs: fmt(data.runs.length, "count"), engine: data.engine_commit ?? "?", date })}
					</p>
				</header>
				<section className="glass pf-checks__table" aria-label={t("checks.summary")}>
					<div className="pf-checks__row pf-checks__row--head" aria-hidden="true">
						<span>{t("checks.col.metric")}</span>
						<span>{t("checks.col.target")}</span>
						<span>{t("checks.col.result")}</span>
						<span>{t("checks.col.interval")}</span>
						<span>{t("checks.col.status")}</span>
					</div>
					{METRICS.map((m) => {
						const s = summary[m];
						if (!s) return null;
						const v = verdict(m, s);
						return (
							<div key={m} className="pf-checks__row">
								<b>{t(`checks.metric.${m}`)}</b>
								<span className="pf-dim">{s.target}</span>
								<span>
									{s.n ? t("checks.result", { hits: fmt(s.hits, "count"), n: fmt(s.n, "count") }) : t("checks.none")}
									{s.abstained ? <small className="pf-dim"> {t("checks.abstained", { n: fmt(s.abstained, "count") })}</small> : null}
								</span>
								<span>{s.wilson95 ? fmtRange(s.wilson95[0] ?? 0, s.wilson95[1] ?? 1, "probability") : t("value.none")}</span>
								<FitMark fit={TONE[v]} label={t(`checks.verdict.${v}`)} />
							</div>
						);
					})}
				</section>
				<section className="pf-checks__list" aria-label={t("checks.each")}>
					<h2 className="pf-h2">{t("checks.each")}</h2>
					<ul className="pf-chips">
						{checks.map((c, i) => {
							const unit = UNIT[c.of ?? c.metric];
							const p = c.predicted;
							const said =
								c.metric === "oom"
									? t(`checks.oom.${c.verdict ?? "abstained"}`)
									: p?.point !== undefined && unit
										? `${fmt(p.point, unit)} (${fmtRange(p.lo ?? p.point, p.hi ?? p.point, unit)})`
										: t("checks.abstain");
							const got = typeof c.measured === "number" && unit ? fmt(c.measured, unit) : String(c.measured);
							return (
								<li key={`${i}:${c.run}:${c.probe}:${c.metric}`} className={`pf-chip pf-chip--${checkFit(c)}`} title={c.run}>
									<FitMark fit={checkFit(c)} label={t(`checks.metric.${c.of ? "interval" : c.metric}`)} />
									<span className="pf-chip__where">
										{c.probe} · {c.gpu ?? "?"} · {settings(c.knobs)}
									</span>
									<span className="pf-chip__nums">{t("checks.pair", { got, said })}</span>
								</li>
							);
						})}
					</ul>
					{data.skipped.length ? (
						<details className="pf-checks__skipped">
							<summary>{t("checks.skipped", { n: fmt(data.skipped.length, "count") })}</summary>
							<ul>
								{data.skipped.map((s, i) => (
									<li key={i} className="pf-dim">
										{"probe" in s ? `${s.probe} · ${t(`checks.metric.${(s as { metric?: string }).metric ?? "peak_memory"}`)}: ` : ""}
										{s.why}
									</li>
								))}
							</ul>
						</details>
					) : null}
				</section>
			</div>
		</div>
	);
}
