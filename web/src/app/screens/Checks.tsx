/**
 * Checks: how B's engine holds up against real probe measurements it never saw (review item 2,
 * the coverage page). The numbers come from api/scripts/coverage.py over A's real runs and ship
 * with the build; the page shows labels, numbers, intervals and status words, with the method
 * behind an "i". A separate screen, never part of the cockpit.
 */
import data from "../../coverage/coverage.json";
import vsRaw from "../../coverage/vs_model.json";
import { type MetricSummary, verdict, type Verdict } from "../../coverage/verdict";
import { fmt, fmtRange } from "../../lib/fmt";
import { t } from "../../lib/i18n";
import { Info } from "../../workspace/components/Info";
import { FitMark } from "../../workspace/components/StatusMark";
import type { Fit } from "../../workspace/status";

/** B's comparison (vs_model.json, from docs/validation/preflight-vs-model.md): the same code asked to Muse Spark without probes. */
type Side = { step_s?: unknown; peak_gib?: unknown; p_oom?: number | null; why?: string; step_why?: string };
const vs = vsRaw as unknown as {
	summary: string;
	caveats: string[];
	rows: Array<{
		script: string;
		label: string;
		gpu: string;
		device: string;
		full_run: { ran_out_of_memory: boolean; reserved_when_it_ran_out_gib: number | null };
		preflight: Side;
		muse_spark: Side;
		ratio: { preflight_step: number | null; muse_step: number | null; preflight_peak: number | null; muse_peak: number | null };
	}>;
};

/** A prediction over what the full run did, "0.916×", "18.0×": three significant figures, so the table
 * shows what B's summary says (1.36×, and 1.10× within 10% beside 1.14× outside it). */
function ratio(x: number): string {
	return `${new Intl.NumberFormat("en-GB", { minimumSignificantDigits: 3, maximumSignificantDigits: 3 }).format(x)}×`;
}

/** Preflight's and Muse Spark's ratio for one quantity, or why there is none (B's words). */
function Pair({ pf, ms, pfWhy, msWhy, oom, reserved, peak }: { pf: number | null; ms: number | null; pfWhy?: string; msWhy?: string; oom: boolean; reserved?: number | null; peak?: boolean }) {
	// A run that ran out of memory never showed its full peak: nothing to compare against on either side.
	if (peak && oom)
		return <small className="pf-dim">{reserved ? t("checks.ask.no_peak_at", { gib: new Intl.NumberFormat("en-GB", { maximumFractionDigits: 1 }).format(reserved) }) : t("checks.ask.no_peak")}</small>;
	return (
		<span className="pf-checks__pair">
			<span>{t("checks.ask.preflight", { v: pf !== null ? ratio(pf) : (pfWhy ?? t("checks.ask.no_answer")) })}</span>
			<span>{t("checks.ask.muse", { v: ms !== null ? ratio(ms) : (msWhy ?? t("checks.ask.no_answer")) })}</span>
		</span>
	);
}

const METRICS = ["oom", "step_time", "peak_memory", "interval"] as const;
const TONE: Record<Verdict, Fit> = { meets: "fits", no_misses: "unknown", unsettled: "unknown", unchecked: "unknown", below: "fails", missed: "fails" };

interface Check {
	run: string;
	probe: string;
	gpu: string | null;
	knobs: Record<string, unknown>;
	metric: string;
	of?: string;
	measured: number | string;
	predicted: { status?: string; point?: number; lo?: number; hi?: number } | null;
	predicted_peak?: { point?: number } | null;
	error?: number;
	hit?: boolean;
	/** How the memory prediction was formed (coverage.py memory_target). */
	basis?: string;
	verdict?: string;
	/** sized_for_seq_len checks: the probe's longest padded row and its seq_len. */
	longest_row?: number;
	seq_len?: number;
}

const UNIT: Record<string, string> = { step_time: "s", "time.step": "s", peak_memory: "bytes", "memory.peak": "bytes" };

function settings(knobs: Record<string, unknown>): string {
	return Object.entries(knobs)
		.map(([k, v]) => `${t(`knob.${k}`)} ${String(v)}`)
		.join(" · ");
}

function checkFit(c: Check): Fit {
	if (c.metric === "oom") return c.verdict === "caught" || c.verdict === "warned" ? "fits" : c.verdict === "missed" ? "fails" : "unknown";
	if (c.hit === undefined) return "unknown";
	return c.hit ? "fits" : "fails";
}

function target(m: string, s: MetricSummary): string {
	const bar = `${fmt(s.rate_bar, "probability")}${s.rate_bar_source === "cited" ? "" : ` ${t(`checks.source.${s.rate_bar_source}`)}`}`;
	if (m === "oom") return t("checks.target.oom");
	if (s.tolerance) return t("checks.target.tolerance", { tol: fmt(s.tolerance, "probability"), bar });
	return t("checks.target.rate", { bar });
}

function skippedText(s: { run: string; probe?: string; metric?: string; why: string; codes?: string[] }): string {
	if (s.why === "nothing_measured") return t("checks.why.nothing_measured", { run: s.run, codes: (s.codes ?? []).map((c) => t(`error.${c}`)).join(", ") });
	return `${s.probe ?? ""} · ${t(`checks.metric.${s.metric ?? "peak_memory"}`)}: ${t(`checks.why.${s.why}`)}`;
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
						{t("checks.meta", { measured: fmt(data.runs_measured.length, "count"), runs: fmt(data.runs.length, "count"), engine: data.engine_commit ?? "?", date })}
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
								<span className="pf-dim" data-label={t("checks.col.target")}>
									{target(m, s)}
								</span>
								<span data-label={t("checks.col.result")}>
									{s.n ? t("checks.result", { hits: fmt(s.hits, "count"), n: fmt(s.n, "count") }) : t("checks.none")}
									{s.abstained ? <small className="pf-dim"> {t("checks.abstained", { n: fmt(s.abstained, "count") })}</small> : null}
									{s.sized_for_seq_len ? <small className="pf-dim"> {t("checks.sized_for_seq_len_n", { n: fmt(s.sized_for_seq_len, "count") })}</small> : null}
								</span>
								<span data-label={t("checks.col.interval")}>{s.wilson95 ? fmtRange(s.wilson95[0] ?? 0, s.wilson95[1] ?? 1, "probability") : t("value.none")}</span>
								<FitMark fit={TONE[v]} label={t(`checks.verdict.${v}`)} />
							</div>
						);
					})}
				</section>
				<section className="pf-checks__ask" aria-labelledby="pf-checks-ask">
					<h2 className="pf-h2" id="pf-checks-ask">
						{t("checks.ask.title")}
					</h2>
					<p>{vs.summary}</p>
					<p className="pf-dim pf-checks__note">{t("checks.ask.how")}</p>
					<div className="glass pf-checks__table" role="table" aria-labelledby="pf-checks-ask">
						<div className="pf-checks__row pf-checks__row--ask pf-checks__row--head" role="row">
							<span role="columnheader">{t("checks.ask.col.script")}</span>
							<span role="columnheader">{t("checks.ask.col.step")}</span>
							<span role="columnheader">{t("checks.ask.col.peak")}</span>
							<span role="columnheader">{t("checks.ask.col.oom")}</span>
						</div>
						{vs.rows.map((r) => (
							<div key={r.script} className="pf-checks__row pf-checks__row--ask" role="row">
								<span role="rowheader">
									<b>{r.label}</b>
									<small className="pf-dim">
										{" "}
										{r.device.replace(/^NVIDIA /, "") === r.gpu ? t("checks.ask.on", { gpu: r.gpu }) : t("checks.ask.on_other", { gpu: r.gpu, ran: r.device.replace(/^NVIDIA /, "") })}
									</small>
								</span>
								<span role="cell" data-label={t("checks.ask.col.step")}>
									<Pair pf={r.ratio.preflight_step} ms={r.ratio.muse_step} pfWhy={r.preflight.step_why ?? r.preflight.why} msWhy={r.muse_spark.step_why} oom={r.full_run.ran_out_of_memory} />
								</span>
								<span role="cell" data-label={t("checks.ask.col.peak")}>
									<Pair pf={r.ratio.preflight_peak} ms={r.ratio.muse_peak} pfWhy={r.preflight.why} oom={r.full_run.ran_out_of_memory} reserved={r.full_run.reserved_when_it_ran_out_gib} peak />
								</span>
								<span role="cell" data-label={t("checks.ask.col.oom")}>
									<span className="pf-checks__pair">
										<span>{t("checks.ask.preflight", { v: r.preflight.p_oom === null || r.preflight.p_oom === undefined ? t("checks.ask.no_answer") : fmt(r.preflight.p_oom, "probability") })}</span>
										<span>{t("checks.ask.muse", { v: fmt(r.muse_spark.p_oom, "probability") })}</span>
										<small className="pf-dim">{t(r.full_run.ran_out_of_memory ? "checks.ask.ran_out" : "checks.ask.did_not")}</small>
									</span>
								</span>
							</div>
						))}
					</div>
					<ul className="pf-dim pf-checks__caveats">
						{vs.caveats.map((c) => (
							<li key={c}>{c}</li>
						))}
					</ul>
				</section>
				<section className="pf-checks__list" aria-label={t("checks.each")}>
					<h2 className="pf-h2">{t("checks.each")}</h2>
					<ul className="pf-chips">
						{checks.map((c, i) => {
							const unit = UNIT[c.of ?? c.metric];
							const p = c.predicted;
							const said =
								c.verdict === "sized_for_seq_len"
									? t("checks.sized_for_seq_len", { longest: fmt(c.longest_row ?? 0, "count"), seq: fmt(c.seq_len ?? 0, "count") })
									: c.metric === "oom"
									? [t(`checks.oom.${c.verdict ?? "abstained"}`), c.predicted_peak?.point ? t("checks.oom.peak", { peak: fmt(c.predicted_peak.point, "bytes") }) : null].filter(Boolean).join(", ")
									: p?.point !== undefined && unit
										? t(c.basis === "peak_less_eval_and_context" ? "checks.predicted_no_eval" : "checks.predicted", {
												value: `${fmt(p.point, unit)} (${fmtRange(p.lo ?? p.point, p.hi ?? p.point, unit)})`,
											})
										: t("checks.abstain");
							const got = c.metric === "oom" ? t(String(c.measured) === "oom.eval" ? "checks.measured.oom_eval" : "checks.measured.oom") : typeof c.measured === "number" && unit ? fmt(c.measured, unit) : String(c.measured);
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
										{skippedText(s as never)}
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
