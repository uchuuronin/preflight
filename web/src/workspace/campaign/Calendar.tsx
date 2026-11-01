/**
 * Campaign calendar with team lanes, drawn from B's campaign plan (campaign/plan.ts). One row per
 * lane with the runs B put there; runs B cut for the budget or couldn't date sit below, each with
 * B's reason. When B's plan carries each run's start, the lanes are drawn against time (bar to the
 * run's point, whisker to its upper end); until then, each lane lists its runs in B's order with
 * their durations. The totals and the finish are B's estimates with their intervals, or B's reason
 * for not giving one. Behind `c.campaign_calendar`.
 */
import { fmt, fmtRange } from "../../lib/fmt";
import { hasKey, t } from "../../lib/i18n";
import { Info } from "../components/Info";
import type { Estimate } from "../record/types";
import { trusted } from "../status";
import { type CampaignPlan, finishBand, hasSchedule, lanesOf, type PlanCell, slotsOf, startOf, unitOf } from "./plan";

function said(e: Estimate | null | undefined): string {
	if (!e || !trusted(e)) return e?.reason ? t("campaign.abstain", { why: e.reason }) : t("value.none");
	const unit = unitOf(e);
	return `${fmt(e.point, unit)} (${fmtRange(e.lo ?? e.point, e.hi ?? e.point, unit)})`;
}

function cellLabel(c: PlanCell): string {
	return Object.entries(c.knobs)
		.map(([k, v]) => `${hasKey(`knob.${k}`) ? t(`knob.${k}`) : k} ${String(v)}`)
		.join(" · ");
}

function when(iso: string | undefined): string {
	if (!iso) return "";
	return new Intl.DateTimeFormat("en-GB", { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit", timeZone: "UTC" }).format(new Date(iso));
}

/** The finish band needs all three dates, each a real date: a record with only some says the finish in seconds instead. */
function band(q: Record<string, string>): boolean {
	return ["p10", "p50", "p90"].every((k) => typeof q[k] === "string" && Number.isFinite(Date.parse(q[k]!)));
}

function Duration({ c }: { c: PlanCell }) {
	// The record's own reason when it gives one; the generic line only when it doesn't.
	if (!trusted(c.time)) return <span className="pf-cal__abstain">{c.time.reason ? t("campaign.abstain", { why: c.time.reason }) : t("campaign.no_time")}</span>;
	return (
		<span>
			{fmt(c.time.point, "s_short")} <small className="pf-dim">({fmtRange(c.time.lo ?? c.time.point, c.time.hi ?? c.time.point, "s_short")})</small>
			{c.segments && c.segments > 1 ? <small className="pf-dim"> · {t("campaign.segments", { n: fmt(c.segments, "count") })}</small> : null}
		</span>
	);
}

export function CampaignCalendar({ plan }: { plan: CampaignPlan }) {
	const { rows, unplaced } = lanesOf(plan);
	const timed = hasSchedule(plan);
	const fin = timed && band(plan.completion) ? finishBand(plan) : null;
	const span = timed ? Math.max(1, fin?.p90 ?? 0, ...plan.cells.filter((c) => c.lane && !c.cut && trusted(c.time)).map((c) => (startOf(c) ?? 0) + (c.time.hi ?? c.time.point ?? 0))) : 1;
	const pct = (s: number) => `${Math.min(100, (s / span) * 100).toFixed(2)}%`;
	const q = plan.completion;
	return (
		<section id="pf-campaign" className="glass pf-band pf-cal" aria-label={t("campaign.title")}>
			<div className="pf-band__head">
				<b>{t("campaign.title")}</b>
				{plan.card.notes ? (
					<Info label={t("campaign.title")} align="end">
						{plan.card.notes}
					</Info>
				) : null}
			</div>
			{/* The guide's step speaks here while its tour is on the calendar (GuidePanel portals into it). */}
			<div id="pf-campaign-guide" />
			<dl className="pf-cal__totals">
				<div>
					<dt>{t("campaign.cost")}</dt>
					<dd>{said(plan.usd)}</dd>
				</div>
				<div>
					<dt>{t("campaign.gpu_hours")}</dt>
					<dd>{said(plan.gpu_hours)}</dd>
				</div>
				<div>
					<dt>{t("campaign.finish")}</dt>
					<dd>{band(q) ? t("campaign.finish_band", { p10: when(q.p10), p50: when(q.p50), p90: when(q.p90) }) : said(plan.finish_s)}</dd>
				</div>
			</dl>
			<ol className="pf-cal__lanes">
				{rows.map((r) => (
					<li key={r.lane} className="pf-cal__lane">
						<span className="pf-cal__who">{r.lane}</span>
						{r.cells.length === 0 ? (
							<span className="pf-dim">{t("campaign.lane_empty")}</span>
						) : timed ? (
							<div className="pf-cal__slots">
								{/* One track per slot the lane holds, so runs going at once sit on their own rows. */}
								{slotsOf(r.cells).map((slot, i) => (
									<div key={i} className="pf-cal__track" role="list">
										{slot.map((c) => (
											<div key={c.id} role="listitem" className="pf-cal__bar" style={{ left: pct(startOf(c) ?? 0), width: pct(c.time.point ?? 0) }} title={`${cellLabel(c)}: ${said(c.time)}`}>
												<i className="pf-cal__whisker" style={{ width: `${(((c.time.hi ?? 0) - (c.time.point ?? 0)) / Math.max(1, c.time.point ?? 1)) * 100}%` }} />
												<span>{cellLabel(c)}</span>
											</div>
										))}
									</div>
								))}
							</div>
						) : (
							<ul className="pf-cal__runs">
								{r.cells.map((c) => (
									<li key={c.id} className={trusted(c.time) ? undefined : "is-abstain"}>
										<span className="pf-cal__run">{cellLabel(c)}</span>
										<Duration c={c} />
									</li>
								))}
							</ul>
						)}
					</li>
				))}
				{fin ? (
					<li className="pf-cal__lane pf-cal__finish">
						<span className="pf-cal__who">{t("campaign.finish_row")}</span>
						<div className="pf-cal__track" role="img" aria-label={t("campaign.finish_band", { p10: when(q.p10), p50: when(q.p50), p90: when(q.p90) })}>
							{/* The middle 80% of simulated finishes, with the most likely one marked. */}
							<span className="pf-cal__band" style={{ left: pct(fin.p10), width: pct(fin.p90 - fin.p10) }} />
							<i className="pf-cal__p50" style={{ left: pct(fin.p50) }} />
						</div>
					</li>
				) : null}
			</ol>
			{unplaced.length ? (
				<div className="pf-cal__unplaced">
					<b>{t("campaign.unplaced", { n: fmt(unplaced.length, "count") })}</b>
					<ul className="pf-cal__runs">
						{unplaced.map((c) => (
							<li key={c.id} className={c.cut ? "is-cut" : "is-abstain"}>
								<span className="pf-cal__run">{cellLabel(c)}</span>
								<span className="pf-dim">{c.cut ? c.reason || t("campaign.cut") : c.time.reason || t("campaign.no_time")}</span>
							</li>
						))}
					</ul>
				</div>
			) : null}
			{plan.notes.length ? (
				<ul className="pf-cal__notes">
					{plan.notes.map((n) => (
						<li key={n}>{n}</li>
					))}
				</ul>
			) : null}
		</section>
	);
}
