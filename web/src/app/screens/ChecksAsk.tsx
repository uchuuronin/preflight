/**
 * Checks: the same code, asked to Muse Spark without probes (B's vs_model.json, from
 * docs/validation/preflight-vs-model.md). Read first as B's three tiles, then as two miss charts
 * (step time, peak memory) that say each miss in words, Preflight a circle and Muse Spark a diamond
 * so colour is never the only cue. The numbers, the out-of-memory calls and B's caveats follow, folded.
 */
import vsRaw from "../../coverage/vs_model.json";
import { fmt } from "../../lib/fmt";
import { t } from "../../lib/i18n";

type Est = { point: number; lo?: number; hi?: number; status?: string } | null | undefined;
type Side = { step_s?: Est; peak_gib?: Est; p_oom?: number | null; why?: string; step_why?: string };
type Row = {
	script: string;
	label: string;
	gpu: string;
	device: string;
	full_run: { step_s: number | null; peak_gib: number | null; ran_out_of_memory: boolean; reserved_when_it_ran_out_gib: number | null };
	preflight: Side;
	muse_spark: Side;
	ratio: { preflight_step: number | null; muse_step: number | null; preflight_peak: number | null; muse_peak: number | null };
};
const vs = vsRaw as unknown as { summary: string; caveats: string[]; rows: Row[] };

const nf = (opts: Intl.NumberFormatOptions) => new Intl.NumberFormat("en-GB", opts);
const sig3 = nf({ minimumSignificantDigits: 3, maximumSignificantDigits: 3 });
const one = nf({ maximumFractionDigits: 1 });

/** "0.916×", "18.0×": three significant figures, so the table says what B's summary says. */
export function ratio(x: number): string {
	return `${sig3.format(x)}×`;
}

// ---------------------------------------------------------------- the miss chart

/**
 * How far off each guess was, read at a glance (Chiya, Nov 3: B's dot plot read as a log axis). The
 * centre line is what the full run did. Each guess is a bar out from it: right for too slow (or too
 * high), left for too fast (or too low), its length the size of the miss on a scale that reads the
 * same both ways (2x too slow is as far right as half the time is left; see `reach`). The miss is said in
 * words at the end of every lane, so nothing has to be read off an axis; the shaded strip is the
 * target. Preflight's bar ends in a circle, Muse Spark's in a diamond, so colour is never the only cue.
 */
type Metric = "step" | "peak";
const TARGET: Record<Metric, number> = { step: 0.15, peak: 0.1 };
/** The scale's edge, either way: Muse Spark's 18x step miss sits inside it. */
const EDGE = 24;
const GUIDES = [2, 10];
/**
 * Half the track, as a fraction: 0 at the full run, 1 at EDGE times off either way. The square root of
 * the log stretches the small misses so a 20% one is visible beside an 18x one; the 2x and 10x guides
 * mark the scale, and the words carry the number.
 */
const reach = (x: number) => {
	const r = Math.min(1, Math.abs(Math.log(x)) / Math.log(EDGE));
	return Math.sign(Math.log(x)) * Math.sqrt(r);
};
const at = (x: number) => 50 + 50 * reach(x);

function estOf(side: Side, m: Metric): Est {
	return m === "step" ? side.step_s : side.peak_gib;
}
function actualOf(r: Row, m: Metric): number | null {
	return m === "step" ? r.full_run.step_s : r.full_run.peak_gib;
}
function ratioOf(r: Row, who: "preflight" | "muse", m: Metric): number | null {
	return r.ratio[`${who}_${m}` as keyof Row["ratio"]];
}

/** Why a row has nothing to plot for this side, in B's words. */
function whyNot(r: Row, who: "preflight" | "muse", m: Metric): string {
	if (m === "peak" && r.full_run.ran_out_of_memory) {
		const gib = r.full_run.reserved_when_it_ran_out_gib;
		return gib ? t("checks.ask.no_peak_at", { gib: one.format(gib) }) : t("checks.ask.no_peak");
	}
	const side = who === "preflight" ? r.preflight : r.muse_spark;
	return (m === "step" ? (side.step_why ?? side.why) : side.why) ?? t("checks.ask.no_answer");
}

/** The miss in words: "8% too fast", "18× too slow", "within 10%" for a peak inside the target. */
export function missWords(x: number, m: Metric): string {
	const high = x >= 1;
	const dir = m === "step" ? (high ? "slow" : "fast") : high ? "high" : "low";
	if (x >= 2 || x <= 0.5) return t(`checks.ask.miss.times_${dir}`, { v: one.format(high ? x : 1 / x) });
	return t(`checks.ask.miss.pct_${dir}`, { v: Math.round(Math.abs(x - 1) * 100) });
}

function Lane({ r, who, m }: { r: Row; who: "preflight" | "muse"; m: Metric }) {
	const x = ratioOf(r, who, m);
	const name = t(who === "preflight" ? "checks.ask.side.preflight" : "checks.ask.side.muse");
	if (x === null) {
		return (
			<li className={`pf-miss__lane pf-miss__lane--${who} is-empty`}>
				<span className={`pf-ask__key pf-ask__key--${who}`} aria-hidden="true" />
				<span className="pf-miss__why">
					<span className="pf-visually-hidden">{name}: </span>
					{whyNot(r, who, m)}
				</span>
			</li>
		);
	}
	const e = estOf(who === "preflight" ? r.preflight : r.muse_spark, m);
	const act = actualOf(r, m);
	const unit = m === "step" ? "s" : "GiB";
	const num = (v: number) => (m === "step" ? sig3.format(v) : one.format(v));
	const detail = t("checks.ask.tip", {
		who: name,
		pred: `${num(e?.point ?? 0)} ${unit}`,
		range: e?.lo !== undefined && e?.hi !== undefined ? t("checks.ask.tip_range", { lo: num(e.lo), hi: num(e.hi), unit }) : "",
		actual: act ? `${num(act)} ${unit}` : "",
		off: missWords(x, m),
	});
	const right = x >= 1;
	const left = Math.min(50, at(x));
	const width = Math.max(0.8, Math.abs(at(x) - 50));
	const onTarget = Math.abs(x - 1) <= TARGET[m];
	return (
		<li className={`pf-miss__lane pf-miss__lane--${who}${onTarget ? " is-on" : ""}`} title={detail}>
			<span className={`pf-ask__key pf-ask__key--${who}`} aria-hidden="true" />
			<span className="pf-miss__track" aria-hidden="true">
				<span className={`pf-miss__bar${right ? " is-right" : " is-left"}`} style={{ left: `${left}%`, width: `${width}%` }}>
					<span className="pf-miss__end" />
				</span>
			</span>
			<span className="pf-miss__words">
				<span className="pf-visually-hidden">{name}: </span>
				{missWords(x, m)}
			</span>
		</li>
	);
}

function MissChart({ m }: { m: Metric }) {
	const band = TARGET[m];
	const rows = vs.rows.filter((r) => !(m === "peak" && r.full_run.ran_out_of_memory));
	const ranOut = m === "peak" ? vs.rows.filter((r) => r.full_run.ran_out_of_memory) : [];
	const lo = m === "step" ? "checks.ask.dir.fast" : "checks.ask.dir.low";
	const hi = m === "step" ? "checks.ask.dir.slow" : "checks.ask.dir.high";
	return (
		<figure className="glass pf-miss">
			<figcaption className="pf-miss__cap">
				<b>{t(m === "step" ? "checks.ask.miss.step" : "checks.ask.miss.peak")}</b>
				<small>{t("checks.ask.band", { pct: Math.round(band * 100) })}</small>
			</figcaption>
			<div className="pf-miss__scale" aria-hidden="true">
				<span className="pf-miss__key" />
				<span className="pf-miss__track pf-miss__track--scale">
					<span className="pf-miss__dir is-left">{t(lo)}</span>
					<span className="pf-miss__dir is-mid">{t("checks.ask.dir.real")}</span>
					<span className="pf-miss__dir is-right">{t(hi)}</span>
				</span>
				<span className="pf-miss__words" />
			</div>
			<ul className="pf-miss__rows">
				{rows.map((r) => (
					<li key={r.script} className="pf-miss__row">
						<span className="pf-miss__name">
							{r.label}
							{r.device.replace(/^NVIDIA /, "") !== r.gpu ? <small> · {t("checks.ask.ran_on", { ran: r.device.replace(/^NVIDIA /, "") })}</small> : null}
						</span>
						<ul className="pf-miss__lanes" style={{ ["--band-l" as string]: `${at(1 - band)}%`, ["--band-w" as string]: `${at(1 + band) - at(1 - band)}%` }}>
							<Lane r={r} who="preflight" m={m} />
							<Lane r={r} who="muse" m={m} />
						</ul>
					</li>
				))}
			</ul>
			<div className="pf-miss__guides" aria-hidden="true">
				<span className="pf-miss__key" />
				<span className="pf-miss__track pf-miss__track--scale">
					{GUIDES.flatMap((g) => [
						<span key={`l${g}`} style={{ left: `${at(1 / g)}%` }}>{`${g}×`}</span>,
						<span key={`r${g}`} style={{ left: `${at(g)}%` }}>{`${g}×`}</span>,
					])}
				</span>
				<span className="pf-miss__words" />
			</div>
			{ranOut.length ? (
				<p className="pf-miss__note">{ranOut.map((r) => t("checks.ask.gap_both", { script: r.label, why: whyNot(r, "preflight", "peak") })).join(". ")}.</p>
			) : null}
		</figure>
	);
}

// ---------------------------------------------------------------- tiles

function stepRange(who: "preflight" | "muse"): string {
	const both = vs.rows.filter((r) => r.ratio.preflight_step !== null && r.ratio.muse_step !== null);
	const xs = both.map((r) => ratioOf(r, who, "step") as number);
	if (!xs.length) return t("value.none");
	if (who === "muse") return t("checks.ask.tile.times", { a: one.format(Math.min(...xs)), b: Math.round(Math.max(...xs)) });
	const p = xs.map((x) => Math.round(Math.abs(x - 1) * 100));
	return t("checks.ask.tile.pct", { a: Math.min(...p), b: Math.max(...p) });
}

function within(who: "preflight" | "muse"): string {
	const both = vs.rows.filter((r) => r.ratio.preflight_peak !== null && r.ratio.muse_peak !== null);
	const k = both.filter((r) => Math.abs((ratioOf(r, who, "peak") as number) - 1) <= TARGET.peak).length;
	return t("checks.ask.tile.of", { k, n: both.length });
}

function oomCalls(who: "preflight" | "muse"): string {
	const ran = vs.rows.filter((r) => r.full_run.ran_out_of_memory);
	const side = (r: Row) => (who === "preflight" ? r.preflight : r.muse_spark).p_oom;
	const caught = ran.filter((r) => (side(r) ?? -1) >= 0.8).length;
	const none = ran.filter((r) => side(r) === null || side(r) === undefined).length;
	const missed = ran.length - caught - none;
	return [caught && t("checks.ask.tile.caught", { n: caught }), missed && t("checks.ask.tile.missed", { n: missed }), none && t("checks.ask.tile.no_answer", { n: none })]
		.filter(Boolean)
		.join(", ");
}

function Tile({ title, pf, ms }: { title: string; pf: string; ms: string }) {
	return (
		<div className="glass pf-ask__tile">
			<span className="pf-ask__tile-title">{title}</span>
			<span className="pf-ask__tile-row">
				<span className="pf-ask__key pf-ask__key--preflight" aria-hidden="true" />
				<span className="pf-ask__tile-who">{t("checks.ask.side.preflight")}</span>
				<b>{pf}</b>
			</span>
			<span className="pf-ask__tile-row">
				<span className="pf-ask__key pf-ask__key--muse" aria-hidden="true" />
				<span className="pf-ask__tile-who">{t("checks.ask.side.muse")}</span>
				<b>{ms}</b>
			</span>
		</div>
	);
}

// ---------------------------------------------------------------- the section

export function ChecksAsk() {
	const ran = vs.rows.filter((r) => r.full_run.ran_out_of_memory);
	return (
		<section className="pf-checks__ask" aria-labelledby="pf-checks-ask">
			<h2 className="pf-h2" id="pf-checks-ask">
				{t("checks.ask.title")}
			</h2>
			<p className="pf-ask__lede">{t("checks.ask.lede")}</p>
			<div className="pf-ask__tiles">
				<Tile title={t("checks.ask.tile.step")} pf={stepRange("preflight")} ms={stepRange("muse")} />
				<Tile title={t("checks.ask.tile.peak")} pf={within("preflight")} ms={within("muse")} />
				<Tile title={t("checks.ask.tile.oom", { n: ran.length })} pf={oomCalls("preflight")} ms={oomCalls("muse")} />
			</div>
			<div className="pf-ask__legend">
				<span>
					<span className="pf-ask__key pf-ask__key--preflight" aria-hidden="true" /> {t("checks.ask.legend.preflight")}
				</span>
				<span>
					<span className="pf-ask__key pf-ask__key--muse" aria-hidden="true" /> {t("checks.ask.legend.muse")}
				</span>
				<small>{t("checks.ask.how")}</small>
			</div>
			<div className="pf-ask__plots">
				<MissChart m="step" />
				<MissChart m="peak" />
			</div>
			<p className="pf-dim pf-ask__summary">{vs.summary}</p>
			<details className="pf-ask__more">
				<summary>{t("checks.ask.numbers")}</summary>
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
							{(["step", "peak"] as const).map((m) => (
								<span key={m} role="cell" data-label={t(m === "step" ? "checks.ask.col.step" : "checks.ask.col.peak")}>
									{m === "peak" && r.full_run.ran_out_of_memory ? (
										<small className="pf-dim">{whyNot(r, "preflight", "peak")}</small>
									) : (
										<span className="pf-checks__pair">
											{(["preflight", "muse"] as const).map((w) => {
												const x = ratioOf(r, w, m);
												return <span key={w}>{t(w === "preflight" ? "checks.ask.preflight" : "checks.ask.muse", { v: x !== null ? ratio(x) : whyNot(r, w, m) })}</span>;
											})}
										</span>
									)}
								</span>
							))}
							<span role="cell" data-label={t("checks.ask.col.oom")}>
								<span className="pf-checks__pair">
									<span>{t("checks.ask.preflight", { v: r.preflight.p_oom === null || r.preflight.p_oom === undefined ? t("checks.ask.no_answer") : fmt(r.preflight.p_oom, "probability") })}</span>
									<span>{t("checks.ask.muse", { v: fmt(r.muse_spark.p_oom ?? 0, "probability") })}</span>
									<small className="pf-dim">{t(r.full_run.ran_out_of_memory ? "checks.ask.ran_out" : "checks.ask.did_not")}</small>
								</span>
							</span>
						</div>
					))}
				</div>
			</details>
			<details className="pf-ask__more">
				<summary>{t("checks.ask.caveats")}</summary>
				<ul className="pf-dim pf-checks__caveats">
					{vs.caveats.map((c) => (
						<li key={c}>{c}</li>
					))}
				</ul>
			</details>
		</section>
	);
}
