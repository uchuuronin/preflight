/**
 * Checks: the same code, asked to Muse Spark without probes (B's vs_model.json, from
 * docs/validation/preflight-vs-model.md). Read first as three tiles, then as a dot plot of each
 * prediction over what the full run did, both sides on one log scale (step time and peak memory as
 * two small multiples, never two axes), Preflight a circle and Muse Spark a diamond so colour is
 * never the only cue. The numbers, the out-of-memory calls and B's caveats follow, as written.
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

/** How far off, in words a stranger reads: "8% low", "36% high", "18× too high". */
function off(x: number): string {
	if (x >= 2) return t("checks.ask.off_times_high", { v: one.format(x) });
	if (x <= 0.5) return t("checks.ask.off_times_low", { v: one.format(1 / x) });
	const pct = Math.round(Math.abs(x - 1) * 100);
	return t(x >= 1 ? "checks.ask.off_pct_high" : "checks.ask.off_pct_low", { v: pct });
}

// ---------------------------------------------------------------- the dot plot

/** One shared log scale: Muse Spark's gnn step interval reaches 36×, Preflight's retrieval peak 0.43×. */
const LO = 0.25;
const HI = 64;
const TICKS = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64];
const pos = (x: number) => (100 * (Math.log(Math.min(HI, Math.max(LO, x))) - Math.log(LO))) / (Math.log(HI) - Math.log(LO));
const tickLabel = (x: number) => (x < 1 ? `${x === 0.25 ? "¼" : "½"}×` : `${x}×`);

type Metric = "step" | "peak";
const TARGET: Record<Metric, number> = { step: 0.15, peak: 0.1 };

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

function Mark({ r, who, m }: { r: Row; who: "preflight" | "muse"; m: Metric }) {
	const x = ratioOf(r, who, m);
	if (x === null) return null;
	const e = estOf(who === "preflight" ? r.preflight : r.muse_spark, m);
	const act = actualOf(r, m);
	const lo = e?.lo !== undefined && act ? e.lo / act : null;
	const hi = e?.hi !== undefined && act ? e.hi / act : null;
	const name = t(who === "preflight" ? "checks.ask.side.preflight" : "checks.ask.side.muse");
	const unit = m === "step" ? "s" : "GiB";
	const num = (v: number) => (m === "step" ? sig3.format(v) : one.format(v));
	const tip = t("checks.ask.tip", {
		who: name,
		pred: `${num(e?.point ?? 0)} ${unit}`,
		range: e?.lo !== undefined && e?.hi !== undefined ? t("checks.ask.tip_range", { lo: num(e.lo), hi: num(e.hi), unit }) : "",
		actual: act ? `${num(act)} ${unit}` : "",
		off: off(x),
	});
	return (
		<span className={`pf-ask__lane pf-ask__lane--${who}`}>
			{lo !== null && hi !== null ? <span className="pf-ask__whisker" style={{ left: `${pos(lo)}%`, width: `${Math.max(0.6, pos(hi) - pos(lo))}%` }} aria-hidden="true" /> : null}
			<span className="pf-ask__dot" style={{ left: `${pos(x)}%` }} role="img" tabIndex={0} aria-label={tip} data-tip={tip}>
				<span className={`pf-ask__shape pf-ask__shape--${who}`} aria-hidden="true" />
			</span>
			{x >= 2 || x <= 0.5 ? (
				// selective direct labels: only the far ones, past the end of their range line
				x >= 1 ? (
					<span className="pf-ask__direct" style={{ left: `${pos(Math.max(x, hi ?? x))}%` }} aria-hidden="true">
						{ratio(x)}
					</span>
				) : (
					<span className="pf-ask__direct pf-ask__direct--left" style={{ left: `${pos(Math.min(x, lo ?? x))}%` }} aria-hidden="true">
						{ratio(x)}
					</span>
				)
			) : null}
		</span>
	);
}

function Plot({ m }: { m: Metric }) {
	const shown = vs.rows.filter((r) => ratioOf(r, "preflight", m) !== null || ratioOf(r, "muse", m) !== null);
	// what isn't plotted, once per script: one line when both sides miss for the same reason
	const gaps = vs.rows.flatMap((r) => {
		const missing = (["preflight", "muse"] as const).filter((w) => ratioOf(r, w, m) === null);
		if (!missing.length) return [];
		const whys = missing.map((w) => whyNot(r, w, m));
		if (missing.length === 2 && whys[0] === whys[1]) return [{ key: r.script, text: t("checks.ask.gap_both", { script: r.label, why: whys[0] }) }];
		return missing.map((w, i) => ({ key: `${r.script}-${w}`, text: t("checks.ask.gap", { who: t(w === "preflight" ? "checks.ask.side.preflight" : "checks.ask.side.muse"), script: r.label, why: whys[i] }) }));
	});
	const band = TARGET[m];
	return (
		<figure className="pf-ask__plot">
			<figcaption className="pf-ask__cap">
				<b>{t(m === "step" ? "checks.ask.col.step" : "checks.ask.col.peak")}</b>
				<small className="pf-dim">{t("checks.ask.band", { pct: Math.round(band * 100) })}</small>
			</figcaption>
			<div className="pf-ask__grid" aria-hidden="true">
				<span className="pf-ask__name" />
				<span className="pf-ask__ticks">
					{TICKS.map((x) => (
						<span key={x} className={x === 1 ? "pf-ask__tick is-one" : "pf-ask__tick"} style={{ left: `${pos(x)}%` }}>
							{tickLabel(x)}
						</span>
					))}
				</span>
			</div>
			<ul className="pf-ask__rows">
				{shown.map((r) => (
					<li key={r.script} className="pf-ask__grid">
						<span className="pf-ask__name">
							{r.label}
							{r.device.replace(/^NVIDIA /, "") !== r.gpu ? <small className="pf-dim"> · {t("checks.ask.ran_on", { ran: r.device.replace(/^NVIDIA /, "") })}</small> : null}
						</span>
						<span className="pf-ask__track">
							<span className="pf-ask__target" style={{ left: `${pos(1 - band)}%`, width: `${pos(1 + band) - pos(1 - band)}%` }} aria-hidden="true" />
							<span className="pf-ask__one" style={{ left: `${pos(1)}%` }} aria-hidden="true" />
							<Mark r={r} who="preflight" m={m} />
							<Mark r={r} who="muse" m={m} />
						</span>
					</li>
				))}
			</ul>
			{gaps.length ? (
				<ul className="pf-ask__gaps pf-dim">
					{gaps.map((g) => (
						<li key={g.key}>{g.text}</li>
					))}
				</ul>
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
			<div className="glass pf-ask__chart">
				<div className="pf-ask__legend">
					<span>
						<span className="pf-ask__key pf-ask__key--preflight" aria-hidden="true" /> {t("checks.ask.legend.preflight")}
					</span>
					<span>
						<span className="pf-ask__key pf-ask__key--muse" aria-hidden="true" /> {t("checks.ask.legend.muse")}
					</span>
					<small className="pf-dim">{t("checks.ask.how")}</small>
				</div>
				<div className="pf-ask__plots">
					<Plot m="step" />
					<Plot m="peak" />
				</div>
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
