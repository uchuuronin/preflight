/**
 * The RL lens's two views, from the measured probe and B's estimates, nothing else:
 *
 * - Reward spread: one cell per sampled prompt, filled by its pass rate. A prompt whose answers all
 *   score the same (all wrong, or all right) gives GRPO no signal, since every answer's advantage is
 *   zero. The headline is the probe's own count with its Wilson 95% interval; beside it, B's
 *   prediction at the current group size (`rl.zero_spread`), which moves with the group-size knob.
 * - Completion length against the cap: mean, p95 and longest, measured; truncation as measured when
 *   the probe read the cap, otherwise "not measured" with B's estimate (`rl.truncation`) beside it.
 */
import { wilson } from "../../addons/_stats/wilson";
import { fmt, fmtRange } from "../../lib/fmt";
import { t } from "../../lib/i18n";
import { target } from "../record/model";
import type { Estimate } from "../record/types";
import { rlProbe, spreadCount } from "../rl";
import { useWorkspace } from "../state";
import { trusted } from "../status";

function est(e: Estimate | undefined): string {
	if (!e || !trusted(e)) return e?.reason ? t("rl.not_predicted", { why: e.reason }) : t("value.none");
	return t("rl.estimate", { point: fmt(e.point, "probability"), range: fmtRange(e.lo ?? e.point, e.hi ?? e.point, "probability") });
}

export function RlBand() {
	const { record, estimates, state } = useWorkspace();
	const probe = rlProbe(record);
	if (record.job.lens !== "rl" || !probe?.rl) return null;
	const rl = probe.rl;
	const rates = rl.pass_rate_per_prompt ?? [];
	const count = spreadCount(probe);
	const w = count ? wilson(count.k, count.n) : null;
	const groupNow = state.knobs.group_size ?? rl.group_size;
	const spreadT = target(record, "rl.zero_spread");
	const truncT = target(record, "rl.truncation");
	const gen = probe.generation;
	const len = rl.response_length;
	const cap = typeof state.knobs.max_completion_len === "number" ? state.knobs.max_completion_len : (gen?.max_new_tokens ?? null);
	const correct = typeof rl.group_size === "number" ? Math.round(rates.reduce((a, b) => a + b, 0) * rl.group_size) : null;
	const answers = typeof rl.group_size === "number" ? rates.length * rl.group_size : null;
	const truncMeasured = typeof rl.truncation_rate === "number";
	const scaleMax = Math.max(cap ?? 0, len?.max ?? 0, 1);

	return (
		<section id="pf-rl" className="glass pf-band pf-rl" aria-label={t("rl.title")}>
			<div className="pf-band__head">
				<b>{t("rl.title")}</b>
			</div>
			<div id="pf-rl-guide" />
			<div className="pf-rl__part">
				<h3 className="pf-rl__h">{t("rl.spread.title")}</h3>
				{w ? (
					<p className="pf-rl__lead">
						{t("rl.spread.measured", { k: fmt(w.k, "count"), n: fmt(w.n, "count"), share: fmt(w.p, "probability"), lo: fmt(w.lo, "probability"), hi: fmt(w.hi, "probability") })}
						<span className="pf-prov">{t("provenance.measured")}</span>
					</p>
				) : null}
				<ol className="pf-rl__cells" aria-label={t("rl.spread.cells", { n: fmt(rates.length, "count") })}>
					{rates.map((p, i) => {
						const flat = p <= 0 || p >= 1;
						return (
							<li
								key={i}
								className={`pf-rl__cell${flat ? " is-flat" : ""}`}
								title={t(flat ? "rl.spread.cell_flat" : "rl.spread.cell", { i: i + 1, rate: fmt(p, "probability") })}
								style={{ ["--pass" as string]: `${Math.round(p * 100)}%` }}
							/>
						);
					})}
				</ol>
				<p className="pf-rl__note">
					{correct !== null && answers ? t("rl.spread.correct", { correct: fmt(correct, "count"), answers: fmt(answers, "count") }) : null}{" "}
					{t("rl.spread.why")}
				</p>
				{spreadT ? (
					<p className="pf-rl__pred">
						{t("rl.spread.predicted", { g: fmt(Number(groupNow ?? 0), "count") })} <b>{est(estimates[spreadT.id])}</b>
					</p>
				) : null}
			</div>
			<div className="pf-rl__part">
				<h3 className="pf-rl__h">{t("rl.length.title")}</h3>
				{len ? (
					<div className="pf-rl__lengths" role="img" aria-label={t("rl.length.aria", { mean: fmt(len.mean ?? 0, "count"), p95: fmt(len.p95 ?? 0, "count"), max: fmt(len.max ?? 0, "count"), cap: cap !== null ? fmt(cap, "count") : t("value.none") })}>
						<span className="pf-rl__lenbar" style={{ width: `${((len.p95 ?? 0) / scaleMax) * 100}%` }} />
						<i className="pf-rl__mark is-mean" style={{ left: `${((len.mean ?? 0) / scaleMax) * 100}%` }} />
						<i className="pf-rl__mark is-max" style={{ left: `${((len.max ?? 0) / scaleMax) * 100}%` }} />
						{cap !== null ? <i className="pf-rl__cap" style={{ left: `${(cap / scaleMax) * 100}%` }} /> : null}
					</div>
				) : null}
				{len ? (
					<p className="pf-rl__note">
						{t("rl.length.measured", { mean: fmt(len.mean ?? 0, "count"), p95: fmt(len.p95 ?? 0, "count"), max: fmt(len.max ?? 0, "count") })}
						{cap !== null ? ` ${t("rl.length.cap", { cap: fmt(cap, "count") })}` : ""}
					</p>
				) : null}
				<p className="pf-rl__pred">
					{truncMeasured ? (
						<>
							{t("rl.trunc.measured", { share: fmt(rl.truncation_rate as number, "probability") })}
							<span className="pf-prov">{t("provenance.measured")}</span>
						</>
					) : (
						<>
							{t("rl.trunc.not_measured")} {truncT ? <b>{est(estimates[truncT.id])}</b> : null}
						</>
					)}
				</p>
			</div>
		</section>
	);
}
