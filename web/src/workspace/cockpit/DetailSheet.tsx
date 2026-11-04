/**
 * Click a node: its breakdown. Numbers come from the provider (predictions at the current
 * knobs) or from a probe, and the sheet says which: "measured at batch 4" is not a prediction
 * for batch 8. Groups list what they hold; findings say what to do about them.
 */
import { type KeyboardEvent, useId, useRef, useState } from "react";
import { fmt, fmtSpread } from "../../lib/fmt";
import { hasKey, t } from "../../lib/i18n";
import { usePanels } from "../../registry";
import { valueText } from "../../guide/GuidePanel";
import { useOrbitLayout } from "../map/Orbit";
import { resolveMetric } from "../provider/metrics";
import { isFixture, target } from "../record/model";
import { useWorkspace } from "../state";
import { bounds, trusted } from "../status";
import { Provenance } from "../components/StatusMark";
import { AddonBoundary } from "../../components/AddonBoundary";
import { FixFinding } from "../../fix/FixFinding";
import { FindingCitations, findingSourceKey, plainDetail } from "./findingSource";
import { useWhyAbstain } from "./whyAbstain";
import { splitQuote } from "./Findings";
import type { Finding, Knob } from "../record/types";

/** A metric path with no label of its own, as words: "probe.eval.peak_allocated_bytes" reads "Eval peak allocated". */
function plainMetric(key: string): string {
	const parts = key.split(".").filter((p) => p !== "probe");
	const words = parts.join(" ").replace(/_/g, " ").replace(/\bbytes\b|\bfraction\b|\bmean\b|\bs\b/g, "").trim();
	return words ? words[0]!.toUpperCase() + words.slice(1) : key;
}

export function DetailSheet() {
	const { record, estimates, state, actions } = useWorkspace();
	const why = useWhyAbstain();
	const layout = useOrbitLayout();
	const drawer = usePanels("drawer", record.job.lens);
	const node = layout.nodes.find((n) => n.id === state.focus);
	if (!node) return null;
	const graphNode = node.graphId ? record.graph.nodes.find((n) => n.id === node.graphId) : undefined;
	const children = graphNode ? record.graph.nodes.filter((n) => n.parent === graphNode.id) : [];
	const findings = record.findings.filter((f) => f.node_id && f.node_id === node.graphId);
	const phaseKey = node.phase ? `time.phase.${node.phase}` : null;
	const metricKeys = [...new Set([...(graphNode?.metrics ?? []), ...(phaseKey && estimates[phaseKey] ? [phaseKey] : [])])];
	const knobs = new Map(record.knobs.map((k) => [k.id, k]));
	const probeWhere = (pk: Record<string, unknown> | undefined) =>
		Object.entries(pk ?? {})
			.slice(0, 2)
			.map(([k, v]) => `${t(knobs.get(k)?.label_key ?? k).toLowerCase()} ${valueText(knobs.get(k), v as never)}`)
			.join(", ");
	const eyebrow = node.branch
		? t(`detail.branch.${node.phase === "eval" ? "eval" : "save"}`)
		: node.share !== null
			? t("detail.share", { share: fmt(node.share, "fraction") })
			: t("detail.loop");
	const title = node.graphId ? node.label : hasKey(`phase.${node.phase}`) ? t(`phase.${node.phase}`) : node.label;

	return (
		<section className="glass pf-detail" aria-label={t("detail.label", { name: title })}>
			{/* Back sits beside the eyebrow, not under the sheet, so it costs no row (Chiya, Nov 3). */}
			<div className="pf-detail__top">
				<span className="pf-eyebrow">{eyebrow}</span>
				<button type="button" className="pf-detail__back" aria-label={t("detail.back")} title={t("detail.back")} onClick={() => actions.focus(null)}>
					<span aria-hidden="true">×</span>
				</button>
			</div>
			<h3>{title}</h3>
			{node.phase && hasKey(`detail.about.${node.phase}`) ? <p>{t(`detail.about.${node.phase}`)}</p> : null}
			{children.length ? (
				<div className="pf-sub">
					{children.map((c, i) => (
						<span key={c.id} className="pf-sub__item">
							{i > 0 ? <i aria-hidden="true">→</i> : null}
							<span>
								{c.label}
								{(c.repeat ?? 1) > 1 ? ` ×${c.repeat}` : ""}
							</span>
						</span>
					))}
				</div>
			) : null}
			{metricKeys.length ? (
				<dl className="pf-detail__metrics">
					{metricKeys.map((key) => {
						const m = resolveMetric(record, state.knobs, estimates, key);
						const tg = target(record, key);
						const label = tg ? t(tg.label_key) : hasKey(`metric.${key}`) ? t(`metric.${key}`) : plainMetric(key);
						const est = estimates[key];
						return (
							<div key={key}>
								<dt>{label}</dt>
								<dd>
									{m ? (
										<>
											<strong>{fmt(m.value, m.unit)}</strong>
											{trusted(est) && est.lo !== null && est.hi !== null && fmtSpread(...bounds(est), est.unit) ? (
												<span className="pf-dim">{fmtSpread(...bounds(est), est.unit)}</span>
											) : null}
											{m.kind === "prediction" && est ? <Provenance status={est.status} /> : null}
											{m.kind === "measured_here" ? <Provenance status="measured" /> : null}
											{m.kind === "measured_elsewhere" ? (
												<span className="pf-dim">{t(isFixture(record) ? "detail.illustrative_at" : "detail.measured_at", { where: probeWhere(m.probeKnobs) })}</span>
											) : null}
										</>
									) : (
										<span className="pf-dim">{est?.status === "abstain" ? why(est.reason) : t("detail.no_value")}</span>
									)}
								</dd>
							</div>
						);
					})}
				</dl>
			) : null}
			{findings.length ? <FindingTabs key={node.id} findings={findings} knobs={knobs} onOpenKnob={(id) => actions.openKnob(id)} /> : null}
			{drawer.map((p) => (
				<AddonBoundary key={p.id} id={p.id}>
					<p.component nodeId={node.id} record={record} knobs={state.knobs} estimates={estimates} />
				</AddonBoundary>
			))}
		</section>
	);
}

const RANK: Record<string, number> = { critical: 0, error: 1, warn: 2, info: 3 };

/**
 * A node's findings, one at a time (Chiya, Nov 3): every finding open at once made the sheet taller than
 * the map and pushed the knobs down, and a scroll box read as chopped. The worst comes first; a row of
 * numbered marks above it (the severity's mark and colour, the title on hover and to a screen reader)
 * pages through the rest, so the sheet's height is one finding's and nothing is cropped. A probe's log
 * line, the longest part, is one click away.
 */
function FindingTabs({ findings, knobs, onOpenKnob }: { findings: Finding[]; knobs: Map<string, Knob>; onOpenKnob: (id: string) => void }) {
	const sorted = [...findings].sort((a, b) => (RANK[a.severity] ?? 9) - (RANK[b.severity] ?? 9));
	const [at, setAt] = useState(0);
	const base = useId();
	const tabs = useRef<(HTMLButtonElement | null)[]>([]);
	const f = sorted[Math.min(at, sorted.length - 1)]!;
	const go = (i: number) => {
		const n = (i + sorted.length) % sorted.length;
		setAt(n);
		tabs.current[n]?.focus();
	};
	const onKey = (e: KeyboardEvent) => {
		if (e.key === "ArrowRight" || e.key === "ArrowDown") go(at + 1);
		else if (e.key === "ArrowLeft" || e.key === "ArrowUp") go(at - 1);
		else if (e.key === "Home") go(0);
		else if (e.key === "End") go(sorted.length - 1);
		else return;
		e.preventDefault();
	};
	const plain = plainDetail(f.detail);
	const { text, quote } = splitQuote(plain.text);
	const knob = f.knob_id ? knobs.get(f.knob_id) : undefined;
	const panel = (
		<div className={`pf-finding pf-finding--${f.severity}`} {...(sorted.length > 1 ? { role: "tabpanel", id: `${base}-p`, "aria-labelledby": `${base}-t${at}` } : {})}>
			<b>{f.title}</b>
			{text ? <p>{text}</p> : null}
			{plain.unsupported ? <p className="pf-dim">{t("findings.claim_unsupported")}</p> : null}
			{quote ? (
				<details className="pf-finding__log">
					<summary>{t("detail.log")}</summary>
					<code>{quote}</code>
				</details>
			) : null}
			<p className="pf-dim">
				{t(findingSourceKey(f))}
				{knob ? ` · ${t("detail.fix_with", { knob: t(knob.label_key).toLowerCase() })}` : ""}
			</p>
			<FindingCitations finding={f} />
			<div className="pf-finding__acts">
				{knob ? (
					<button type="button" className="pf-btn pf-btn--small" onClick={() => onOpenKnob(knob.id)}>
						{t("detail.open_knob", { knob: t(knob.label_key).toLowerCase() })}
					</button>
				) : null}
				<FixFinding finding={f} />
			</div>
		</div>
	);
	if (sorted.length === 1) return <div className="pf-detail__findings">{panel}</div>;
	return (
		<div className="pf-detail__findings is-tabbed">
			<div className="pf-ftabs">
				<span className="pf-ftabs__count" aria-hidden="true">
					{t("detail.finding_of", { i: at + 1, n: sorted.length })}
				</span>
				<div className="pf-ftabs__marks" role="tablist" aria-label={t("detail.findings", { n: sorted.length })} onKeyDown={onKey}>
				{sorted.map((x, i) => (
					<button
						key={x.id}
						ref={(el) => {
							tabs.current[i] = el;
						}}
						type="button"
						role="tab"
						id={`${base}-t${i}`}
						aria-selected={i === at}
						aria-controls={`${base}-p`}
						tabIndex={i === at ? 0 : -1}
						className={`pf-ftab pf-ftab--${x.severity}${i === at ? " is-on" : ""}`}
						aria-label={x.title}
						title={x.title}
						onClick={() => setAt(i)}
					>
						<span aria-hidden="true">{i + 1}</span>
					</button>
				))}
				</div>
			</div>
			{panel}
		</div>
	);
}
