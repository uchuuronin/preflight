/**
 * Click a node: its breakdown. Numbers come from the provider (predictions at the current
 * knobs) or from a probe, and the sheet says which: "measured at batch 4" is not a prediction
 * for batch 8. Groups list what they hold; findings say what to do about them.
 */
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
			<span className="pf-eyebrow">{eyebrow}</span>
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
			{findings.length ? (
				<div className="pf-detail__findings">
					{findings.map((f) => (
						<div key={f.id} className={`pf-finding pf-finding--${f.severity}`}>
							<b>{f.title}</b>
							{f.detail ? <p>{plainDetail(f.detail).text}</p> : null}
							{plainDetail(f.detail).unsupported ? <p className="pf-dim">{t("findings.claim_unsupported")}</p> : null}
							<p className="pf-dim">
								{t(findingSourceKey(f))}
								{f.knob_id && knobs.get(f.knob_id) ? ` · ${t("detail.fix_with", { knob: t(knobs.get(f.knob_id)?.label_key ?? f.knob_id).toLowerCase() })}` : ""}
							</p>
							<FindingCitations finding={f} />
							{f.knob_id && knobs.has(f.knob_id) ? (
								<button type="button" className="pf-btn pf-btn--small" onClick={() => actions.openKnob(f.knob_id ?? null)}>
									{t("detail.open_knob", { knob: t(knobs.get(f.knob_id)?.label_key ?? f.knob_id).toLowerCase() })}
								</button>
							) : null}
							<FixFinding finding={f} />
						</div>
					))}
				</div>
			) : null}
			{drawer.map((p) => (
				<AddonBoundary key={p.id} id={p.id}>
					<p.component nodeId={node.id} record={record} knobs={state.knobs} estimates={estimates} />
				</AddonBoundary>
			))}
			<button type="button" className="pf-btn pf-btn--small" onClick={() => actions.focus(null)}>
				{t("detail.back")}
			</button>
		</section>
	);
}
