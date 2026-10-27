import { fmt, fmtRange } from "../../lib/fmt";
import { hasKey, t } from "../../lib/i18n";
import { usePanels } from "../../registry";
import { useWorkspace } from "../state";
import { trusted } from "../status";
import { Provenance } from "./StatusMark";

/** Click a node: its breakdown, its findings, then anything an add-on docks here. */
export function NodeDrawer({ nodeId, onClose }: { nodeId: string; onClose: () => void }) {
	const { record, estimates, state } = useWorkspace();
	const panels = usePanels("drawer", record.job.lens);
	const node = record.graph.nodes.find((n) => n.id === nodeId);
	if (!node) return null;
	const targets = new Map(record.predictions.targets.map((x) => [x.id, x]));
	const metricIds = Object.entries(node.metrics ?? {}).filter(([, v]) => !!v) as Array<[string, string]>;
	const findings = record.findings.filter((f) => f.node === nodeId);
	return (
		<aside className="pf-drawer" aria-label={t("drawer.label", { node: node.label })}>
			<header className="pf-drawer__head">
				<h2>{node.label_key && hasKey(node.label_key) ? t(node.label_key) : node.label}</h2>
				<button type="button" className="pf-btn pf-btn--quiet" onClick={onClose}>
					{t("action.close")}
				</button>
			</header>
			{metricIds.length ? (
				<dl className="pf-drawer__metrics">
					{metricIds.map(([kind, id]) => {
						const e = estimates[id];
						const tg = targets.get(id);
						if (!e || !tg) return null;
						return (
							<div key={id}>
								<dt>{t(`drawer.metric.${kind}`)}</dt>
								<dd>
									{trusted(e) ? (
										<>
											<strong>{fmt(e.point, tg.unit)}</strong>{" "}
											<span className="pf-muted">{fmtRange(e.lo, e.hi, tg.unit)}</span>
										</>
									) : (
										t("readout.abstain")
									)}
									<Provenance status={e.status} />
								</dd>
							</div>
						);
					})}
				</dl>
			) : (
				<p className="pf-muted">{t("drawer.no_metrics")}</p>
			)}
			{findings.map((f) => (
				<section key={f.id} className={`pf-finding is-${f.severity}`}>
					<h3>{t(f.title)}</h3>
					<p>{t(f.evidence)}</p>
					<p className="pf-muted">{t(`finding.source.${f.source}`)}</p>
				</section>
			))}
			{panels.map((p) => (
				<p.component key={p.id} nodeId={nodeId} record={record} knobs={state.knobs} estimates={estimates} />
			))}
		</aside>
	);
}
