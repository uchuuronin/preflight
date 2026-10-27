/**
 * Memory stack: what fills the GPU, as a stacked bar against a capacity line.
 * Segments are direct-labelled (three series colours sit under 3:1 on the panel, so the
 * label, not the colour, carries identity) and a table view sits underneath.
 */
import { fmt, fmtRange } from "../../lib/fmt";
import { t } from "../../lib/i18n";
import type { WorkspaceContextValue } from "../../registry";
import { useWorkspace } from "../state";
import { capacityBytes, memoryFit, trusted } from "../status";
import { FitMark } from "../components/StatusMark";

/** Fixed order: categorical slots never cycle. */
export const MEMORY_TERMS = [
	{ id: "mem_weights", key: "term.weights", slot: 1 },
	{ id: "mem_grads_optimizer", key: "term.grads_optimizer", slot: 2 },
	{ id: "mem_activations", key: "term.activations", slot: 3 },
	{ id: "mem_logits", key: "term.logits", slot: 4 },
	{ id: "mem_eval_spike", key: "term.eval_spike", slot: 5 },
	{ id: "mem_kv_cache", key: "term.kv_cache", slot: 6 },
] as const;

export function MemoryStackBar({ compact = false }: { compact?: boolean }) {
	const { record, estimates, state } = useWorkspace();
	const cap = capacityBytes(record, estimates, String(state.knobs.gpu ?? ""));
	const terms = MEMORY_TERMS.map((m) => ({ ...m, est: estimates[m.id] })).filter(
		(m) => trusted(m.est) && m.est.point > 0,
	);
	if (!terms.length || !cap) return null;
	const total = terms.reduce((s, m) => s + (m.est?.point ?? 0), 0);
	const scaleMax = Math.max(cap * 1.08, total * 1.02);
	const fit = memoryFit(estimates, record);
	return (
		<figure className={`pf-memstack${compact ? " is-compact" : ""}`}>
			<figcaption className="pf-memstack__caption">
				<span>{t("view.memory.caption")}</span>
				<FitMark fit={fit} />
			</figcaption>
			<div className="pf-memstack__track" role="img" aria-label={t("view.memory.aria", { total: fmt(total, "bytes"), cap: fmt(cap, "bytes") })}>
				{terms.map((m) => {
					const w = ((m.est?.point ?? 0) / scaleMax) * 100;
					const highlighted =
						state.highlight.includes(`term:${m.id}`) ||
						(m.id === "mem_eval_spike" && state.highlight.includes("eval")) ||
						(m.id === "mem_activations" && state.highlight.includes("blocks"));
					return (
						<span
							key={m.id}
							className={`pf-memstack__seg${highlighted ? " is-accent" : ""}`}
							style={{ width: `${w}%`, background: `var(--pf-series-${m.slot})` }}
							title={`${t(m.key)}: ${fmt(m.est?.point, "bytes")}`}
						>
							{!compact && w > 9 ? <span className="pf-memstack__label">{t(m.key)}</span> : null}
						</span>
					);
				})}
				<span className="pf-memstack__cap" style={{ left: `${(cap / scaleMax) * 100}%` }}>
					<span>{t("view.memory.capacity", { cap: fmt(cap, "bytes") })}</span>
				</span>
			</div>
			<ul className="pf-memstack__legend">
				{terms.map((m) => (
					<li key={m.id}>
						<i style={{ background: `var(--pf-series-${m.slot})` }} />
						{t(m.key)} <span className="pf-muted">{fmt(m.est?.point, "bytes")}</span>
					</li>
				))}
			</ul>
		</figure>
	);
}

/** The full view, registered as a panel: the bar plus its table. */
export function MemoryStackView(_props: WorkspaceContextValue) {
	const { estimates } = useWorkspace();
	return (
		<div className="pf-view">
			<MemoryStackBar />
			<table className="pf-table">
				<caption className="pf-visually-hidden">{t("view.memory.table")}</caption>
				<thead>
					<tr>
						<th scope="col">{t("table.term")}</th>
						<th scope="col">{t("table.estimate")}</th>
						<th scope="col">{t("table.interval")}</th>
						<th scope="col">{t("table.status")}</th>
					</tr>
				</thead>
				<tbody>
					{MEMORY_TERMS.map((m) => {
						const e = estimates[m.id];
						if (!e) return null;
						return (
							<tr key={m.id}>
								<th scope="row">{t(m.key)}</th>
								<td>{trusted(e) ? fmt(e.point, "bytes") : t("readout.abstain")}</td>
								<td>{trusted(e) ? fmtRange(e.lo, e.hi, "bytes") : ""}</td>
								<td>{t(`provenance.${e.status}`)}</td>
							</tr>
						);
					})}
				</tbody>
			</table>
		</div>
	);
}
