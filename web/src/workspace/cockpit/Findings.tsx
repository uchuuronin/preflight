/**
 * Findings with no place on the map (Chiya, Nov 1; D's request for the triage shot): the detail
 * sheet lists only a node's own findings, so a finding without a node, or on a node the map
 * doesn't draw, showed nowhere. Muse Spark Open's triage writes info findings with no node; A's
 * diagnosis and claim checks will land here too. Each row: its severity in a word and a mark, its
 * text, the evidence quoted (triage's log line is quoted inside its detail, `(log: "...")`, which
 * A's code has already checked is a real line of the log), and who found it.
 */
import { fmt } from "../../lib/fmt";
import { hasKey, t } from "../../lib/i18n";
import { FixFinding } from "../../fix/FixFinding";
import { useOrbitLayout } from "../map/Orbit";
import type { LoadedRecord } from "../record/types";
import { useWorkspace } from "../state";

type Finding = LoadedRecord["findings"][number];

const RANK: Record<Finding["severity"], number> = { critical: 3, error: 2, warn: 1, info: 0 };

/** Findings no drawn node carries: no node, or a node the map doesn't draw (most severe first). */
export function looseFindings(record: LoadedRecord, drawn: ReadonlySet<string>): Finding[] {
	return record.findings.filter((f) => !f.node_id || !drawn.has(f.node_id)).sort((a, b) => RANK[b.severity] - RANK[a.severity]);
}

/**
 * A finding's detail split into what it says and the log line it quotes. A's triage writes
 * `summary (log: "line")`, or `summary (no log line quoted)` when the model quoted nothing real.
 */
export function splitQuote(detail: string | null | undefined): { text: string; quote: string | null; unquoted: boolean } {
	const d = (detail ?? "").trim();
	const m = /^([\s\S]*?)\s*\(log: "([\s\S]*)"\)$/.exec(d);
	if (m) return { text: m[1]!.trim(), quote: m[2]!, unquoted: false };
	const n = /^([\s\S]*?)\s*\(no log line quoted\)$/.exec(d);
	if (n) return { text: n[1]!.trim(), quote: null, unquoted: true };
	return { text: d, quote: null, unquoted: false };
}

/**
 * Evidence worth a line: only metrics with a name a visitor can read (an i18n label), never a raw
 * metric id; triage's own marker isn't evidence. Quotes in British single marks.
 */
function evidenceLines(f: Finding): string[] {
	return (f.evidence ?? [])
		.filter((e) => e.metric !== "triage" && e.value !== null && e.value !== undefined && hasKey(`metric.${e.metric}`))
		.map((e) => {
			const where = e.probe_id ? t("findings.at_probe", { probe: e.probe_id }) : "";
			const value = typeof e.value === "number" ? fmt(e.value, "number") : `‘${e.value}’`;
			const limit = typeof e.threshold === "number" ? t("findings.threshold", { threshold: fmt(e.threshold, "number") }) : "";
			return `${t(`metric.${e.metric}`)}: ${value}${limit}${where}`;
		});
}

export function useLooseFindings(): Finding[] {
	const { record } = useWorkspace();
	const layout = useOrbitLayout();
	const drawn = new Set(layout.nodes.map((n) => n.graphId).filter((x): x is string => !!x));
	return looseFindings(record, drawn);
}

/** In the cockpit's header: how many findings sit off the map, and a jump to them. */
export function FindingsChip() {
	const loose = useLooseFindings();
	if (!loose.length) return null;
	return (
		<button
			type="button"
			className="pf-tag pf-tag--button pf-findings-chip"
			onClick={() => {
				const el = document.getElementById("pf-findings");
				el?.scrollIntoView({ behavior: window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "start" });
				el?.focus({ preventScroll: true });
			}}
		>
			{t(loose.length === 1 ? "findings.chip_one" : "findings.chip", { n: loose.length })}
		</button>
	);
}

export function FindingsBand() {
	const loose = useLooseFindings();
	if (!loose.length) return null;
	return (
		<section id="pf-findings" className="glass pf-band pf-findings" aria-labelledby="pf-findings-title" tabIndex={-1}>
			<div className="pf-band__head">
				<b id="pf-findings-title">{t("findings.title")}</b>
				<span className="pf-dim">{t("findings.lede")}</span>
			</div>
			<ul className="pf-findings__list">
				{loose.map((f) => {
					const { text, quote, unquoted } = splitQuote(f.detail);
					const lines = evidenceLines(f);
					return (
						<li key={f.id} className={`pf-finding pf-finding--${f.severity}`}>
							<span className={`pf-findings__sev is-${f.severity}`}>{t(`findings.severity.${f.severity}`)}</span>
							<b>{f.title}</b>
							{text ? <p>{text}</p> : null}
							{quote ? (
								<blockquote className="pf-findings__quote">
									<code>{quote}</code>
								</blockquote>
							) : unquoted ? (
								<p className="pf-dim">{t("findings.no_quote")}</p>
							) : null}
							{lines.length ? (
								<ul className="pf-findings__evidence" aria-label={t("findings.evidence")}>
									{lines.map((l, i) => (
										<li key={i}>{l}</li>
									))}
								</ul>
							) : null}
							<p className="pf-findings__source">{t(`finding.source.${f.source}`)}</p>
							<FixFinding finding={f} />
						</li>
					);
				})}
			</ul>
		</section>
	);
}
