/**
 * Who found a finding, as the line under it says, and the sources it cites.
 *
 * A's harness marks findings by source, and some sources carry more than one model:
 * - every model triage is `spark_open_triage`, whichever model read the log, with the model named
 *   in the title (python/preflight/harness/triage.py `_who`: "Muse Glimmer on probe …" for a
 *   private check-in). The title is the per-finding signal: a record can hold both, when probes run
 *   later from a private record go to Spark Open. Live check-in, Nov 2 (record
 *   01M6EG0284PWZDRXGGQVVZYVND): Glimmer triaged it, and the line said Spark Open.
 * - a Muse Spark diagnosis is `spark_diagnosis`; Spark Open's check of it (diagnosis_check.py)
 *   either adds an evidence entry `spark_open_check` (code writes it, not the model, so it's read
 *   rather than the detail, whose words Muse Spark wrote), or turns the finding into
 *   `diagnosis.abstain` ("Not backed by a probe number: abstain, or run a probe"), which also covers
 *   claims the check didn't reach (its claim, time or dollar limits).
 */
import { t } from "../../lib/i18n";
import type { Finding } from "../record/types";

export function findingSourceKey(f: Pick<Finding, "source" | "title" | "code" | "evidence"> & Partial<Pick<Finding, "detail">>): string {
	if (f.source === "spark_open_triage" && f.title.startsWith("Muse Glimmer ")) return "finding.source.glimmer_triage";
	if (f.source === "spark_diagnosis") {
		if (f.code === "diagnosis.abstain") return "finding.source.spark_diagnosis_abstain";
		if ((f.evidence ?? []).some((e) => e.metric === "spark_open_check")) return "finding.source.spark_diagnosis_checked";
	}
	return `finding.source.${f.source}`;
}

/** Only web links open: a citation without an http(s) address is shown as its title. */
function safeUrl(url: string | null | undefined): string | null {
	if (!url) return null;
	try {
		const u = new URL(url);
		return u.protocol === "https:" || u.protocol === "http:" ? u.href : null;
	} catch {
		return null;
	}
}

/** A finding's sources (Muse Spark's web search, when it ran), as links. */
export function FindingCitations({ finding }: { finding: Pick<Finding, "citations"> }) {
	const cites = finding.citations ?? [];
	if (!cites.length) return null;
	return (
		<ul className="pf-findings__cites" aria-label={t("findings.citations")}>
			{cites.map((c, i) => {
				const href = safeUrl(c.url);
				return (
					<li key={i}>
						{href ? (
							<a href={href} target="_blank" rel="noopener noreferrer">
								{c.title || new URL(href).hostname}
							</a>
						) : (
							c.title
						)}
					</li>
				);
			})}
		</ul>
	);
}
