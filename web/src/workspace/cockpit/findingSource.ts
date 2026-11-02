/**
 * Who found a finding, as the finding's line says it. A's harness writes every model triage with
 * `source: "spark_open_triage"`, whichever model read the log, and names the model in the title
 * (python/preflight/harness/triage.py `_who`: "Muse Glimmer on probe …" for a private check-in,
 * "Spark Open on probe …" otherwise). The title is the per-finding signal: a record can hold both,
 * when probes run later from a private record go to Spark Open. Live check-in, Nov 2 (record
 * 01M6EG0284PWZDRXGGQVVZYVND): Glimmer triaged it, and the line said Spark Open.
 */
import type { Finding } from "../record/types";

export function findingSourceKey(f: Pick<Finding, "source" | "title">): string {
	if (f.source === "spark_open_triage" && f.title.startsWith("Muse Glimmer ")) return "finding.source.glimmer_triage";
	return `finding.source.${f.source}`;
}
