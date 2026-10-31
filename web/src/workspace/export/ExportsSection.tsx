/**
 * "Take it with you": the cockpit section that holds the exports panels add-ons register
 * (registry.registerExportsPanel). C builds the plan once per setting and hands it to each;
 * nothing renders until `c.exports` is on and at least one panel is registered.
 */
import { useMemo } from "react";
import { AddonBoundary } from "../../components/AddonBoundary";
import { flag } from "../../lib/flags";
import { t } from "../../lib/i18n";
import { useExportsPanels } from "../../registry";
import { useWorkspace } from "../state";
import { exportPlan } from "./plan";

function appBase(): string | null {
	return typeof window === "undefined" ? null : window.location.origin + window.location.pathname;
}

export function ExportsSection() {
	const { record, recordRef, state, estimates } = useWorkspace();
	const panels = useExportsPanels(record.job.lens);
	const on = flag("c.exports") && panels.length > 0;
	// A sample's ref is its id and a real record's is its share token: either opens it read-only.
	const plan = useMemo(() => (on ? exportPlan(record, state.knobs, estimates, recordRef || null, appBase()) : null), [on, record, state.knobs, estimates, recordRef]);
	if (!plan) return null;
	return (
		<section className="pf-below pf-exports" aria-label={t("exports.label")}>
			<h2 className="pf-h2">{t("exports.title")}</h2>
			{panels.map((p) => (
				<AddonBoundary key={p.id} id={p.id}>
					<p.component record={record} knobs={state.knobs} estimates={estimates} plan={plan} />
				</AddonBoundary>
			))}
		</section>
	);
}
