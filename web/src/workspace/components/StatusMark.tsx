import { t } from "../../lib/i18n";
import type { EstimateStatus } from "../record/types";
import type { Fit } from "../status";

/** Icon shape plus a word: never colour alone. */
export function FitMark({ fit, label }: { fit: Fit; label?: string }) {
	return (
		<span className={`pf-status pf-status--${fit}`}>
			<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
				{fit === "fits" ? <circle cx="6" cy="6" r="4" /> : null}
				{fit === "tight" ? <path d="M6 1.5 L10.5 10 H1.5 Z" /> : null}
				{fit === "fails" ? <path d="M2.5 2.5 L9.5 9.5 M9.5 2.5 L2.5 9.5" /> : null}
				{fit === "unknown" ? <path d="M2.5 6 H9.5" /> : null}
			</svg>
			<span>{label ?? t(`fit.${fit}`)}</span>
		</span>
	);
}

/** Measured / predicted / extrapolated / not trusted, as a short word with its line style. */
export function Provenance({ status }: { status: EstimateStatus }) {
	return (
		<span className={`pf-prov pf-prov--${status}`} title={t(`provenance.${status}.help`)}>
			<span className="pf-prov__line" aria-hidden="true" />
			{t(`provenance.${status}`)}
		</span>
	);
}
