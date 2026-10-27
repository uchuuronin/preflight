import { t } from "../../lib/i18n";
import type { Fit } from "../status";
import type { EstimateStatus } from "../record/types";

/** Icon shape plus a word: never colour alone. */
export function FitMark({ fit, compact = false }: { fit: Fit; compact?: boolean }) {
	return (
		<span className={`pf-fit pf-fit--${fit}`}>
			<svg width="12" height="12" viewBox="0 0 12 12" aria-hidden="true">
				{fit === "fits" && <circle cx="6" cy="6" r="4.5" />}
				{fit === "tight" && <path d="M6 1.2 L10.8 10.2 H1.2 Z" />}
				{fit === "fails" && <path d="M2 2 L10 10 M10 2 L2 10" strokeWidth="2.2" />}
				{fit === "unknown" && <path d="M2 6 H10" strokeWidth="2" />}
			</svg>
			{!compact && <span>{t(`fit.${fit}`)}</span>}
		</span>
	);
}

/** Measured / predicted / extrapolated / abstain, as a short word with its line style. */
export function Provenance({ status }: { status: EstimateStatus }) {
	return (
		<span className={`pf-prov pf-prov--${status}`} title={t(`provenance.${status}.help`)}>
			<span className="pf-prov__line" aria-hidden="true" />
			{t(`provenance.${status}`)}
		</span>
	);
}
