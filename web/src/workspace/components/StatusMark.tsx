import { useId, useState } from "react";
import { t } from "../../lib/i18n";
import { isFixture } from "../record/model";
import type { EstimateStatus } from "../record/types";
import { useOptionalWorkspace } from "../state";
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

/** The word for where a number comes from; on a sample (a fixture record) every number that
 * isn't an abstention is illustrative, whatever its status, since nothing about it was measured. */
export function provenanceKey(status: EstimateStatus, illustrative: boolean): string {
	return illustrative && status !== "abstain" ? "provenance.illustrative" : `provenance.${status}`;
}

/**
 * Measured / predicted / extrapolated / not measured here, as a short word with its line style. The
 * word explains itself in one line on hover, on keyboard focus and on a tap (Chiya, Nov 2: four words
 * are a lot to learn, and a title attribute never shows on a phone).
 */
export function Provenance({ status }: { status: EstimateStatus }) {
	const ws = useOptionalWorkspace();
	const key = provenanceKey(status, ws ? isFixture(ws.record) : false);
	const tip = useId();
	const [open, setOpen] = useState(false);
	return (
		<span className={`pf-prov-wrap${open ? " is-open" : ""}`}>
			<button
				type="button"
				className={`pf-prov pf-prov--${status}`}
				aria-describedby={tip}
				aria-expanded={open}
				onClick={(e) => {
					e.stopPropagation(); // a tap on the word inside a clickable dial explains the word, nothing else
					setOpen((o) => !o);
				}}
				onBlur={() => setOpen(false)}
				onKeyDown={(e) => {
					if (e.key === "Escape") setOpen(false);
				}}
			>
				<span className="pf-prov__line" aria-hidden="true" />
				{t(key)}
			</button>
			<span id={tip} role="tooltip" className="pf-prov__tip">
				{t(`${key}.help`)}
			</span>
		</span>
	);
}
