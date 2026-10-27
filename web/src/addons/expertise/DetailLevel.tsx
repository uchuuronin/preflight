/**
 * The detail-level setting, docked in the Logbook's pilot card (registry slot
 * "logbook-settings"). A radio group: arrow keys move between levels, and only the chosen
 * level is in the tab order.
 */
import { useRef, type KeyboardEvent } from "react";
import { t } from "../../lib/i18n";
import { TIERS, setTier, useTier, type Tier } from "./store";

const NEXT: Record<string, number> = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 };

export function DetailLevel() {
	const tier = useTier();
	const refs = useRef<Array<HTMLButtonElement | null>>([]);

	function move(e: KeyboardEvent<HTMLButtonElement>, from: number) {
		const step = NEXT[e.key];
		if (step === undefined) return;
		e.preventDefault();
		const to = (from + step + TIERS.length) % TIERS.length;
		const next = TIERS[to] as Tier;
		setTier(next);
		refs.current[to]?.focus();
	}

	return (
		<div className="pf-addon-expertise">
			<div className="pf-choice" role="radiogroup" aria-label={t("addon.expertise.dial.title")}>
				{TIERS.map((level, i) => (
					<button
						key={level}
						ref={(el) => {
							refs.current[i] = el;
						}}
						type="button"
						role="radio"
						aria-checked={level === tier}
						tabIndex={level === tier ? 0 : -1}
						onClick={() => setTier(level)}
						onKeyDown={(e) => move(e, i)}
					>
						{t(`addon.expertise.tier.${level}`)}
					</button>
				))}
			</div>
			<p className="pf-addon-expertise__hint" aria-live="polite">
				{t(`addon.expertise.tier.${tier}.hint`)}
			</p>
		</div>
	);
}
