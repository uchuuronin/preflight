/**
 * Glossary card, docked in every knob's control through the registry's knob decorations.
 * A native disclosure, so it needs no script to open and reads well to screen readers.
 * It follows the detail level; "More detail" shows the next level's line without changing
 * the setting.
 */
import { useEffect, useState } from "react";
import { t } from "../../lib/i18n";
import type { Knob } from "../../workspace/record/types";
import { glossaryEntry } from "./glossary";
import { TIERS, useTier } from "./store";

export function GlossaryCard({ knob }: { knob: Knob }) {
	const tier = useTier();
	const [extra, setExtra] = useState(0);
	useEffect(() => setExtra(0), [tier, knob.id]);

	const entry = glossaryEntry(knob.id, tier);
	if (!entry) return null;

	const start = TIERS.indexOf(tier);
	const deeper = TIERS.slice(start + 1, start + 1 + extra);
	const canGoDeeper = start + extra < TIERS.length - 1;

	return (
		<details className="pf-addon-gloss">
			<summary>{t("addon.expertise.card.summary")}</summary>
			<div className="pf-addon-gloss__body">
				<p>{t(entry.bodyKey)}</p>
				{deeper.map((level) => {
					const e = glossaryEntry(knob.id, level);
					return e ? (
						<p key={level} className="pf-addon-gloss__deeper">
							{t(e.bodyKey)}
						</p>
					) : null;
				})}
				{entry.actKey ? (
					<p className="pf-addon-gloss__act">
						<b>{t("addon.expertise.card.act")}</b> {t(entry.actKey)}
					</p>
				) : null}
				{canGoDeeper ? (
					<button type="button" className="pf-link" onClick={() => setExtra((n) => n + 1)}>
						{t("addon.expertise.card.more")}
					</button>
				) : null}
			</div>
		</details>
	);
}
