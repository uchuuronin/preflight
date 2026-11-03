/**
 * D1 · The beginner's line (plainWords.ts), docked in the cockpit head above the status line, at
 * the "New to this" level only. Not a live region: the status line under it already announces a
 * change, and two announcements of the same move would talk over each other. Body-text colour on
 * the cockpit's sky (11.8:1 at its lightest under the head, `theme/tokens.ts`).
 */
import { t } from "../../lib/i18n";
import type { WorkspaceContextValue } from "../../registry";
import { plainLine } from "./plainWords";
import { useTier } from "./store";

export function PlainLine({ record, knobs, estimates }: WorkspaceContextValue) {
	const tier = useTier();
	if (tier !== "new") return null;
	const text = plainLine(record, estimates, knobs);
	if (!text) return null;
	return (
		// A visually hidden lead-in, not aria-label: a label on a plain paragraph isn't read out reliably.
		<p className="pf-addon-plainline">
			<span className="pf-visually-hidden">{t("addon.expertise.plain.label")}: </span>
			{text}
		</p>
	);
}
