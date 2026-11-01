import { hasKey, t } from "../../lib/i18n";

/**
 * Why a number isn't shown. A reason is either one of the provider's codes ("off_grid") with a
 * line of its own, or B's engine's own sentence ("probe p1-eval ran out of memory at these
 * settings", engine/api.py), shown as written with a capital and a full stop.
 */
export function abstainText(reason: string | null | undefined): string {
	const r = (reason ?? "").trim();
	if (!r) return t("abstain.off_grid");
	if (/^[a-z_]+$/.test(r)) return t(hasKey(`abstain.${r}`) ? `abstain.${r}` : "abstain.off_grid");
	const s = r[0].toUpperCase() + r.slice(1);
	return /[.!?]$/.test(s) ? s : `${s}.`;
}
