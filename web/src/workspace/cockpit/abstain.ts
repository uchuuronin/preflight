import { hasKey, t } from "../../lib/i18n";

/** A provider or engine code, as the live guide's checks read one (liveGuide.ts): no spaces. */
const CODE = /^[a-z][a-z0-9_.]{0,47}$/;

/**
 * Why a number isn't shown, or why it is only extrapolated. A reason is either a code
 * ("off_grid", "outside_probed_range") with a line of its own, or B's engine's own sentence
 * ("probe p1-eval ran out of memory at these settings", engine/api.py), shown as written with
 * a full stop. Its first letter is raised only when the first word is a plain word, so a sentence
 * that starts with a knob id ("eval_batch_size changes…", "bf16 has…") keeps the id as it is.
 * A code with no line falls back to the general one, never to the code itself.
 */
export function abstainText(reason: string | null | undefined): string {
	const r = (reason ?? "").trim();
	if (!r) return t("abstain.off_grid");
	if (CODE.test(r)) return t(hasKey(`abstain.${r}`) ? `abstain.${r}` : "abstain.off_grid");
	const first = r.split(/\s/, 1)[0] ?? "";
	const s = /^[a-z]+$/.test(first) ? r[0]!.toUpperCase() + r.slice(1) : r;
	return /[.!?]$/.test(s) ? s : `${s}.`;
}
