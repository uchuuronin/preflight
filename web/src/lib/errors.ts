/**
 * User-facing copy for a probe or job failure, keyed by A's stable error codes
 * (schema/record.py: oom, oom.eval, missing_package, ...). An unknown sub-code falls back to its
 * family ("oom.fsdp" reads as "oom"), then to a generic line; never a raw code on its own.
 */
import { hasKey, t } from "./i18n";

export interface ErrorCopy {
	code: string | null;
	message: string;
	hint: string;
}

export function errorCopy(code: string | null | undefined, raw?: string | null): ErrorCopy {
	const tries = code ? [code, code.split(".")[0] ?? code] : [];
	for (const c of tries) {
		if (hasKey(`error.${c}`)) {
			return { code: c, message: t(`error.${c}`), hint: hasKey(`error.${c}.hint`) ? t(`error.${c}.hint`) : "" };
		}
	}
	return { code: code ?? null, message: raw?.trim() || t("error.generic"), hint: t("error.generic.hint") };
}

/** A job event's message is an i18n key or a plain log line. */
export function eventText(message: string | null | undefined): { text: string; hint: string } | null {
	if (!message) return null;
	if (hasKey(message)) return { text: t(message), hint: hasKey(`${message}.hint`) ? t(`${message}.hint`) : "" };
	return { text: message, hint: "" };
}
