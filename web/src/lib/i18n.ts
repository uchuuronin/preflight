/**
 * t(key, vars): every user-facing string goes through here.
 *
 * English lives in web/src/i18n/en.json (owned by C). Any other web/src/i18n/<lang>.json
 * file (owned by D) is picked up automatically, so adding a language never edits this file.
 * Missing keys fall back to English, then to the key itself, so a gap is visible, never blank.
 */
import en from "../i18n/en.json";

type Dict = Record<string, string>;

const others = import.meta.glob<{ default: Dict }>("../i18n/*.json", { eager: true });

const dictionaries = new Map<string, Dict>([["en", en as Dict]]);
for (const [path, mod] of Object.entries(others)) {
	const lang = path.match(/([\w-]+)\.json$/)?.[1];
	if (lang && lang !== "en") dictionaries.set(lang, mod.default);
}

let current = "en";
const listeners = new Set<() => void>();

export function setLanguage(lang: string): void {
	current = dictionaries.has(lang) ? lang : "en";
	if (typeof document !== "undefined") document.documentElement.lang = current;
	for (const l of listeners) l();
}

export function language(): string {
	return current;
}

export function languages(): string[] {
	return [...dictionaries.keys()];
}

export function onLanguageChange(fn: () => void): () => void {
	listeners.add(fn);
	return () => listeners.delete(fn);
}

/** Add or extend a dictionary at runtime (an add-on with its own strings). */
export function registerStrings(lang: string, strings: Dict): void {
	dictionaries.set(lang, { ...(dictionaries.get(lang) ?? {}), ...strings });
}

export function hasKey(key: string): boolean {
	return key in (dictionaries.get(current) ?? {}) || key in (dictionaries.get("en") ?? {});
}

/** Translate a key. {name} placeholders are filled from vars. */
export function t(key: string, vars?: Record<string, string | number>): string {
	const raw = dictionaries.get(current)?.[key] ?? dictionaries.get("en")?.[key] ?? key;
	if (!vars) return raw;
	return raw.replace(/\{(\w+)\}/g, (m, name: string) => (name in vars ? String(vars[name]) : m));
}
