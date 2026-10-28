/**
 * Feature flags. Defaults come from the repo's flags.json (flat `name: bool`, every feature
 * off until its validation row passes). Overrides, later wins:
 *   1. flags.<mode>.json at the repo root, where <mode> is the Vite mode (development,
 *      preview, production);
 *   2. VITE_FLAGS, a JSON object, per deploy environment;
 *   3. ?flags=a,b,-c in the URL, outside production builds only.
 */
import defaults from "../../../flags.json";

type FlagMap = Record<string, boolean>;

function onlyFlags(raw: Record<string, unknown>): FlagMap {
	const out: FlagMap = {};
	for (const [k, v] of Object.entries(raw)) if (!k.startsWith("$") && typeof v === "boolean") out[k] = v;
	return out;
}

const perMode = import.meta.glob<{ default: Record<string, unknown> }>("../../../flags.*.json", { eager: true });

function fromMode(): FlagMap {
	const entry = Object.entries(perMode).find(([path]) => path.endsWith(`/flags.${import.meta.env.MODE}.json`));
	return entry ? onlyFlags(entry[1].default) : {};
}

function fromEnv(): FlagMap {
	const raw = import.meta.env.VITE_FLAGS as string | undefined;
	if (!raw) return {};
	try {
		return onlyFlags(JSON.parse(raw) as Record<string, unknown>);
	} catch {
		return {};
	}
}

function fromUrl(): FlagMap {
	if (typeof window === "undefined" || import.meta.env.PROD) return {};
	const raw = new URLSearchParams(window.location.search).get("flags");
	if (!raw) return {};
	const out: FlagMap = {};
	for (const part of raw.split(",")) {
		const name = part.trim();
		if (!name) continue;
		if (name.startsWith("-")) out[name.slice(1)] = false;
		else out[name] = true;
	}
	return out;
}

const resolved: FlagMap = {
	...onlyFlags(defaults as Record<string, unknown>),
	...fromMode(),
	...fromEnv(),
	...fromUrl(),
};

export function flag(name: string): boolean {
	return resolved[name] === true;
}

export function allFlags(): Readonly<FlagMap> {
	return resolved;
}
