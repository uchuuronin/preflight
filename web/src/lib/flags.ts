/**
 * Feature flags. Defaults come from the repo's flags.json (every feature off until its
 * validation row passes). Overrides, in order: VITE_FLAGS (JSON, per environment), then
 * ?flags=a,b,-c in the URL (preview and local only).
 */
import defaults from "../../../flags.json";

type FlagMap = Record<string, boolean>;

function fromEnv(): FlagMap {
	const raw = import.meta.env.VITE_FLAGS as string | undefined;
	if (!raw) return {};
	try {
		return JSON.parse(raw) as FlagMap;
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

const resolved: FlagMap = { ...(defaults.flags as FlagMap), ...fromEnv(), ...fromUrl() };

export function flag(name: string): boolean {
	return resolved[name] === true;
}

export function allFlags(): Readonly<FlagMap> {
	return resolved;
}
