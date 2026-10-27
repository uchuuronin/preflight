/**
 * Detail level (D1, the expertise dial): new, practitioner or expert. One per browser,
 * kept in localStorage only. Storage can be missing or throw (private windows, blocked
 * site data), so every access is guarded and the app works at the default without it.
 */
import { useSyncExternalStore } from "react";

export type Tier = "new" | "practitioner" | "expert";

export const TIERS: readonly Tier[] = ["new", "practitioner", "expert"];
export const DEFAULT_TIER: Tier = "practitioner";
export const STORAGE_KEY = "pf.expertise.v1";

export function parseTier(raw: string | null | undefined): Tier | null {
	return (TIERS as readonly string[]).includes(raw ?? "") ? (raw as Tier) : null;
}

function readSaved(): Tier {
	try {
		return parseTier(localStorage.getItem(STORAGE_KEY)) ?? DEFAULT_TIER;
	} catch {
		return DEFAULT_TIER;
	}
}

let current: Tier = readSaved();
const listeners = new Set<() => void>();

export function getTier(): Tier {
	return current;
}

export function setTier(tier: Tier): void {
	if (tier === current) return;
	current = tier;
	try {
		localStorage.setItem(STORAGE_KEY, tier);
	} catch {
		// Not saved; the choice still holds for this visit.
	}
	for (const l of listeners) l();
}

export function subscribeTier(fn: () => void): () => void {
	listeners.add(fn);
	return () => {
		listeners.delete(fn);
	};
}

export function useTier(): Tier {
	return useSyncExternalStore(subscribeTier, getTier, getTier);
}

/** Re-read storage, as a fresh page load would. Tests only. */
export function resetTierForTests(): void {
	current = readSaved();
	listeners.clear();
}
