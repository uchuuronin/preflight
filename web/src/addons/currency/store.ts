/**
 * Local currency (D3): "auto" follows the browser's region, or a chosen code. Kept per browser
 * in localStorage, guarded for blocked storage. Applying calls C's setCurrency(code, rate),
 * so every fmt(value, "usd") on screen converts; nothing else changes.
 */
import { useSyncExternalStore } from "react";
import { setCurrency } from "../../lib/fmt";
import { perUsd } from "./rates";
import { currencyForLocales } from "./region";

export type Choice = "auto" | string;

export const STORAGE_KEY = "pf.currency.v1";

function browserLocales(): readonly string[] {
	if (typeof navigator === "undefined") return [];
	return navigator.languages?.length ? navigator.languages : [navigator.language];
}

function readSaved(): Choice {
	try {
		const raw = localStorage.getItem(STORAGE_KEY);
		return raw && perUsd(raw) !== null ? raw : "auto";
	} catch {
		return "auto";
	}
}

let locales: readonly string[] = browserLocales();
let choice: Choice = readSaved();
const listeners = new Set<() => void>();

export function getChoice(): Choice {
	return choice;
}

/** The browser region's currency, when the table carries it. */
export function autoCurrency(): string | null {
	return currencyForLocales(locales);
}

export function resolvedCurrency(): string {
	return choice === "auto" ? (autoCurrency() ?? "USD") : choice;
}

export function applyCurrency(): void {
	const code = resolvedCurrency();
	setCurrency(code, perUsd(code) ?? 1);
}

export function setChoice(next: Choice): void {
	if (next !== "auto" && perUsd(next) === null) return;
	if (next === choice) return;
	choice = next;
	try {
		if (next === "auto") localStorage.removeItem(STORAGE_KEY);
		else localStorage.setItem(STORAGE_KEY, next);
	} catch {
		// Not saved; the choice still holds for this visit.
	}
	applyCurrency();
	for (const l of listeners) l();
}

export function subscribeCurrency(fn: () => void): () => void {
	listeners.add(fn);
	return () => {
		listeners.delete(fn);
	};
}

export function useCurrencyChoice(): Choice {
	return useSyncExternalStore(subscribeCurrency, getChoice, getChoice);
}

/** Re-read storage and locales, as a fresh page load would. Tests only. */
export function resetCurrencyForTests(testLocales?: readonly string[]): void {
	locales = testLocales ?? browserLocales();
	choice = readSaved();
	listeners.clear();
}
