/**
 * Exchange rates for D3. Preflight's prices are in US dollars (B's table); the ECB publishes
 * rates per euro, so a rate per dollar is (currency per euro) / (dollars per euro).
 * One dated snapshot, shown on screen with its date, never fetched live.
 * [cited: ECB euro foreign exchange reference rates, 26 Oct 2026; rates.json carries the link]
 */
import data from "./rates.json";

export interface RateTable {
	as_of: string;
	source: string;
	url: string;
	per_eur: Record<string, number>;
}

export const RATES: RateTable = data;

/** Every currency the table can convert to: US dollars first, then by code. */
export const CURRENCIES: readonly string[] = [
	"USD",
	...[...Object.keys(RATES.per_eur).filter((c) => c !== "USD"), "EUR"].sort(),
];

/** Units of `code` per US dollar, or null when the table doesn't carry it. */
export function perUsd(code: string): number | null {
	if (code === "USD") return 1;
	const usdPerEur = RATES.per_eur.USD;
	if (!usdPerEur) return null;
	if (code === "EUR") return 1 / usdPerEur;
	const perEur = RATES.per_eur[code];
	return perEur === undefined ? null : perEur / usdPerEur;
}
