/**
 * The visitor's currency from their browser's locales (navigator.languages), for D3's
 * "Automatic" choice. Only currencies the rate table carries; anything else gives null.
 */
import { perUsd } from "./rates";

// Country to currency [cited: ISO 4217; euro area per the ECB, 20 members since Croatia, 2023].
const EURO_AREA = ["AT", "BE", "CY", "DE", "EE", "ES", "FI", "FR", "GR", "HR", "IE", "IT", "LT", "LU", "LV", "MT", "NL", "PT", "SI", "SK"];

const BY_REGION: Record<string, string> = {
	...Object.fromEntries(EURO_AREA.map((r) => [r, "EUR"])),
	US: "USD",
	GB: "GBP",
	JP: "JPY",
	CZ: "CZK",
	DK: "DKK",
	HU: "HUF",
	PL: "PLN",
	RO: "RON",
	SE: "SEK",
	CH: "CHF",
	LI: "CHF",
	IS: "ISK",
	NO: "NOK",
	TR: "TRY",
	AU: "AUD",
	BR: "BRL",
	CA: "CAD",
	CN: "CNY",
	HK: "HKD",
	ID: "IDR",
	IL: "ILS",
	IN: "INR",
	KR: "KRW",
	MX: "MXN",
	MY: "MYR",
	NZ: "NZD",
	PH: "PHP",
	SG: "SGD",
	TH: "THB",
	ZA: "ZAR",
};

function regionOf(locale: string): string | null {
	for (const part of locale.split(/[-_]/).slice(1)) {
		if (/^[A-Za-z]{2}$/.test(part)) return part.toUpperCase();
	}
	return null;
}

export function currencyForLocales(locales: readonly string[]): string | null {
	for (const locale of locales) {
		const region = regionOf(locale);
		if (!region) continue;
		const code = BY_REGION[region];
		return code && perUsd(code) !== null ? code : null;
	}
	return null;
}
