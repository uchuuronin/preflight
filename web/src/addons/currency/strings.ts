/**
 * English copy for the currency add-on, registered into C's t() under addon.currency.*.
 * Currency names are D's own (British sentence case) so they translate with the rest (D7).
 */
import { registerStrings } from "../../lib/i18n";

const NAMES: Record<string, string> = {
	USD: "US dollar",
	EUR: "Euro",
	GBP: "Pound sterling",
	JPY: "Japanese yen",
	INR: "Indian rupee",
	CNY: "Chinese yuan",
	AUD: "Australian dollar",
	BRL: "Brazilian real",
	CAD: "Canadian dollar",
	CHF: "Swiss franc",
	CZK: "Czech koruna",
	DKK: "Danish krone",
	HKD: "Hong Kong dollar",
	HUF: "Hungarian forint",
	IDR: "Indonesian rupiah",
	ILS: "Israeli new shekel",
	ISK: "Icelandic krona",
	KRW: "South Korean won",
	MXN: "Mexican peso",
	MYR: "Malaysian ringgit",
	NOK: "Norwegian krone",
	NZD: "New Zealand dollar",
	PHP: "Philippine peso",
	PLN: "Polish zloty",
	RON: "Romanian leu",
	SEK: "Swedish krona",
	SGD: "Singapore dollar",
	THB: "Thai baht",
	TRY: "Turkish lira",
	ZAR: "South African rand",
};

export const CURRENCY_EN: Record<string, string> = {
	"addon.currency.title": "Show costs in",
	"addon.currency.auto": "Automatic: {name}",
	"addon.currency.option": "{name} ({code})",
	"addon.currency.note": "Prices are in US dollars, converted at the rate of {date} from the",
	"addon.currency.source": "European Central Bank",
	...Object.fromEntries(Object.entries(NAMES).map(([code, name]) => [`addon.currency.name.${code}`, name])),
};

export function registerCurrencyStrings(): void {
	registerStrings("en", CURRENCY_EN);
}
