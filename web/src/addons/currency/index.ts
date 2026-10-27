/**
 * D3 · Local currency. Imported automatically by C's loadAddons().
 *
 * Every price stays in US dollars in the record (B's table); this converts at display time
 * through C's setCurrency(), at the ECB rate of a stated date, to the visitor's currency
 * (from the browser's region) or one they pick in the Logbook. Behind flag d.currency.
 * Regional GPU prices are data for B's table: python/preflight/addons/regional_prices.py.
 */
import { flag } from "../../lib/flags";
import { registerAppPanel } from "../../registry";
import { CurrencySetting } from "./CurrencySetting";
import { applyCurrency } from "./store";
import { registerCurrencyStrings } from "./strings";
import "./currency.css";

export const FLAG = "d.currency";

registerCurrencyStrings();

/** Apply the visitor's currency, only when the add-on is on. */
export function start(enabled: boolean = flag(FLAG)): void {
	if (enabled) applyCurrency();
}

start();

registerAppPanel({
	id: "currency.setting",
	slot: "logbook-settings",
	titleKey: "addon.currency.title",
	order: 20,
	flag: FLAG,
	component: CurrencySetting,
});
