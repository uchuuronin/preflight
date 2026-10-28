import { renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { currencyCode, setCurrency } from "../../lib/fmt";
import { t } from "../../lib/i18n";
import { useRegistry } from "../../registry";
import { CurrencySetting } from "./CurrencySetting";
import { FLAG, start } from "./index";
import { resetCurrencyForTests } from "./store";

describe("currency add-on", () => {
	beforeEach(() => {
		localStorage.clear();
		resetCurrencyForTests(["de-DE"]);
		setCurrency("USD", 1);
	});
	afterEach(() => setCurrency("USD", 1));

	it("docks the setting in the Logbook behind its flag", () => {
		const { result } = renderHook(() => useRegistry());
		const panel = result.current.appPanels.find((p) => p.id === "currency.setting");
		expect(panel).toMatchObject({ slot: "logbook-settings", flag: FLAG, component: CurrencySetting });
		expect(t(panel!.titleKey)).toBe("Show costs in");
		expect(FLAG).toBe("d.currency");
	});

	it("applies the visitor's currency at start when the flag is on", () => {
		start(true);
		expect(currencyCode()).toBe("EUR");
	});

	it("leaves every price in US dollars when the flag is off", () => {
		start(false);
		expect(currencyCode()).toBe("USD");
	});
});
