import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { currencyCode, fmt, setCurrency } from "../../lib/fmt";
import { STORAGE_KEY, applyCurrency, getChoice, resetCurrencyForTests, resolvedCurrency, setChoice, subscribeCurrency } from "./store";

describe("currency store", () => {
	beforeEach(() => {
		localStorage.clear();
		resetCurrencyForTests(["en-US"]);
		setCurrency("USD", 1);
	});
	afterEach(() => {
		vi.restoreAllMocks();
		setCurrency("USD", 1);
	});

	it("starts on automatic and follows the browser's region", () => {
		resetCurrencyForTests(["en-IN"]);
		expect(getChoice()).toBe("auto");
		expect(resolvedCurrency()).toBe("INR");
	});

	it("falls back to US dollars when the region isn't carried", () => {
		resetCurrencyForTests(["ru-RU"]);
		expect(resolvedCurrency()).toBe("USD");
	});

	it("converts every dollar figure once applied", () => {
		resetCurrencyForTests(["en-GB"]);
		applyCurrency();
		expect(currencyCode()).toBe("GBP");
		// $100 at 0.8382 / 1.1188 GBP per USD
		expect(fmt(100, "usd")).toBe("£75");
	});

	it("remembers a chosen currency and applies it at once", () => {
		setChoice("EUR");
		expect(localStorage.getItem(STORAGE_KEY)).toBe("EUR");
		expect(currencyCode()).toBe("EUR");
		resetCurrencyForTests(["en-IN"]);
		expect(resolvedCurrency()).toBe("EUR");
	});

	it("ignores a saved code the table doesn't carry", () => {
		localStorage.setItem(STORAGE_KEY, "RUB");
		resetCurrencyForTests(["en-US"]);
		expect(getChoice()).toBe("auto");
	});

	it("tells listeners when the choice changes, and only then", () => {
		const fn = vi.fn();
		const off = subscribeCurrency(fn);
		setChoice("JPY");
		setChoice("JPY");
		expect(fn).toHaveBeenCalledTimes(1);
		off();
		setChoice("auto");
		expect(fn).toHaveBeenCalledTimes(1);
	});

	it("keeps working when storage throws", () => {
		vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
			throw new Error("blocked");
		});
		vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
			throw new Error("blocked");
		});
		resetCurrencyForTests(["en-US"]);
		setChoice("CAD");
		expect(resolvedCurrency()).toBe("CAD");
		expect(currencyCode()).toBe("CAD");
	});
});
