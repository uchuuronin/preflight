import { describe, expect, it } from "vitest";
import { CURRENCIES, RATES, perUsd } from "./rates";
import { currencyForLocales } from "./region";

describe("exchange rates", () => {
	it("are dated and sourced", () => {
		expect(RATES.as_of).toBe("2026-10-26");
		expect(RATES.source).toMatch(/European Central Bank/);
		expect(RATES.url).toMatch(/^https:\/\/www\.ecb\.europa\.eu\//);
	});

	it("convert from US dollars through the euro", () => {
		expect(perUsd("USD")).toBe(1);
		expect(perUsd("EUR")).toBeCloseTo(1 / 1.1188, 6);
		expect(perUsd("INR")).toBeCloseTo(106.4935 / 1.1188, 6);
		expect(perUsd("GBP")).toBeCloseTo(0.8382 / 1.1188, 6);
	});

	it("refuse a currency they don't carry", () => {
		expect(perUsd("RUB")).toBeNull();
		expect(perUsd("usd")).toBeNull();
	});

	it("offer US dollars first, then the rest in code order", () => {
		expect(CURRENCIES[0]).toBe("USD");
		expect(CURRENCIES).toContain("EUR");
		expect(CURRENCIES).toHaveLength(30);
		const rest = CURRENCIES.slice(1);
		expect(rest).toEqual([...rest].sort());
	});
});

describe("currency from the browser's region", () => {
	it("reads the region from the first locale that has one", () => {
		expect(currencyForLocales(["en-IN"])).toBe("INR");
		expect(currencyForLocales(["en", "de-DE"])).toBe("EUR");
		expect(currencyForLocales(["en-GB"])).toBe("GBP");
		expect(currencyForLocales(["ja-JP"])).toBe("JPY");
	});

	it("covers every euro-area country", () => {
		for (const c of ["AT", "BE", "CY", "DE", "EE", "ES", "FI", "FR", "GR", "HR", "IE", "IT", "LT", "LU", "LV", "MT", "NL", "PT", "SI", "SK"]) {
			expect(currencyForLocales([`xx-${c}`]), c).toBe("EUR");
		}
	});

	it("handles script subtags and underscores", () => {
		expect(currencyForLocales(["zh-Hans-CN"])).toBe("CNY");
		expect(currencyForLocales(["pt_BR"])).toBe("BRL");
	});

	it("returns null when no region is known or carried", () => {
		expect(currencyForLocales(["en"])).toBeNull();
		expect(currencyForLocales(["ru-RU"])).toBeNull();
		expect(currencyForLocales([])).toBeNull();
	});
});
