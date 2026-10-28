import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { currencyCode, setCurrency } from "../../lib/fmt";
import { CurrencySetting } from "./CurrencySetting";
import { getChoice, resetCurrencyForTests } from "./store";
import { registerCurrencyStrings } from "./strings";

describe("CurrencySetting", () => {
	beforeEach(() => {
		localStorage.clear();
		resetCurrencyForTests(["en-IN"]);
		setCurrency("USD", 1);
		registerCurrencyStrings();
	});
	afterEach(() => {
		cleanup();
		setCurrency("USD", 1);
	});

	it("is a labelled menu starting on automatic, naming what that means", () => {
		render(<CurrencySetting />);
		const menu = screen.getByRole("combobox", { name: "Show costs in" }) as HTMLSelectElement;
		expect(menu.value).toBe("auto");
		expect(menu.options[0]?.textContent).toBe("Automatic: Indian rupee");
	});

	it("lists every carried currency by name and code", () => {
		render(<CurrencySetting />);
		const menu = screen.getByRole("combobox", { name: "Show costs in" }) as HTMLSelectElement;
		const labels = [...menu.options].map((o) => o.textContent);
		expect(labels).toHaveLength(31);
		expect(labels).toContain("US dollar (USD)");
		expect(labels).toContain("Euro (EUR)");
		expect(labels).toContain("Japanese yen (JPY)");
	});

	it("switches currency on change", () => {
		render(<CurrencySetting />);
		fireEvent.change(screen.getByRole("combobox", { name: "Show costs in" }), { target: { value: "EUR" } });
		expect(getChoice()).toBe("EUR");
		expect(currencyCode()).toBe("EUR");
	});

	it("says what the figures are based on, with the date and a link to the source", () => {
		render(<CurrencySetting />);
		expect(screen.getByText(/Prices are in US dollars, converted at the rate of 26 Oct 2026/)).toBeTruthy();
		const link = screen.getByRole("link", { name: "European Central Bank" });
		expect(link.getAttribute("href")).toMatch(/^https:\/\/www\.ecb\.europa\.eu\//);
		expect(link.getAttribute("rel")).toBe("noopener noreferrer");
	});

	it("drops the note when costs are already in US dollars", () => {
		resetCurrencyForTests(["en-US"]);
		render(<CurrencySetting />);
		expect(screen.queryByText(/converted at the rate/)).toBeNull();
	});
});
