import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { DetailLevel } from "./DetailLevel";
import { getTier, resetTierForTests } from "./store";
import { registerExpertiseStrings } from "./strings";

describe("DetailLevel", () => {
	beforeEach(() => {
		localStorage.clear();
		resetTierForTests();
		registerExpertiseStrings();
	});
	afterEach(cleanup);

	it("offers the three levels as one radio group, practitioner chosen", () => {
		render(<DetailLevel />);
		const group = screen.getByRole("radiogroup", { name: "Detail level" });
		const options = screen.getAllByRole("radio");
		expect(group).toBeTruthy();
		expect(options.map((o) => o.textContent)).toEqual(["New to this", "Practitioner", "Expert"]);
		expect(screen.getByRole("radio", { name: /Practitioner/ }).getAttribute("aria-checked")).toBe("true");
	});

	it("explains what the chosen level shows", () => {
		render(<DetailLevel />);
		expect(screen.getByText("Standard terms and the usual settings")).toBeTruthy();
	});

	it("switches level on press and says so", () => {
		render(<DetailLevel />);
		fireEvent.click(screen.getByRole("radio", { name: /New to this/ }));
		expect(getTier()).toBe("new");
		expect(screen.getByRole("radio", { name: /New to this/ }).getAttribute("aria-checked")).toBe("true");
		expect(screen.getByText("Plain words, only the settings that matter most")).toBeTruthy();
	});

	it("moves between levels with the arrow keys", () => {
		render(<DetailLevel />);
		const practitioner = screen.getByRole("radio", { name: /Practitioner/ });
		fireEvent.keyDown(practitioner, { key: "ArrowRight" });
		expect(getTier()).toBe("expert");
		fireEvent.keyDown(screen.getByRole("radio", { name: /Expert/ }), { key: "ArrowRight" });
		expect(getTier()).toBe("new");
	});

	it("keeps only the chosen option in the tab order", () => {
		render(<DetailLevel />);
		const tabbable = screen.getAllByRole("radio").filter((o) => o.getAttribute("tabindex") === "0");
		expect(tabbable.map((o) => o.textContent)).toEqual(["Practitioner"]);
	});
});
