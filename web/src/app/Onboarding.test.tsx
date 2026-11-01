// @vitest-environment jsdom
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

let on = true;
vi.mock("../lib/flags", () => ({ flag: (name: string) => name === "c.onboarding" && on }));

import en from "../i18n/en.json";
import { Onboarding, STEPS } from "./Onboarding";

function page() {
	document.body.innerHTML = '<div class="pf-orbit"></div><div class="pf-dials"></div><div class="pf-dock__row"></div><section class="pf-guide"></section>';
}

beforeEach(() => {
	localStorage.clear();
	on = true;
	page();
});
afterEach(() => cleanup());

describe("first-visit notes", () => {
	it("walks the four parts once, lighting each, and not again", () => {
		const { unmount } = render(<Onboarding />);
		for (const [i, s] of STEPS.entries()) {
			expect(screen.getByText(en[`onboard.${s.id}.title` as keyof typeof en])).toBeTruthy();
			expect(document.querySelector(s.target)?.classList.contains("pf-onboard-lit")).toBe(true);
			fireEvent.click(screen.getByRole("button", { name: i === STEPS.length - 1 ? en["onboard.done"] : en["onboard.next"] }));
		}
		expect(screen.queryByRole("dialog")).toBeNull();
		expect(document.querySelectorAll(".pf-onboard-lit")).toHaveLength(0);
		unmount();
		render(<Onboarding />);
		expect(screen.queryByRole("dialog")).toBeNull();
	});

	it("ends on Skip or Escape, and stays away with the flag off", () => {
		render(<Onboarding />);
		fireEvent.keyDown(screen.getByRole("dialog"), { key: "Escape" });
		expect(screen.queryByRole("dialog")).toBeNull();
		cleanup();
		localStorage.clear();
		on = false;
		render(<Onboarding />);
		expect(screen.queryByRole("dialog")).toBeNull();
	});

	it("still shows when storage throws (a private window)", () => {
		const get = vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
			throw new Error("blocked");
		});
		render(<Onboarding />);
		expect(screen.getByRole("dialog")).toBeTruthy();
		get.mockRestore();
	});
});
