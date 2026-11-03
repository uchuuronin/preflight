// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import en from "../../i18n/en.json";
import { KNOBS_ID, KnobsJump } from "../cockpit/Dials";
import { Provenance } from "./StatusMark";

afterEach(() => {
	cleanup();
	vi.unstubAllGlobals();
});

const tip = () => document.body.querySelector(".pf-prov__tip");

describe("a provenance word explains itself", () => {
	it("keeps its own name, with the line as its description", () => {
		render(<Provenance status="predicted" />);
		const word = screen.getByRole("button", { name: en["provenance.predicted"] });
		expect(document.getElementById(word.getAttribute("aria-describedby") ?? "")?.textContent).toBe(en["provenance.predicted.help"]);
	});

	it("opens on a tap and closes on a second tap, a tap elsewhere or Escape, even while focused", () => {
		render(<Provenance status="extrapolated" />);
		const word = screen.getByRole("button");
		fireEvent.click(word);
		expect(tip()?.textContent).toBe(en["provenance.extrapolated.help"]);
		fireEvent.click(word);
		expect(tip()).toBeNull();
		fireEvent.focus(word);
		expect(tip()).not.toBeNull();
		fireEvent.keyDown(word, { key: "Escape" });
		expect(tip()).toBeNull(); // dismissible while focus stays on the word
		fireEvent.click(word);
		fireEvent.pointerDown(document.body);
		expect(tip()).toBeNull();
	});

	it("opens on a phone's tap, which sends pointer and focus events before the click", () => {
		render(<Provenance status="predicted" />);
		const word = screen.getByRole("button");
		fireEvent.pointerEnter(word, { pointerType: "touch" });
		fireEvent.focus(word);
		fireEvent.click(word);
		expect(tip()).not.toBeNull();
	});

	it("stays open when a hovered word is clicked, and closes on the pointer leaving only if hovered", () => {
		render(<Provenance status="predicted" />);
		const word = screen.getByRole("button");
		fireEvent.pointerEnter(word, { pointerType: "mouse" });
		expect(tip()).not.toBeNull();
		fireEvent.pointerLeave(word, { pointerType: "mouse" });
		expect(tip()).toBeNull();
		fireEvent.pointerEnter(word, { pointerType: "mouse" });
		fireEvent.click(word);
		fireEvent.pointerLeave(word, { pointerType: "mouse" });
		expect(tip()).not.toBeNull();
	});

	it("stays inside the window near its right edge", () => {
		render(<Provenance status="measured" />);
		const word = screen.getByRole("button");
		word.getBoundingClientRect = () => ({ left: window.innerWidth - 20, right: window.innerWidth - 4, top: 10, bottom: 24, width: 16, height: 14, x: 0, y: 0, toJSON: () => ({}) });
		fireEvent.pointerEnter(word, { pointerType: "mouse" });
		const el = tip() as HTMLElement;
		expect(parseFloat(el.style.left) + parseFloat(el.style.maxWidth)).toBeLessThanOrEqual(window.innerWidth - 8);
	});
});

describe("the Knobs jump", () => {
	const observe = (ratio: number, top: number) => {
		let cb: IntersectionObserverCallback = () => {};
		vi.stubGlobal(
			"IntersectionObserver",
			class {
				constructor(c: IntersectionObserverCallback) {
					cb = c;
				}
				observe() {}
				disconnect() {}
			},
		);
		const rail = document.createElement("div");
		rail.id = KNOBS_ID;
		document.body.appendChild(rail);
		render(<KnobsJump />);
		act(() => cb([{ intersectionRatio: ratio, boundingClientRect: { top } } as IntersectionObserverEntry], {} as IntersectionObserver));
		rail.remove();
	};

	it("shows while under half the rail is in view, pointing to where it is", () => {
		observe(0.3, 600);
		expect(screen.getByRole("button").textContent).toContain("↓");
		cleanup();
		observe(0, -400);
		expect(screen.getByRole("button").textContent).toContain("↑");
	});

	it("hides once half the rail shows", () => {
		observe(0.6, 300);
		expect(screen.queryByRole("button")).toBeNull();
	});
});
