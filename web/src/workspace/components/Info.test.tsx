import { act, cleanup, fireEvent, render } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { Info } from "./Info";

// jsdom has no layout: the popover's box is stubbed where a 390 px phone would put it.
function boxAt(left: number, width = 300) {
	return vi.spyOn(HTMLElement.prototype, "getBoundingClientRect").mockImplementation(function (this: HTMLElement) {
		return (this.classList.contains("pf-info__pop") ? { left, right: left + width, top: 0, bottom: 80, width, height: 80, x: left, y: 0 } : { left: 0, right: 24, top: 0, bottom: 24, width: 24, height: 24, x: 0, y: 0 }) as DOMRect;
	});
}

describe("the i popover on a phone (PM's audit, Nov 3)", () => {
	afterEach(() => {
		cleanup();
		vi.restoreAllMocks();
	});
	const width = () => vi.spyOn(document.documentElement, "clientWidth", "get").mockReturnValue(390);

	it("shifts an end-aligned popover that runs off the left edge back to 12 px", () => {
		width();
		boxAt(-241);
		const { container } = render(<Info label="Flight plan" align="end">text</Info>);
		act(() => fireEvent.click(container.querySelector("button") as HTMLElement));
		expect((container.querySelector(".pf-info__pop") as HTMLElement).style.translate).toBe("253px 0px");
		act(() => window.dispatchEvent(new Event("resize"))); // the same shift again stays applied
		expect((container.querySelector(".pf-info__pop") as HTMLElement).style.translate).toBe("253px 0px");
	});
	it("shifts a start-aligned popover that runs off the right edge back inside", () => {
		width();
		boxAt(200);
		const { container } = render(<Info label="Map" align="start">text</Info>);
		act(() => fireEvent.click(container.querySelector("button") as HTMLElement));
		expect((container.querySelector(".pf-info__pop") as HTMLElement).style.translate).toBe("-122px 0px");
	});
	it("leaves a popover that fits where its alignment put it", () => {
		width();
		boxAt(40);
		const { container } = render(<Info label="Fit">text</Info>);
		act(() => fireEvent.click(container.querySelector("button") as HTMLElement));
		expect((container.querySelector(".pf-info__pop") as HTMLElement).style.translate).toBe("");
	});
});
