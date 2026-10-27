import { describe, expect, it } from "vitest";
import { wilson } from "./wilson";

// Reference values from the closed form with z = 1.959964 (Wilson 1927), checked by hand.
describe("wilson", () => {
	it("matches the textbook interval for 8 of 10", () => {
		const w = wilson(8, 10);
		expect(w.p).toBe(0.8);
		expect(w.lo).toBeCloseTo(0.4902, 3);
		expect(w.hi).toBeCloseTo(0.9433, 3);
	});

	it("stays inside 0 and 1 at the edges", () => {
		const all = wilson(6, 6);
		expect(all.hi).toBe(1);
		expect(all.lo).toBeCloseTo(0.6097, 3);
		const none = wilson(0, 6);
		expect(none.lo).toBe(0);
		expect(none.hi).toBeCloseTo(0.3903, 3);
	});

	it("refuses an empty or impossible count", () => {
		expect(() => wilson(0, 0)).toThrow();
		expect(() => wilson(5, 4)).toThrow();
	});
});
