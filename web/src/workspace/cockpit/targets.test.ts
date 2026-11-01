import { describe, expect, it } from "vitest";
import { targetsApart } from "./Bands";

const ev = (at: number) => ({ kind: "eval" as const, step: Math.round(at * 1000), at });

describe("flight-plan marks as targets (audit finding 8)", () => {
	it("keeps marks that are 24 px apart pressable, draws the rest, and always keeps the failing eval", () => {
		const marks = [ev(0.1), ev(0.15), ev(0.2), ev(0.25), ev(0.3)];
		// 300 px wide: marks are 15 px apart, so every other one is a target.
		const kept = targetsApart(marks, 300, () => false);
		expect([...kept]).toEqual([0, 2, 4]);
		const withFail = targetsApart(marks, 300, (m) => m.at === 0.15);
		expect(withFail.has(1)).toBe(true);
		expect(withFail.has(0)).toBe(false); // too close to the failing eval
		// Wide enough: all of them.
		expect(targetsApart(marks, 1000, () => false).size).toBe(5);
		expect(targetsApart(marks, 0, () => false).size).toBe(5);
	});
});
