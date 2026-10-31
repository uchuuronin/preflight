import { describe, expect, it } from "vitest";
import { FRAMES, normInv, outcomes, Z90 } from "./hops";
import type { Estimate } from "./record/types";

const est = (point: number, lo: number, hi: number, status: Estimate["status"] = "predicted") => ({ point, lo, hi, status }) as Estimate;

describe("possible outcomes come only from the estimate's own interval", () => {
	it("inverts the normal CDF to textbook values", () => {
		expect(normInv(0.5)).toBeCloseTo(0, 9);
		expect(normInv(0.95)).toBeCloseTo(Z90, 7);
		expect(normInv(0.025)).toBeCloseTo(-1.959964, 5);
		expect(normInv(0.001)).toBeCloseTo(-3.090232, 5);
	});

	it("lands one frame in twenty past each end of the 90% interval, as the interval says", () => {
		const draws = outcomes(est(10, 6, 20));
		expect(draws).toHaveLength(FRAMES);
		const sorted = [...draws].sort((a, b) => a - b);
		expect(draws.filter((d) => d < 6)).toHaveLength(1);
		expect(draws.filter((d) => d > 20)).toHaveLength(1);
		expect(sorted[1]!).toBeGreaterThan(6);
		expect(sorted.at(-2)!).toBeLessThan(20);
		// The lopsided interval stays lopsided: half the frames below the point, half above.
		expect(draws.filter((d) => d < 10)).toHaveLength(FRAMES / 2);
		expect(sorted[FRAMES - 1]! - 10).toBeGreaterThan(10 - sorted[0]!);
	});

	it("draws each quantile once, in an order that jumps across the spread", () => {
		const draws = outcomes(est(10, 6, 20));
		expect(new Set(draws).size).toBe(FRAMES);
		const steps = draws.slice(1).map((d, i) => Math.abs(d - draws[i]!));
		const sorted = [...draws].sort((a, b) => a - b);
		const sweep = sorted.slice(1).map((d, i) => d - sorted[i]!);
		expect(Math.min(...steps)).toBeGreaterThan(Math.max(...sweep) / 4);
	});

	it("shows nothing when the estimate abstains or has no interval", () => {
		expect(outcomes(est(10, 6, 20, "abstain"))).toEqual([]);
		expect(outcomes(est(10, 10, 10))).toEqual([]);
		expect(outcomes(undefined)).toEqual([]);
	});

	it("stays at or above zero when the interval touches it", () => {
		const draws = outcomes(est(0.2, 0, 1));
		expect(Math.min(...draws)).toBeGreaterThanOrEqual(0);
		expect(draws).toHaveLength(FRAMES);
	});

	it("puts frames past a limit about as often as the interval does", () => {
		// The 95th percentile at 20: a limit there is passed by 5% of outcomes, one frame in 20.
		const over = (cap: number) => outcomes(est(10, 6, 20)).filter((d) => d > cap).length;
		expect(over(20)).toBe(1);
		expect(over(10)).toBe(FRAMES / 2);
	});
});
