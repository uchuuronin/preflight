import { describe, expect, it } from "vitest";
import { fmt } from "./fmt";

describe("fmt", () => {
	it("formats the engine's gpu_hours like gpu_h (B's request, Nov 1)", () => {
		expect(fmt(18.3, "gpu_hours")).toBe(fmt(18.3, "gpu_h"));
		expect(fmt(18.3, "gpu_hours")).toMatch(/GPU-hours/);
	});

	it("shows fractions of a cent in usd_small, cents from ten cents up", () => {
		expect(fmt(0.0042, "usd_small")).toMatch(/0\.0042/);
		expect(fmt(0.8, "usd_small")).toMatch(/0\.80$/);
		expect(fmt(0, "usd_small")).toMatch(/0\.00$/);
	});
});
