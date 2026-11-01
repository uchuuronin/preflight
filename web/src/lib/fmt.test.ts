import { describe, expect, it } from "vitest";
import { fmt, fmtDelta } from "./fmt";

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

describe("fmtDelta", () => {
	it("says a change too small to show is almost nothing, never +US$0 (live site, Nov 1)", () => {
		expect(fmtDelta(0.001, "usd")).toBe("almost nothing");
		expect(fmtDelta(-0.001, "usd")).toBe("almost nothing");
		expect(fmtDelta(2, "usd")).toMatch(/^\+US\$2/);
		expect(fmtDelta(0, "usd")).toBe("No change");
	});
});
