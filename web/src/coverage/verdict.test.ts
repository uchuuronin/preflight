import { describe, expect, it } from "vitest";
import data from "./coverage.json";
import { verdict } from "./verdict";

const s = (n: number, hits: number, w: number[] | null, bar = 0.88) => ({ rate_bar: bar, rate_bar_source: "cited", n, hits, rate: n ? hits / n : null, wilson95: w, abstained: 0 });

describe("coverage verdicts", () => {
	it("calls nothing from no checks", () => {
		expect(verdict("step_time", s(0, 0, null))).toBe("unchecked");
	});

	it("needs the whole Wilson interval past the bar to call a pass or a failure", () => {
		expect(verdict("interval", s(3, 3, [0.4385, 1]))).toBe("unsettled");
		expect(verdict("interval", s(3, 0, [0, 0.5615]))).toBe("below");
		expect(verdict("interval", s(100, 97, [0.9154, 0.9897]))).toBe("meets");
	});

	it("treats one missed out-of-memory as the verdict, and otherwise says none missed so far", () => {
		expect(verdict("oom", s(5, 4, [0.3755, 0.9638], 1))).toBe("missed");
		expect(verdict("oom", s(1, 1, [0.2065, 1], 1))).toBe("no_misses");
	});

	it("reads the generated file the page ships, bars and their sources included", () => {
		for (const [metric, summary] of Object.entries(data.summary)) {
			expect(summary.rate_bar).toBeGreaterThan(0);
			expect(["cited", "guess", "derived", "measured"]).toContain(summary.rate_bar_source);
			expect(["unchecked", "unsettled", "meets", "below", "missed", "no_misses"]).toContain(verdict(metric, summary));
		}
	});
});
