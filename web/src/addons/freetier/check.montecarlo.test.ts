import { describe, expect, it } from "vitest";
import type { Estimate } from "../../workspace/record/types";
import { checkPreset } from "./check";
import { PRESETS, type Preset } from "./presets";

// calibration: simulated. The truth below is drawn from the same interval form the rule reads, so
// this shows the rule is consistent with its inputs, not that it is accurate; its numbers are never
// quoted. It stays as a guard against a rule that ignores the interval (the point-estimate rule
// fails it). Accuracy comes from the zoo's real runs (docs/decisions/freetier-one-session.md).
// The decision behind the one-session rule, run against checkPreset itself. Predicted run times 2 to 20 h on a 12 h Colab session; true time spread
// around the prediction as the records' 90% intervals say (hi/point 1.15, or 1.40 for one in
// four) [measured from A's fixture records, not real runs], optionally 1.5x wider than claimed;
// 5 to 30 min of setup each session [guess, as SETUP_S].

function rng(seed: number) {
	return () => {
		seed = (seed * 1664525 + 1013904223) >>> 0;
		return seed / 2 ** 32;
	};
}

function normal(rand: () => number): number {
	const u = Math.max(rand(), 1e-12);
	return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

function simulate(widen: number, n = 40_000) {
	const colab = PRESETS.find((p) => p.id === "colab") as Preset;
	const rand = rng(11);
	const mem = { point: 8e9, lo: 7e9, hi: 9e9, unit: "bytes", status: "predicted", model_id: "mc", source: "grid", level: 0.9 } as Estimate;
	let says = 0;
	let saysRight = 0;
	let fits = 0;
	let fitsFound = 0;
	for (let i = 0; i < n; i++) {
		const point = Math.exp(Math.log(2 * 3600) + rand() * Math.log(10));
		const ratio = rand() < 0.75 ? 1.15 : 1.4;
		const sigma = Math.log(ratio) / 1.645;
		const truth = point * Math.exp(normal(rand) * sigma * widen);
		const setup = (5 + rand() * 25) * 60;
		const time = { point, lo: point / ratio, hi: point * ratio, unit: "s", status: "predicted", model_id: "mc", source: "grid", level: 0.9 } as Estimate;
		const one = checkPreset(colab, { gpu: "T4", memory: mem, time }).kind === "one_session";
		const really = truth + setup <= colab.sessionCapS;
		if (one) {
			says++;
			if (really) saysRight++;
		}
		if (really) {
			fits++;
			if (one) fitsFound++;
		}
	}
	return { precision: saysRight / says, recall: fitsFound / fits };
}

describe("one-session rule (Monte Carlo)", () => {
	it("is almost never wrong when it says one session", { timeout: 60_000 }, () => {
		const { precision, recall } = simulate(1);
		expect(precision).toBeGreaterThanOrEqual(0.995);
		expect(recall).toBeGreaterThanOrEqual(0.85);
	});

	it("stays above 99% even if the intervals are 1.5x too narrow", { timeout: 60_000 }, () => {
		expect(simulate(1.5).precision).toBeGreaterThanOrEqual(0.99);
	});
});
