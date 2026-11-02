import { describe, expect, it } from "vitest";
import en from "../i18n/en.json";
import type { LoadedRecord, Prediction } from "../workspace/record/types";
import { changeWords } from "./changeWords";

const cost = { id: "cost.total", unit: "usd", label_key: "target.cost.total", readout: "cost" as const };
const est = (point: number, lo: number, hi: number) => ({ point, lo, hi, unit: "usd", status: "predicted", model_id: "m", source: "grid" }) as never;

function ctx(after: [number, number, number]) {
	const before: Prediction = { "cost.total": est(10, 9, 11) };
	return {
		record: {} as LoadedRecord,
		estimates: before,
		knobs: { gpu_type: "L4" },
		provider: { predict: (_r: unknown, k: Record<string, unknown>) => (k.gpu_type === "L4" ? before : { "cost.total": est(...after) }) as Prediction },
	};
}

describe("how a move's effect on the cost is said (B, Nov 2)", () => {
	const move = { knobId: "gpu_type", to: "B200-SXM-180GB", delta: -0.17 };

	it("says about the same cost when the two cost ranges overlap", () => {
		const w = changeWords(ctx([9.83, 9.1, 10.6]), move, "L4", cost);
		expect(w.change).toBe(en["guide.change.same_cost"]);
		expect(w.would_change).toBe(en["guide.would_change.same_cost"]);
		expect(w.change).not.toMatch(/%|-|\+/);
	});

	it("gives the signed change when the ranges don't overlap", () => {
		const w = changeWords(ctx([6, 5.5, 6.5]), { ...move, delta: -4 }, "L4", cost);
		expect(w.change).toMatch(/^changes /);
		expect(w.change).not.toBe(en["guide.change.same_cost"]);
	});
});
