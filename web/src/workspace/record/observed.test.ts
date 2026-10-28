import { describe, expect, it } from "vitest";
import type { Probe } from "./types";
import { probeObserves } from "./observed";

const probe = {
	id: "p",
	knobs: {},
	status: "ok",
	steps: 6,
	memory: { peak_allocated_bytes: 10, by_term: { weights: 4, activations: 3 } },
	step_time: { n_steps: 6, mean_s: 0.4, p95_s: 0.5, by_phase: { forward: 0.3 } },
	throughput: { tokens_per_s: 1200, mfu: null },
	cost: { usd: 0.1 },
} as unknown as Probe;

describe("what a probe observed", () => {
	it("counts the memory peak, attributed terms, phases and throughput", () => {
		for (const id of ["memory.peak", "memory.weights", "memory.term.activations", "time.phase.forward", "time.step", "throughput.tokens_per_s"]) {
			expect(probeObserves(probe, id), id).toBe(true);
		}
	});

	it("never counts totals, risks, missing fields or failed probes", () => {
		for (const id of ["time.total", "cost.total", "risk.oom", "memory.logits", "throughput.mfu", "time.phase.eval", "tokens.total"]) {
			expect(probeObserves(probe, id), id).toBe(false);
		}
		expect(probeObserves({ ...probe, status: "oom" } as Probe, "memory.peak")).toBe(false);
		expect(probeObserves({ ...probe, steps: 1 } as Probe, "memory.peak")).toBe(false);
	});
});
