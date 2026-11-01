import { describe, expect, it } from "vitest";
import smoke from "../../../../api/tests/fixtures/a-smoke-l4-main-4.record.json";
import type { Estimate } from "../../workspace/record/types";
import { checkPreset, gpuMemoryBytes } from "./check";
import real from "./fixtures/real-l4-runs.json";
import { PRESETS } from "./presets";

// D4 on A's measured L4 records (Nov 1): retrieval, GNN and vision (origin/remote-runs, estimates
// copied unchanged into fixtures/real-l4-runs.json) and the Oct 28 LoRA smoke record. None of them
// can score "finishes in one session": the engine abstains on time.total for all three ("the run's
// length isn't known"), the LoRA record has no estimates, and no full run on these was timed to the
// end (the LoRA full run, a-gt-lora-l4-5, runs out of memory at step 200). So the simulated check
// (check.montecarlo.test.ts) stays; what these records can check is that D4 never claims a session
// count without a trusted time, and judges memory against the measured peak.

type Run = { lens: string; gpu: string; estimates: Record<string, Estimate | undefined> };
const runs = (real as unknown as { runs: Run[] }).runs;

describe("free-tier verdicts on A's real L4 records", () => {
	it("covers the three new records, each on an L4", () => {
		expect(runs.map((r) => r.lens)).toEqual(["retrieval", "gnn", "vision"]);
		for (const r of runs) expect(r.gpu).toBe("L4");
	});

	it("never says one session, or how many, while the engine has no trusted run time", () => {
		for (const r of runs) {
			expect(r.estimates["time.total"]?.status).toBe("abstain");
			for (const p of PRESETS) {
				const v = checkPreset(p, { gpu: r.gpu, memory: r.estimates["memory.peak"], time: r.estimates["time.total"], cost: r.estimates["cost.total"] });
				expect(["one_session", "sessions"]).not.toContain(v.kind);
			}
		}
	});

	it("judges memory on the record's own peak: the GNN's measured 5.0 GiB (4.5 to 5.6) fits every free GPU", () => {
		const gnn = runs.find((r) => r.lens === "gnn") as Run;
		const peak = gnn.estimates["memory.peak"] as Estimate;
		expect(peak.status).toBe("measured");
		for (const p of PRESETS) {
			const v = checkPreset(p, { gpu: gnn.gpu, memory: peak, time: gnn.estimates["time.total"] });
			expect(v.kind).not.toBe("no_memory");
			// The tier's roomiest GPU really holds it, by B's capacity table.
			expect(Math.max(...p.gpus.map((g) => gpuMemoryBytes(g) ?? 0))).toBeGreaterThan(peak.hi as number);
		}
	});

	it("abstains where the engine gave no trusted memory (vision) and on a record with no estimates (LoRA)", () => {
		const vision = runs.find((r) => r.lens === "vision") as Run;
		const lora = (smoke as unknown as { predictions: { estimates?: Record<string, Estimate> } }).predictions.estimates ?? {};
		for (const p of PRESETS) {
			expect(checkPreset(p, { gpu: "L4", memory: vision.estimates["memory.peak"], time: vision.estimates["time.total"] }).kind).toBe("abstain");
			expect(checkPreset(p, { gpu: "L4", memory: lora["memory.peak"], time: lora["time.total"] }).kind).toBe("abstain");
		}
	});
});
