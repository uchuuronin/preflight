import { describe, expect, it } from "vitest";
import goldenRaw from "../../../../schema/examples/finetune.json";
import devRaw from "../fixtures/dev-finetune.record.json";
import { fitOf } from "../status";
import { capacityBytes, currentKnobs } from "./model";
import { FAILS_AT, FITS_BELOW, gpuSpec, memorySegments, memoryTermBytes, oomStatus, resolveGpu } from "./terms";
import { type Estimate, type FlightRecord, normaliseRecord, type Prediction } from "./types";

const dev = normaliseRecord(devRaw as unknown as FlightRecord);
const golden = normaliseRecord(goldenRaw as unknown as FlightRecord);
const GIB = 1024 ** 3;

function est(point: number, unit = "bytes"): Estimate {
	return { point, lo: point, hi: point, unit, status: "predicted", model_id: "t", source: "grid", reason: null } as unknown as Estimate;
}

describe("OOM cut-offs match engine.memory", () => {
	it("fits below 1%, fails at 80% or more, tight between", () => {
		expect(FITS_BELOW).toBe(0.01);
		expect(FAILS_AT).toBe(0.8);
		expect(oomStatus(0.0099)).toBe("fits");
		expect(oomStatus(0.01)).toBe("tight");
		expect(oomStatus(0.79)).toBe("tight");
		expect(oomStatus(0.8)).toBe("fails");
	});

	it("drives the fit word everywhere", () => {
		const knobs = currentKnobs(dev);
		expect(fitOf(dev, { "risk.oom": est(0.005, "fraction") }, knobs)).toBe("fits");
		expect(fitOf(dev, { "risk.oom": est(0.3, "fraction") }, knobs)).toBe("tight");
		expect(fitOf(dev, { "risk.oom": est(0.95, "fraction") }, knobs)).toBe("fails");
	});
});

describe("memory terms", () => {
	it("reads B's ids and the older ones", () => {
		expect(memoryTermBytes({ "memory.eval_spike": est(3 * GIB) }, "eval_spike")).toBe(3 * GIB);
		expect(memoryTermBytes({ "memory.term.eval": est(2 * GIB) }, "eval_spike")).toBe(2 * GIB);
		expect(memoryTermBytes({ "memory.term.weights": est(GIB) }, "weights")).toBe(GIB);
		expect(memoryTermBytes({ "memory.eval_spike": est(-GIB) }, "eval_spike")).toBe(0);
	});

	it("groups all twelve terms into the ring's six slots", () => {
		const p: Prediction = {
			"memory.weights": est(10 * GIB),
			"memory.master_weights": est(1 * GIB),
			"memory.frozen_models": est(2 * GIB),
			"memory.context": est(1 * GIB),
			"memory.gradients": est(1 * GIB),
			"memory.optimizer": est(2 * GIB),
			"memory.comm_buffers": est(1 * GIB),
			"memory.activations": est(20 * GIB),
			"memory.logits": est(4 * GIB),
			"memory.kv_cache": est(6 * GIB),
			"memory.generation_engine": est(0),
			"memory.eval_spike": est(2 * GIB),
		};
		const { segments, breakdown } = memorySegments(p, 50 * GIB);
		expect(breakdown).toBe(true);
		const by = Object.fromEntries(segments.map((s) => [s.key, s.bytes / GIB]));
		expect(by).toEqual({ weights: 14, training: 4, activations: 20, logits: 4, generation: 6, eval: 2 });
		expect(segments.find((s) => s.key === "weights")?.parts.map((x) => x.term)).toEqual(["weights", "master_weights", "frozen_models", "context"]);
	});

	it("scales terms down to the peak when they overshoot it, and shows the rest as other when short", () => {
		const p: Prediction = { "memory.weights": est(30 * GIB), "memory.activations": est(30 * GIB) };
		const over = memorySegments(p, 40 * GIB).segments;
		expect(over.reduce((a, s) => a + s.bytes, 0)).toBeCloseTo(40 * GIB, 0);
		const short = memorySegments(p, 80 * GIB).segments;
		expect(short.find((s) => s.key === "other")?.bytes).toBeCloseTo(20 * GIB, 0);
	});

	it("falls back to one total segment without a breakdown", () => {
		expect(memorySegments({}, 12 * GIB).segments).toEqual([{ key: "total", bytes: 12 * GIB, slot: 1, parts: [] }]);
		expect(memorySegments({}, null).segments).toEqual([]);
	});
});

describe("GPU names from B's table", () => {
	it("resolves canonical names, aliases and Modal names", () => {
		expect(resolveGpu("A100-SXM-80GB")).toBe("A100-SXM-80GB");
		expect(resolveGpu("a100-sxm-80gb")).toBe("A100-SXM-80GB");
		expect(resolveGpu("A100-80GB")).toBe("A100-SXM-80GB");
		expect(resolveGpu("H100")).toBe("H100-SXM-80GB");
		expect(resolveGpu("not a gpu")).toBeNull();
		expect(gpuSpec("MI300X")?.vendor).toBe("amd");
	});

	it("gives capacity for the GPU the knobs choose, the record's own measurement first", () => {
		expect(capacityBytes(dev, currentKnobs(dev))).toBe(dev.env?.gpu?.memory_bytes);
		expect(capacityBytes(dev, { ...currentKnobs(dev), gpu_type: "L40S" })).toBe(48 * GIB);
		expect(capacityBytes(golden, currentKnobs(golden))).toBeGreaterThan(0);
	});
});
