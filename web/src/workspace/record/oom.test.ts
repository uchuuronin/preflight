import { describe, expect, it } from "vitest";
import devRaw from "../fixtures/dev-finetune.record.json";
import { GridProvider } from "../provider/GridProvider";
import { capacityBytes, currentKnobs } from "./model";
import { pOom, pOomFromMemory, Z90 } from "./oom";
import { type FlightRecord, normaliseRecord } from "./types";

const dev = normaliseRecord(devRaw as unknown as FlightRecord);
const provider = new GridProvider();

describe("P(OOM) from memory", () => {
	it("matches B's engine.memory.p_oom", () => {
		// Reference values from python/preflight/engine/memory.py, Oct 27.
		expect(pOom(70e9, 0.1 / Z90, 80e9)).toBeCloseTo(0.014031807479844774, 6);
		expect(pOom(79e9, 0.1 / Z90, 80e9)).toBeCloseTo(0.4180429804072979, 6);
		expect(pOom(90e9, 0.15 / Z90, 80e9)).toBeCloseTo(0.901747358719027, 6);
		expect(pOom(40e9, 0.2 / Z90, 80e9)).toBeLessThan(1e-7);
		expect(pOom(80e9, 0, 80e9)).toBe(0);
	});

	it("reads the spread off the wider side of a 90% band", () => {
		const e = { point: 70e9, lo: 70e9 * Math.exp(-0.05), hi: 70e9 * Math.exp(0.1), unit: "bytes", status: "predicted" as const, model_id: "m", source: "grid" as const };
		expect(pOomFromMemory(e, 80e9)).toBeCloseTo(0.014031807479844774, 6);
		expect(pOomFromMemory({ ...e, status: "abstain", point: null }, 80e9)).toBeNull();
		expect(pOomFromMemory(e, null)).toBeNull();
	});
});

describe("risk between grid points", () => {
	it("comes from interpolated memory, not from interpolating the probability", () => {
		// Sequence length 3000 sits between the grid's 2048 and 3072.
		const knobs = { ...currentKnobs(dev), eval_batch_size: 8, seq_len: 3000 };
		const p = provider.predict(dev, knobs);
		const derived = pOomFromMemory(p["memory.peak"], capacityBytes(dev, knobs));
		expect(p["risk.oom"]?.reason).toBe("from_memory");
		expect(p["risk.oom"]?.point).toBeCloseTo(derived ?? -1, 9);
	});

	it("keeps the engine's own value on a grid point", () => {
		const p = provider.predict(dev, { eval_batch_size: 8, seq_len: 3072 });
		expect(p["risk.oom"]?.reason).not.toBe("from_memory");
	});
});
