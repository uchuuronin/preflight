import { describe, expect, it } from "vitest";
import record from "../fixtures/handmade-finetune.record.json";
import type { FlightRecord } from "../record/types";
import { halfToFloat } from "./float16";
import { GridProvider } from "./GridProvider";
import { currentKnobs } from "./PredictionProvider";

const rec = record as unknown as FlightRecord;
const provider = new GridProvider();
const GIB = 1024 ** 3;

describe("float16", () => {
	it("decodes normal, subnormal and special values", () => {
		expect(halfToFloat(0x3c00)).toBe(1);
		expect(halfToFloat(0xc000)).toBe(-2);
		expect(halfToFloat(0x7bff)).toBe(65504);
		expect(halfToFloat(0x0001)).toBeCloseTo(5.96e-8, 10);
		expect(halfToFloat(0x7c00)).toBe(Number.POSITIVE_INFINITY);
		expect(Number.isNaN(halfToFloat(0x7e00))).toBe(true);
	});
});

describe("GridProvider.predict", () => {
	it("flags the planted eval OOM at the record's own settings", () => {
		const p = provider.predict(rec, currentKnobs(rec));
		expect(p.p_oom?.point).toBeGreaterThan(0.9);
		expect(p.peak_memory?.point).toBeGreaterThan(80 * GIB);
	});

	it("applies the grid scale so bytes come back in bytes", () => {
		const p = provider.predict(rec, { eval_batch_size: 8 });
		const peak = p.peak_memory?.point ?? 0;
		expect(peak).toBeGreaterThan(60 * GIB);
		expect(peak).toBeLessThan(72 * GIB);
		expect(p.peak_memory?.lo).toBeLessThanOrEqual(peak);
		expect(p.peak_memory?.hi).toBeGreaterThanOrEqual(peak);
	});

	it("interpolates between grid points", () => {
		const a = provider.predict(rec, { eval_batch_size: 8, seq_len: 1024 }).mem_activations?.point ?? 0;
		const b = provider.predict(rec, { eval_batch_size: 8, seq_len: 1536 }).mem_activations?.point ?? 0;
		const mid = provider.predict(rec, { eval_batch_size: 8, seq_len: 1280 }).mem_activations?.point ?? 0;
		expect(mid).toBeGreaterThan(a);
		expect(mid).toBeLessThan(b);
		expect(mid).toBeCloseTo((a + b) / 2, -6);
	});

	it("marks values outside the probed range as extrapolated", () => {
		const inside = provider.predict(rec, { eval_batch_size: 8, gpu: "A100-80GB" });
		const outside = provider.predict(rec, { eval_batch_size: 8, gpu: "H100" });
		expect(["predicted", "measured"]).toContain(inside.time_to_finish?.status);
		expect(outside.time_to_finish?.status).toBe("extrapolated");
	});

	it("abstains past 2x the largest probed batch x sequence length", () => {
		const p = provider.predict(rec, { eval_batch_size: 8, batch_size: 16, seq_len: 4096 });
		expect(p.peak_memory?.status).toBe("abstain");
		expect(p.peak_memory?.reason).toBe("outside_trusted_range");
	});

	it("abstains off-grid instead of guessing", () => {
		const p = provider.predict(rec, { lora_rank: 64 });
		expect(p.peak_memory?.status).toBe("abstain");
		expect(p.peak_memory?.reason).toBe("off_grid");
	});

	it("labels an exact probe point as measured", () => {
		const p = provider.predict(rec, {
			seq_len: 512,
			batch_size: 4,
			eval_batch_size: 8,
			gpu: "A100-80GB",
			grad_ckpt: false,
		});
		expect(p.peak_memory?.status).toBe("measured");
	});
});

describe("GridProvider.sensitivity", () => {
	it("puts the OOM fix first when the run is failing", () => {
		const ranked = provider.sensitivity(rec, "cost");
		expect(ranked[0]?.knob_id).toBe("eval_batch_size");
	});

	it("never moves a pinned knob", () => {
		const ranked = provider.sensitivity(rec, "cost", { pinned: new Set(["eval_batch_size"]) });
		expect(ranked.map((r) => r.knob_id)).not.toContain("eval_batch_size");
	});

	it("ranks cost levers once the run fits", () => {
		const ranked = provider.sensitivity(rec, "cost", { knobs: { eval_batch_size: 8 } });
		expect(ranked.length).toBeGreaterThan(0);
		for (const r of ranked) expect(r.delta).toBeLessThan(0);
		for (const r of ranked) {
			const p = provider.predict(rec, { eval_batch_size: 8, [r.knob_id]: r.best_value });
			expect(p.p_oom?.point ?? 0).toBeLessThanOrEqual(0.1);
		}
	});
});

describe("GridProvider.sensitivity, failing runs", () => {
	it("picks the cheapest of the moves that clear an OOM, not the one with the lowest residual risk", () => {
		const top = provider.sensitivity(rec, "cost")[0];
		expect(top?.knob_id).toBe("eval_batch_size");
		expect(top?.best_value).toBe(8);
	});
});
