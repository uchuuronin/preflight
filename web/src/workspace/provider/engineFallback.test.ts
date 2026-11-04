import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import raw from "../fixtures/lora-live-nov1.record.json";
import { currentKnobs } from "../record/model";
import { type FlightRecord, normaliseRecord, type Prediction } from "../record/types";
import { askEngine, engineVersion, mergeEngine, needsEngine, resetEngineCache, withEngine } from "./engineFallback";
import { GridProvider } from "./GridProvider";

let on = true;
let local = false;
vi.mock("../../lib/flags", () => ({ flag: (name: string) => (name === "c.engine_fallback" && on) || (name === "c.local_predict" && local) }));

const record = normaliseRecord(raw as unknown as FlightRecord);
const grid = new GridProvider();
const at = (k: Record<string, string | number | boolean>) => ({ ...currentKnobs(record), ...k });
// The engine's answer on this record (api/tests/test_predict.py): lr 2e-4 steps as 1e-4 does.
const engineStep = { point: 1.709, lo: 1.535, hi: 1.902, unit: "s", status: "predicted", model_id: "step-time@0.1.0", source: "exact" };

function answer(cells: number, status = 200) {
	return vi.fn(async (_url: string, init: RequestInit) => {
		const n = (JSON.parse(String(init.body)) as { cells: unknown[] }).cells.length;
		expect(n).toBe(cells);
		return new Response(JSON.stringify({ cells: Array.from({ length: n }, () => ({ "time.step": engineStep })) }), { status });
	});
}

beforeEach(() => {
	on = true;
	local = false;
	resetEngineCache();
});
afterEach(() => vi.unstubAllGlobals());

describe("the engine where the grids hold no value (Chiya, Nov 2)", () => {
	it("the grids abstain at lr 2e-4 on the LoRA record, for want of a grid value", () => {
		const p = grid.predict(record, at({ batch_size: 4, lr: 0.0002 }));
		expect(p["time.step"]?.status).toBe("abstain");
		expect(needsEngine(p)).toBe(true);
	});

	it("takes the engine's estimate, with its status, only for targets the grids abstained on", () => {
		const g: Prediction = {
			"time.step": { point: null, lo: null, hi: null, unit: "s", status: "abstain", reason: "off_grid" } as never,
			"memory.peak": { point: 9, lo: 8, hi: 10, unit: "bytes", status: "predicted" } as never,
			"risk.oom": { point: null, lo: null, hi: null, unit: "p", status: "abstain", reason: "no_model" } as never,
		};
		const e: Prediction = { "time.step": engineStep as never, "memory.peak": { point: 99 } as never, "risk.oom": { point: 0.5, status: "predicted" } as never };
		const m = mergeEngine(g, e);
		expect(m["time.step"]).toEqual(engineStep);
		expect(mergeEngine({ "risk.oom": { point: null, unit: "p", status: "abstain", reason: "off_grid" } as never }, { "risk.oom": { point: 1, unit: "p", status: "measured" } as never })["risk.oom"]?.status).toBe("predicted");
		expect(mergeEngine({ "time.step": { point: null, unit: "s", status: "abstain", reason: "off_grid" } as never }, { "time.step": { point: 1, unit: "ms", status: "predicted" } as never })["time.step"]?.status).toBe("abstain");
		expect(m["memory.peak"]?.point).toBe(9); // the grid's value stands
		expect(m["risk.oom"]?.status).toBe("abstain"); // an abstention for another reason stands
	});

	it("asks once per setting, then every read of predict has the engine's step", async () => {
		const fetch = answer(1);
		vi.stubGlobal("fetch", fetch);
		const p = withEngine(grid, "lora-qwen");
		const knobs = at({ batch_size: 4, lr: 0.0002 });
		expect(p.predict(record, knobs)["time.step"]?.status).toBe("abstain"); // never waits
		await askEngine("lora-qwen", record, [knobs, knobs]);
		await askEngine("lora-qwen", record, [knobs]);
		expect(fetch).toHaveBeenCalledTimes(1);
		expect(String(fetch.mock.calls[0]?.[0])).toBe("/v1/records/lora-qwen/predict");
		expect(withEngine(grid, "lora-qwen").predict(record, knobs)["time.step"]).toMatchObject({ point: 1.709, status: "predicted" });
	});

	it("asks again after a busy answer, and never for a live file or with the flag off", async () => {
		const busy = answer(1, 429);
		vi.stubGlobal("fetch", busy);
		const knobs = at({ batch_size: 4, lr: 0.0002 });
		await askEngine("lora-qwen", record, [knobs]);
		await askEngine("lora-qwen", record, [knobs]);
		expect(busy).toHaveBeenCalledTimes(2);
		const v = engineVersion();
		await askEngine("lora-qwen", record, [knobs]);
		expect(engineVersion()).toBe(v); // a busy reply doesn't re-render, so it can't loop
		await askEngine("live", record, [knobs]);
		on = false;
		await askEngine("lora-qwen", record, [at({ lr: 0.0003 })]);
		expect(busy).toHaveBeenCalledTimes(3);
		expect(withEngine(grid, "lora-qwen")).toBe(grid);
	});

	it("sends a record from disk with its settings when local predictions are on (PM, Nov 3)", async () => {
		local = true;
		const fetch = answer(1);
		vi.stubGlobal("fetch", fetch);
		const knobs = at({ batch_size: 4, lr: 0.0002 });
		await askEngine("live", record, [knobs]);
		expect(String(fetch.mock.calls[0]?.[0])).toBe("/v1/predict");
		const body = JSON.parse(String(fetch.mock.calls[0]?.[1]?.body)) as { record: { record_id: string }; cells: unknown[] };
		expect(body.record.record_id).toBe(record.record_id);
		expect(withEngine(grid, "live").predict(record, knobs)["time.step"]).toMatchObject({ point: 1.709, status: "predicted" });
	});
});
