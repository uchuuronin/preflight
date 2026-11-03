/**
 * Contract 0.6.4 `Grid.independent_of` (B, behind engine.carry_free_knobs): a grid answers at any value
 * of the knobs its target doesn't depend on. On the Nov 2 LoRA record the grids hold the learning rate
 * at 1e-4, so at 2e-4 the cockpit abstained, though the step time doesn't move with it.
 */
import { describe, expect, it } from "vitest";
import raw from "../fixtures/lora-live-nov1.record.json";
import { currentKnobs } from "../record/model";
import { type FlightRecord, type LoadedRecord, normaliseRecord } from "../record/types";
import { freeFor, freeKnobs, GridProvider } from "./GridProvider";

const base = normaliseRecord(raw as unknown as FlightRecord);
const FREE = ["lr", "warmup_steps", "lora_alpha", "lora_dropout"];

/** The record as B fills it with the flag on: every grid names the knobs its target ignores. */
function withFree(rec: LoadedRecord, free: string[]): LoadedRecord {
	const grids = rec.predictions.grids.map((g) => ({ ...g, independent_of: free }));
	return { ...rec, predictions: { ...rec.predictions, grids } } as LoadedRecord;
}

const grid = new GridProvider();
const at = (rec: LoadedRecord, k: Record<string, string | number | boolean>) => ({ ...currentKnobs(rec), ...k });

describe("Grid.independent_of (0.6.4)", () => {
	it("without it, a moved learning rate leaves every grid, as before", () => {
		const p = grid.predict(base, at(base, { batch_size: 2, lr: 0.0002 }));
		expect(p["time.total"]?.status).toBe("abstain");
	});

	it("with it, the batch-size grid answers at lr 2e-4 as at 1e-4, labelled predicted since no probe ran there", () => {
		const rec = withFree(base, FREE);
		const one = grid.predict(rec, at(rec, { batch_size: 2, lr: 0.0001 }));
		const two = grid.predict(rec, at(rec, { batch_size: 2, lr: 0.0002 }));
		expect(two["time.total"]?.status).not.toBe("abstain");
		expect(two["time.total"]?.point).toBeCloseTo(one["time.total"]!.point!, 9);
		expect(two["time.total"]?.status).not.toBe("measured");
	});

	it("only the knobs it names: a knob the target follows still leaves the grid", () => {
		const rec = withFree(base, ["lr"]);
		// warm-up is free in B's list but not in this grid's: moved with the batch size, no grid answers.
		const p = grid.predict(rec, at(rec, { batch_size: 2, lr: 0.0002, warmup_steps: 999 }));
		expect(p["time.total"]?.status).toBe("abstain");
	});

	it("with only a free knob moved, a target with no grid of its own keeps B's estimate (review, Nov 2)", () => {
		const rec = withFree(base, FREE);
		const here = grid.predict(rec, currentKnobs(rec));
		const moved = grid.predict(rec, at(rec, { lr: 0.0002 }));
		// the LoRA record has no time.step grid: before, it went to "Not measured here" while time.total stayed
		expect(rec.predictions.grids.some((g) => g.target_id === "time.step")).toBe(false);
		expect(moved["time.step"]?.status).toBe("predicted");
		expect(moved["time.step"]?.point).toBeCloseTo(here["time.step"]!.point!, 9);
		expect(moved["time.total"]?.status).not.toBe("abstain");
		// without independent_of the old behaviour stands
		expect(grid.predict(base, at(base, { lr: 0.0002 }))["time.step"]?.status).toBe("abstain");
	});

	it("follows B's families: a knob free for time but not memory leaves memory's estimate", () => {
		const grids = base.predictions.grids.map((g) => ({ ...g, independent_of: g.target_id.startsWith("memory") || g.target_id === "risk.oom" ? [] : ["lr"] }));
		const rec = { ...base, predictions: { ...base.predictions, grids } } as LoadedRecord;
		const moved = grid.predict(rec, at(rec, { lr: 0.0002 }));
		expect(moved["time.step"]?.status).toBe("predicted");
		expect(moved["memory.peak"]?.status).toBe("abstain");
	});

	it("a grid with the free knob on an axis doesn't take it out of the family's free set (review, Nov 2)", () => {
		const rec = withFree(base, FREE);
		// B never names an axis knob: on g.time.total.lr the list leaves lr out
		const grids = rec.predictions.grids.map((g) => (g.axes.some((a) => a.knob_id === "lr") ? { ...g, independent_of: FREE.filter((k) => k !== "lr") } : g));
		const r2 = { ...rec, predictions: { ...rec.predictions, grids } } as LoadedRecord;
		expect([...freeFor(r2, "time.step")].sort()).toEqual([...FREE].sort());
		expect(grid.predict(r2, at(r2, { lr: 0.0002 }))["time.step"]?.status).toBe("predicted");
	});

	it("reads an absent or null list as no free knobs", () => {
		expect(freeKnobs({}).size).toBe(0);
		expect(freeKnobs({ independent_of: null }).size).toBe(0);
		expect([...freeKnobs({ independent_of: ["lr"] })]).toEqual(["lr"]);
	});
});
