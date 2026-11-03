/**
 * Contract 0.6.4 `Grid.independent_of` (B, behind engine.carry_free_knobs): a grid answers at any value
 * of the knobs its target doesn't depend on. On the Nov 2 LoRA record the grids hold the learning rate
 * at 1e-4, so at 2e-4 the cockpit abstained, though the step time doesn't move with it.
 */
import { describe, expect, it } from "vitest";
import raw from "../fixtures/lora-live-nov1.record.json";
import { currentKnobs } from "../record/model";
import { type FlightRecord, type LoadedRecord, normaliseRecord } from "../record/types";
import { freeKnobs, GridProvider } from "./GridProvider";

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

	it("reads an absent or null list as no free knobs", () => {
		expect(freeKnobs({}).size).toBe(0);
		expect(freeKnobs({ independent_of: null }).size).toBe(0);
		expect([...freeKnobs({ independent_of: ["lr"] })]).toEqual(["lr"]);
	});
});
