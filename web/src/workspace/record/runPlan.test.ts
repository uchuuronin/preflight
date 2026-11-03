import { describe, expect, it } from "vitest";
import gnnRaw from "../fixtures/gnn_pyg-a10-nov2.record.json";
import { currentKnobs, runPlan } from "./model";
import { type FlightRecord, normaliseRecord } from "./types";

const gnn = normaliseRecord(gnnRaw as unknown as FlightRecord);

describe("runPlan, a map's step count that counts epochs (B's engine fix, Nov 3)", () => {
	it("keeps the record's own steps at its own values", () => {
		expect(runPlan(gnn).totalSteps).toBe(gnn.job.script_map?.training?.max_steps);
	});
	it("scales with epochs and with the examples a step takes, as the engine does", () => {
		const base = currentKnobs(gnn);
		const steps = gnn.job.script_map!.training!.max_steps!;
		const epochs = Number(base.epochs);
		expect(runPlan(gnn, { ...base, epochs: epochs * 2 }).totalSteps).toBe(steps * 2);
		expect(runPlan(gnn, { ...base, batch_size: Number(base.batch_size) * 2 }).totalSteps).toBe(Math.ceil(steps / 2));
	});
	it("epochs 0 is no steps, as in B's engine", () => {
		expect(runPlan(gnn, { ...currentKnobs(gnn), epochs: 0 }).totalSteps).toBe(0);
	});

	it("never rescales a trainer's enforced step cap", () => {
		const hf = { ...gnn, job: { ...gnn.job, script_map: { ...gnn.job.script_map!, framework: "hf_trainer" } } } as typeof gnn;
		const base = currentKnobs(hf);
		expect(runPlan(hf, { ...base, epochs: Number(base.epochs) * 2 }).totalSteps).toBe(hf.job.script_map!.training!.max_steps);
	});
});
