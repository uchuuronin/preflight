// @vitest-environment jsdom
import { afterEach, describe, expect, it } from "vitest";
import goldenRaw from "../../../schema/examples/finetune.json";
import agentRaw from "./fixtures/agent-eval-oct31.record.json";
import campaignRaw from "./fixtures/campaign-grid.record.json";
import { planFromRecord } from "./campaign/plan";
import { exportPlan } from "./export/plan";
import { GridProvider } from "./provider/GridProvider";
import { currentKnobs, targetFor } from "./record/model";
import { type FlightRecord, normaliseRecord } from "./record/types";
import { appliesHere, loadRate, rateFactor, saveRate, tableRate, withRate } from "./rate";

const gpuRec = normaliseRecord(goldenRaw as unknown as FlightRecord);
const apiRec = normaliseRecord(agentRaw as unknown as FlightRecord);
const base = new GridProvider();

afterEach(() => localStorage.clear());

describe("the table's rate", () => {
	it("is B's row for the GPU at this setting, and the token mix's blended rate for an API run", () => {
		const g = tableRate(gpuRec, currentKnobs(gpuRec));
		expect(g?.kind).toBe("gpu");
		expect(g && g.kind === "gpu" ? g.usdPerHour : null).toBe(0.8);
		const a = tableRate(apiRec, currentKnobs(apiRec));
		expect(a?.kind).toBe("tokens");
		// The record's own blended rate (job_cost.lines[0].rate_usd), from the same rows and mix.
		expect(a && a.kind === "tokens" ? a.usdPerToken : null).toBeCloseTo(apiRec.job_cost!.lines[0]!.rate_usd!, 8);
	});
});

describe("the visitor's rate", () => {
	const knobs = currentKnobs(gpuRec);
	const cost = targetFor(gpuRec, "cost")!.id;

	it("rescales every dollar estimate by visitor over table, interval and all", () => {
		const before = base.predict(gpuRec, knobs)[cost]!;
		const after = withRate(base, { kind: "gpu", gpu: "L4", usdPerHour: 1.6 }).predict(gpuRec, knobs)[cost]!;
		expect(after.point).toBeCloseTo(before.point! * 2, 9);
		expect(after.lo).toBeCloseTo(before.lo! * 2, 9);
		expect(after.hi).toBeCloseTo(before.hi! * 2, 9);
	});

	it("takes $0 (an assigned GPU): every dollar is zero, and no cost move is offered as a saving", () => {
		const free = withRate(base, { kind: "gpu", gpu: "L4", usdPerHour: 0 });
		expect(free.predict(gpuRec, knobs)[cost]!.point).toBe(0);
		const moves = free.moves(gpuRec, { kind: "min_cost" }, { knobs });
		expect(moves.filter((m) => m.targetId === cost)).toHaveLength(0);
	});

	it("leaves another GPU at the table's rate", () => {
		const o = { kind: "gpu" as const, gpu: "L4", usdPerHour: 0 };
		const gpuKnob = gpuRec.knobs.find((k) => k.id === "gpu_type");
		const other = (gpuKnob?.domain.values ?? []).find((v) => v !== "L4");
		if (!other) return;
		expect(rateFactor(gpuRec, { ...knobs, gpu_type: other }, o)).toBe(1);
		expect(appliesHere(gpuRec, { ...knobs, gpu_type: other }, o)).toBe(false);
	});

	it("prices an API run's input and output tokens at their own rates, mixed as the sample mixed them", () => {
		const k = currentKnobs(apiRec);
		const c = targetFor(apiRec, "cost")!.id;
		const before = base.predict(apiRec, k)[c]!.point!;
		// Double both rates: double the cost.
		const after = withRate(base, { kind: "tokens", model: "muse-spark-1.3", inputPerM: 2.5, outputPerM: 8.5 }).predict(apiRec, k)[c]!.point!;
		expect(after).toBeCloseTo(before * 2, 6);
	});

	it("goes with the export plan, so a config or sbatch file quotes the same rate", () => {
		const k = currentKnobs(gpuRec);
		expect(exportPlan(gpuRec, k, {}, null, null).rate).toMatchObject({ kind: "gpu", usdPerHour: 0.8, source: "table" });
		expect(exportPlan(gpuRec, k, {}, null, null, { kind: "gpu", gpu: "L4", usdPerHour: 0 }).rate).toMatchObject({ usdPerHour: 0, source: "visitor" });
	});

	it("rescales the campaign's totals too", () => {
		const rec = normaliseRecord(campaignRaw as unknown as FlightRecord);
		const plain = planFromRecord(rec.campaign!, rec.models)!;
		const half = planFromRecord(rec.campaign!, rec.models, () => 0.5)!;
		expect(half.usd.point).toBeCloseTo(plain.usd.point! / 2, 9);
		expect(half.gpu_hours).toEqual(plain.gpu_hours);
	});

	it("is kept per job in this browser, and a malformed entry is ignored", () => {
		saveRate("lora-qwen", { kind: "gpu", gpu: "L4", usdPerHour: 0 });
		expect(loadRate("lora-qwen")).toEqual({ kind: "gpu", gpu: "L4", usdPerHour: 0 });
		expect(loadRate("agent-eval")).toBeNull();
		localStorage.setItem("pf.rate.v1", JSON.stringify({ x: { kind: "gpu", gpu: "L4", usdPerHour: -1 } }));
		expect(loadRate("x")).toBeNull();
		saveRate("lora-qwen", null);
		expect(loadRate("lora-qwen")).toBeNull();
	});
});
