// @vitest-environment jsdom
import { afterEach, describe, expect, it } from "vitest";
import goldenRaw from "../../../schema/examples/finetune.json";
import devRaw from "./fixtures/dev-finetune.record.json";
import agentRaw from "./fixtures/agent-eval-oct31.record.json";
import campaignRaw from "./fixtures/campaign-grid-golden.record.json";
import { planFromRecord } from "./campaign/plan";
import { exportPlan } from "./export/plan";
import { GridProvider } from "./provider/GridProvider";
import { currentKnobs, targetFor } from "./record/model";
import { type FlightRecord, normaliseRecord } from "./record/types";
import { appliesHere, estimateFactor, loadRate, rateFactor, saveRate, tableRate, withRate } from "./rate";

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
		expect(a && a.kind === "tokens" ? a.usdPerToken : null).toBeCloseTo(apiRec.job_cost?.lines?.[0]?.rate_usd ?? Number.NaN, 8);
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
		// Moves that clear or lower a predicted failure stay (whatever the rate); none saves money.
		const moves = free.moves(gpuRec, { kind: "min_cost" }, { knobs });
		const riskNow = base.predict(gpuRec, knobs)["risk.oom"]?.point ?? 0;
		for (const m of moves.filter((x) => x.targetId === cost)) {
			expect(m.delta).toBeGreaterThanOrEqual(0);
			expect(m.clearsFailure || (m.risk !== null && m.risk < riskNow)).toBe(true);
		}
	});

	it("keeps the moves that clear a predicted failure, whatever the rate (review, Nov 1)", () => {
		const dev = normaliseRecord(devRaw as unknown as FlightRecord);
		const k = currentKnobs(dev);
		const clearing = base.moves(dev, { kind: "min_cost" }, { knobs: k }).filter((m) => m.clearsFailure);
		expect(clearing.length).toBeGreaterThan(0);
		const gpu = String(k.gpu_type);
		for (const rate of [0, 2.5]) {
			const got = withRate(base, { kind: "gpu", gpu, usdPerHour: rate }).moves(dev, { kind: "min_cost" }, { knobs: k });
			expect(got.filter((m) => m.clearsFailure).map((m) => m.knobId)).toEqual(clearing.map((m) => m.knobId));
			expect(got[0]?.clearsFailure).toBe(true);
		}
	});

	it("keeps risk-lowering moves on a failing run with nothing that clears it, priced at the visitor's rate", () => {
		const fail = "risk.oom";
		const costId = targetFor(gpuRec, "cost")!.id;
		const k = currentKnobs(gpuRec);
		const est = (point: number, unit: string) => ({ point, lo: point, hi: point, unit, status: "predicted", model_id: "m", source: "exact" }) as never;
		// A stub engine: the run is failing (90%), and the only move lowers the risk but costs more.
		const stub = {
			name: "grid" as const,
			ready: () => true,
			predict: (_r: unknown, kn: Record<string, unknown>) => ({ [costId]: est(kn.batch_size === 1 ? 2 : 1, "usd"), [fail]: est(kn.batch_size === 1 ? 0.5 : 0.9, "probability") }),
			sensitivity: () => [],
			moves: () => [{ knobId: "batch_size", to: 1, targetId: costId, delta: 1, clearsFailure: false, fit: "tight" as const, risk: 0.5, relative: 1 }],
		};
		const rec = { ...gpuRec, predictions: { ...gpuRec.predictions, targets: [...gpuRec.predictions.targets.filter((t) => t.id !== fail), { id: fail, label_key: "x", unit: "probability", readout: "risk" } as never] } };
		const got = withRate(stub as never, { kind: "gpu", gpu: "L4", usdPerHour: 1.6 }).moves(rec, { kind: "min_cost" }, { knobs: k });
		expect(got.map((m) => m.knobId)).toEqual(["batch_size"]);
		// Its cost change at the visitor's rate (twice the table's): 2 x (2 - 1).
		expect(got[0]!.delta).toBeCloseTo(2, 9);
	});

	it("leaves another GPU at the table's rate", () => {
		const o = { kind: "gpu" as const, gpu: "L4", usdPerHour: 0 };
		const gpuKnob = gpuRec.knobs.find((k) => k.id === "gpu_type");
		const other = (gpuKnob?.domain?.values ?? []).find((v) => v !== "L4");
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

	it("scales an API run's per-token prices, never a rate field, which B doesn't write for token costs (Nov 1)", () => {
		const k = currentKnobs(apiRec);
		const c = targetFor(apiRec, "cost")!.id;
		const o = { kind: "tokens" as const, model: "muse-spark-1.3", inputPerM: 2.5, outputPerM: 8.5 };
		const plain = base.predict(apiRec, k)[c]!;
		// Even if an estimate carried a rate, the token path is the per-token ratio.
		const withField = { ...plain, rate: { quantity: "tokens.total", usd_per_unit: 99, unit: "tokens" } } as never;
		expect(estimateFactor(apiRec, k, withField, o)).toBeCloseTo(rateFactor(apiRec, k, o), 12);
		expect(rateFactor(apiRec, k, o)).toBeCloseTo(2, 6);
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

	it("divides by the rate an estimate names (0.6.0 Estimate.rate), not the table's, for this GPU only", () => {
		const k = currentKnobs(gpuRec);
		const e = { point: 10, lo: 8, hi: 12, unit: "usd", status: "predicted", model_id: "m", source: "exact", rate: { quantity: "cost.gpu_hours", usd_per_unit: 2, unit: "gpu_hours" } } as never;
		// Priced at US$2/h; the visitor pays US$1/h: half.
		expect(estimateFactor(gpuRec, k, e, { kind: "gpu", gpu: "L4", usdPerHour: 1 })).toBeCloseTo(0.5, 9);
		// No rate on the estimate: the table's (US$0.80/h on L4).
		const plain = { point: 10, lo: 8, hi: 12, unit: "usd", status: "predicted", model_id: "m", source: "exact" } as never;
		expect(estimateFactor(gpuRec, k, plain, { kind: "gpu", gpu: "L4", usdPerHour: 1.6 })).toBeCloseTo(2, 9);
		// A rate for another GPU leaves the estimate's dollars alone; no visitor rate, no change.
		expect(estimateFactor(gpuRec, k, e, { kind: "gpu", gpu: "H100", usdPerHour: 1 })).toBe(1);
		expect(estimateFactor(gpuRec, k, e, null)).toBe(1);
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
