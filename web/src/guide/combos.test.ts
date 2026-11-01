import { describe, expect, it } from "vitest";
import devRaw from "../workspace/fixtures/dev-finetune.record.json";
import type { Move, PredictionProvider } from "../workspace/provider/PredictionProvider";
import { currentKnobs } from "../workspace/record/model";
import { type FlightRecord, type KnobValue, type LoadedRecord, normaliseRecord, type Prediction } from "../workspace/record/types";
import { bestCombo, comboVisit } from "./combos";
import { nextStep } from "./ruleTour";

const GIB = 2 ** 30;
const dev = normaliseRecord(devRaw as unknown as FlightRecord);

/**
 * The live LoRA record's shape (Nov 1): at eval batch 32 the eval runs out; eval batch 4 alone
 * still leaves training too big at seq 1,024; eval batch 4 with seq 512 fits. B's multi-axis grid
 * covers eval_batch_size x seq_len with eval_no_grad held true.
 */
function lora(): LoadedRecord {
	const grid = (target_id: string) => ({
		id: `g.${target_id}.eval_batch_size+seq_len`,
		target_id,
		unit: target_id === "memory.peak" ? "bytes" : "probability",
		axes: [
			{ knob_id: "eval_batch_size", values: [4, 8, 32], scale: "log" },
			{ knob_id: "seq_len", values: [512, 1024], scale: "log" },
		],
		fixed: { eval_no_grad: true },
		model_id: "memory-analytical@test",
	});
	return { ...dev, predictions: { ...dev.predictions, grids: [grid("memory.peak"), grid("risk.oom")] } } as unknown as LoadedRecord;
}

/** Peak memory as the combination gives it: what the grid would hold, as a function. */
function peakOf(k: Readonly<Record<string, KnobValue>>): number {
	const evalPart = k.eval_no_grad === true ? 0.5 : 6;
	const train = k.seq_len === 512 ? 15 : 24;
	return (train + evalPart * (Number(k.eval_batch_size) / 4)) * GIB;
}

const fake: PredictionProvider = {
	name: "grid",
	ready: () => true,
	predict(_r, knobs) {
		const full = { ...currentKnobs(lora()), ...knobs };
		const peak = peakOf(full);
		const risk = peak > 22 * GIB ? 0.99 : 0.001;
		const est = (point: number, unit: string) => ({ point, lo: point, hi: point, unit, status: "predicted" as const, model_id: "t" });
		return { "memory.peak": est(peak, "bytes"), "risk.oom": est(risk, "probability"), "time.total": { ...est(0, "s"), point: null, status: "abstain" as const } } as unknown as Prediction;
	},
	sensitivity: () => [],
	moves: (): Move[] => [],
};

describe("combined moves", () => {
	it("finds the fewest knobs that clear the failure together, holding the grid's fixed knobs", () => {
		const r = lora();
		const c = bestCombo(r, fake, currentKnobs(r), { kind: "min_cost" });
		expect(c).not.toBeNull();
		const set = Object.fromEntries(c!.changes.map((x) => [x.knobId, x.to]));
		// eval_no_grad comes from the grid's fixed values: it is part of the setting it predicts.
		expect(set).toMatchObject({ seq_len: 512, eval_no_grad: true });
		expect(c!.risk).toBeLessThan(0.01);
		expect(c!.peak).toBeLessThan(22 * GIB);
	});

	it("is the tour's step when no single knob clears it, offered once", () => {
		const r = lora();
		const ctx = { record: r, provider: fake, knobs: currentKnobs(r), pinned: new Set<string>(), objective: { kind: "min_cost" as const }, visited: new Set<string>(["sample:more"]) };
		const step = nextStep(ctx);
		expect(step.kind).toBe("combo");
		expect(step.textKey).toBe("guide.combo.line");
		const again = nextStep({ ...ctx, visited: new Set([...ctx.visited, comboVisit(step.combo!)]) });
		// Another cell that also fits may follow; the one turned down never comes back.
		expect(again.combo ? comboVisit(again.combo) : "").not.toBe(comboVisit(step.combo!));
	});

	it("never changes a pinned knob", () => {
		const r = lora();
		expect(bestCombo(r, fake, currentKnobs(r), { kind: "min_cost" }, { pinned: new Set(["seq_len"]) })).toBeNull();
	});

	it("offers nothing without a multi-axis grid", () => {
		const flat = { ...dev, predictions: { ...dev.predictions, grids: dev.predictions.grids.filter((g) => g.axes.length < 2) } } as LoadedRecord;
		expect(bestCombo(flat, fake, currentKnobs(flat), { kind: "min_cost" })).toBeNull();
	});
});

describe("the live guide hears about a combined move", () => {
	it("returns it beside the ranking when no single knob fits", async () => {
		const { runBrowserCall } = await import("./liveGuide");
		const { fakeActions } = await import("./testkit");
		const r = lora();
		const f = fakeActions();
		const ctx = { record: r, provider: fake, actions: f.actions, knobs: currentKnobs(r), pinned: new Set<string>(), requestApproval: () => {} };
		const out = runBrowserCall({ call_id: "s", name: "sensitivity", arguments: { objective: "min_cost" }, runs_in: "browser" }, ctx, false);
		expect(out.tool === "sensitivity" && out.combined?.knobs).toMatchObject({ seq_len: 512, eval_no_grad: true });
	});
});
