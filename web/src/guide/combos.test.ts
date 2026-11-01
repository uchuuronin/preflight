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
 * still leaves training too big at seq 1,024; eval batch 4 with seq 512 fits. B's pair grid covers
 * eval_batch_size x seq_len with every other knob held where it is.
 */
function lora(): LoadedRecord {
	const grid = (target_id: string) => ({
		id: `g.${target_id}.eval_batch_size+seq_len.clears`,
		target_id,
		unit: target_id === "memory.peak" ? "bytes" : "probability",
		axes: [
			{ knob_id: "eval_batch_size", values: [4, 8, 32], scale: "log" },
			{ knob_id: "seq_len", values: [512, 1024], scale: "log" },
		],
		fixed: { eval_no_grad: false },
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
	it("finds the knobs that clear the failure together", () => {
		const r = lora();
		const c = bestCombo(r, fake, currentKnobs(r), { kind: "min_cost" });
		expect(c).not.toBeNull();
		const set = Object.fromEntries(c!.changes.map((x) => [x.knobId, x.to]));
		expect(set).toEqual({ eval_batch_size: 4, seq_len: 512 });
		expect(c!.risk).toBeLessThan(0.01);
		expect(c!.fit).toBe("fits");
		expect(c!.peak).toBeLessThan(22 * GIB);
		// Words in the grid's order (its axes, then what it holds), and a knob named as not
		// enough alone only because its own change was predicted not to clear.
		const ids = c!.changes.map((x) => x.knobId);
		const order = ["eval_batch_size", "seq_len"];
		expect(ids).toEqual([...ids].sort((a, b) => order.indexOf(a) - order.indexOf(b)));
		expect(c!.alone).not.toBeNull();
		expect(fake.predict(r, { ...currentKnobs(r), [c!.alone!]: c!.changes.find((x) => x.knobId === c!.alone)!.to })["risk.oom"]!.point).toBeGreaterThanOrEqual(0.01);
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

	it("skips a pair grid once the visitor has moved a knob it holds", () => {
		const r = lora();
		expect(bestCombo(r, fake, { ...currentKnobs(r), eval_no_grad: true }, { kind: "min_cost" })).toBeNull();
	});

	it("offers a tight combination only when it beats the best single move", () => {
		const r = lora();
		const near: PredictionProvider = {
			...fake,
			predict(rec, knobs) {
				const out = fake.predict(rec, knobs);
				const p = out["risk.oom"]!;
				return { ...out, "risk.oom": { ...p, point: p.point! < 0.01 ? 0.03 : p.point } } as Prediction;
			},
		};
		expect(bestCombo(r, near, currentKnobs(r), { kind: "min_cost" }, { below: 0.02 })).toBeNull();
		expect(bestCombo(r, near, currentKnobs(r), { kind: "min_cost" }, { below: 0.04 })?.fit).toBe("tight");
	});

	it("never changes a pinned knob", () => {
		const r = lora();
		expect(bestCombo(r, fake, currentKnobs(r), { kind: "min_cost" }, { pinned: new Set(["seq_len"]) })).toBeNull();
	});

	it("reads only B's pair grids, not a record's own multi-axis grids", () => {
		const r = lora();
		const own = { ...r, predictions: { ...r.predictions, grids: r.predictions.grids.map((g) => ({ ...g, id: g.id.replace(/\.clears$/, "") })) } } as LoadedRecord;
		expect(bestCombo(own, fake, currentKnobs(own), { kind: "min_cost" })).toBeNull();
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
		expect(out.tool === "sensitivity" && out.combined).toMatchObject({ knobs: { eval_batch_size: 4, seq_len: 512 }, fit: "fits" });
	});
});

describe("a combination B's pair grid clears only to 'unlikely' (P at most 0.05)", () => {
	it("is offered as tight, to measure first", () => {
		const r = lora();
		const near: PredictionProvider = {
			...fake,
			predict(rec, knobs) {
				const out = fake.predict(rec, knobs);
				const p = out["risk.oom"]!;
				return { ...out, "risk.oom": { ...p, point: p.point! < 0.01 ? 0.03 : p.point } } as Prediction;
			},
		};
		const c = bestCombo(r, near, currentKnobs(r), { kind: "min_cost" });
		expect(c?.fit).toBe("tight");
		expect(c?.risk).toBeCloseTo(0.03);
	});
});
