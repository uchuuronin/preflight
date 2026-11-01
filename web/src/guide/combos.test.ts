import { describe, expect, it } from "vitest";
import devRaw from "../workspace/fixtures/dev-finetune.record.json";
import type { Move, PredictionProvider } from "../workspace/provider/PredictionProvider";
import { currentKnobs } from "../workspace/record/model";
import { type FlightRecord, type KnobValue, type LoadedRecord, normaliseRecord, type Prediction } from "../workspace/record/types";
import { bestCombo, comboKeys, comboVisit, comboWords, rowsCut, rowsWords } from "./combos";
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

describe("ranking combined moves (PM, Nov 1)", () => {
	// The live LoRA record's choice: eval batch 4 with seq_len 512, or eval without gradients with
	// seq_len 256. Both come close; the first shortens the sequence less, and the eval knobs don't
	// change what the model learns.
	function two(): LoadedRecord {
		const grid = (target_id: string, a: { knob_id: string; values: KnobValue[] }, fixed: Record<string, KnobValue>) => ({
			id: `g.${target_id}.${a.knob_id}_seq_len.clears`,
			target_id,
			unit: target_id === "memory.peak" ? "bytes" : "probability",
			axes: [
				{ ...a, scale: "linear" },
				{ knob_id: "seq_len", values: [256, 512, 1024, 2048], scale: "log" },
			],
			fixed,
			model_id: "memory-analytical@test",
		});
		const grids = ["memory.peak", "risk.oom"].flatMap((id) => [
			grid(id, { knob_id: "eval_no_grad", values: [false, true] }, { eval_batch_size: 32 }),
			grid(id, { knob_id: "eval_batch_size", values: [4, 8, 16, 32] }, { eval_no_grad: false }),
		]);
		const probes = [
			{ ...dev.probes[0]!, knobs: { ...dev.probes[0]!.knobs, seq_len: 2048 }, extras: { dataset_max_tokens: 2048, dataset_p999_tokens: 1156, dataset_p50_tokens: 264, dataset_rows_scanned: 50724, dataset_rows_total: 50724 } },
		];
		return { ...dev, probes, predictions: { ...dev.predictions, grids } } as unknown as LoadedRecord;
	}
	const clears = (k: Readonly<Record<string, KnobValue>>) =>
		(k.eval_no_grad === true && Number(k.seq_len) <= 512) || (Number(k.eval_batch_size) <= 4 && Number(k.seq_len) <= 1024);
	const near: PredictionProvider = {
		...fake,
		predict(rec, knobs) {
			const full = { ...currentKnobs(rec), ...knobs };
			const est = (point: number, unit: string) => ({ point, lo: point, hi: point, unit, status: "predicted" as const, model_id: "t" });
			return { "memory.peak": est(clears(full) ? 18 * GIB : 30 * GIB, "bytes"), "risk.oom": est(clears(full) ? 0.025 : 0.97, "probability") } as unknown as Prediction;
		},
	};

	it("shortens the sequence as little as it can, and prefers the knobs that don't change what's learnt", () => {
		const r = two();
		// Eval without gradients with seq 512 is the fewest steps in all (1 + 2, B's order); eval
		// batch 4 with seq 1,024 is more (3 + 1) but shortens the sequence one step, not two.
		const c = bestCombo(r, near, currentKnobs(r), { kind: "min_cost" });
		expect(Object.fromEntries(c!.changes.map((x) => [x.knobId, x.to]))).toEqual({ eval_batch_size: 4, seq_len: 1024 });
		expect(c!.fit).toBe("tight");
		// From a seq_len between the axis's values, the farther cell still counts farther.
		const off = bestCombo(r, near, { ...currentKnobs(r), seq_len: 1536 }, { kind: "min_cost" });
		expect(Object.fromEntries(off!.changes.map((x) => [x.knobId, x.to]))).toEqual({ eval_batch_size: 4, seq_len: 1024 });
	});

	it("says what the shorter sequence cuts, by the rows A measured", () => {
		const r = two();
		const c = bestCombo(r, near, currentKnobs(r), { kind: "min_cost" })!;
		expect(c.cut).toMatchObject({ to: 1024, kind: "some", p999: 1156, p50: 264, rows: 50724 });
		expect(comboKeys(c).whyKey).toBe("guide.combo.why_rows");
		expect(comboWords(r, currentKnobs(r), c).rows_cut).toBe(
			"A sequence length of 1,024 cuts short every row longer than that. Half the rows are at most 264 tokens, and no more than 1 in 1,000 is longer than 1,156.",
		);
		const at = (to: number, extras: Record<string, number> = {}) =>
			rowsCut({ ...r, probes: [{ ...r.probes[0]!, extras: { ...r.probes[0]!.extras, ...extras } }] } as LoadedRecord, currentKnobs(r), [{ knobId: "seq_len", to }]);
		expect(at(1200)?.kind).toBe("few");
		expect(at(128)?.kind).toBe("most");
		expect(at(1900, { dataset_max_tokens: 1800 })?.kind).toBe("none");
		// A 99.9th percentile at the probe's own length was cut there: it says nothing about the rows.
		expect(at(1024, { dataset_p999_tokens: 2048 })).toBeNull();
		// A scan that stopped early speaks only for the rows it read.
		expect(at(1200, { dataset_rows_scanned: 30000 })).toBeNull();
		expect(rowsCut({ ...r, probes: [] } as LoadedRecord, currentKnobs(r), [{ knobId: "seq_len", to: 1024 }])).toBeNull();
		expect(rowsWords({ cut: at(1200) })).toBe("A sequence length of 1,200 cuts at most 1 row in 1,000 short (no more than 50 of 50,724): the 99.9th percentile row is 1,156 tokens.");
	});
});
