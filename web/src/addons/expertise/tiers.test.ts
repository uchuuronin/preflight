import { describe, expect, it } from "vitest";
import agentRaw from "../../../../schema/examples/agent.json";
import finetuneRaw from "../../../../schema/examples/finetune.json";
import pretrainRaw from "../../../../schema/examples/pretrain.json";
import devRaw from "../../workspace/fixtures/dev-finetune.record.json";
import { GridProvider } from "../../workspace/provider/GridProvider";
import { knobChoices } from "../../workspace/record/model";
import { type FlightRecord, type KnobValue, type ObjectiveKind, normaliseRecord } from "../../workspace/record/types";
import { BASELINE, MAX_NEW_KNOBS, NEW_TOP_SENSITIVE, labelKeyFor, visibleKnobs } from "./tiers";

const FT_KNOBS = ["batch_size", "seq_len", "grad_accum", "eval_batch_size", "lora_r", "gradient_checkpointing", "precision", "packing", "eval_no_grad", "gpu_type"];

const base = {
	lens: "finetune" as const,
	knobIds: FT_KNOBS,
	pinned: new Set<string>(),
	highlighted: new Set<string>(),
	ranked: [] as string[],
};

describe("visibleKnobs", () => {
	it("shows every knob to practitioners and experts, in record order", () => {
		expect(visibleKnobs({ ...base, tier: "practitioner" })).toEqual(FT_KNOBS);
		expect(visibleKnobs({ ...base, tier: "expert" })).toEqual(FT_KNOBS);
	});

	it("shows the lens baseline at the new level", () => {
		expect(visibleKnobs({ ...base, tier: "new" })).toEqual(["batch_size", "seq_len", "gpu_type"]);
	});

	it("adds the top two knobs by impact at the new level", () => {
		expect(NEW_TOP_SENSITIVE).toBe(2);
		const shown = visibleKnobs({ ...base, tier: "new", ranked: ["eval_batch_size", "packing", "lora_r"] });
		expect(shown).toEqual(["batch_size", "seq_len", "eval_batch_size", "packing", "gpu_type"]);
	});

	it("always shows pinned knobs and knobs the guide is on", () => {
		const shown = visibleKnobs({
			...base,
			tier: "new",
			pinned: new Set(["precision"]),
			highlighted: new Set(["eval_no_grad"]),
		});
		expect(shown).toContain("precision");
		expect(shown).toContain("eval_no_grad");
	});

	it("skips baseline ids the record does not declare", () => {
		expect(visibleKnobs({ ...base, tier: "new", knobIds: ["seq_len", "lora_r"] })).toEqual(["seq_len"]);
	});

	it("falls back to the top knobs by impact for a lens with no baseline", () => {
		const shown = visibleKnobs({ ...base, lens: "gnn", tier: "new", knobIds: ["fan_out", "hops", "seed_batch"], ranked: ["hops", "fan_out"] });
		expect(shown).toEqual(["fan_out", "hops"]);
	});

	it("has a baseline for every lens A has sample records for", () => {
		for (const lens of ["finetune", "pretrain", "agent"] as const) expect(BASELINE[lens]?.length).toBe(3);
	});
});

describe("labelKeyFor", () => {
	const known = new Set(["addon.expertise.label.eval_batch_size.new"]);
	it("uses the plain label at the new level when there is one", () => {
		expect(labelKeyFor("eval_batch_size", "knob.eval_batch_size", "new", known)).toBe("addon.expertise.label.eval_batch_size.new");
	});
	it("keeps the record's label otherwise", () => {
		expect(labelKeyFor("eval_batch_size", "knob.eval_batch_size", "practitioner", known)).toBe("knob.eval_batch_size");
		expect(labelKeyFor("lora_r", "knob.lora_r", "new", known)).toBe("knob.lora_r");
	});
});

// The decision behind NEW_TOP_SENSITIVE, kept as a regression check (docs/decisions/expertise-new-level.md).
// Random settings over each sample record's gridded knobs, three goals; the new level must show the knob
// that clears a predicted failure every time, and the best knob for any goal at least 95% of the time,
// with no more than MAX_NEW_KNOBS on screen. Rerun when B's engine replaces the fixture predictions.
describe("new level on the sample records (Monte Carlo)", () => {
	const GOALS: ObjectiveKind[] = ["min_cost", "min_time", "fit_memory"];
	const SAMPLES = 300;

	function rng(seed: number) {
		return () => {
			seed = (seed * 1664525 + 1013904223) >>> 0;
			return seed / 2 ** 32;
		};
	}

	it("keeps the fix in view and the screen small", { timeout: 60_000 }, () => {
		const provider = new GridProvider();
		let best = 0;
		let bestHit = 0;
		let fixes = 0;
		let fixHit = 0;
		let most = 0;
		for (const raw of [devRaw, finetuneRaw, pretrainRaw, agentRaw]) {
			const record = normaliseRecord(raw as unknown as FlightRecord);
			const onAxis = new Set(record.predictions.grids.flatMap((g) => g.axes.map((a) => a.knob_id)));
			const rand = rng(7);
			for (let i = 0; i < SAMPLES; i++) {
				const knobs: Record<string, KnobValue> = {};
				for (const k of record.knobs) {
					const c = onAxis.has(k.id) ? knobChoices(record, k) : [];
					knobs[k.id] = c.length ? (c[Math.floor(rand() * c.length)] as KnobValue) : k.current;
				}
				const ranked = provider.sensitivity(record, { kind: "min_cost" }, { knobs }).map((x) => x.knob_id);
				const shown = new Set(
					visibleKnobs({
						lens: record.job.lens,
						tier: "new",
						knobIds: record.knobs.map((k) => k.id),
						pinned: new Set(),
						highlighted: new Set(),
						ranked,
					}),
				);
				most = Math.max(most, shown.size);
				for (const goal of GOALS) {
					const moves = provider.moves(record, { kind: goal }, { knobs });
					const top = moves[0]?.knobId;
					if (top) {
						best++;
						if (shown.has(top)) bestHit++;
					}
					const fix = moves.find((m) => m.clearsFailure)?.knobId;
					if (fix) {
						fixes++;
						if (shown.has(fix)) fixHit++;
					}
				}
			}
		}
		expect(best).toBeGreaterThan(500);
		expect(fixes).toBeGreaterThan(100);
		expect(fixHit / fixes).toBe(1);
		expect(bestHit / best).toBeGreaterThanOrEqual(0.95);
		expect(most).toBeLessThanOrEqual(MAX_NEW_KNOBS);
	});
});
