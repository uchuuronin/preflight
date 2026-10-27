import { describe, expect, it } from "vitest";
import { BASELINE, NEW_TOP_SENSITIVE, labelKeyFor, visibleKnobs } from "./tiers";

const FT_KNOBS = ["batch_size", "seq_len", "grad_accum", "eval_batch_size", "lora_r", "gradient_checkpointing", "precision", "packing", "eval_no_grad", "gpu_type"];

const base = {
	lens: "finetune" as const,
	knobIds: FT_KNOBS,
	pinned: new Set<string>(),
	highlighted: new Set<string>(),
	touched: new Set<string>(),
	failing: new Set<string>(),
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

	it("always shows the knobs behind a current failure, even off the grid and unranked", () => {
		const shown = visibleKnobs({ ...base, tier: "new", failing: new Set(["eval_no_grad", "eval_batch_size"]), ranked: ["packing", "lora_r"] });
		expect(shown).toContain("eval_no_grad");
		expect(shown).toContain("eval_batch_size");
	});

	it("keeps a knob the visitor touched, even after they put it back", () => {
		expect(visibleKnobs({ ...base, tier: "new", touched: new Set(["lora_r"]) })).toContain("lora_r");
	});

	it("takes its top knobs from the ranking it is given, in that order", () => {
		const byTime = visibleKnobs({ ...base, tier: "new", ranked: ["precision", "packing", "lora_r"] });
		expect(byTime).toContain("precision");
		expect(byTime).toContain("packing");
		expect(byTime).not.toContain("lora_r");
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
