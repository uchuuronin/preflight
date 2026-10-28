import { describe, expect, it } from "vitest";
import { hasKey } from "../../lib/i18n";
import census from "./census.json";
import { registerExpertiseStrings } from "./strings";
import { BASELINE, MAX_NEW_KNOBS, NEW_TOP_SENSITIVE, labelKeyFor, visibleKnobs } from "./tiers";

const FT_KNOBS = ["batch_size", "seq_len", "grad_accum", "eval_batch_size", "lora_r", "gradient_checkpointing", "precision", "packing", "eval_no_grad", "gpu_type"];

const base = {
	lens: "finetune" as const,
	knobIds: FT_KNOBS,
	pinned: new Set<string>(),
	highlighted: new Set<string>(),
	touched: new Set<string>(),
	opened: new Set<string>(),
	failing: new Set<string>(),
	warnings: new Set<string>(),
	warningsListed: true,
	ranked: [] as string[],
};

describe("visibleKnobs", () => {
	it("shows every knob to practitioners and experts, in record order", () => {
		expect(visibleKnobs({ ...base, tier: "practitioner" })).toEqual(FT_KNOBS);
		expect(visibleKnobs({ ...base, tier: "expert" })).toEqual(FT_KNOBS);
	});

	it("fills the new level from the lens baseline, most common first, up to five", () => {
		expect(visibleKnobs({ ...base, tier: "new" })).toEqual(["batch_size", "seq_len", "grad_accum", "eval_batch_size", "lora_r"]);
	});

	it("adds the top two knobs by impact at the new level", () => {
		expect(NEW_TOP_SENSITIVE).toBe(2);
		const shown = visibleKnobs({ ...base, tier: "new", ranked: ["eval_batch_size", "packing", "lora_r"] });
		expect(shown).toEqual(["batch_size", "seq_len", "grad_accum", "eval_batch_size", "packing"]);
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

	it("picks at most five knobs by itself: failures, then the goal's top two, then the baseline", () => {
		expect(MAX_NEW_KNOBS).toBe(5);
		const shown = visibleKnobs({
			...base,
			tier: "new",
			failing: new Set(["eval_no_grad", "eval_batch_size"]),
			ranked: ["lora_r", "precision", "packing"],
		});
		expect(shown).toHaveLength(5);
		expect(shown).toEqual(["batch_size", "eval_batch_size", "lora_r", "precision", "eval_no_grad"]);
	});

	it("shows warnings' knobs only once the visitor opens them", () => {
		expect(visibleKnobs({ ...base, tier: "new" })).not.toContain("packing");
		expect(visibleKnobs({ ...base, tier: "new", opened: new Set(["packing"]) })).toContain("packing");
	});

	it("keeps warnings' knobs in the dock, never dropped, while no list shows the warnings", () => {
		const shown = visibleKnobs({
			...base,
			tier: "new",
			failing: new Set(["eval_no_grad", "eval_batch_size"]),
			warnings: new Set(["packing"]),
			warningsListed: false,
			ranked: ["lora_r", "precision", "grad_accum"],
		});
		expect(shown).toContain("packing");
		expect(shown).toEqual(["eval_batch_size", "lora_r", "precision", "packing", "eval_no_grad"]);
	});

	it("hides warnings' knobs once a list shows the warnings", () => {
		const shown = visibleKnobs({ ...base, tier: "new", warnings: new Set(["packing"]), warningsListed: true });
		expect(shown).not.toContain("packing");
	});

	it("never drops a failure's knob to stay under the cap", () => {
		const six = ["batch_size", "seq_len", "grad_accum", "eval_batch_size", "lora_r", "packing"];
		const shown = visibleKnobs({ ...base, tier: "new", failing: new Set(six), ranked: ["precision"] });
		expect(shown).toEqual(six);
	});

	it("adds what the visitor opened, moved or pinned on top of the five", () => {
		const shown = visibleKnobs({
			...base,
			tier: "new",
			failing: new Set(["eval_no_grad", "eval_batch_size"]),
			ranked: ["lora_r", "precision"],
			opened: new Set(["packing"]),
			touched: new Set(["grad_accum"]),
			pinned: new Set(["gradient_checkpointing"]),
		});
		expect(shown).toHaveLength(8);
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
		expect(visibleKnobs({ ...base, tier: "new", knobIds: ["seq_len", "eval_no_grad"] })).toEqual(["seq_len"]);
	});

	it("falls back to the top knobs by impact for a lens with no baseline", () => {
		const shown = visibleKnobs({ ...base, lens: "gnn", tier: "new", knobIds: ["fan_out", "hops", "seed_batch"], ranked: ["hops", "fan_out"] });
		expect(shown).toEqual(["fan_out", "hops"]);
	});

	it("has a baseline for every lens A has sample records for", () => {
		for (const lens of ["finetune", "pretrain", "agent"] as const) expect(BASELINE[lens]?.length).toBeGreaterThanOrEqual(3);
	});

	it("takes each training lens's baseline from the census ranking", () => {
		for (const [lens, row] of Object.entries(census.lenses)) expect(BASELINE[lens as keyof typeof BASELINE]).toEqual(row.ranking);
		expect(Object.keys(census.lenses).sort()).toEqual(["finetune", "pretrain", "rl", "vision"]);
	});

	it("names only knob ids the app knows, in every baseline", () => {
		registerExpertiseStrings();
		const named = (id: string) => hasKey(`knob.${id}`) || hasKey(`addon.expertise.label.${id}.new`);
		for (const ids of Object.values(BASELINE)) for (const id of ids ?? []) expect(named(id), id).toBe(true);
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
