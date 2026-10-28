import { beforeEach, describe, expect, it } from "vitest";
import finetuneRaw from "../../../../schema/examples/finetune.json";
import type { KnobViewContext } from "../../registry";
import { currentKnobs } from "../../workspace/record/model";
import { type FlightRecord, normaliseRecord } from "../../workspace/record/types";
import { createKnobView, failingKnobs } from "./knobView";
import { worthChecking } from "./worthChecking";
import { resetTierForTests, setTier } from "./store";
import { registerExpertiseStrings } from "./strings";

const record = normaliseRecord(finetuneRaw as unknown as FlightRecord);

function ctx(over: Partial<KnobViewContext> = {}): KnobViewContext {
	return {
		record,
		lens: record.job.lens,
		knobs: currentKnobs(record),
		pinned: new Set(),
		highlight: new Set(),
		ranked: [],
		active: null,
		...over,
	};
}

/** As C's dock would pass it once it lists the view's findings under the row. */
function listed(over: Partial<KnobViewContext> = {}): KnobViewContext {
	return { ...ctx(over), findingsShown: true } as KnobViewContext;
}

function shownIds(view: ReturnType<typeof createKnobView>, c: KnobViewContext): string[] {
	return view.filter!(record.knobs, c).map((k) => k.id);
}

describe("failingKnobs", () => {
	it("takes the knobs of error and critical findings, not warnings", () => {
		expect([...failingKnobs(record)].sort()).toEqual(["eval_batch_size", "eval_no_grad"]);
	});
});

describe("expertise knob view", () => {
	beforeEach(() => {
		localStorage.clear();
		resetTierForTests();
		registerExpertiseStrings();
	});

	it("filters nothing for practitioners and experts", () => {
		const view = createKnobView();
		expect(shownIds(view, ctx())).toHaveLength(record.knobs.length);
		setTier("expert");
		expect(shownIds(view, ctx())).toHaveLength(record.knobs.length);
	});

	it("at the new level shows the failure's knobs and the goal's top two, filled from the baseline to five", () => {
		setTier("new");
		const shown = shownIds(createKnobView(), listed({ ranked: ["lora_r", "precision", "grad_accum"] }));
		expect(shown.sort()).toEqual(["batch_size", "eval_batch_size", "eval_no_grad", "lora_r", "precision"]);
	});

	it("on the golden fine-tune today shows the failures, the warning, then batch size and sequence length", () => {
		setTier("new");
		expect(shownIds(createKnobView(), ctx()).sort()).toEqual(["batch_size", "eval_batch_size", "eval_no_grad", "packing", "seq_len"]);
	});

	it("keeps a warning's knob in the dock while C's list isn't showing the warnings", () => {
		setTier("new");
		expect(shownIds(createKnobView(), ctx())).toContain("packing");
		expect(shownIds(createKnobView(), { ...ctx(), findingsShown: false } as KnobViewContext)).toContain("packing");
	});

	it("once C's list shows the warnings, reveals a warning's knob when opened and keeps it after closing", () => {
		setTier("new");
		const view = createKnobView();
		expect(shownIds(view, listed())).not.toContain("packing");
		expect(shownIds(view, listed({ active: "packing" }))).toContain("packing");
		expect(shownIds(view, listed({ active: null }))).toContain("packing");
	});

	it("hands C the warnings to list and the list's heading", () => {
		const view = createKnobView();
		expect(view.findings(ctx())).toEqual(worthChecking(record));
		expect(view.findingsKey(3)).toBe("addon.expertise.worth_checking.other");
	});

	it("keeps a knob the visitor moved and then put back", () => {
		setTier("new");
		const view = createKnobView();
		const moved = { ...currentKnobs(record), grad_accum: 8 };
		expect(shownIds(view, ctx({ knobs: moved }))).toContain("grad_accum");
		expect(shownIds(view, ctx())).toContain("grad_accum");
	});

	it("remembers touched knobs per record, not across records", () => {
		setTier("new");
		const view = createKnobView();
		const other = { ...record, record_id: "another" };
		const fresh = createKnobView().filter!(other.knobs, ctx({ record: other })).map((k) => k.id);
		shownIds(view, ctx({ knobs: { ...currentKnobs(record), grad_accum: 8 } }));
		expect(shownIds(view, ctx())).toContain("grad_accum");
		expect(view.filter!(other.knobs, ctx({ record: other })).map((k) => k.id)).toEqual(fresh);
	});


	it("names knobs plainly at the new level and by code name at expert", () => {
		const view = createKnobView();
		const k = record.knobs.find((x) => x.id === "eval_batch_size")!;
		expect(view.labelKey!(k, ctx())).toBe("knob.eval_batch_size");
		setTier("new");
		expect(view.labelKey!(k, ctx())).toBe("addon.expertise.label.eval_batch_size.new");
		setTier("expert");
		expect(view.labelKey!(k, ctx())).toBe("addon.expertise.label.eval_batch_size.expert");
	});

	it("names its show-all capsule", () => {
		expect(createKnobView().moreKey).toBe("addon.expertise.more");
	});
});
