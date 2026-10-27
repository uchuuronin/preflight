import { beforeEach, describe, expect, it } from "vitest";
import finetuneRaw from "../../../../schema/examples/finetune.json";
import type { KnobViewContext } from "../../registry";
import { currentKnobs } from "../../workspace/record/model";
import { type FlightRecord, normaliseRecord } from "../../workspace/record/types";
import { createKnobView, failingKnobs } from "./knobView";
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

	it("at the new level shows the baseline, the top two for the goal and the failure's knobs", () => {
		setTier("new");
		const shown = shownIds(createKnobView(), ctx({ ranked: ["lora_r", "precision", "packing"] }));
		expect(shown).toEqual(expect.arrayContaining(["batch_size", "seq_len", "lora_r", "precision", "eval_batch_size", "eval_no_grad"]));
		expect(shown).not.toContain("packing");
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
		shownIds(view, ctx({ knobs: { ...currentKnobs(record), grad_accum: 8 } }));
		const other = { ...record, record_id: "another" };
		expect(view.filter!(other.knobs, ctx({ record: other })).map((k) => k.id)).not.toContain("grad_accum");
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
