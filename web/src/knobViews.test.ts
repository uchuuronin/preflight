import { describe, expect, it, vi } from "vitest";
import devRaw from "./workspace/fixtures/dev-finetune.record.json";
import { applyKnobViews, type KnobViewContext, type KnobViewEntry } from "./registry";
import { currentKnobs } from "./workspace/record/model";
import { type FlightRecord, type Knob, normaliseRecord } from "./workspace/record/types";

const dev = normaliseRecord(devRaw as unknown as FlightRecord);
const ctx = (over: Partial<KnobViewContext> = {}): KnobViewContext => ({
	record: dev,
	lens: dev.job.lens,
	knobs: currentKnobs(dev),
	pinned: new Set(),
	highlight: new Set(),
	ranked: [],
	active: null,
	...over,
});
const onlyFirstTwo: KnobViewEntry = { id: "new", filter: (ks: Knob[]) => ks.slice(0, 2), moreKey: "addon.more" };
const ids = (ks: Knob[]) => ks.map((k) => k.id);

describe("knob views (D's expertise dial)", () => {
	it("shows every knob when no view is registered", () => {
		expect(applyKnobViews(dev.knobs, ctx(), []).shown).toHaveLength(dev.knobs.length);
	});

	it("filters the dock and reports what it hid", () => {
		const r = applyKnobViews(dev.knobs, ctx(), [onlyFirstTwo]);
		expect(ids(r.shown)).toEqual(ids(dev.knobs.slice(0, 2)));
		expect(r.hidden).toHaveLength(dev.knobs.length - 2);
		expect(r.moreKey).toBe("addon.more");
	});

	it("never hides a pinned, guided, open or moved knob", () => {
		const last = dev.knobs[dev.knobs.length - 1] as Knob;
		const third = dev.knobs[2] as Knob;
		const fourth = dev.knobs[3] as Knob;
		const moved = dev.knobs[4] as Knob;
		const knobs = { ...currentKnobs(dev), [moved.id]: "something else" };
		const r = applyKnobViews(dev.knobs, ctx({ pinned: new Set([last.id]), highlight: new Set([third.id]), active: fourth.id, knobs }), [onlyFirstTwo]);
		for (const k of [last, third, fourth, moved]) expect(ids(r.shown)).toContain(k.id);
	});

	it("uses a view's label keys only when the key exists", () => {
		const view: KnobViewEntry = { id: "labels", labelKey: (k) => (k.id === "batch_size" ? "known.key" : "missing.key") };
		const r = applyKnobViews(dev.knobs, ctx(), [view], (key) => key === "known.key");
		expect(r.labelKeys).toEqual({ batch_size: "known.key" });
	});

	it("skips a view that throws and keeps every knob", () => {
		const err = vi.spyOn(console, "error").mockImplementation(() => {});
		const r = applyKnobViews(dev.knobs, ctx(), [{ id: "boom", filter: () => { throw new Error("x"); } }]);
		expect(r.shown).toHaveLength(dev.knobs.length);
		expect(err).toHaveBeenCalledTimes(1);
		err.mockRestore();
	});

	it("ignores knobs a view invents and respects flags", () => {
		const fake = { ...(dev.knobs[0] as Knob), id: "not_a_knob" };
		expect(ids(applyKnobViews(dev.knobs, ctx(), [{ id: "inv", filter: () => [fake] }]).shown)).not.toContain("not_a_knob");
		expect(applyKnobViews(dev.knobs, ctx(), [{ ...onlyFirstTwo, flag: "flag.that.is.off" }]).shown).toHaveLength(dev.knobs.length);
	});
});
