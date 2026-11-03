/**
 * The range test's curve (D's spec, Nov 2), on D's two real curves from the A10 (GitHub run 41490183749):
 * the LoRA zoo script (steepest fall 7.9e-5, blew up at 7.6e-3, its own rate 1e-4) and nanoGPT.
 */
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { t } from "../../lib/i18n";
import real from "../fixtures/lr-range-nov2.json";
import { isRangeTest, nearestRate, rangeTestOf, rateVerdict, readRangeTest, smoothed } from "./rangeTest";
import { RangeTestCurve } from "./RangeTest";

afterEach(cleanup);

const lora = real.finetune_lora_hf;
const probe = (extras: Record<string, number>) => ({ id: "p1-lr", knobs: {}, extras });
const CHOICES = [0.0000125, 0.000025, 0.00005, 0.0001, 0.0002, 0.0004, 0.0008];

describe("reading the range test off the record", () => {
	it("reads D's LoRA curve: 68 points with rates, and the three marked rates", () => {
		const r = readRangeTest(probe(lora.extras), "A10");
		expect(r.checked).toBe(true);
		expect(r.points).toHaveLength(68);
		expect(r.points.every((p) => p.lr !== null && p.lr > 0)).toBe(true);
		expect(r.suggested).toBeCloseTo(7.92e-5, 6);
		expect(r.diverged).toBeCloseTo(7.56e-3, 4);
		expect(r.minLoss).toBeCloseTo(7.74e-4, 5);
	});

	it("finds the probe on a record by its mark, not its id, and only that one", () => {
		const rec = { probes: [{ id: "p1", knobs: {}, extras: {} }, { id: "p9", knobs: {}, extras: lora.extras }], env: { gpu: { name: "A10" } } } as never;
		expect(rangeTestOf(rec)?.probeId).toBe("p9");
		expect(rangeTestOf({ probes: [{ id: "p1", knobs: {}, extras: {} }], env: null } as never)).toBeNull();
		expect(isRangeTest({ extras: lora.extras } as never)).toBe(true);
		expect(isRangeTest({ extras: {} } as never)).toBe(false);
	});

	it("keeps the losses but no rates when the ramp isn't checked, with the reason", () => {
		const unchecked: Record<string, number> = { lr_range: 1, "lr_range.unverified": 1 };
		for (const [k, v] of Object.entries(lora.extras)) if (k.startsWith("lr_range.loss.")) unchecked[k] = v as number;
		const r = readRangeTest(probe(unchecked));
		expect(r.checked).toBe(false);
		expect(r.points).toHaveLength(68);
		expect(r.points.every((p) => p.lr === null)).toBe(true);
		expect(r.suggested).toBeNull();
		expect(r.reasons).toEqual(["unverified"]);
	});

	it("gives no suggestion on a flat curve, even if a rate were left", () => {
		const r = readRangeTest(probe({ ...lora.extras, "lr_range.flat": 1 }));
		expect(r.flat).toBe(true);
		expect(r.suggested).toBeNull();
	});
});

describe("the smoothed curve", () => {
	it("matches D's _smooth (probes/lr_range.py) to six places", () => {
		const want = [2.564041, 2.46066, 2.385725, 2.276674, 2.179804, 2.11536, 2.013793];
		smoothed([3.0, 2.5, 2.7, 2.0, 1.6, 1.9, 1.2]).forEach((v, i) => expect(v).toBeCloseTo(want[i]!, 5));
	});
});

describe("where a rate sits on the curve", () => {
	const r = readRangeTest(probe(lora.extras));
	it("says the script's own 1e-4 is in the falling stretch", () => expect(rateVerdict(r, 0.0001)).toBe("near"));
	it("past the lowest loss, and past the blow-up", () => {
		expect(rateVerdict(r, 0.002)).toBe("past_lowest");
		expect(rateVerdict(r, 0.01)).toBe("past_divergence");
	});
	it("well below the steepest fall", () => expect(rateVerdict(r, 0.000005)).toBe("far_below"));
	it("nearest knob choice on a log scale", () => {
		expect(nearestRate(CHOICES, 7.9e-5)).toBe(0.0001);
		expect(nearestRate(CHOICES, 3.6e-5)).toBe(0.00005); // log distance 0.14 against 0.16 to 0.000025
		expect(nearestRate([], 1e-4)).toBeNull();
	});
});

describe("the curve in the learning-rate editor", () => {
	it("marks the steepest fall, the blow-up and the run's rate, and sets the knob from its button", () => {
		const onPick = vi.fn();
		render(<RangeTestCurve test={readRangeTest(probe(lora.extras), "A10")} value={0.0004} choices={CHOICES} onPick={onPick} />);
		const fig = screen.getByRole("img");
		expect(fig.getAttribute("aria-label")).toContain("0.000079");
		expect(fig.getAttribute("aria-label")).toContain("0.0076");
		expect(document.querySelector(".pf-lrr__blowup")).not.toBeNull();
		expect(document.querySelectorAll(".pf-lrr__mark")).toHaveLength(3);
		expect(screen.getByText(t("lrr.verdict.near", { v: "0.0004" }))).toBeTruthy();
		fireEvent.click(screen.getByRole("button", { name: t("lrr.use", { v: "0.0001" }) }));
		expect(onPick).toHaveBeenCalledWith(0.0001);
	});

	it("marks the run's rate red past the blow-up, and says so", () => {
		render(<RangeTestCurve test={readRangeTest(probe(lora.extras))} value={0.01} choices={[...CHOICES, 0.01]} onPick={() => {}} />);
		expect(document.querySelector(".pf-lrr__mark.is-yours-bad")).not.toBeNull();
		expect(screen.getByText(t("lrr.verdict.past_divergence", { v: "0.01" }))).toBeTruthy();
	});

	it("draws an unchecked ramp by step, with no rate marks and the reason in words", () => {
		const unchecked: Record<string, number> = { lr_range: 1, "lr_range.not_set": 1 };
		for (const [k, v] of Object.entries(lora.extras)) if (k.startsWith("lr_range.loss.")) unchecked[k] = v as number;
		render(<RangeTestCurve test={readRangeTest(probe(unchecked))} value={0.0001} choices={CHOICES} onPick={() => {}} />);
		expect(document.querySelectorAll(".pf-lrr__mark")).toHaveLength(0);
		expect(screen.getByText(t("lrr.unchecked.not_set"))).toBeTruthy();
		expect(screen.queryByRole("button")).toBeNull();
	});

	it("says No clear slope on a flat curve", () => {
		render(<RangeTestCurve test={readRangeTest(probe({ ...lora.extras, "lr_range.flat": 1 }))} value={0.0001} choices={CHOICES} onPick={() => {}} />);
		expect(screen.getByText(t("lrr.flat"))).toBeTruthy();
	});

	it("draws nanoGPT's curve too, which ran to the end of the ramp", () => {
		render(<RangeTestCurve test={readRangeTest(probe(real.pretrain_nanogpt.extras))} value={real.pretrain_nanogpt.own_lr} choices={[0.0001, 0.0003, 0.0006, 0.001]} onPick={() => {}} />);
		expect(document.querySelector(".pf-lrr__line")?.getAttribute("d")?.split("L").length).toBe(97);
	});
});
