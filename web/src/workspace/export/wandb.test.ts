import { describe, expect, it } from "vitest";
import devRaw from "../fixtures/dev-finetune.record.json";
import { GridProvider } from "../provider/GridProvider";
import { currentKnobs } from "../record/model";
import { type FlightRecord, normaliseRecord } from "../record/types";
import { exportPlan } from "./plan";
import { wandbFile } from "./wandb";

const dev = normaliseRecord(devRaw as unknown as FlightRecord);
const knobs = currentKnobs(dev);
const plan = exportPlan(dev, knobs, new GridProvider().predict(dev, knobs), "lora-7b", "https://preflight.example/");

/** What Python's json.loads reads: the text between r''' and '''. */
function embedded(text: string): Record<string, unknown> & { readouts: Array<Record<string, unknown>> } {
	const m = /json\.loads\(r'''([\s\S]*?)'''\)/.exec(text);
	if (!m) throw new Error("no payload");
	return JSON.parse(m[1]!);
}

describe("the Weights & Biases export", () => {
	it("carries the plan's own numbers, and no text in it can close the Python string", () => {
		const title = "Bob's run with ''' and \\ in it";
		const text = wandbFile({ ...plan, title });
		expect(text).not.toMatch(/api_key|WANDB_API_KEY/);
		const body = text.slice(text.indexOf("r'''") + 4, text.lastIndexOf("'''"));
		expect(body).not.toContain("'");
		const data = embedded(text);
		expect(data.title).toBe(title);
		expect(data.config).toEqual(plan.config);
		expect(data.readouts.map((r) => [r.target, r.point, r.lo, r.hi, r.status])).toEqual(plan.readouts.map((r) => [r.targetId, r.point, r.lo, r.hi, r.status]));
		expect(data.readouts.every((r) => /^[A-Za-z0-9_]*$/.test(String(r.unit)))).toBe(true);
	});
});

describe("a sample's Weights & Biases file", () => {
	it("says its numbers are examples, in the file and in the run's summary", () => {
		const text = wandbFile(plan, true);
		expect(embedded(text).illustrative).toBe(true);
		expect(text).toMatch(/examples, not measured on your job/);
		expect(text).toContain('run.summary["preflight/illustrative"] = PLAN["illustrative"]');
		expect(embedded(wandbFile(plan)).illustrative).toBe(false);
		expect(wandbFile(plan)).not.toMatch(/From a Preflight sample/);
		// The link is mentioned only where there is one.
		expect(wandbFile(plan)).toMatch(/link to this Preflight record/);
		expect(wandbFile({ ...plan, shareUrl: null })).not.toMatch(/link to this Preflight record/);
	});
});
