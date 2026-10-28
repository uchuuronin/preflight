import { describe, expect, it } from "vitest";
import devRaw from "../fixtures/dev-finetune.record.json";
import { GridProvider } from "../provider/GridProvider";
import { currentKnobs } from "../record/model";
import { type FlightRecord, normaliseRecord } from "../record/types";
import { statusLine } from "./statusLine";

const dev = normaliseRecord(devRaw as unknown as FlightRecord);
const provider = new GridProvider();
const line = (knobs: Record<string, unknown>) => {
	const k = { ...currentKnobs(dev), ...knobs } as Record<string, string | number | boolean>;
	return statusLine(dev, provider.predict(dev, k), k);
};

describe("the status line", () => {
	it("leads with the failure and its numbers when the run won't fit", () => {
		const s = line({});
		expect(s.tone).toBe("fails");
		expect(s.text).toMatch(/^Runs out of memory on A100-SXM-80GB: peaks at .+ GiB of 79.2 GiB\.$/);
	});

	it("gives time and cost with their ranges, and the budget, when it fits", () => {
		const s = line({ eval_batch_size: 8 });
		expect(s.tone).toBe("fits");
		expect(s.text).toMatch(/^Fits on A100-SXM-80GB\. It finishes in .+ \(.+ to .+\) and costs US\$.+\. (Within|May go over|Over) the US\$60 budget\.$/);
	});

	it("never quotes a number the readouts don't show", () => {
		const k = { ...currentKnobs(dev), eval_batch_size: 8 } as Record<string, string | number | boolean>;
		const p = provider.predict(dev, k);
		const text = statusLine(dev, p, k).text;
		expect(text).not.toMatch(/NaN|undefined|null/);
	});
});
