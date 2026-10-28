import { describe, expect, it } from "vitest";
import devRaw from "../fixtures/dev-finetune.record.json";
import { GridProvider } from "../provider/GridProvider";
import { type FlightRecord, normaliseRecord } from "./types";
import { dialScale, LOG_SPAN_MAX, memoryStackScale, scaleFraction } from "./scale";

const dev = normaliseRecord(devRaw as unknown as FlightRecord);
const provider = new GridProvider();

describe("fixed dial scales", () => {
	it("is the same object all session: flying the knobs never rescales a dial", () => {
		const a = dialScale(dev, "time.total", "s");
		provider.predict(dev, { batch_size: 1, seq_len: 4096 });
		expect(dialScale(dev, "time.total", "s")).toBe(a);
		expect(a?.kind).toBe("log");
		expect((a?.hi ?? 0) / (a?.lo ?? 1)).toBeLessThanOrEqual(LOG_SPAN_MAX + 1e-6);
	});

	it("keeps every setting the grid covers on the dial", () => {
		const s = dialScale(dev, "cost.total", "usd");
		if (!s) throw new Error("no scale");
		for (const batch of [1, 4, 32]) {
			for (const seq of [256, 2048, 4096]) {
				const e = provider.predict(dev, { batch_size: batch, seq_len: seq })["cost.total"];
				if (e?.status === "abstain" || e?.hi == null) continue;
				expect(scaleFraction(s, e.hi).off).toBe(false);
			}
		}
	});

	it("uses a linear scale to a limit the visitor set, and 0 to 1 for a probability", () => {
		expect(dialScale(dev, "cost.total", "usd", 50)).toEqual({ kind: "linear", lo: 0, hi: 50, source: "limit" });
		expect(dialScale(dev, "risk.oom", "probability")?.hi).toBe(1);
		expect(scaleFraction({ kind: "linear", lo: 0, hi: 50, source: "limit" }, 75)).toEqual({ frac: 1, off: true });
	});

	it("sizes the memory stack to the biggest GPU the record can choose", () => {
		const s = memoryStackScale(dev);
		expect(s).toBeGreaterThanOrEqual(80e9);
		expect(memoryStackScale(dev)).toBe(s);
	});
});
