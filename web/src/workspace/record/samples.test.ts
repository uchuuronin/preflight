import { describe, expect, it } from "vitest";
import { currentRef, loadRecord, SAMPLES } from "./load";
import { isFixture } from "./model";
import { normaliseRecord } from "./types";

describe("the samples' label", () => {
	it("says illustrative exactly for the fixture samples and measured for real runs", async () => {
		for (const s of SAMPLES) expect(isFixture(normaliseRecord(await s.load())), s.id).toBe(s.numbers === "illustrative");
		expect(SAMPLES.some((s) => s.numbers === "measured")).toBe(true);
	});
	it("opens a retired sample's old link as its measured replacement (PM, Nov 3)", async () => {
		expect(currentRef("lora-7b")).toBe("lora-qwen");
		expect(currentRef("agent")).toBe("agent-eval");
		const old = await loadRecord("lora-7b");
		const now = await loadRecord("lora-qwen");
		expect(old.record_id).toBe(now.record_id);
		for (const s of SAMPLES) expect(isFixture(normaliseRecord(await s.load())), s.id).toBe(false);
	});
});
