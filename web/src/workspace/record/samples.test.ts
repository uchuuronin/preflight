import { describe, expect, it } from "vitest";
import { SAMPLES } from "./load";
import { isFixture } from "./model";
import { normaliseRecord } from "./types";

describe("the samples' label", () => {
	it("says illustrative exactly for the fixture samples and measured for real runs", async () => {
		for (const s of SAMPLES) expect(isFixture(normaliseRecord(await s.load())), s.id).toBe(s.numbers === "illustrative");
		expect(SAMPLES.some((s) => s.numbers === "measured")).toBe(true);
	});
});
