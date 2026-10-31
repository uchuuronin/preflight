import { describe, expect, it } from "vitest";
import en from "../../i18n/en.json";
import { SAMPLES } from "./load";
import { isFixture } from "./model";
import { normaliseRecord } from "./types";

describe("the samples' label", () => {
	it("calls samples illustrative only while every sample is a fixture", async () => {
		// Departures says "Samples with illustrative numbers"; a real measured sample would make that
		// untrue, and this test asks for the label to change with it.
		expect(en["newflight.departures.sub"]).toMatch(/illustrative/i);
		for (const s of SAMPLES) expect(isFixture(normaliseRecord(await s.load())), s.id).toBe(true);
	});
});
