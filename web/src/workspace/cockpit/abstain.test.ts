import { describe, expect, it } from "vitest";
import en from "../../i18n/en.json";
import { abstainText } from "./abstain";

describe("why a number isn't shown", () => {
	it("says a provider code's own line", () => {
		expect(abstainText("outside_grid")).toBe(en["abstain.outside_grid"]);
		expect(abstainText(undefined)).toBe(en["abstain.off_grid"]);
	});

	it("never shows a code it has no line for", () => {
		expect(abstainText("some_new_code")).toBe(en["abstain.off_grid"]);
	});

	it("shows the engine's own sentence as a sentence, not as a key", () => {
		// B's engine/api.py, for a record whose eval probe ran out of memory (live check, Nov 1).
		expect(abstainText("probe p1-eval ran out of memory at these settings")).toBe("Probe p1-eval ran out of memory at these settings.");
		expect(abstainText("Already a sentence.")).toBe("Already a sentence.");
	});
});
