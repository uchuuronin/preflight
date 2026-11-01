import { describe, expect, it } from "vitest";
import en from "../../i18n/en.json";
import { abstainText } from "./abstain";

describe("why a number isn't shown", () => {
	it("says a provider code's own line", () => {
		expect(abstainText("outside_grid")).toBe(en["abstain.outside_grid"]);
		expect(abstainText(undefined)).toBe(en["abstain.off_grid"]);
	});

	it("never shows a code it has no line for, whatever its shape", () => {
		for (const code of ["some_new_code", "probe.oom", "fp8_off"]) expect(abstainText(code)).toBe(en["abstain.off_grid"]);
		for (const empty of [null, "", "   "]) expect(abstainText(empty)).toBe(en["abstain.off_grid"]);
	});

	it("shows the engine's own sentence as a sentence, not as a key", () => {
		// B's engine/api.py, for a record whose eval probe ran out of memory (live check, Nov 1).
		expect(abstainText("probe p1-eval ran out of memory at these settings")).toBe("Probe p1-eval ran out of memory at these settings.");
		expect(abstainText("Already a sentence.")).toBe("Already a sentence.");
		// A sentence that opens with a knob id keeps the id as written.
		expect(abstainText("eval_batch_size changes memory only through the eval pass")).toBe("eval_batch_size changes memory only through the eval pass.");
		expect(abstainText("bf16 has no fast path on L4")).toBe("bf16 has no fast path on L4.");
	});
});
