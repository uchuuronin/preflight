// @vitest-environment jsdom
import { describe, expect, it } from "vitest";
import { verdictText } from "./FixRace";
import { verdictFor } from "./verified";

describe("a fix lane's verdict (PM, Nov 3)", () => {
	it("shows Preflight's Nov 2 checks of the same fixes on the LoRA sample", () => {
		const single = verdictFor("lora-qwen", { template_id: "eval.wrap_no_grad", value: null });
		expect(verdictText(single!)).toBe("Rejected: the re-run still ran out of memory.");
		expect(single!.checkedOn).toBe("2026-11-02");
		const stack = verdictFor("lora-qwen", { template_id: "eval.no_grad_and_batch", value: 8 });
		expect(verdictText(stack!)).toBe("Passed: eval peak 10.4 GiB, loss within 0.11%.");
	});
	it("claims nothing for another value, another fix or another record", () => {
		expect(verdictFor("lora-qwen", { template_id: "eval.no_grad_and_batch", value: 4 })).toBeNull();
		expect(verdictFor("lora-qwen", { template_id: "eval.reduce_batch", value: 8 })).toBeNull();
		expect(verdictFor("01SOMEONES0WN0RECORD000000", { template_id: "eval.wrap_no_grad", value: null })).toBeNull();
	});
	it("takes the race's own verify result first, when the API sends one", () => {
		const v = verdictFor("x", { template_id: "eval.wrap_no_grad", value: null, verify: { passed: false, reason: "oom" } });
		expect(v).toEqual({ passed: false, reason: "oom" });
	});
});
