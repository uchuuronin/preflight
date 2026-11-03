import { describe, expect, it } from "vitest";
import { t } from "../../lib/i18n";
import { flightTitle } from "./Logbook";

describe("a logbook flight's name (Chiya, Nov 2: 'agent' and 'finetune' showed as titles)", () => {
	it("keeps a record's own title", () => {
		expect(flightTitle({ ref: "x", title: "GRPO on GSM8K", lens: "rl" })).toBe("GRPO on GSM8K");
	});
	it("gives a sample with no title its sample name", () => {
		expect(flightTitle({ ref: "agent-eval", title: "agent", lens: "agent" })).toBe(t("sample.agent_eval"));
		expect(flightTitle({ ref: "lora-qwen", title: "finetune", lens: "finetune" })).toBe(t("sample.lora_qwen"));
	});
	it("names any other untitled record by its kind of job", () => {
		expect(flightTitle({ ref: "SHARETOKENxxxxxxxx", title: "finetune", lens: "finetune" })).toBe(t("lens.finetune"));
	});
});
