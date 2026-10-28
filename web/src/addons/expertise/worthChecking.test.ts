import { beforeAll, describe, expect, it } from "vitest";
import agentRaw from "../../../../schema/examples/agent.json";
import finetuneRaw from "../../../../schema/examples/finetune.json";
import pretrainRaw from "../../../../schema/examples/pretrain.json";
import { t } from "../../lib/i18n";
import { type FlightRecord, normaliseRecord } from "../../workspace/record/types";
import { registerExpertiseStrings } from "./strings";
import { worthChecking, worthCheckingKey } from "./worthChecking";

const load = (raw: unknown) => normaliseRecord(raw as FlightRecord);

describe("worthChecking", () => {
	beforeAll(registerExpertiseStrings);

	it("lists the record's warnings, not its failures or notes", () => {
		expect(worthChecking(load(finetuneRaw)).map((f) => f.code)).toEqual(["data.padding_waste"]);
		expect(worthChecking(load(pretrainRaw)).map((f) => f.code)).toEqual(["precision.fp32", "ckpt.sparse"]);
		expect(worthChecking(load(agentRaw)).map((f) => f.code)).toEqual(["agent.no_prompt_caching"]);
	});

	it("names the knob that fixes each warning, when the record declares it", () => {
		expect(worthChecking(load(pretrainRaw)).map((f) => f.knob_id)).toEqual(["precision", "checkpoint_every"]);
	});

	it("drops a knob the record doesn't declare, but keeps the warning", () => {
		const record = load(finetuneRaw);
		const odd = { ...record, findings: record.findings.map((f) => (f.severity === "warn" ? { ...f, knob_id: "not_a_knob" } : f)) };
		expect(worthChecking(odd)).toEqual([expect.objectContaining({ code: "data.padding_waste", knob_id: null })]);
	});

	it("heads the list in plain words, singular and plural", () => {
		expect(t(worthCheckingKey(1), { n: 1 })).toBe("1 thing worth checking");
		expect(t(worthCheckingKey(3), { n: 3 })).toBe("3 things worth checking");
	});
});
