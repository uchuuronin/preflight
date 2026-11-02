import { describe, expect, it } from "vitest";
import agentEvalRaw from "../workspace/fixtures/agent-eval-oct31.record.json";
import devRaw from "../workspace/fixtures/dev-finetune.record.json";
import { GridProvider } from "../workspace/provider/GridProvider";
import { currentKnobs } from "../workspace/record/model";
import { type FlightRecord, normaliseRecord } from "../workspace/record/types";
import { candidatesFor, MAX_CANDIDATES, TEMPLATES, template } from "./candidates";
import { startRace } from "./race";

const dev = normaliseRecord(devRaw as unknown as FlightRecord);
const agent = normaliseRecord(agentEvalRaw as unknown as FlightRecord);
const provider = new GridProvider();
const ctx = (record = dev, pinned = new Set<string>()) => ({ record, provider, knobs: currentKnobs(record), pinned, objective: { kind: "min_cost" as const } });
const finding = (record: typeof dev, id: string) => record.findings.find((f) => f.id === id)!;

describe("which fixes race for a finding", () => {
	it("races the eval out-of-memory fix against the other fixes for the same number", () => {
		const c = candidatesFor(finding(dev, "f2"), ctx());
		expect(c[0]?.template_id).toBe("eval.reduce_batch");
		expect(c.length).toBeGreaterThan(1);
		expect(c.length).toBeLessThanOrEqual(MAX_CANDIDATES);
		const metric = template("eval.reduce_batch")?.metric;
		for (const x of c) {
			const t = template(x.template_id)!;
			expect(t.metric).toBe(metric);
			expect(t.lenses).toContain("finetune");
			if (t.needs_value) expect(x.value).toBeDefined();
		}
		// The value the eval batch takes is one the engine says fits, not a made-up one.
		const v = c[0]?.value as number;
		expect(v).toBeLessThan(currentKnobs(dev).eval_batch_size as number);
	});

	it("leaves a pinned knob alone, and races nothing if the finding's own fix needs it", () => {
		expect(candidatesFor(finding(dev, "f2"), ctx(dev, new Set(["eval_batch_size"])))).toEqual([]);
		const c = candidatesFor(finding(dev, "f2"), ctx(dev, new Set(["gradient_checkpointing"])));
		expect(c.map((x) => x.template_id)).not.toContain("memory.gradient_checkpointing");
	});

	it("caps the agent's retry loop in the code, with the cap shown as the value", () => {
		const c = candidatesFor(agent.findings[0]!, ctx(agent));
		expect(c).toEqual([{ template_id: "agent.cap_tool_retries", value: 3 }]);
	});

	it("knows every template the API knows", () => {
		expect(TEMPLATES).toHaveLength(13); // 12, and the eval stack (Nov 2)
		expect(candidatesFor({ ...agent.findings[0]!, fix_template: "no.such" }, ctx(agent))).toEqual([]);
	});
});

describe("the race client", () => {
	it("says what the API refused with its code", async () => {
		const post = (async () => new Response(JSON.stringify({ error: { code: "not_your_code", message: "x" } }), { status: 403 })) as unknown as typeof fetch;
		expect(await startRace("tok", null, [{ template_id: "eval.wrap_no_grad" }], post)).toEqual({ ok: false, code: "not_your_code", message: "x" });
		const down = (async () => {
			throw new TypeError("offline");
		}) as unknown as typeof fetch;
		expect((await startRace("tok", null, [], down)).ok).toBe(false);
	});
});
