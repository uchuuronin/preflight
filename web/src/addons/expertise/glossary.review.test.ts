import { describe, expect, it } from "vitest";
import { GLOSSARY, KNOB_TERMS } from "./glossary";
import { TIERS } from "./store";

// The Oct 27 review's glossary list (Contracts tab, "D"), kept as checks so the fixes stay fixed.
function all(id: string): string {
	const t = GLOSSARY[id];
	if (!t) throw new Error(`no term ${id}`);
	return [t.new, t.practitioner, t.expert, t.act ?? ""].join(" ");
}

describe("glossary after the Oct 27 review", () => {
	it("marks features that aren't built yet as planned, at every level", () => {
		for (const id of ["interval", "verify", "repetition_exposure", "seeds_needed", "cost_frontier"]) {
			for (const tier of TIERS) expect(GLOSSARY[id]?.[tier], `${id}.${tier}`).toMatch(/^Planned/);
		}
		expect(GLOSSARY.divergence_risk?.expert).toMatch(/classifier is planned/);
	});

	it("defines measured, predicted and extrapolated as the contract does", () => {
		const t = all("measured_predicted");
		expect(t).toMatch(/a probe observed/);
		expect(t).toMatch(/any other knob/);
		expect(t).not.toMatch(/2×/);
	});

	it("says T4 and P100 train in fp16 with loss scaling", () => {
		expect(all("precision")).toMatch(/T4 and P100/);
		expect(all("precision")).toMatch(/fp16 with (dynamic )?loss scaling/);
	});

	it("pairs a lower batch with more gradient accumulation", () => {
		expect(GLOSSARY.oom?.act).toMatch(/raise gradient accumulation/i);
		expect(GLOSSARY.batch_size?.act).toMatch(/raise gradient accumulation/i);
	});

	it("gives MFU's FLOPs per token for LoRA as well as full training", () => {
		expect(GLOSSARY.mfu?.expert).toMatch(/6N/);
		expect(GLOSSARY.mfu?.expert).toMatch(/4N with LoRA/);
	});

	it("says the sampled subgraph grows with the product of fan-outs", () => {
		expect(GLOSSARY.fan_out?.practitioner).toMatch(/product of the fan-outs/);
		expect(GLOSSARY.fan_out?.practitioner).not.toMatch(/hops × fan-out/);
	});

	it("gives the rate-limited wall time as the largest of the three limits", () => {
		const f = /max\(total tokens \/ tokens per minute, requests \/ requests per minute/;
		expect(GLOSSARY.concurrency?.expert).toMatch(f);
		expect(GLOSSARY.rate_limit?.expert).toMatch(f);
	});

	it("explains Hugging Face's eval logit accumulation and its fix", () => {
		expect(all("eval_batch")).toMatch(/keeps every eval batch's logits on the GPU/);
		expect(GLOSSARY.eval_batch?.act).toMatch(/eval_accumulation_steps/);
	});

	it("explains ZeRO and FSDP, and maps the sharding knobs to it", () => {
		expect(all("zero_fsdp")).toMatch(/stage 3/);
		expect(all("fsdp_mode")).toMatch(/ZeRO-3/);
		expect(KNOB_TERMS.zero_stage).toBe("zero_fsdp");
	});
});
