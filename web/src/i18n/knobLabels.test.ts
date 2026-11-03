import { describe, expect, it } from "vitest";
import en from "./en.json";

// B's list (from-b.md, Nov 3): knob ids the engine reads that had no label on the site.
const ADDED = ["dtype", "torch_dtype", "ptdtype", "compile", "prefetch_factor", "group_by_length", "pin_memory", "nodes", "inter_node_gbps", "pipeline_parallel", "rl_iterations", "request_rate", "max_batched_tokens", "deepspeed_config", "tf32", "dump_attentions"];

describe("knob labels", () => {
	it("every knob on B's list has a plain label", () => {
		const strings = en as Record<string, string>;
		for (const id of ADDED) expect(strings[`knob.${id}`], id).toMatch(/\S/);
	});
});
