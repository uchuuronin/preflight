import { describe, expect, it, vi } from "vitest";
import record from "../workspace/fixtures/handmade-finetune.record.json";
import type { FlightRecord, KnobValue } from "../workspace/record/types";
import { GridProvider } from "../workspace/provider/GridProvider";
import { currentKnobs } from "../workspace/provider/PredictionProvider";
import type { WorkspaceActions } from "../workspace/state";
import { GuideFallback, runLiveTurn } from "./liveGuide";
import type { ToolContext } from "./tools";

const rec = record as unknown as FlightRecord;

function ctx(pinned: Set<string>, moves: Array<[string, KnobValue]>): () => ToolContext {
	const actions = {
		setKnob: (id: string, v: KnobValue) => {
			moves.push([id, v]);
			return true;
		},
		pin: (id: string, on: boolean) => (on ? pinned.add(id) : pinned.delete(id)),
		highlight: vi.fn(),
		propose: vi.fn(),
	} as unknown as WorkspaceActions;
	return () => ({ record: rec, provider: new GridProvider(), actions, knobs: currentKnobs(rec), pinned, requestApproval: vi.fn() });
}

function reply(body: unknown, status = 200): Response {
	return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

describe("live guide loop", () => {
	it("runs tool calls in the browser and sends their results back", async () => {
		const bodies: Array<{ input: Array<Record<string, unknown>>; tools: Array<{ name: string }> }> = [];
		const responses = [
			{ text: "", calls: [{ call_id: "c1", name: "set_knob", arguments: { knob_id: "eval_batch_size", value: 8, why: "oom" } }], provenance: {} },
			{ text: "Your eval pass was the risk; eval batch 8 fits.", calls: [], provenance: {} },
		];
		const post = vi.fn(async (b: unknown) => {
			bodies.push(b as (typeof bodies)[number]);
			return reply(responses.shift());
		});
		const moves: Array<[string, KnobValue]> = [];
		const out = await runLiveTurn([], "cheapest by Friday", ctx(new Set(), moves), { post });
		expect(moves).toEqual([["eval_batch_size", 8]]);
		expect(out.text).toContain("eval batch 8");
		const second = bodies[1]?.input ?? [];
		const output = second.find((i) => i.type === "function_call_output");
		expect(output?.call_id).toBe("c1");
		expect(JSON.parse(String(output?.output)).ok).toBe(true);
	});

	it("rebuilds tools every round so a pin takes effect mid-turn", async () => {
		const pinned = new Set<string>();
		const bodies: Array<{ tools: Array<{ name: string; parameters: { properties?: { knob_id?: { enum?: string[] } } } }> }> = [];
		const responses = [
			{ text: "", calls: [{ call_id: "c1", name: "pin", arguments: { knob_id: "seq_len", pinned: true } }], provenance: {} },
			{ text: "Kept your sequence length.", calls: [], provenance: {} },
		];
		const post = vi.fn(async (b: unknown) => {
			bodies.push(b as (typeof bodies)[number]);
			return reply(responses.shift());
		});
		await runLiveTurn([], "keep sequence length", ctx(pinned, []), { post });
		const enumAfter = bodies[1]?.tools.find((t) => t.name === "set_knob")?.parameters.properties?.knob_id?.enum ?? [];
		expect(enumAfter).not.toContain("seq_len");
	});

	it("hands over to the rule-based tour on a fallback response", async () => {
		const post = vi.fn(async () =>
			reply({ error: { code: "guide_unavailable", message: "off", fallback: "rule_based", reason: "disabled" } }, 503),
		);
		await expect(runLiveTurn([], "hi", ctx(new Set(), []), { post })).rejects.toBeInstanceOf(GuideFallback);
	});
});
