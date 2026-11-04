import { describe, expect, it, vi } from "vitest";
import devRaw from "../workspace/fixtures/dev-finetune.record.json";
import { GridProvider } from "../workspace/provider/GridProvider";
import { currentKnobs } from "../workspace/record/model";
import { type FlightRecord, type KnobValue, normaliseRecord } from "../workspace/record/types";
import { type GuideCall, GuideFallback, guardSetKnob, LiveTour, type RoundOut, runBrowserCall, type TourContext } from "./liveGuide";
import { fakeActions } from "./testkit";

const dev = normaliseRecord(devRaw as unknown as FlightRecord);
const provider = new GridProvider();

function makeCtx(over: Partial<{ knobs: Record<string, KnobValue>; pinned: Set<string> }> = {}) {
	const pinned = over.pinned ?? new Set<string>();
	const f = fakeActions(pinned);
	let knobs = { ...currentKnobs(dev), ...(over.knobs ?? {}) };
	const actions = {
		...f.actions,
		setKnob: (id: string, v: KnobValue, by?: "user" | "guide") => {
			const ok = f.actions.setKnob(id, v, by);
			if (ok) knobs = { ...knobs, [id]: v };
			return ok;
		},
	};
	const ctx = (): TourContext => ({ record: dev, provider, actions, knobs, pinned, objective: { kind: "min_cost" }, requestApproval: vi.fn() });
	return { ctx, moves: f.moves, trace: f.trace };
}

function round(over: Partial<RoundOut>): RoundOut {
	return { text: "", withheld: null, calls: [], awaiting: "visitor", state: "s", provenance: {}, ...over };
}

function scripted(...rounds: RoundOut[]) {
	const bodies: Array<Record<string, unknown>> = [];
	const post = vi.fn(async (b: unknown) => {
		bodies.push(b as Record<string, unknown>);
		const next = rounds.shift();
		return new Response(JSON.stringify(next), { status: 200, headers: { "Content-Type": "application/json" } });
	});
	return { post, bodies };
}

const call = (name: string, args: Record<string, unknown>, id = name): GuideCall => ({ call_id: id, name, arguments: args, runs_in: "browser" });

describe("the live tour loop", () => {
	it("runs browser tools on its own and stops when the visitor must decide", async () => {
		const { ctx, moves } = makeCtx();
		const { post, bodies } = scripted(
			round({ calls: [call("sensitivity", { objective: "min_cost" }), call("highlight", { nodes: ["eval", "nope"] })], awaiting: "results", state: "s1" }),
			round({ calls: [call("set_knob", { knob_id: "eval_batch_size", value: 8, why: "oom" })], awaiting: "results", state: "s2" }),
			round({ text: "Eval batch 8 fits.", awaiting: "decision", state: "s3" }),
		);
		const views: string[] = [];
		const tour = new LiveTour("lora-7b", ctx, (v) => views.push(v.awaiting), post);
		const view = await tour.start();
		expect(view.awaiting).toBe("decision");
		expect(view.text).toBe("Eval batch 8 fits.");
		expect(moves).toEqual([["eval_batch_size", 8]]);
		expect(bodies[0]).toMatchObject({ record: "lora-7b", start: true });
		expect(bodies[0]).not.toHaveProperty("input");
		expect(bodies[0]).not.toHaveProperty("tools");
		const second = bodies[1] as { state: string; results: Array<Record<string, unknown>> };
		expect(second.state).toBe("s1");
		expect(second.results.map((r) => r.tool)).toEqual(["sensitivity", "highlight"]);
		expect(second.results[1]).toMatchObject({ nodes: ["eval"] });
		const third = bodies[2] as { results: Array<Record<string, unknown>>; setting: { knobs: Record<string, KnobValue> } };
		expect(third.results[0]).toMatchObject({ tool: "set_knob", moved: true, from_value: 32, to_value: 8, refused: null });
		expect(third.setting.knobs.eval_batch_size).toBe(8);
	});

	it("sends the visitor's decision and approvals as typed facts", async () => {
		const { ctx } = makeCtx();
		const { post, bodies } = scripted(round({ text: "Next." }), round({ text: "Queued." }));
		const tour = new LiveTour("lora-7b", ctx, () => {}, post);
		await tour.decide({ knob_id: "eval_batch_size", action: "reverted", value: 32 });
		await tour.approve("c9", true);
		expect(bodies[0]).toMatchObject({ decision: { knob_id: "eval_batch_size", action: "reverted", value: 32 } });
		expect(bodies[1]).toMatchObject({ approvals: [{ call_id: "c9", approved: true }] });
	});

	it("hands over to the rule-based tour when the server says so", async () => {
		const { ctx } = makeCtx();
		const post = vi.fn(async () => new Response(JSON.stringify({ error: { message: "off", fallback: "rule_based", reason: "daily_cap" } }), { status: 503 }));
		const tour = new LiveTour("lora-7b", ctx, () => {}, post);
		await expect(tour.start()).rejects.toBeInstanceOf(GuideFallback);
	});

	it("hands a record opened from disk to the rule-based tour without asking the server (Chiya, Nov 3)", async () => {
		const { ctx } = makeCtx();
		const post = vi.fn();
		const tour = new LiveTour("live", ctx, () => {}, post);
		await expect(tour.start()).rejects.toThrow("The live guide reads records checked in on the site");
		expect(post).not.toHaveBeenCalled();
	});
});

describe("the browser's set_knob guard", () => {
	it("refuses pinned knobs, values outside the knob, and moves that don't fit", () => {
		expect(guardSetKnob(makeCtx({ pinned: new Set(["eval_batch_size"]) }).ctx(), "eval_batch_size", 8)).toBe("pinned");
		expect(guardSetKnob(makeCtx().ctx(), "eval_batch_size", 7)).toBe("not_in_domain");
		expect(guardSetKnob(makeCtx().ctx(), "nope", 1)).toBe("not_in_domain");
		expect(guardSetKnob(makeCtx().ctx(), "eval_batch_size", 32)).toBe("would_fail");
		expect(guardSetKnob(makeCtx().ctx(), "eval_batch_size", 8)).toBeNull();
	});

	it("refuses settings the model can't vouch for", () => {
		// The run as checked in is failing, so a result-changing knob may move, but not to a worse fit.
		const r = guardSetKnob(makeCtx().ctx(), "seq_len", 4096);
		expect(["not_trusted", "would_fail", "would_be_tight"]).toContain(r);
		// Once it fits, longer sequences are a trade-off to put to the person, never a move.
		expect(guardSetKnob(makeCtx({ knobs: { eval_batch_size: 8 } }).ctx(), "seq_len", 4096)).toBe("changes_results");
	});

	it("moves at most one knob per round", () => {
		const { ctx, moves } = makeCtx();
		const first = runBrowserCall(call("set_knob", { knob_id: "eval_batch_size", value: 8 }, "a"), ctx(), false);
		const second = runBrowserCall(call("set_knob", { knob_id: "batch_size", value: 8 }, "b"), ctx(), true);
		expect(first).toMatchObject({ moved: true });
		expect(second).toMatchObject({ moved: false, refused: "one_per_round" });
		expect(moves).toHaveLength(1);
	});

	it("ranks with fit labels and readouts the server can type-check", () => {
		const { ctx } = makeCtx({ knobs: { eval_batch_size: 8 } });
		const out = runBrowserCall(call("sensitivity", { objective: "min_cost" }), ctx(), false);
		if (out.tool !== "sensitivity") throw new Error("wrong tool");
		for (const r of out.ranked) {
			expect(r.impact).toBeGreaterThanOrEqual(0);
			expect(r.impact).toBeLessThanOrEqual(1);
			expect(["fits", "tight"]).toContain(r.fit);
		}
		const p = runBrowserCall(call("predict", { knobs: { eval_batch_size: 8, made_up: 1 } }), ctx(), false);
		if (p.tool !== "predict") throw new Error("wrong tool");
		for (const r of Object.values(p.readouts)) expect(r.unit).toMatch(/^[A-Za-z/_%]{0,16}$/);
	});
});

describe("the live guide on the measured LoRA sample (PM, Nov 3)", () => {
	it("offers a combination on the L4 before any GPU change, and ranks no GPU change beside it", async () => {
		const { SAMPLES } = await import("../workspace/record/load");
		const rec = normaliseRecord((await SAMPLES.find((s) => s.id === "lora-qwen")!.load()) as FlightRecord);
		const provider = new GridProvider();
		const { actions } = fakeActions();
		const base = { record: rec, provider, actions, knobs: currentKnobs(rec), pinned: new Set<string>(), objective: { kind: "min_cost" as const }, requestApproval: vi.fn() };
		const call = { name: "sensitivity", call_id: "s1", arguments: { objective: "min_cost" } } as unknown as GuideCall;
		const out = runBrowserCall(call, base, false) as { combined?: { knobs: Record<string, KnobValue> }; ranked: Array<{ knob_id: string }> };
		expect(out.combined).toBeTruthy();
		expect(Object.keys(out.combined!.knobs)).not.toContain("gpu_type");
		expect(out.ranked.map((r) => r.knob_id)).not.toContain("gpu_type");
		// After the person declines a combination, the GPU change is back.
		const again = runBrowserCall(call, { ...base, combosDeclined: new Set(["combo:x"]) }, false) as { ranked: Array<{ knob_id: string }> };
		expect(again.ranked.map((r) => r.knob_id)).toContain("gpu_type");
	});
});
