import { describe, expect, it } from "vitest";
import raceRaw from "../../../api/preflight_api/fix/races/nov2_stack.json";
import liveRaw from "../workspace/fixtures/lora-live-nov1.record.json";
import { bestCombo } from "../guide/combos";
import { GridProvider } from "../workspace/provider/GridProvider";
import { currentKnobs } from "../workspace/record/model";
import { type FlightRecord, normaliseRecord } from "../workspace/record/types";
import { candidatesFor, liveCandidates, stackedCandidate } from "./candidates";

const live = normaliseRecord(liveRaw as unknown as FlightRecord);
const provider = new GridProvider();
const ctx = { record: live, provider, knobs: currentKnobs(live), pinned: new Set<string>(), objective: { kind: "min_cost" as const } };
const evalFinding = live.findings.find((f) => f.fix_template === "eval.wrap_no_grad")!;

describe("stacking the eval fixes when neither passes alone (Chiya, Nov 2)", () => {
	it("takes the eval batch from the guide's own combined move", () => {
		const combo = bestCombo(live, provider, ctx.knobs, ctx.objective, { pinned: ctx.pinned });
		const guideBatch = combo?.changes.find((c) => c.knobId === "eval_batch_size")?.to;
		expect(guideBatch).toBeDefined();
		expect(stackedCandidate(evalFinding, ctx)).toEqual({ template_id: "eval.no_grad_and_batch", value: guideBatch });
	});

	it("never races the stack in the first round, only after its parts fail", () => {
		expect(candidatesFor(evalFinding, ctx).map((c) => c.template_id)).not.toContain("eval.no_grad_and_batch");
	});

	it("is ruled out by a pinned eval knob", () => {
		expect(stackedCandidate(evalFinding, { ...ctx, pinned: new Set(["eval_batch_size"]) })).toBeNull();
		expect(stackedCandidate(evalFinding, { ...ctx, pinned: new Set(["eval_no_grad"]) })).toBeNull();
	});

	it("the Nov 2 race used exactly the guide's value (api/preflight_api/fix/races/nov2_stack.json)", () => {
		const race = raceRaw as { round1: unknown[]; stack: { template_id: string; value: unknown } };
		expect(race.round1).toEqual(candidatesFor(evalFinding, ctx));
		expect(race.stack).toEqual(stackedCandidate(evalFinding, ctx));
	});

	it("live, where nothing is verified, races the stack beside the single fix when no single move clears", () => {
		expect(liveCandidates(evalFinding, ctx)).toEqual([...candidatesFor(evalFinding, ctx), stackedCandidate(evalFinding, ctx)]);
	});
});

describe("Muse Code's summary on screen", () => {
	it("keeps a link's words and drops its sandbox path", async () => {
		const { plainSaid } = await import("./FixRace");
		expect(plainSaid("averaging in [train.py](/fix/wt/c1/train.py:74). Done.")).toBe("averaging in train.py. Done.");
	});
});
