import { describe, expect, it, vi } from "vitest";
import goldenRaw from "../../../schema/examples/finetune.json";
import devRaw from "../workspace/fixtures/dev-finetune.record.json";
import { GridProvider } from "../workspace/provider/GridProvider";
import { currentKnobs } from "../workspace/record/model";
import { type FlightRecord, type KnobValue, normaliseRecord } from "../workspace/record/types";
import type { TraceEntry } from "../workspace/state";
import { nextStep } from "./ruleTour";
import { fakeActions } from "./testkit";
import { buildTools, executeTool, runTool, type ToolContext } from "./tools";

const dev = normaliseRecord(devRaw as unknown as FlightRecord);
const golden = normaliseRecord(goldenRaw as unknown as FlightRecord);
const provider = new GridProvider();
const cost = { kind: "min_cost" as const };

function ctx(record = dev, pinned = new Set<string>()): ToolContext & { moves: Array<[string, KnobValue]>; trace: TraceEntry[]; approve: ReturnType<typeof vi.fn> } {
	const f = fakeActions(pinned);
	const approve = vi.fn();
	return { record, provider, actions: f.actions, knobs: currentKnobs(record), pinned, requestApproval: approve, moves: f.moves, trace: f.trace, approve };
}

describe("guide tool schema", () => {
	it("removes pinned knobs from set_knob's enum every turn", () => {
		const tools = buildTools(dev, new Set(["eval_batch_size", "gpu_type"]));
		const setKnob = tools.find((x) => x.name === "set_knob");
		const props = setKnob?.parameters.properties as { knob_id: { enum: string[] } };
		expect(props.knob_id.enum).not.toContain("eval_batch_size");
		expect(props.knob_id.enum).not.toContain("gpu_type");
		expect(props.knob_id.enum).toContain("batch_size");
	});

	it("drops set_knob entirely when everything is pinned", () => {
		expect(buildTools(dev, new Set(dev.knobs.map((k) => k.id))).some((x) => x.name === "set_knob")).toBe(false);
	});

	it("refuses a set_knob on a pinned knob even if the model sends one", () => {
		const c = ctx(dev, new Set(["eval_batch_size"]));
		const res = executeTool("set_knob", { knob_id: "eval_batch_size", value: 8, why: "x" }, c);
		expect(res.ok).toBe(false);
		expect(c.moves).toHaveLength(0);
	});

	it("never runs a money-spending tool, only asks for approval", () => {
		const c = ctx();
		for (const name of ["run_probe", "propose_patch", "run_verify"]) {
			expect(executeTool(name, {}, c)).toEqual({ ok: true, result: { status: "awaiting_person_approval" } });
		}
		expect(c.approve).toHaveBeenCalledTimes(3);
	});

	it("puts every call and its real result in the trace", () => {
		const c = ctx();
		runTool("predict", { knobs: { eval_batch_size: 8 } }, c);
		runTool("set_knob", { knob_id: "nope", value: 1, why: "x" }, c);
		expect(c.trace).toHaveLength(2);
		expect(c.trace[0]?.tool).toBe("predict");
		expect(c.trace[0]?.result).toContain("memory.peak");
		expect(c.trace[1]?.ok).toBe(false);
	});
});

describe("rule-based tour", () => {
	const base = (record = dev) => ({ record, provider, knobs: currentKnobs(record), objective: cost, visited: new Set<string>() });

	it("opens on the OOM fix and lights the eval node", () => {
		const step = nextStep({ ...base(), pinned: new Set() });
		expect(step.kind).toBe("move");
		expect(step.move?.knobId).toBe("eval_batch_size");
		expect(step.move?.to).toBe(8);
		expect(step.nodes).toContain("eval");
	});

	it("names the pin in the way instead of moving it", () => {
		const step = nextStep({ ...base(), pinned: new Set(["eval_batch_size", "gpu_type", "gradient_checkpointing", "batch_size", "seq_len"]) });
		expect(step.kind).toBe("blocked_by_pin");
		expect(step.pinnedKnob).toBe("eval_batch_size");
	});

	it("doesn't offer shorter sequences as a saving once the run fits", () => {
		const knobs = { ...currentKnobs(dev), eval_batch_size: 8 };
		const visited = new Set<string>();
		for (let i = 0; i < 6; i++) {
			const step = nextStep({ ...base(), knobs, pinned: new Set(), visited });
			if (step.kind !== "move" || !step.move) break;
			expect(step.move.knobId).not.toBe("seq_len");
			visited.add(step.move.knobId);
		}
	});

	it("says honestly that nothing safe can be moved when no grid covers the fix", () => {
		const step = nextStep({ ...base(golden), pinned: new Set() });
		expect(step.kind).toBe("nothing_safe");
		expect(step.nodes).toContain("eval");
	});
});

describe("measure-first offers (tight settings)", () => {
	const fitsMove = { knobId: "batch_size", to: 8, targetId: "cost.total", delta: -0.5, clearsFailure: false, fit: "fits", risk: 0.002, relative: -0.05 } as const;
	const tightMove = { knobId: "eval_batch_size", to: 16, targetId: "cost.total", delta: -3, clearsFailure: false, fit: "tight", risk: 0.2, relative: -0.3 } as const;
	const stub = (risk: number) => ({
		name: "grid" as const,
		ready: () => true,
		predict: () => ({ "risk.oom": { point: risk, lo: risk, hi: risk, unit: "fraction", status: "predicted", model_id: "t", source: "grid", reason: null } }) as never,
		sensitivity: () => [],
		moves: (_r: unknown, _o: unknown, opts?: { includeTight?: boolean }) => (opts?.includeTight ? [tightMove, fitsMove] : [fitsMove]) as never,
	});
	const ctxFor = (risk: number, visited = new Set<string>()) => ({
		record: dev,
		provider: stub(risk),
		knobs: currentKnobs(dev),
		pinned: new Set<string>(),
		objective: cost,
		visited,
	});

	it("offers to measure a tight setting that saves 10% more than the best fitting move", () => {
		const step = nextStep(ctxFor(0.001));
		expect(step.kind).toBe("probe_offer");
		expect(step.move?.knobId).toBe("eval_batch_size");
	});

	it("moves on to the fitting move once the offer is skipped", () => {
		const step = nextStep(ctxFor(0.001, new Set(["probe:eval_batch_size"])));
		expect(step.kind).toBe("move");
		expect(step.move?.fit).toBe("fits");
	});

	it("never offers a tight setting while the run itself is at risk", () => {
		expect(nextStep(ctxFor(0.3)).kind).not.toBe("probe_offer");
	});

	it("uses the 'would stop' wording only when the run likely fails", () => {
		const tight = nextStep({ ...ctxFor(0.3), provider: { ...stub(0.3), moves: () => [{ ...fitsMove, clearsFailure: true }] as never } });
		expect(tight.textKey).toBe("guide.rule.generic.fix.line");
	});
});
