import { describe, expect, it, vi } from "vitest";
import agentRaw from "../../../schema/examples/agent.json";
import goldenRaw from "../../../schema/examples/finetune.json";
import devRaw from "../workspace/fixtures/dev-finetune.record.json";
import { GridProvider } from "../workspace/provider/GridProvider";
import { currentKnobs, knobChoices, targetFor } from "../workspace/record/model";
import { type FlightRecord, type KnobValue, normaliseRecord } from "../workspace/record/types";
import type { TraceEntry } from "../workspace/state";
import { guardSetKnob, runBrowserCall } from "./liveGuide";
import { findingVisit, nextStep, nodesForKnob, SAMPLE_VISIT } from "./ruleTour";
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
		// Built here, not read off the golden's numbers (B's refill makes its eval batch 4 fit, and a
		// B200 clear the eval, Nov 2): every grid over the eval knobs and the GPU taken away, so no
		// grid covers a fix.
		const fixKnobs = new Set(["eval_batch_size", "eval_no_grad", "eval_every", "gpu_type"]);
		const noFix = { ...golden, predictions: { ...golden.predictions, grids: golden.predictions.grids.filter((g) => !g.axes.some((a) => fixKnobs.has(a.knob_id))) } };
		const step = nextStep({ ...base(noFix), pinned: new Set() });
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
	// Every dev finding names a knob of the record, so none is a finding step: these run unseeded.
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


describe("sample_more (agent runs whose cost range straddles the budget)", () => {
	const golden = normaliseRecord(agentRaw as unknown as FlightRecord);
	// The golden agent record's own cost range may or may not straddle its budget (fixture numbers
	// do, B's engine abstains under 10 tasks), so these tests set a range and a budget across it.
	const costId = targetFor(golden, "cost")!.id;
	const agent = {
		...golden,
		job: { ...golden.job, constraints: { ...golden.job.constraints, budget_usd: 150 } },
		predictions: {
			...golden.predictions,
			grids: golden.predictions.grids.filter((g) => g.target_id !== costId),
			estimates: { ...golden.predictions.estimates, [costId]: { point: 84, lo: 46, hi: 159, unit: "usd", status: "predicted", model_id: "m", source: "exact" } as never },
		},
	};

	it("is in the tool schema for agent runs only", () => {
		expect(buildTools(agent, new Set()).some((x) => x.name === "sample_more")).toBe(true);
		expect(buildTools(dev, new Set()).some((x) => x.name === "sample_more")).toBe(false);
	});

	it("asks for the person's click when the range straddles the budget, never runs on its own", () => {
		const c = ctx(agent);
		const res = executeTool("sample_more", { reason: "budget" }, c);
		expect(res).toEqual({ ok: true, result: { status: "awaiting_person_approval", tasks: 20 } });
		expect(c.approve).toHaveBeenCalledWith("sample_more", expect.objectContaining({ tasks: 20, budget: 150 }));
	});

	it("refuses when the range is entirely under or over the budget", () => {
		const under = { ...agent, job: { ...agent.job, constraints: { ...agent.job.constraints, budget_usd: 1000 } } };
		const c = ctx(under);
		expect(executeTool("sample_more", { reason: "x" }, c).ok).toBe(false);
		expect(c.approve).not.toHaveBeenCalled();
	});

	it("opens the tour with the sample offer, and moves on when it is skipped", () => {
		const base = { record: agent, provider, knobs: currentKnobs(agent), pinned: new Set<string>(), objective: cost };
		expect(nextStep({ ...base, visited: new Set() }).kind).toBe("sample_offer");
		expect(nextStep({ ...base, visited: new Set([SAMPLE_VISIT]) }).kind).not.toBe("sample_offer");
	});
});

import agentEvalRaw from "../workspace/fixtures/agent-eval-oct31.record.json";

describe("the agent tour on A's measured run (Oct 31)", () => {
	const real = normaliseRecord(agentEvalRaw as unknown as FlightRecord);
	const base = { record: real, provider, knobs: currentKnobs(real), pinned: new Set<string>(), objective: cost };

	it("opens on the retry loop found in the code and lights the tools node", () => {
		const step = nextStep({ ...base, visited: new Set() });
		expect(step.kind).toBe("finding");
		expect(step.finding?.code).toBe("agent.tool_retry_loop");
		expect(step.finding?.fix_template).toBe("agent.cap_tool_retries");
		expect(step.nodes).toEqual(["tools"]);
	});

	it("never offers running fewer tasks as a saving: an eval's tasks are its result", () => {
		const after = nextStep({ ...base, visited: new Set(real.findings.map(findingVisit)) });
		expect(after.move?.knobId).not.toBe("n_tasks");
		expect(after.kind).toBe("done");
		// The rule tour's tools are held to the same rule...
		const c = ctx(real);
		const res = executeTool("set_knob", { knob_id: "n_tasks", value: 62, why: "cheaper" }, c);
		expect(res.ok).toBe(false);
		const ranked = executeTool("sensitivity", { objective: "min_cost" }, c);
		expect(ranked.ok && (ranked.result as Array<{ knob_id: string; changes_results: boolean }>).find((k) => k.knob_id === "n_tasks")?.changes_results).toBe(true);
		// ...and so is the live guide, whose browser half runs its own set_knob and sensitivity.
		const choices = knobChoices(real, real.knobs.find((k) => k.id === "n_tasks")!);
		const fewer = choices.find((v) => typeof v === "number" && v < 500);
		expect(fewer).toBeDefined();
		expect(guardSetKnob(c, "n_tasks", fewer)).toBe("changes_results");
		const live = runBrowserCall({ call_id: "s", name: "sensitivity", arguments: { objective: "min_cost" }, runs_in: "browser" }, c, false);
		expect(live.tool === "sensitivity" && live.ranked.map((r) => r.knob_id)).not.toContain("n_tasks");
	});

	it("shows only findings no knob shows, critical ones included", () => {
		// Every finding on the other samples names a knob of its record: where a move ranks that
		// knob, the moves show it, so the tour opens with a move.
		for (const r of [dev, golden]) {
			const step = nextStep({ record: r, provider, knobs: currentKnobs(r), pinned: new Set(), objective: cost, visited: new Set() });
			expect(step.kind).not.toBe("finding");
		}
		// The agent example ranks no move for cost, so its retry-loop finding (on a knob no move
		// ranks) is said rather than ending the tour at "done" without it.
		const agent = normaliseRecord(agentRaw as unknown as FlightRecord);
		const said = nextStep({ record: agent, provider, knobs: currentKnobs(agent), pinned: new Set(), objective: cost, visited: new Set() });
		expect(said.kind).toBe("finding");
		expect(said.textKey).toBe("guide.finding.knob");
		expect(said.finding?.knob_id).toBe("tool_retry_cap");
		// A critical finding whose fix is in the code is shown like an error.
		const crit = { ...real, findings: real.findings.map((f) => ({ ...f, severity: "critical" as const })) };
		expect(nextStep({ ...base, record: crit, visited: new Set() }).kind).toBe("finding");
	});

	it("respects a pin on the knob a fix would need", () => {
		const c = ctx(real, new Set(["concurrency"]));
		expect(executeTool("set_knob", { knob_id: "concurrency", value: 8, why: "x" }, c).ok).toBe(false);
		expect(buildTools(real, new Set(["concurrency"])).find((x) => x.name === "set_knob")?.parameters.properties).toBeTruthy();
	});
});

describe("the nodes a knob lights", () => {
	const withAffects = (affects: string[] | undefined) =>
		({ ...dev, findings: [], knobs: dev.knobs.map((k) => (k.id === "eval_batch_size" ? { ...k, affects } : k)) }) as typeof dev;

	it("uses the record's own affects, not the table, and keeps only nodes on its map", () => {
		const [first, second] = dev.graph.nodes.map((n) => n.id).filter((n) => n !== "eval");
		expect(nodesForKnob(withAffects([second, "not_on_this_map", first]), "eval_batch_size")).toEqual([second, first]);
		// A knob the record says acts on nothing lights nothing.
		expect(nodesForKnob(withAffects([]), "eval_batch_size")).toEqual([]);
	});

	it("falls back to the table for a record from before affects", () => {
		expect(nodesForKnob(withAffects(undefined), "eval_batch_size")).toEqual(["eval"]);
	});
});
