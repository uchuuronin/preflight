import { describe, expect, it } from "vitest";
import devRaw from "../workspace/fixtures/dev-finetune.record.json";
import { GridProvider } from "../workspace/provider/GridProvider";
import { currentKnobs } from "../workspace/record/model";
import { type FlightRecord, type KnobValue, normaliseRecord, type ObjectiveKind } from "../workspace/record/types";
import { breaksLimit, type Limits, NO_LIMITS } from "./limits";
import { guardSetKnob, runBrowserCall } from "./liveGuide";
import { findingVisit, nextStep, type TourStep } from "./ruleTour";
import { fakeActions } from "./testkit";
import { executeTool } from "./tools";

const lora = normaliseRecord(devRaw as unknown as FlightRecord);
const provider = new GridProvider();
const A100 = "A100-SXM-80GB";
const AIMS: ObjectiveKind[] = ["min_cost", "min_time", "fit_memory", "min_risk", "max_throughput"];
const a100Only: Limits = { ...NO_LIMITS, allowed: { gpu_type: [A100] } };

/** The rule tour as the visitor would take it: apply every move it offers, up to the step cap. */
function tour(kind: ObjectiveKind, limits: Limits, knobs = currentKnobs(lora)) {
	const steps: TourStep[] = [];
	const visited = new Set<string>(["sample:more"]);
	let k: Record<string, KnobValue> = { ...knobs };
	for (let i = 0; i < 8; i++) {
		const s = nextStep({ record: lora, provider, knobs: k, pinned: new Set(), objective: { kind }, visited, limits });
		steps.push(s);
		if (s.kind !== "move" || !s.move) break;
		k = { ...k, [s.move.knobId]: s.move.to };
		visited.add(s.move.knobId);
	}
	return { steps, knobs: k };
}

function liveCtx(limits: Limits, knobs = currentKnobs(lora)) {
	const f = fakeActions();
	return { record: lora, provider, actions: f.actions, knobs, pinned: new Set<string>(), limits, requestApproval: () => {} };
}

describe("limits are hard limits for both tours", () => {
	it("with the GPU kept on A100, no tour for any aim suggests another GPU", () => {
		// Without the limit the tour does move the GPU (to H100), so the limit is what stops it.
		expect(tour("min_cost", NO_LIMITS).steps.some((s) => s.move?.knobId === "gpu_type")).toBe(true);
		for (const kind of AIMS) {
			const { steps, knobs } = tour(kind, a100Only);
			expect(steps.filter((s) => s.move?.knobId === "gpu_type")).toEqual([]);
			expect(knobs.gpu_type).toBe(A100);
			// The live guide's browser half: ranked without it, and a move to it refused.
			const c = liveCtx(a100Only);
			const ranked = runBrowserCall({ call_id: "s", name: "sensitivity", arguments: { objective: kind }, runs_in: "browser" }, c, false);
			expect(ranked.tool === "sensitivity" && ranked.ranked.map((r) => r.knob_id)).not.toContain("gpu_type");
			expect(guardSetKnob(c, "gpu_type", "H100-SXM-80GB")).toBe("outside_limits");
			const tool = executeTool("sensitivity", { objective: kind }, c);
			expect(tool.ok && (tool.result as Array<{ knob_id: string }>).map((r) => r.knob_id)).not.toContain("gpu_type");
			expect(executeTool("set_knob", { knob_id: "gpu_type", value: "L40S", why: "x" }, c).ok).toBe(false);
		}
	});

	it("'Not this' on batch size moves to another lever, or says plainly that none is left", () => {
		// Unconstrained, the tour runs eval batch, GPU, then batch size.
		const free = tour("min_cost", NO_LIMITS).steps.map((s) => s.move?.knobId ?? s.kind);
		// ...then a problem found on a knob no move ranks (the tour's last stop before done).
		expect(free).toEqual(["eval_batch_size", "gpu_type", "batch_size", "finding"]);
		// Turned down at the GPU step, the next suggestion is a different lever: batch size.
		const fixed = { ...currentKnobs(lora), eval_batch_size: 8 };
		const declinedGpu: Limits = { ...NO_LIMITS, excluded: { gpu_type: ["H100-SXM-80GB"] }, declined: ["gpu_type"] };
		const afterGpu = nextStep({ record: lora, provider, knobs: fixed, pinned: new Set(), objective: { kind: "min_cost" }, visited: new Set(["sample:more", "eval_batch_size", "gpu_type"]), limits: declinedGpu });
		expect(afterGpu.move?.knobId).toBe("batch_size");
		// Turned down at batch size on A100: batch size is the last lever there, so the tour ends,
		// and the live guide's ranking no longer offers batch size at any value.
		const at = tour("min_cost", a100Only).steps.find((s) => s.move?.knobId === "batch_size");
		expect(at?.move?.to).toBe(8);
		const declined: Limits = { ...a100Only, excluded: { batch_size: [8] }, declined: ["batch_size"] };
		const next = nextStep({ record: lora, provider, knobs: fixed, pinned: new Set(), objective: { kind: "min_cost" }, visited: new Set(["sample:more", "eval_batch_size", "batch_size"]), limits: declined });
		expect(next.move?.knobId).not.toBe("batch_size");
		// Once its findings are read, nothing is left.
		const read = new Set(["sample:more", "eval_batch_size", "batch_size", ...lora.findings.map(findingVisit)]);
		expect(nextStep({ record: lora, provider, knobs: fixed, pinned: new Set(), objective: { kind: "min_cost" }, visited: read, limits: declined }).kind).toBe("done");
		expect(["done", "finding"]).toContain(next.kind);
		const c = liveCtx(declined, fixed);
		const ranked = runBrowserCall({ call_id: "s", name: "sensitivity", arguments: { objective: "min_cost" }, runs_in: "browser" }, c, false);
		expect(ranked.tool === "sensitivity" && ranked.ranked.map((r) => r.knob_id)).not.toContain("batch_size");
		expect(guardSetKnob(c, "batch_size", 8)).toBe("outside_limits");
		// Where another lever exists, "Not this" leads to it: eval batch turned down at the start.
		const declinedEval: Limits = { ...NO_LIMITS, excluded: { eval_batch_size: [8] }, declined: ["eval_batch_size"] };
		const other = nextStep({ record: lora, provider, knobs: currentKnobs(lora), pinned: new Set(), objective: { kind: "min_cost" }, visited: new Set(["sample:more", "eval_batch_size"]), limits: declinedEval });
		expect(other.kind).toBe("move");
		expect(other.move?.knobId).not.toBe("eval_batch_size");
	});

	it("a budget, a deadline and a quota refuse moves past them, but not moves back under them", () => {
		const knobs = { ...currentKnobs(lora), eval_batch_size: 8 };
		const before = provider.predict(lora, knobs);
		const cost = before["cost.total"]?.point as number;
		const after = provider.predict(lora, { ...knobs, gpu_type: "L40S" });
		const pricier = (after["cost.total"]?.point as number) > cost;
		const tight: Limits = { ...NO_LIMITS, budgetUsd: cost * 1.0001 };
		if (pricier) expect(breaksLimit(lora, tight, "gpu_type", "L40S", after, before, knobs)).toBe("over_budget");
		// Already over the budget: a move that lowers the cost is still allowed.
		const under: Limits = { ...NO_LIMITS, budgetUsd: cost / 10 };
		const cheaper = provider.predict(lora, { ...knobs, batch_size: 8 });
		expect((cheaper["cost.total"]?.point as number) < cost).toBe(true);
		expect(breaksLimit(lora, under, "batch_size", 8, cheaper, before, knobs)).toBeNull();
		const time = before["time.total"]?.point as number;
		expect(breaksLimit(lora, { ...NO_LIMITS, deadlineS: time / 10 }, "batch_size", 8, cheaper, before, knobs)).toBeNull();
		expect(breaksLimit(lora, { ...NO_LIMITS, gpuHours: 0.001 }, "gpu_type", "L40S", after, before, knobs)).not.toBeNull();
	});
});

describe("the review's gaps: fixes, frozen knobs, the guide's pin tool", () => {
	it("a code fix never sets a value the visitor ruled out or a knob they froze", async () => {
		const { candidatesFor } = await import("../fix/candidates");
		const f2 = lora.findings.find((f) => f.id === "f2")!;
		const base = { record: lora, provider, knobs: currentKnobs(lora), pinned: new Set<string>(), objective: { kind: "min_cost" as const } };
		const free = candidatesFor(f2, { ...base, limits: NO_LIMITS });
		const first = free[0]!;
		expect(first.template_id).toBe("eval.reduce_batch");
		const ruledOut = candidatesFor(f2, { ...base, limits: { ...NO_LIMITS, excluded: { eval_batch_size: [first.value as number] } } });
		expect(ruledOut.find((c) => c.template_id === "eval.reduce_batch")?.value).not.toBe(first.value);
		const frozen = candidatesFor(f2, { ...base, limits: { ...NO_LIMITS, frozen: ["gradient_checkpointing"] } });
		expect(frozen.map((c) => c.template_id)).not.toContain("memory.gradient_checkpointing");
		const noOn = candidatesFor(f2, { ...base, limits: { ...NO_LIMITS, excluded: { gradient_checkpointing: [true] } } });
		expect(noOn.map((c) => c.template_id)).not.toContain("memory.gradient_checkpointing");
	});

	it("'Don't change' on the GPU holds in every step, the pin-in-the-way step included", () => {
		const frozen: Limits = { ...NO_LIMITS, frozen: ["gpu_type"] };
		for (const kind of AIMS) {
			const { steps, knobs } = tour(kind, frozen);
			expect(steps.filter((s) => s.move?.knobId === "gpu_type" || s.pinnedKnob === "gpu_type")).toEqual([]);
			expect(knobs.gpu_type).toBe(A100);
		}
		// With every other lever pinned, the step that names a pin in the way never names the GPU.
		const pinned = new Set(lora.knobs.map((k) => k.id).filter((k) => k !== "gpu_type"));
		const s = nextStep({ record: lora, provider, knobs: { ...currentKnobs(lora), eval_batch_size: 8 }, pinned, objective: { kind: "min_cost" }, visited: new Set(["sample:more"]), limits: frozen });
		expect(s.pinnedKnob).not.toBe("gpu_type");
		expect(s.move?.knobId).not.toBe("gpu_type");
	});

	it("the guide can't unpin a knob the visitor said not to change", () => {
		const c = liveCtx({ ...NO_LIMITS, frozen: ["gpu_type"] });
		c.pinned.add("gpu_type");
		const out = runBrowserCall({ call_id: "p", name: "pin", arguments: { knob_id: "gpu_type", pinned: false }, runs_in: "browser" }, c, false);
		expect(out.tool === "pin" && out.pinned).toBe(true);
		expect(executeTool("pin", { knob_id: "gpu_type", pinned: false }, c).ok).toBe(false);
		expect(guardSetKnob(c, "gpu_type", "H100-SXM-80GB")).not.toBeNull();
	});

	it("a move whose range, not just its middle, passes the budget is refused", () => {
		const est = (point: number, hi: number) => ({ "cost.total": { point, lo: point * 0.8, hi, unit: "usd", status: "predicted", model_id: "t", source: "grid", reason: null } }) as never;
		const knobs = currentKnobs(lora);
		// Middle US$38, top of the range US$45, budget US$40: refused.
		expect(breaksLimit(lora, { ...NO_LIMITS, budgetUsd: 40 }, "batch_size", 8, est(38, 45), est(30, 35), knobs)).toBe("over_budget");
		// The whole range under US$40: fine.
		expect(breaksLimit(lora, { ...NO_LIMITS, budgetUsd: 40 }, "batch_size", 8, est(30, 39), est(30, 35), knobs)).toBeNull();
		// Already past it and coming down: allowed.
		expect(breaksLimit(lora, { ...NO_LIMITS, budgetUsd: 40 }, "batch_size", 8, est(42, 48), est(50, 60), knobs)).toBeNull();
	});
});

describe("releasing a pin needs the visitor's click", () => {
	it("the guide's unpin only asks; a frozen knob isn't even asked about", () => {
		const asked: string[] = [];
		const c = { ...liveCtx(NO_LIMITS), requestUnpin: (id: string) => asked.push(id) };
		c.pinned.add("batch_size");
		const out = runBrowserCall({ call_id: "p", name: "pin", arguments: { knob_id: "batch_size", pinned: false }, runs_in: "browser" }, c, false);
		expect(out.tool === "pin" && out.pinned).toBe(true);
		expect(c.pinned.has("batch_size")).toBe(true);
		expect(asked).toEqual(["batch_size"]);
		const f = { ...liveCtx({ ...NO_LIMITS, frozen: ["gpu_type"] }), requestUnpin: (id: string) => asked.push(id) };
		f.pinned.add("gpu_type");
		runBrowserCall({ call_id: "p", name: "pin", arguments: { knob_id: "gpu_type", pinned: false }, runs_in: "browser" }, f, false);
		expect(asked).toEqual(["batch_size"]);
	});
});
