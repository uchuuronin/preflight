import { describe, expect, it, vi } from "vitest";
import record from "../workspace/fixtures/handmade-finetune.record.json";
import type { FlightRecord, KnobValue } from "../workspace/record/types";
import { GridProvider } from "../workspace/provider/GridProvider";
import { currentKnobs } from "../workspace/provider/PredictionProvider";
import type { WorkspaceActions } from "../workspace/state";
import { nextStep } from "./ruleTour";
import { buildTools, executeTool } from "./tools";

const rec = record as unknown as FlightRecord;
const provider = new GridProvider();

function fakeActions(): WorkspaceActions & { moves: Array<[string, KnobValue]>; approvals: string[] } {
	const moves: Array<[string, KnobValue]> = [];
	return {
		moves,
		approvals: [],
		setKnob: (id, v) => {
			moves.push([id, v]);
			return true;
		},
		pin: vi.fn(),
		highlight: vi.fn(),
		propose: vi.fn(),
		snapshot: vi.fn(() => ({ id: "s1", name: "s1", knobs: {}, estimates: {} })),
		compare: vi.fn(() => []),
		showGhost: vi.fn(),
		setPlayhead: vi.fn(),
		reset: vi.fn(),
	};
}

describe("guide tool schema", () => {
	it("removes pinned knobs from set_knob's enum every turn", () => {
		const tools = buildTools(rec, new Set(["eval_batch_size", "gpu"]));
		const setKnob = tools.find((x) => x.name === "set_knob");
		const props = setKnob?.parameters.properties as { knob_id: { enum: string[] } };
		expect(props.knob_id.enum).not.toContain("eval_batch_size");
		expect(props.knob_id.enum).not.toContain("gpu");
		expect(props.knob_id.enum).toContain("batch_size");
	});

	it("drops set_knob entirely when everything is pinned", () => {
		const tools = buildTools(rec, new Set(rec.knobs.map((k) => k.id)));
		expect(tools.some((x) => x.name === "set_knob")).toBe(false);
	});

	it("refuses a set_knob on a pinned knob even if the model sends one", () => {
		const actions = fakeActions();
		const res = executeTool(
			"set_knob",
			{ knob_id: "eval_batch_size", value: 8, why: "x" },
			{ record: rec, provider, actions, knobs: currentKnobs(rec), pinned: new Set(["eval_batch_size"]), requestApproval: vi.fn() },
		);
		expect(res.ok).toBe(false);
		expect(actions.moves).toHaveLength(0);
	});

	it("never runs a money-spending tool, only asks for approval", () => {
		const actions = fakeActions();
		const requestApproval = vi.fn();
		for (const name of ["run_probe", "propose_patch", "run_verify"]) {
			const res = executeTool(name, {}, { record: rec, provider, actions, knobs: currentKnobs(rec), pinned: new Set(), requestApproval });
			expect(res).toEqual({ ok: true, result: { status: "awaiting_person_approval" } });
		}
		expect(requestApproval).toHaveBeenCalledTimes(3);
	});
});

describe("rule-based tour", () => {
	const base = { record: rec, provider, knobs: currentKnobs(rec), objective: "cost" as const, visited: new Set<string>() };

	it("opens on the OOM fix", () => {
		const step = nextStep({ ...base, pinned: new Set() });
		expect(step.kind).toBe("move");
		expect(step.knobId).toBe("eval_batch_size");
		expect(step.nodes).toContain("eval");
	});

	it("never offers a pinned knob, and names the pin when it blocks the fix", () => {
		const pinned = new Set(rec.knobs.map((k) => k.id).filter((id) => id !== "lora_rank"));
		const step = nextStep({ ...base, pinned });
		expect(step.kind).toBe("blocked_by_pin");
		expect(step.pinnedKnob).toBe("eval_batch_size");
	});

	it("doesn't offer shorter sequences as a cost saving once the run fits", () => {
		const knobs = { ...base.knobs, eval_batch_size: 8 };
		const visited = new Set<string>();
		for (let i = 0; i < 6; i++) {
			const step = nextStep({ ...base, knobs, pinned: new Set(), visited });
			if (step.kind !== "move" || !step.knobId) break;
			expect(step.knobId).not.toBe("seq_len");
			visited.add(step.knobId);
		}
	});
});
