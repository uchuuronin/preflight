// @vitest-environment jsdom
import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import grpoRaw from "../fixtures/grpo-l4.record.json";
import devRaw from "../fixtures/dev-finetune.record.json";
import { GridProvider } from "../provider/GridProvider";
import { currentKnobs } from "../record/model";
import { type FlightRecord, normaliseRecord } from "../record/types";
import { WorkspaceProvider } from "../state";
import { rlProbe, spreadCount } from "../rl";
import { RlBand } from "./RlBand";
import { nextStep, rlVisit, type TourContext } from "../../guide/ruleTour";
import { SAMPLES } from "../record/load";

const grpo = normaliseRecord(grpoRaw as unknown as FlightRecord);
const open = (record = grpo) =>
	render(
		<WorkspaceProvider record={record} provider={new GridProvider()}>
			<RlBand />
		</WorkspaceProvider>,
	);

afterEach(cleanup);

describe("the GRPO sample is A's measured L4 run (PF510)", () => {
	it("carries A's measured reward signal, and truncation as not measured", () => {
		expect(SAMPLES.find((s) => s.id === "grpo")?.numbers).toBe("measured");
		const p = rlProbe(grpo)!;
		expect(p.rl?.pass_rate_per_prompt).toHaveLength(20);
		expect(p.rl?.group_size).toBe(8);
		expect(spreadCount(p)).toEqual({ k: 15, n: 20 });
		// A's log: the cap wasn't read on this run, so its 0.0 isn't a measurement.
		expect(p.rl?.truncation_rate ?? null).toBeNull();
		expect(grpo.share_token ?? null).toBeNull();
	});
});

describe("the RL views", () => {
	it("say the measured share with its Wilson interval, and mark each flat prompt", () => {
		const { container } = open();
		const lead = container.querySelector(".pf-rl__lead")!.textContent!;
		expect(lead).toContain("15 of 20 prompts");
		expect(lead).toMatch(/53%.*89%/);
		expect(container.querySelectorAll(".pf-rl__cell")).toHaveLength(20);
		expect(container.querySelectorAll(".pf-rl__cell.is-flat")).toHaveLength(15);
		expect(container.textContent).toContain("7 of 160 answers were correct");
	});

	it("never shows a truncation share the run didn't measure, only the engine's estimate", () => {
		const { container } = open();
		expect(container.textContent).toContain("Truncation wasn't measured on this run");
		expect(container.querySelector(".pf-rl__cap")).toBeTruthy();
	});

	it("draws nothing on a run that isn't RL", () => {
		const { container } = open(normaliseRecord(devRaw as unknown as FlightRecord));
		expect(container.querySelector("#pf-rl")).toBeNull();
	});
});

describe("the GRPO tour", () => {
	const ctx = (visited: Set<string>): TourContext => ({
		record: grpo,
		provider: new GridProvider(),
		knobs: currentKnobs(grpo),
		pinned: new Set(),
		objective: { kind: "min_time" },
		visited,
	});

	it("walks the reward spread, then completion length against the cap, before any move", () => {
		const v = new Set<string>();
		const a = nextStep(ctx(v));
		expect([a.kind, a.rlPart, a.textKey]).toEqual(["rl", "spread", "guide.rl.spread"]);
		v.add(rlVisit("spread"));
		const b = nextStep(ctx(v));
		// The longest completion reached the 512 cap on this run.
		expect([b.kind, b.rlPart, b.textKey]).toEqual(["rl", "length", "guide.rl.length"]);
		v.add(rlVisit("length"));
		expect(nextStep(ctx(v)).kind).not.toBe("rl");
	});

	it("judges the cap against the one the run was measured at, not the knob as it is now", () => {
		// Raising the knob doesn't make a completion that hit the measured cap "under" it.
		const raised = { ...currentKnobs(grpo), max_completion_len: 1024 };
		expect(nextStep({ ...ctx(new Set([rlVisit("spread")])), knobs: raised }).textKey).toBe("guide.rl.length");
		// A run whose longest completion stayed under its cap says so.
		const shorter = {
			...grpo,
			probes: grpo.probes.map((p) => (p.rl ? { ...p, rl: { ...p.rl, response_length: { ...p.rl.response_length!, max: 400 } } } : p)),
		};
		expect(nextStep({ ...ctx(new Set([rlVisit("spread")])), record: shorter }).textKey).toBe("guide.rl.length_under");
	});
});
