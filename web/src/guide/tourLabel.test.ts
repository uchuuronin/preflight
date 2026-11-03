/**
 * The tour proposes only a move that meets its own label (PM, Nov 2: under "Lowest cost" the LoRA
 * sample's first step was L4 → B300, +US$2.09). Walked step by step on every sample and both aims: a
 * saving improves the aim; a fix ranked on the aim is the fitting fix that costs the aim least, and
 * says so; a fix the aim can't rank (picked on memory) says that instead.
 */
import { describe, expect, it } from "vitest";
import en from "../i18n/en.json";
import { GridProvider } from "../workspace/provider/GridProvider";
import { SAMPLES } from "../workspace/record/load";
import { currentKnobs, objectiveTarget } from "../workspace/record/model";
import { normaliseRecord } from "../workspace/record/types";
import { campaignVisit, isGpuMove, nextStep, rlVisit, SAMPLE_VISIT } from "./ruleTour";

const provider = new GridProvider();
const AIMS = ["min_cost", "min_time"] as const;
const LEAST = ["guide.rule.generic.fix.why", "guide.rule.gpu_type.fix.why"];

describe("the rule-based tour keeps to its aim", () => {
	let byMemory = 0;
	for (const sample of SAMPLES) {
		for (const kind of AIMS) {
			it(`${sample.id}, ${kind}`, async () => {
				const record = normaliseRecord(await sample.load());
				const objective = { kind } as never;
				const goal = objectiveTarget(record, kind);
				if (!goal) return; // this lens has no such aim
				const sign = goal.better === "higher" ? -1 : 1;
				const visited = new Set<string>();
				for (let i = 0; i < 12; i++) {
					const ctx = { record, provider, knobs: currentKnobs(record), objective, visited, pinned: new Set<string>() };
					const step = nextStep(ctx as never);
					if (step.kind === "rl" && step.rlPart) visited.add(rlVisit(step.rlPart));
					else if (step.kind === "campaign" && step.campaignPart) visited.add(campaignVisit(step.campaignPart));
					else if (step.kind === "sample_offer") visited.add(SAMPLE_VISIT);
					else if (step.kind === "move" && step.move) {
						const m = step.move;
						visited.add(m.knobId);
						if (m.targetId !== goal.id) {
							// ranked on memory: its reason mustn't claim anything about the aim
							byMemory++;
							expect(LEAST).not.toContain(step.whyKey);
							continue;
						}
						if (!m.clearsFailure) {
							expect(sign * m.delta, `${m.knobId} → ${String(m.to)} must improve ${kind}`).toBeLessThanOrEqual(0);
							continue;
						}
						const others = provider.moves(record, objective).filter((o) => o.fit === "fits" && o.clearsFailure && !visited.has(o.knobId));
						// A GPU move only when nothing on this GPU clears it (PM, Nov 3); the cheapest within its group.
						if (isGpuMove(m)) expect(others.filter((o) => !isGpuMove(o)).map((o) => o.knobId)).toEqual([]);
						for (const o of others.filter((o) => isGpuMove(o) === isGpuMove(m))) expect(sign * o.delta).toBeGreaterThanOrEqual(sign * m.delta);
					} else break;
				}
			});
		}
	}

	it("meets the memory-ranked case on some sample, so the check above isn't empty", () => {
		expect(byMemory).toBeGreaterThan(0);
	});

	it("says a fix that costs more is the cheapest way left to fit, not a saving", () => {
		for (const k of LEAST) expect(en[k as keyof typeof en]).toMatch(/least/);
	});
});
