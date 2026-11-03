/**
 * The tour proposes only a move that meets its own label (PM, Nov 2: under "Lowest cost" the LoRA
 * sample's first step was L4 → B300, +US$2.09). A saving step improves the aim; a fix step is the
 * fitting move that clears the failure at the least cost to the aim, and its words say so.
 */
import { describe, expect, it } from "vitest";
import en from "../i18n/en.json";
import { GridProvider } from "../workspace/provider/GridProvider";
import { SAMPLES } from "../workspace/record/load";
import { currentKnobs, objectiveTarget } from "../workspace/record/model";
import { normaliseRecord } from "../workspace/record/types";
import { nextStep } from "./ruleTour";

const provider = new GridProvider();
const AIMS = ["min_cost", "min_time"] as const;

describe("the rule-based tour keeps to its aim", () => {
	for (const sample of SAMPLES) {
		for (const kind of AIMS) {
			it(`${sample.id}, ${kind}`, async () => {
				const record = normaliseRecord(await sample.load());
				const objective = { kind } as never;
				const goal = objectiveTarget(record, kind);
				if (!goal) return; // this lens has no such aim
				const ctx = { record, provider, knobs: currentKnobs(record), objective, visited: new Set<string>(), pinned: new Set<string>() };
				const step = nextStep(ctx as never);
				if (step.kind !== "move" || !step.move) return;
				const m = step.move;
				const sign = goal.better === "higher" ? -1 : 1;
				if (!m.clearsFailure) {
					expect(sign * m.delta, `${m.knobId} → ${String(m.to)} must improve ${kind}`).toBeLessThanOrEqual(0);
					return;
				}
				// A fix: no other fitting move that clears the failure costs the aim less.
				const others = provider.moves(record, objective).filter((o) => o.fit === "fits" && o.clearsFailure);
				for (const o of others) expect(sign * o.delta).toBeGreaterThanOrEqual(sign * m.delta);
			});
		}
	}

	it("says a fix that costs more is the cheapest way to fit, not a saving", () => {
		for (const k of ["guide.rule.gpu_type.fix.why", "guide.rule.generic.fix.why"] as const) expect(en[k]).toMatch(/least/);
	});
});
