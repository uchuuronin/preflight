import { describe, expect, it } from "vitest";
import en from "../i18n/en.json";
import gnnRaw from "../workspace/fixtures/gnn_pyg-nov1.record.json";
import retrievalRaw from "../workspace/fixtures/retrieval_st-nov1.record.json";
import visionRaw from "../workspace/fixtures/vision_timm-nov1.record.json";
import { GridProvider } from "../workspace/provider/GridProvider";
import { currentKnobs } from "../workspace/record/model";
import { type FlightRecord, type LoadedRecord, normaliseRecord } from "../workspace/record/types";
import { findingVisit, nextStep, type TourStep } from "./ruleTour";

const provider = new GridProvider();
const load = (raw: unknown) => normaliseRecord(raw as FlightRecord);

/** The tour as a visitor walks it: each step taken (a move applied, a finding read) until done. */
function walk(record: LoadedRecord, max = 6): TourStep[] {
	const steps: TourStep[] = [];
	const visited = new Set<string>();
	let knobs = currentKnobs(record);
	for (let i = 0; i < max; i++) {
		const step = nextStep({ record, provider, knobs, pinned: new Set(), objective: record.job.objective ?? { kind: "min_time" }, visited });
		steps.push(step);
		if (step.kind === "finding" && step.finding) visited.add(findingVisit(step.finding));
		else if (step.kind === "move" && step.move) {
			visited.add(step.move.knobId);
			knobs = { ...knobs, [step.move.knobId]: step.move.to };
		} else break;
	}
	return steps;
}

// A's L4 smoke records of D's zoo scripts (Nov 1). Each script's planted problem is named in its
// own docstring; the tour has to reach it from the record alone.
describe("each lens's tour reaches its planted problem", () => {
	it("retrieval: the small contrastive batch, on the negatives node", () => {
		const steps = walk(load(retrievalRaw));
		const found = steps.find((s) => s.kind === "finding");
		expect(found?.finding?.code).toBe("retrieval.small_batch");
		expect(found?.textKey).toBe("guide.finding.knob");
		expect(found?.nodes).toContain("negatives");
		// B's 1% chance of running out is said as a chance, never as "likely to fail".
		expect(steps.map((s) => s.textKey)).not.toContain("guide.nothing_safe");
	});

	it("GNN: the outer hops' fan-out, by measured memory when time isn't predicted", () => {
		const record = load(gnnRaw);
		const first = walk(record)[0]!;
		expect(first.kind).toBe("move");
		expect(first.textKey).toBe("guide.rule.aim_unpredicted.line");
		expect(first.move?.knobId).toMatch(/^fanout\.hop[23]$/);
		expect(first.move?.targetId).toBe("memory.peak");
		expect(first.nodes).toContain("sampler");
	});

	it("vision: augmentation on the CPU, on the augment node", () => {
		const steps = walk(load(visionRaw));
		const found = steps.find((s) => s.kind === "finding");
		expect(found?.finding?.code).toBe("vision.cpu_augment");
		expect(found?.nodes).toContain("augment");
	});
});

describe("the lens samples name every knob and readout in words", () => {
	it("has a label for each knob and target the three records carry", () => {
		for (const raw of [retrievalRaw, gnnRaw, visionRaw]) {
			const r = load(raw);
			const keys = [...r.knobs.map((k) => k.label_key), ...r.predictions.targets.map((t) => t.label_key)];
			expect(keys.filter((k) => !(k in en)), r.job.lens).toEqual([]);
		}
	});
});
