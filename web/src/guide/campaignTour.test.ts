import { describe, expect, it } from "vitest";
import campaignRaw from "../workspace/fixtures/campaign-grid.record.json";
import { planFromRecord } from "../workspace/campaign/plan";
import { GridProvider } from "../workspace/provider/GridProvider";
import { currentKnobs } from "../workspace/record/model";
import { type FlightRecord, normaliseRecord } from "../workspace/record/types";
import { SAMPLES } from "../workspace/record/load";
import { campaignParts, campaignVisit, nextStep, type TourContext } from "./ruleTour";

const record = normaliseRecord(campaignRaw as unknown as FlightRecord);
const campaign = planFromRecord(record.campaign!, record.models)!;
const base = (visited: Set<string>, withCampaign = true): TourContext => ({
	record,
	provider: new GridProvider(),
	knobs: currentKnobs(record),
	pinned: new Set(),
	objective: { kind: "min_cost" },
	visited,
	campaign: withCampaign ? campaign : null,
});

describe("the campaign tour on B's plan (sample PF420)", () => {
	it("is a sample, and its plan is B's: dated, with runs cut for the budget", () => {
		expect(SAMPLES.some((s) => s.id === "campaign-grid")).toBe(true);
		expect(campaign.cells).toHaveLength(12);
		expect(campaign.cut_list).toHaveLength(5);
		expect(campaign.finish_s.status).toBe("predicted");
		expect(Object.keys(campaign.completion).sort()).toEqual(["p10", "p50", "p90"]);
	});

	it("walks the plan first, in order: what it costs, what the budget cuts, when it ends", () => {
		const v = new Set<string>();
		const keys: string[] = [];
		for (let i = 0; i < 3; i++) {
			const s = nextStep(base(v));
			expect(s.kind).toBe("campaign");
			keys.push(s.textKey);
			v.add(campaignVisit(s.campaignPart!));
		}
		expect(keys).toEqual(["guide.campaign.grid", "guide.campaign.cut", "guide.campaign.finish"]);
		// Then the run's own tour.
		expect(nextStep(base(v)).kind).not.toBe("campaign");
	});

	it("skips the cut step when nothing was cut, and says why when the finish isn't dated", () => {
		const uncut = { ...campaign, cut_list: [] };
		expect(campaignParts({ campaign: uncut, visited: new Set() })).toEqual(["grid", "finish"]);
		const undated = { ...campaign, completion: {}, finish_s: { ...campaign.finish_s, point: null, lo: null, hi: null, status: "abstain" as const, reason: "in 8% of draws a run never finishes" } };
		const v = new Set([campaignVisit("grid"), campaignVisit("cut")]);
		const s = nextStep({ ...base(v), campaign: undated });
		expect(s.textKey).toBe("guide.campaign.finish_abstain");
	});

	it("doesn't walk a plan that isn't on screen (flag off, or no campaign)", () => {
		expect(nextStep(base(new Set(), false)).kind).not.toBe("campaign");
	});
});
