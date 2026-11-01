// @vitest-environment jsdom
import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import type { Campaign } from "../../../../schema/ts/contract";
import { CampaignCalendar } from "./Calendar";
import datedRaw from "./fixtures/pretrain-dated.plan.json";
import { type CampaignPlan, hasSchedule, lanesOf, planFromRecord, slotsOf } from "./plan";

const dated = datedRaw as unknown as CampaignPlan;

afterEach(cleanup);

/** The dated plan as B's to_campaign() writes it into a 0.5.0 record: floats for per-run cost. */
function asRecord05(plan: CampaignPlan): Campaign {
	return {
		cells: plan.cells.map((c) => ({
			id: c.id,
			knobs: c.knobs,
			gpu_hours: c.gpu_hours?.point ?? null,
			usd: c.usd?.point ?? null,
			segments: c.segments,
			lane: c.lane,
			cut: c.cut,
			time: c.time as never,
		})),
		lanes: plan.lanes,
		cut_list: plan.cut_list,
		completion_quantiles: plan.completion,
		gpu_hours: plan.gpu_hours as never,
		usd: plan.usd as never,
		finish_s: plan.finish_s as never,
		notes: plan.notes,
	};
}

describe("a shared record's campaign draws the same calendar as the live plan", () => {
	it("keeps every lane, run, cut and total from a 0.5.0 record", () => {
		const p = planFromRecord(asRecord05(dated))!;
		expect(p).not.toBeNull();
		expect(lanesOf(p).rows.map((r) => [r.lane, r.cells.map((c) => c.id)])).toEqual(lanesOf(dated).rows.map((r) => [r.lane, r.cells.map((c) => c.id)]));
		expect(lanesOf(p).unplaced.map((c) => c.id)).toEqual(lanesOf(dated).unplaced.map((c) => c.id));
		expect(p.usd).toEqual(dated.usd);
		expect(p.completion).toEqual(dated.completion);
		// A 0.5.0 record has only points per run: no interval, so none is shown as if exact.
		expect(p.cells.every((c) => c.gpu_hours === null && c.usd === null)).toBe(true);
	});

	it("reads 0.6.0's slot, reason and per-run intervals when the record has them", () => {
		const rec = asRecord05(dated);
		const cut = dated.cells.find((c) => c.cut)!;
		rec.cells = rec.cells!.map((c, i) => ({
			...c,
			slot: i % 2,
			reason: c.id === cut.id ? cut.reason : null,
			usd_estimate: dated.cells[i]!.usd,
		})) as never;
		const p = planFromRecord(rec)!;
		expect(p.cells.find((c) => c.id === cut.id)!.reason).toBe(cut.reason);
		expect(p.cells.map((c) => c.usd)).toEqual(dated.cells.map((c) => c.usd ?? null));
		expect(p.cells.map((c) => c.slot)).toEqual(dated.cells.map((_, i) => i % 2));
	});

	it("marks a run with no time in the record as not timed, with a reason, never as zero", () => {
		const rec = asRecord05(dated);
		const first = rec.cells!.find((c) => !c.cut)!;
		delete (first as { time?: unknown }).time;
		const p = planFromRecord(rec)!;
		const c = p.cells.find((x) => x.id === first.id)!;
		expect(c.time.status).toBe("abstain");
		expect(c.time.reason).toBeTruthy();
		const { container } = render(<CampaignCalendar plan={p} />);
		expect(container.querySelectorAll(".pf-cal__runs li.is-abstain").length).toBeGreaterThan(0);
	});

	it("draws nothing for a record without a campaign's runs", () => {
		expect(planFromRecord({ cells: [] })).toBeNull();
	});

	it("gives each slot of a lane its own row when runs go at once", () => {
		const est = (point: number) => ({ point, lo: point, hi: point, unit: "s", status: "predicted", model_id: "m", source: "exact" }) as never;
		const lane = dated.lanes[0]!;
		const two: CampaignPlan = {
			...dated,
			cells: dated.cells.map((c, i) => ({ ...c, lane: c.cut ? c.lane : lane, slot: i % 2, start_s: est(0) })),
		};
		expect(hasSchedule(two)).toBe(true);
		expect(slotsOf(lanesOf(two).rows[0]!.cells)).toHaveLength(2);
		const { container } = render(<CampaignCalendar plan={two} />);
		const firstLane = container.querySelector(".pf-cal__lane");
		expect(firstLane?.querySelectorAll(".pf-cal__track")).toHaveLength(2);
	});
});
