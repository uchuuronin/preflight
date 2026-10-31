// @vitest-environment jsdom
import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import en from "../../i18n/en.json";
import { CampaignCalendar } from "./Calendar";
import abstainingRaw from "./fixtures/pretrain-abstaining.plan.json";
import datedRaw from "./fixtures/pretrain-dated.plan.json";
import { type CampaignPlan, hasSchedule, lanesOf } from "./plan";

// Two real outputs of B's planner (see each fixture's $source).
const abstaining = abstainingRaw as unknown as CampaignPlan;
const dated = datedRaw as unknown as CampaignPlan;

afterEach(cleanup);

describe("the campaign calendar draws B's plan and nothing else", () => {
	it("puts every run in the lane B chose, and the cut runs apart with B's reason", () => {
		const { rows, unplaced } = lanesOf(dated);
		expect(rows.map((r) => r.lane)).toEqual(["chiya", "ana", "ben"]);
		const placed = rows.flatMap((r) => r.cells);
		expect(placed.every((c) => !c.cut)).toBe(true);
		expect(placed.length + unplaced.length).toBe(dated.cells.length);
		expect(unplaced.map((c) => c.id).sort()).toEqual([...dated.cut_list].sort());
		render(<CampaignCalendar plan={dated} />);
		const region = screen.getByRole("region", { name: en["campaign.title"] });
		expect(within(region).getByText(en["campaign.unplaced"].replace("{n}", String(unplaced.length)))).toBeTruthy();
		for (const lane of ["chiya", "ana", "ben"]) expect(within(region).getByText(lane)).toBeTruthy();
	});

	it("gives the finish as B's p10 to p90 when B dated it", () => {
		render(<CampaignCalendar plan={dated} />);
		const finish = screen.getByText(/Most likely/);
		expect(finish.textContent).toMatch(/UTC; 80% between/);
	});

	it("says why when B won't give a total or a finish, and marks the runs with no run time", () => {
		const { container } = render(<CampaignCalendar plan={abstaining} />);
		expect(abstaining.finish_s.status).toBe("abstain");
		expect(screen.getAllByText(/Not predicted: .*can't be dated/).length).toBeGreaterThan(0);
		const noTime = abstaining.cells.filter((c) => c.time.status === "abstain").length;
		expect(container.querySelectorAll(".pf-cal__runs li.is-abstain")).toHaveLength(noTime);
		expect(screen.getByText(abstaining.notes[0]!)).toBeTruthy();
	});

	it("lists runs by lane until B's plan carries start times, then draws them against time", () => {
		expect(hasSchedule(dated)).toBe(false);
		const withStarts: CampaignPlan = { ...dated, cells: dated.cells.map((c, i) => ({ ...c, slot: 0, start_s: i * 1000 })) };
		expect(hasSchedule(withStarts)).toBe(true);
		const { container } = render(<CampaignCalendar plan={withStarts} />);
		expect(container.querySelectorAll(".pf-cal__bar").length).toBe(lanesOf(dated).rows.flatMap((r) => r.cells).length);
	});
});
