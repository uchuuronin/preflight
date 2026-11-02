import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, beforeAll, describe, expect, it } from "vitest";
import devRaw from "../../workspace/fixtures/dev-finetune.record.json";
import { type Estimate, type FlightRecord, type KnobValue, normaliseRecord } from "../../workspace/record/types";
import { FreeTierPanel } from "./FreeTierPanel";
import { registerFreeTierStrings } from "./strings";

const record = normaliseRecord(devRaw as unknown as FlightRecord);
const H = 3600;

function est(point: number, unit: string, ratio = 1.15): Estimate {
	return { point, lo: point / ratio, hi: point * ratio, unit, status: "predicted", model_id: "t", source: "grid", level: 0.9 } as Estimate;
}

function renderPanel(gpu: KnobValue, memGb: number, hours: number) {
	const estimates = { "memory.peak": est(memGb * 1e9, "bytes", 1.1), "time.total": est(hours * H, "s") };
	return render(<FreeTierPanel record={record} knobs={{ ...{}, gpu_type: gpu }} estimates={estimates} />);
}

function row(name: string) {
	return within(screen.getByRole("listitem", { name }));
}

describe("FreeTierPanel", () => {
	beforeAll(registerFreeTierStrings);
	afterEach(cleanup);

	it("lists the three free tiers with their session caps", () => {
		renderPanel("T4", 9, 6);
		expect(screen.getAllByRole("listitem").map((li) => li.getAttribute("aria-label"))).toEqual(["Google Colab (free)", "Kaggle Notebooks", "Modal (Starter credit)", "Lightning AI (free plan)"]);
		expect(row("Google Colab (free)").getByText("12 h sessions")).toBeTruthy();
		expect(row("Lightning AI (free plan)").getByText("4 h sessions")).toBeTruthy();
	});

	it("says one session on a T4 run that fits", () => {
		renderPanel("T4", 9, 6);
		expect(row("Google Colab (free)").getByText("One session")).toBeTruthy();
		expect(row("Kaggle Notebooks").getByText("One session")).toBeTruthy();
	});

	it("says how many sessions and how often to save when it runs long", () => {
		renderPanel("T4", 9, 20);
		const colab = row("Google Colab (free)");
		expect(colab.getByText("2 sessions")).toBeTruthy();
		expect(colab.getByText(/Save a checkpoint at least every 11 h 45 min/)).toBeTruthy();
	});

	it("says it won't fit when peak memory exceeds the free GPU, with both sizes", () => {
		renderPanel("T4", 15, 2);
		const colab = row("Google Colab (free)");
		expect(colab.getByText("Won't fit")).toBeTruthy();
		// The dev record is a sample (fixture models), so its need is illustrative, as on the readouts.
		expect(colab.getByText(/Needs 15.4 GiB at its peak \(illustrative\); a T4 has 14.7 GiB/)).toBeTruthy();
	});

	it("says where the memory need comes from on a real record: the estimate's own status", () => {
		const real = { ...record, record_id: "01REALRUN", predictions: { ...record.predictions, grids: [], estimates: {} } };
		const estimates = { "memory.peak": est(15e9, "bytes", 1.1), "time.total": est(2 * H, "s") };
		render(<FreeTierPanel record={real} knobs={{ gpu_type: "T4" }} estimates={estimates} />);
		expect(row("Google Colab (free)").getByText(/Needs 15.4 GiB at its peak \(predicted\); a T4 has 14.7 GiB/)).toBeTruthy();
	});

	it("asks for a probe on the free GPU when the run was measured elsewhere", () => {
		renderPanel("A100-SXM-80GB", 9, 6);
		expect(row("Kaggle Notebooks").getByText("Measure on T4 or P100")).toBeTruthy();
		expect(row("Lightning AI (free plan)").getByText("Measure on L40S, H100 or H200")).toBeTruthy();
	});

	it("names its sources with the date they were read", () => {
		renderPanel("T4", 9, 6);
		const link = screen.getByRole("link", { name: "Colab FAQ" });
		expect(link.getAttribute("href")).toBe("https://research.google.com/colaboratory/faq.html");
		expect(screen.getAllByText(/read 27 Oct 2026/)).toHaveLength(5);
	});

	it("tells T4 and P100 tiers to train in fp16, firmly when the plan is set to bf16", () => {
		renderPanel("T4", 9, 6);
		expect(row("Kaggle Notebooks").getByText(/No bf16 on T4 or P100: train in fp16 with loss scaling/)).toBeTruthy();
		cleanup();
		const estimates = { "memory.peak": est(9e9, "bytes", 1.1), "time.total": est(6 * H, "s") };
		render(<FreeTierPanel record={record} knobs={{ gpu_type: "T4", precision: "bf16" }} estimates={estimates} />);
		expect(row("Google Colab (free)").getByText(/This plan uses bf16, which a T4 can't run/)).toBeTruthy();
		expect(within(screen.getByRole("listitem", { name: "Modal (Starter credit)" })).queryByText(/fp16/)).toBeNull();
	});

	it("says when a run would use more than a month of Modal's free credit", () => {
		const estimates = { "memory.peak": est(9e9, "bytes", 1.1), "time.total": est(6 * H, "s"), "cost.total": est(40, "usd") };
		render(<FreeTierPanel record={record} knobs={{ gpu_type: "L4" }} estimates={estimates} />);
		expect(row("Modal (Starter credit)").getByText(/more than one month's (US)?\$30 free credit/)).toBeTruthy();
	});

	it("says Colab's GPU is reported, not published", () => {
		renderPanel("T4", 9, 6);
		expect(row("Google Colab (free)").getByText(/Colab doesn't publish its GPU or limits/)).toBeTruthy();
	});
});
