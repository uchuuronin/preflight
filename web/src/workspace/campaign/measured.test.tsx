import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import campaignRaw from "../fixtures/campaign-grid.record.json";
import { type FlightRecord, normaliseRecord } from "../record/types";
import { CampaignCalendar } from "./Calendar";
import { measuredCell, planFromRecord } from "./plan";

// PF420: B's planner on a 12-run grid, with two probes on the A100 (batch 8 and 12 at seq 1024, 60,000 steps).
const record = normaliseRecord(campaignRaw as unknown as FlightRecord);

describe("campaign runs timed by a probe at their setting (PM, Nov 2)", () => {
	afterEach(cleanup);
	it("times a run from the probe at its setting: the probe's step time times the run's steps", () => {
		const m = measuredCell({ knobs: { batch_size: 8, seq_len: 1024, seed: 0 } }, record)!;
		expect(m.status).toBe("measured");
		expect(m.point).toBeCloseTo(0.179 * 60000, 3);
		expect(m.model_id).toBe("probe:p1");
	});
	it("takes the engine's relative range around the measured point", () => {
		const engine = { point: 10000, lo: 8000, hi: 13000, unit: "s", status: "predicted", model_id: "m", source: "exact" } as never;
		const m = measuredCell({ knobs: { batch_size: 8, seq_len: 1024 } }, record, engine)!;
		expect(m.lo! / m.point!).toBeCloseTo(0.8, 6);
		expect(m.hi! / m.point!).toBeCloseTo(1.3, 6);
	});
	it("doesn't time a run from an eval-off re-run, or where a probe at the setting failed", () => {
		const p1 = record.probes.find((p) => p.id === "p1")!;
		const evalOff = { ...record, probes: record.probes.map((p) => (p.id === "p1" ? { ...p, extras: { ...p.extras, eval_off: 1 } } : p)) };
		expect(measuredCell({ knobs: { batch_size: 8, seq_len: 1024 } }, evalOff)).toBeNull();
		const oom = { ...record, probes: [...record.probes, { ...p1, id: "p1-eval", status: "oom" as const }] };
		expect(measuredCell({ knobs: { batch_size: 8, seq_len: 1024 } }, oom)).toBeNull();
		// a probe at another value of a knob the run doesn't set isn't at the run's setting
		const other = { ...record, probes: record.probes.map((p) => (p.id === "p1" ? { ...p, knobs: { ...p.knobs, num_workers: 4 } } : p)) };
		expect(measuredCell({ knobs: { batch_size: 8, seq_len: 1024 } }, other)).toBeNull();
	});
	it("leaves a run with no probe at its setting to the engine", () => {
		expect(measuredCell({ knobs: { batch_size: 8, seq_len: 512, seed: 0 } }, record)).toBeNull();
		// another GPU than the probe's isn't measured
		expect(measuredCell({ knobs: { batch_size: 8, seq_len: 1024, gpu_type: "L4" } }, record)).toBeNull();
		// a run sized by epochs has no steps to multiply
		expect(measuredCell({ knobs: { batch_size: 8 } }, { ...record, knobs: record.knobs.filter((k) => k.id !== "max_steps") })).toBeNull();
	});
	it("uses the measured time, GPU hours and dollars in the plan, and only with the probes given", () => {
		// PF420 predates per-run cost intervals (0.6.0): give each run one, $2.10 an hour of its time.
		const priced = { ...record.campaign!, cells: record.campaign!.cells!.map((c) => ({ ...c, usd_estimate: c.time ? { ...c.time, unit: "usd", point: (c.time.point! / 3600) * 2.1, lo: (c.time.lo! / 3600) * 2.1, hi: (c.time.hi! / 3600) * 2.1 } : null })) };
		const plan = planFromRecord(priced as typeof record.campaign & object, record.models, undefined, record)!;
		const cell = plan.cells.find((c) => c.id === "batch_size=8,seq_len=1024,seed=0")!;
		expect(cell.measured).toBe(true);
		expect(cell.time.point).toBeCloseTo(0.179 * 60000, 3);
		// dollars at the engine's rate keep the engine's status: only the time is measured
		expect(cell.usd?.status).toBe(priced.cells.find((c) => c.id === cell.id)!.usd_estimate!.status);
		expect(cell.usd!.point!).toBeCloseTo((0.179 * 60000 / 3600) * 2.1, 6);
		expect(plan.cells.find((c) => c.id === "batch_size=8,seq_len=512,seed=0")!.measured).toBe(false);
		expect(planFromRecord(record.campaign!, record.models)!.cells.every((c) => !c.measured)).toBe(true);
	});
	it("marks the measured runs on the calendar", () => {
		const plan = planFromRecord(record.campaign!, record.models, undefined, record)!;
		const { container } = render(<CampaignCalendar plan={plan} />);
		const tags = container.querySelectorAll(".pf-cal__measured");
		expect(tags.length).toBe(plan.cells.filter((c) => c.measured && c.lane && !c.cut).length);
		expect(tags.length).toBeGreaterThan(0);
		expect(tags[0]!.textContent).toBe("Measured");
		expect(container.textContent).toContain("the totals, the finish and the start dates are still the engine's");
	});
});
