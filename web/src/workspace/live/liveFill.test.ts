import { describe, expect, it } from "vitest";
import loraRaw from "../fixtures/lora-live-nov1.record.json";
import { GridProvider } from "../provider/GridProvider";
import { currentKnobs } from "../record/model";
import { type FlightRecord, normaliseRecord } from "../record/types";
import { gpuRow } from "../rate";
import { isLiveEstimate, liveFill, liveStillRunning, withLive } from "./liveFill";

type Win = { index: number; wall_s: number; peak?: number };

/** A live record as A's collector writes it, with the parent's predictions emptied where asked. */
function live(windows: Win[], status = "running", duration = 600, drop: string[] = []): FlightRecord {
	const rec = structuredClone(loraRaw as unknown) as FlightRecord;
	rec.job = { ...rec.job, source: { ...rec.job.source, kind: "inflight" } };
	for (const id of drop) delete rec.predictions!.estimates![id];
	rec.predictions!.grids = [];
	rec.probes = [
		...(rec.probes ?? []),
		{
			id: "live",
			knobs: {},
			status,
			steps: windows.at(-1)?.index ?? 0,
			duration_s: duration,
			step_log: windows.map((w) => ({ index: w.index, kind: "timed", wall_s: w.wall_s, ...(w.peak ? { peak_reserved_bytes: w.peak, peak_allocated_bytes: w.peak * 0.95 } : {}) })),
		} as NonNullable<FlightRecord["probes"]>[number],
	];
	return rec;
}

const MiB = 2 ** 20;

describe("a followed run's readings where the record has no prediction (Chiya, Nov 4)", () => {
	it("fills time, cost and memory at the run's own settings, as measured live, one reading each", () => {
		// Chiya's run: 6.33 ms a step, peak 412 MiB, no predictions at these settings
		const raw = live([{ index: 100, wall_s: 0.00633, peak: 412 * MiB }, { index: 200, wall_s: 0.00633, peak: 412 * MiB }], "running", 2.0, ["time.total", "cost.total", "memory.peak"]);
		const record = normaliseRecord(raw);
		const grid = new GridProvider();
		const before = grid.predict(record, currentKnobs(record));
		expect(before["time.total"]?.status ?? "abstain").toBe("abstain");
		const p = withLive(grid, raw).predict(record, currentKnobs(record));
		const projected = 2.0 + (3170 - 200) * 0.00633;
		expect(p["time.total"]).toMatchObject({ status: "measured", point: projected, lo: projected, hi: projected, unit: "s" });
		expect(isLiveEstimate(p["time.total"])).toBe(true);
		const rate = gpuRow(record, currentKnobs(record))!.usd_per_hour;
		expect(p["cost.total"]).toMatchObject({ status: "measured", unit: "usd", rate: { unit: "gpu_hours", usd_per_unit: rate } });
		expect(p["cost.total"]!.point).toBeCloseTo((projected / 3600) * rate, 6);
		expect(p["memory.peak"]).toMatchObject({ status: "measured", point: 412 * MiB, unit: "bytes" });
		expect(liveStillRunning(raw)).toBe(true);
		expect(liveStillRunning(live([{ index: 3170, wall_s: 0.00633, peak: 412 * MiB }], "ok", 20))).toBe(false);
	});

	it("leaves the record's own predictions, other settings, and a run with no reading alone", () => {
		const raw = live([{ index: 100, wall_s: 0.00633, peak: 412 * MiB }], "running", 1.0, ["memory.peak"]);
		const record = normaliseRecord(raw);
		const grid = new GridProvider();
		const own = currentKnobs(record);
		const p = withLive(grid, raw).predict(record, own);
		expect(p["time.total"]?.status).toBe("predicted"); // the record predicts it: not replaced
		expect(isLiveEstimate(p["time.total"])).toBe(false);
		expect(p["memory.peak"]?.status).toBe("measured"); // it didn't: the live peak
		// a knob moved off the run's settings: the live reading says nothing about that setting
		const moved = withLive(grid, raw).predict(record, { ...own, batch_size: 8 });
		expect(moved["memory.peak"]?.status ?? "abstain").toBe("abstain");
		// a run with no windows yet has no reading: the abstention stands
		const empty = live([], "running", 0.5, ["memory.peak", "time.total"]);
		const q = liveFill(normaliseRecord(empty), own, grid.predict(normaliseRecord(empty), own), empty);
		expect(q["memory.peak"]?.status ?? "abstain").toBe("abstain");
		expect(q["time.total"]?.status ?? "abstain").toBe("abstain");
	});
});
