import { describe, expect, it } from "vitest";
import loraRaw from "../fixtures/lora-live-nov1.record.json";
import retrRaw from "../fixtures/retrieval_st-nov1.record.json";
import type { FlightRecord } from "../record/types";
import { asLiveRecord } from "./liveFile";
import { readLive } from "./liveCheck";

type Win = { index: number; wall_s: number; peak?: number };

/** A live record as A's collector writes it (recorder/inflight.py live_record): the parent's
 * probes and predictions, source kind inflight, plus probe `live` with one step-log entry a window. */
function live(raw: unknown, windows: Win[], status = "running", duration = 600): FlightRecord {
	const rec = structuredClone(raw) as FlightRecord;
	rec.job = { ...rec.job, source: { ...rec.job.source, kind: "inflight" } };
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

const lora = loraRaw as unknown as FlightRecord;
const step = lora.predictions!.estimates!["time.step"]!;

describe("the real run against its predicted band (A's collector's checks)", () => {
	it("says nothing while the run is inside its band", () => {
		const mid = (step.lo! + step.hi!) / 2;
		const r = readLive(live(lora, [20, 40, 60].map((i) => ({ index: i, wall_s: mid })), "running", 30 + 60 * mid))!;
		expect(r.alerts).toEqual([]);
		expect(r.stepS).toBeCloseTo(mid);
		expect(r.totalSteps).toBe(3170);
	});

	it("needs two slow windows in a row before it says the steps are slow", () => {
		const slow = step.hi! * 1.4;
		const mid = (step.lo! + step.hi!) / 2;
		const one = readLive(live(lora, [{ index: 20, wall_s: mid }, { index: 40, wall_s: slow }]))!;
		expect(one.alerts.map((a) => a.kind)).not.toContain("step");
		const two = readLive(live(lora, [{ index: 20, wall_s: mid }, { index: 40, wall_s: slow }, { index: 60, wall_s: slow }]))!;
		expect(two.alerts.find((a) => a.kind === "step")).toMatchObject({ side: "above", step: 60 });
	});

	it("says at once when the peak goes over the band's top", () => {
		const retr = retrRaw as unknown as FlightRecord;
		const hi = retr.predictions!.estimates!["memory.peak"]!.hi!;
		const r = readLive(live(retr, [{ index: 20, wall_s: 0.04, peak: hi * 1.1 }]))!;
		expect(r.alerts.map((a) => a.kind)).toEqual(["peak"]);
	});

	it("projects the whole run at the recent pace and says when it runs past the band", () => {
		const slow = step.hi! * 1.4;
		const r = readLive(live(lora, [{ index: 200, wall_s: slow }, { index: 220, wall_s: slow }], "running", 30 + 220 * slow))!;
		expect(r.projectedS).toBeCloseTo(30 + 220 * slow + (3170 - 220) * slow);
		expect(r.alerts.map((a) => a.kind)).toContain("total");
	});
});

describe("opening a live record", () => {
	it("takes only a live record from preflight watch", () => {
		expect(() => asLiveRecord(JSON.stringify(live(lora, [{ index: 20, wall_s: 1 }])))).not.toThrow();
		expect(() => asLiveRecord(JSON.stringify(lora))).toThrow(/isn't a live record/);
		expect(() => asLiveRecord("{")).toThrow(/isn't JSON/);
	});
});
