import { describe, expect, it } from "vitest";
import goldenRaw from "../../../../schema/examples/finetune.json";
import devRaw from "../fixtures/dev-finetune.record.json";
import { currentKnobs, isFixture } from "../record/model";
import { type FlightRecord, normaliseRecord } from "../record/types";
import { GridProvider } from "./GridProvider";
import { resolveMetric } from "./metrics";
import { FAILS_AT, FITS_BELOW } from "../record/terms";

const dev = normaliseRecord(devRaw as unknown as FlightRecord);
const golden = normaliseRecord(goldenRaw as unknown as FlightRecord);
const provider = new GridProvider();
const GIB = 1024 ** 3;
const cost = { kind: "min_cost" as const };

/** The golden record without any grid over `knob` (or for `target`), so moving it is off every grid. */
function withoutGrids(rec: typeof golden, drop: { knob?: string; target?: string }) {
	const grids = rec.predictions.grids.filter((g) => !(drop.knob && g.axes.some((a) => a.knob_id === drop.knob)) && !(drop.target && g.target_id === drop.target));
	return { ...rec, predictions: { ...rec.predictions, grids } };
}

// These read the golden record's own numbers or assert relations, so they hold whether its
// predictions are A's fixture figures or B's engine output (PREFLIGHT_GOLDEN_ENGINE=1).
describe("GridProvider on the golden fine-tune", () => {
	it("returns the record's own estimates at the record's own settings", () => {
		const p = provider.predict(golden, currentKnobs(golden));
		for (const id of ["risk.oom", "time.total"]) {
			const own = golden.predictions.estimates[id]!;
			expect(p[id]?.point).toBeCloseTo(own.point!, 6);
		}
		// Both sets say this run fails on its GPU (the eval pass at batch 32).
		expect(p["risk.oom"]!.point!).toBeGreaterThanOrEqual(FAILS_AT);
	});

	it("reads memory from the grid with its power-of-two scale", () => {
		const k = currentKnobs(golden);
		const peak = (b: number) => provider.predict(golden, { ...k, batch_size: b })["memory.peak"]?.point ?? Number.NaN;
		const [p2, p3, p4] = [peak(2), peak(3), peak(4)];
		expect(p2).toBeGreaterThan(0);
		expect(p4).toBeGreaterThan(0);
		// Between grid points 2 and 4, at log(3) on the log axis (not halfway, as a linear axis would be).
		// Not "rises with batch": B's engine has the eval spike dominate this peak, so it needn't.
		expect(p3).toBeGreaterThanOrEqual(Math.min(p2, p4));
		expect(p3).toBeLessThanOrEqual(Math.max(p2, p4));
		const t = Math.log(3 / 2) / Math.log(4 / 2);
		// Within 0.1% (grid values are stored rounded); halfway would be 0.4% or more off on either set.
		expect(Math.abs(p3 - (p2 + t * (p4 - p2))) / p3).toBeLessThan(1e-3);
	});

	it("abstains when a knob no grid covers moves, instead of keeping the old number", () => {
		const rec = withoutGrids(golden, { knob: "eval_batch_size" });
		const p = provider.predict(rec, { ...currentKnobs(rec), eval_batch_size: 8 });
		expect(p["risk.oom"]?.status).toBe("abstain");
		expect(p["memory.peak"]?.status).toBe("abstain");
		expect(p["risk.oom"]?.reason).toBe("off_grid");
	});

	it("abstains on a target with no grid away from the current settings, and not on a gridded one", () => {
		const rec = withoutGrids(golden, { target: "time.total" });
		const p = provider.predict(rec, { ...currentKnobs(rec), batch_size: 8 });
		expect(p["time.total"]?.status).toBe("abstain");
		expect(p["memory.peak"]?.status).not.toBe("abstain");
	});

	it("is labelled a fixture", () => {
		expect(isFixture(golden)).toBe(true);
	});
});

describe("GridProvider on the dev record", () => {
	it("flags the planted eval OOM at the record's settings", () => {
		const p = provider.predict(dev, currentKnobs(dev));
		expect(p["risk.oom"]?.point).toBeGreaterThan(0.9);
		expect(p["memory.peak"]?.point).toBeGreaterThan(80 * GIB);
	});

	it("interpolates in log space along a log axis", () => {
		const at = (b: number) => provider.predict(dev, { eval_batch_size: 8, batch_size: b })["memory.activations"]?.point ?? 0;
		const a2 = at(2);
		const a4 = at(4);
		const a3 = at(3);
		expect(a3).toBeGreaterThan(a2);
		expect(a3).toBeLessThan(a4);
	});

	it("switches grids when a fixed knob moves", () => {
		const off = provider.predict(dev, { eval_batch_size: 8, batch_size: 8 })["memory.activations"]?.point ?? 0;
		const on = provider.predict(dev, { eval_batch_size: 8, batch_size: 8, gradient_checkpointing: true })[
			"memory.activations"
		]?.point ?? 0;
		expect(on).toBeLessThan(off / 3);
	});

	it("marks an unprobed GPU as extrapolated and a probe point as measured", () => {
		expect(provider.predict(dev, { eval_batch_size: 8, gpu_type: "H100-SXM-80GB" })["time.total"]?.status).toBe("extrapolated");
		expect(provider.predict(dev, { eval_batch_size: 8 })["memory.peak"]?.status).toBe("measured");
	});

	it("calls only what the probe observed measured: never the run's cost, time or risk", () => {
		const p = provider.predict(dev, { eval_batch_size: 8 });
		expect(p["memory.peak"]?.status).toBe("measured");
		expect(p["throughput.tokens_per_s"]?.status).toBe("measured");
		for (const id of ["cost.total", "time.total", "risk.oom"]) expect(p[id]?.status).toBe("predicted");
		// One knob away from the probe, nothing is measured.
		expect(provider.predict(dev, { eval_batch_size: 16 })["memory.peak"]?.status).not.toBe("measured");
	});

	it("drops a 'measured' label from the record's own estimates when no probe here observed it", () => {
		const rec = {
			...dev,
			predictions: {
				...dev.predictions,
				estimates: { ...dev.predictions.estimates, "cost.total": { ...dev.predictions.estimates!["cost.total"]!, status: "measured" as const } },
			},
		};
		expect(provider.predict(rec, {})["cost.total"]?.status).toBe("predicted");
	});

	it("abstains past twice the probed batch x sequence length", () => {
		const p = provider.predict(dev, { eval_batch_size: 8, batch_size: 32, seq_len: 4096 });
		expect(p["memory.peak"]?.status).toBe("abstain");
		expect(p["memory.peak"]?.reason).toBe("outside_trusted_range");
	});
});

describe("sensitivity and moves", () => {
	it("fixes the failure first, with the cheapest move that clears it", () => {
		const [top] = provider.moves(dev, cost);
		expect(top?.knobId).toBe("eval_batch_size");
		expect(top?.to).toBe(8);
		expect(top?.clearsFailure).toBe(true);
	});

	it("returns the contract's KnobImpact shape", () => {
		const [top] = provider.sensitivity(dev, cost);
		expect(top).toMatchObject({ knob_id: "eval_batch_size", impact: 1, direction: "decrease", best_value: 8 });
	});

	it("never moves a pinned knob", () => {
		const ids = provider.moves(dev, cost, { pinned: new Set(["eval_batch_size"]) }).map((m) => m.knobId);
		expect(ids).not.toContain("eval_batch_size");
	});

	it("once the run fits, only offers moves that save and stay safe", () => {
		const knobs = { ...currentKnobs(dev), eval_batch_size: 8 };
		const moves = provider.moves(dev, cost, { knobs });
		expect(moves.length).toBeGreaterThan(0);
		for (const m of moves) {
			expect(m.delta).toBeLessThan(0);
			expect(provider.predict(dev, { ...knobs, [m.knobId]: m.to })["risk.oom"]?.point ?? 0).toBeLessThan(FITS_BELOW);
		}
	});

	it("offers an eval batch move on the golden record only where a grid covers that knob", () => {
		const covered = golden.predictions.grids.some((g) => g.axes.some((a) => a.knob_id === "eval_batch_size"));
		const ids = provider.moves(golden, cost).map((m) => m.knobId);
		if (!covered) expect(ids).not.toContain("eval_batch_size");
		expect(provider.moves(withoutGrids(golden, { knob: "eval_batch_size" }), cost).map((m) => m.knobId)).not.toContain("eval_batch_size");
		// Every move it does offer lands on a prediction, never an abstention.
		const k = currentKnobs(golden);
		for (const m of provider.moves(golden, cost)) expect(provider.predict(golden, { ...k, [m.knobId]: m.to })[m.targetId]?.status).not.toBe("abstain");
	});
});

describe("metric resolver", () => {
	it("reads a probe metric and says where it was measured", () => {
		const m = resolveMetric(golden, currentKnobs(golden), {}, "probe.data.padding_fraction");
		expect(m?.value).toBeCloseTo(0.46, 2);
		expect(m?.kind).toBe("measured_here");
		const far = resolveMetric(golden, { ...currentKnobs(golden), batch_size: 16 }, {}, "memory.by_term.logits");
		expect(far?.kind).toBe("measured_elsewhere");
		expect(far?.unit).toBe("bytes");
	});
});

describe("moves follow B's cut-offs and the measure-first rule", () => {
	const cost = { kind: "min_cost" as const };
	it("never proposes a move into fails, and by default only into fits", () => {
		const knobs = { ...currentKnobs(dev), eval_batch_size: 8 };
		for (const m of provider.moves(dev, cost, { knobs })) {
			expect(m.fit).toBe("fits");
			expect(m.risk ?? 0).toBeLessThan(FITS_BELOW);
		}
		for (const m of provider.moves(dev, cost, { knobs, includeTight: true })) {
			expect(m.risk ?? 0).toBeLessThan(FAILS_AT);
			expect(m.fit === "tight").toBe((m.risk ?? 0) >= FITS_BELOW);
		}
	});

	it("reports each move's change relative to the current aim", () => {
		const knobs = { ...currentKnobs(dev), eval_batch_size: 8 };
		const base = provider.predict(dev, knobs)["cost.total"]?.point ?? 0;
		const m = provider.moves(dev, cost, { knobs })[0];
		expect(m?.relative).toBeCloseTo((m?.delta ?? 0) / base, 6);
	});
});
