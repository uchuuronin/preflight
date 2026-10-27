import { describe, expect, it } from "vitest";
import type { Estimate } from "../../workspace/record/types";
import { SETUP_S, canonicalGpu, checkPreset, gpuMemoryBytes } from "./check";
import { PRESETS, type Preset } from "./presets";

const H = 3600;
const GB = 1e9;

function est(point: number, ratio = 1.15, unit = "s"): Estimate {
	return { point, lo: point / ratio, hi: point * ratio, unit, status: "predicted", model_id: "test", source: "grid", level: 0.9 } as Estimate;
}

const colab = PRESETS.find((p) => p.id === "colab") as Preset;
const kaggle = PRESETS.find((p) => p.id === "kaggle") as Preset;
const lightning = PRESETS.find((p) => p.id === "lightning") as Preset;

describe("free-tier presets", () => {
	it("cover Colab, Kaggle and Lightning, each dated and sourced", () => {
		expect(PRESETS.map((p) => p.id)).toEqual(["colab", "kaggle", "lightning"]);
		for (const p of PRESETS) {
			expect(p.sessionCapS, p.id).toBeGreaterThan(0);
			expect(p.sources.length, p.id).toBeGreaterThan(0);
			for (const s of p.sources) {
				expect(s.url, p.id).toMatch(/^https:\/\//);
				expect(s.read, p.id).toMatch(/^2026-\d\d-\d\d$/);
			}
		}
	});

	it("carry the published session caps", () => {
		expect(colab.sessionCapS).toBe(12 * H);
		expect(kaggle.sessionCapS).toBe(12 * H);
		expect(lightning.sessionCapS).toBe(4 * H);
	});

	it("name GPUs B's table knows", () => {
		for (const p of PRESETS) for (const g of p.gpus) expect(gpuMemoryBytes(g), `${p.id}:${g}`).toBeGreaterThan(0);
	});
});

describe("canonicalGpu", () => {
	it("resolves names and aliases through B's table", () => {
		expect(canonicalGpu("Tesla T4")).toBe("T4");
		expect(canonicalGpu("p100")).toBe("P100-16GB");
		expect(canonicalGpu("A100-SXM-80GB")).toBe("A100-SXM-80GB");
		expect(canonicalGpu("not a gpu")).toBeNull();
		expect(canonicalGpu(null)).toBeNull();
	});
});

describe("checkPreset", () => {
	const fitsMem = est(9 * GB, 1.1, "bytes");

	it("says one session when the slow end plus setup fits the cap", () => {
		const v = checkPreset(colab, { gpu: "T4", memory: fitsMem, time: est(9 * H) });
		expect(v).toMatchObject({ kind: "one_session" });
	});

	it("counts sessions, and how often to save, when it doesn't", () => {
		const v = checkPreset(colab, { gpu: "T4", memory: fitsMem, time: est(20 * H) });
		const usable = 12 * H - SETUP_S;
		expect(v).toEqual({ kind: "sessions", sessions: Math.ceil((20 * H * 1.15) / usable), saveEveryS: usable });
	});

	it("judges by the slow end of the range, not the point", () => {
		// point 11 h fits 12 h, but 11 h x 1.15 + 15 min does not
		expect(checkPreset(colab, { gpu: "T4", memory: fitsMem, time: est(11 * H) }).kind).toBe("sessions");
	});

	it("says it won't fit when peak memory exceeds the free GPU", () => {
		const v = checkPreset(colab, { gpu: "T4", memory: est(15 * GB, 1.1, "bytes"), time: est(2 * H) });
		expect(v).toMatchObject({ kind: "no_memory", needBytes: 15 * GB * 1.1 });
	});

	it("checks memory but not time when the run was measured on another GPU", () => {
		expect(checkPreset(kaggle, { gpu: "A100-SXM-80GB", memory: fitsMem, time: est(2 * H) })).toEqual({ kind: "other_gpu" });
		expect(checkPreset(kaggle, { gpu: "A100-SXM-80GB", memory: est(30 * GB, 1.1, "bytes"), time: est(2 * H) }).kind).toBe("no_memory");
	});

	it("accepts any GPU a platform offers", () => {
		expect(checkPreset(kaggle, { gpu: "P100-16GB", memory: fitsMem, time: est(2 * H) }).kind).toBe("one_session");
	});

	it("abstains when a number it needs isn't trusted", () => {
		const abstain = { ...est(2 * H), status: "abstain" } as Estimate;
		expect(checkPreset(colab, { gpu: "T4", memory: fitsMem, time: abstain }).kind).toBe("abstain");
		expect(checkPreset(colab, { gpu: "T4", memory: undefined, time: est(2 * H) }).kind).toBe("abstain");
	});
});
