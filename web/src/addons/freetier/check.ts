/**
 * "Will this finish in one free session?" (D4)
 *
 * One session: the slow end of the time range (the interval's hi) plus SETUP_S of installs and
 * downloads fits the cap. Picked over four other rules by a simulation whose "true" times were
 * drawn from the same intervals, so its figures are calibration: simulated, never quoted
 * (docs/decisions/freetier-one-session.md). The real check replaces it: one-session verdicts
 * against measured wall-clock on the zoo's full runs, with Wilson intervals, once A has them.
 * Memory is judged against the free GPU's usable capacity from B's table [cited: engine/gpus.py,
 * via the generated gpus.json]. Time is only judged when predicted for a GPU the tier offers.
 */
import gpuTable from "../../workspace/record/gpus.json";
import type { Estimate } from "../../workspace/record/types";
import type { Preset } from "./presets";

/**
 * Allowance per session for installs, downloads and loading the model.
 * [guess: replace with A's measured fetch + install + model-load time from the job events
 * (stages fetching to harnessing) once the zoo scripts run on a GPU, by G2, Sun Nov 1]
 */
export const SETUP_S = 15 * 60;

const TABLE = gpuTable as unknown as {
	gpus: Record<string, { memory_bytes?: number }>;
	aliases: Record<string, string>;
};

export function canonicalGpu(name: string | null | undefined): string | null {
	if (!name) return null;
	if (TABLE.gpus[name]) return name;
	return TABLE.aliases[name.trim().toLowerCase()] ?? null;
}

export function gpuMemoryBytes(name: string): number | null {
	const key = canonicalGpu(name);
	return key ? (TABLE.gpus[key]?.memory_bytes ?? null) : null;
}

/** overCredit appears only for credit tiers: true when the run's cost could pass a month's credit. */
export type Verdict =
	| { kind: "one_session"; overCredit?: boolean }
	| { kind: "sessions"; sessions: number; saveEveryS: number; overCredit?: boolean }
	| { kind: "no_memory"; needBytes: number; haveBytes: number }
	| { kind: "other_gpu" }
	| { kind: "abstain" };

export interface CheckInput {
	/** The GPU the numbers are for (the gpu_type knob, else the record's GPU). */
	gpu: string | null;
	memory: Estimate | undefined;
	time: Estimate | undefined;
	/** The run's cost, for tiers that give a monthly credit. */
	cost?: Estimate | undefined;
}

/** The interval's slow end, or null when the estimate is missing, abstains or has no bound. */
function slowEnd(e: Estimate | undefined): number | null {
	if (!e || e.status === "abstain") return null;
	return typeof e.hi === "number" && Number.isFinite(e.hi) ? e.hi : null;
}

export function checkPreset(preset: Preset, input: CheckInput): Verdict {
	const memHi = slowEnd(input.memory);
	if (memHi === null) return { kind: "abstain" };
	const gpu = canonicalGpu(input.gpu);
	const offered = gpu !== null && preset.gpus.includes(gpu);
	// Memory hardly depends on which GPU runs the job; judge it on the GPU the run would use,
	// or on the roomiest one the tier offers.
	const capacities = (offered ? [gpu] : preset.gpus).map((g) => gpuMemoryBytes(g) ?? 0);
	const haveBytes = Math.max(...capacities);
	if (memHi > haveBytes) return { kind: "no_memory", needBytes: memHi, haveBytes };
	if (!offered) return { kind: "other_gpu" };
	const timeHi = slowEnd(input.time);
	if (timeHi === null) return { kind: "abstain" };
	const credit = creditCheck(preset, input.cost);
	const usable = preset.sessionCapS - SETUP_S;
	if (timeHi + SETUP_S <= preset.sessionCapS) return { kind: "one_session", ...credit };
	return { kind: "sessions", sessions: Math.ceil(timeHi / usable), saveEveryS: usable, ...credit };
}

function creditCheck(preset: Preset, cost: Estimate | undefined): { overCredit?: boolean } {
	if (preset.monthlyCreditUsd === undefined) return {};
	const costHi = slowEnd(cost);
	return costHi === null ? {} : { overCredit: costHi > preset.monthlyCreditUsd };
}
