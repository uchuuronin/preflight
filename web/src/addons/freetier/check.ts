/**
 * "Will this finish in one free session?" (D4)
 *
 * One session: the slow end of the time range (the interval's hi) plus SETUP_S of installs and
 * downloads fits the cap. Chosen by Monte Carlo over five rules (docs/decisions/freetier-one-
 * session.md): wrong about 1 in 500 of its "one session" verdicts, under 1 in 100 if the
 * intervals were 1.5x too narrow. Memory is judged against the free GPU's usable capacity from
 * B's table. Time is only judged when the run was measured or predicted on a GPU the tier offers.
 */
import gpuTable from "../../workspace/record/gpus.json";
import type { Estimate } from "../../workspace/record/types";
import type { Preset } from "./presets";

/** Allowance per session for installs, downloads and loading the model. */
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

export type Verdict =
	| { kind: "one_session" }
	| { kind: "sessions"; sessions: number; saveEveryS: number }
	| { kind: "no_memory"; needBytes: number; haveBytes: number }
	| { kind: "other_gpu" }
	| { kind: "abstain" };

export interface CheckInput {
	/** The GPU the numbers are for (the gpu_type knob, else the record's GPU). */
	gpu: string | null;
	memory: Estimate | undefined;
	time: Estimate | undefined;
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
	const usable = preset.sessionCapS - SETUP_S;
	if (timeHi + SETUP_S <= preset.sessionCapS) return { kind: "one_session" };
	return { kind: "sessions", sessions: Math.ceil(timeHi / usable), saveEveryS: usable };
}
