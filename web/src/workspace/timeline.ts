/**
 * The run as a sequence of events along the flight plan: evals, checkpoints and
 * predicted trouble. Everything here derives from the record and the current estimates;
 * no number is computed outside the PredictionProvider.
 */
import type { FlightRecord, Prediction } from "./record/types";
import { trusted } from "./status";

export type RunEventKind = "eval" | "checkpoint" | "predicted_oom" | "end";

export interface RunEvent {
	kind: RunEventKind;
	step: number;
	/** 0..1 along the run. */
	at: number;
	nodes: string[];
}

const MAX_MARKERS = 40;

export function totalSteps(record: FlightRecord): number {
	return record.job.total_steps ?? 1;
}

export function runEvents(record: FlightRecord, est: Prediction): RunEvent[] {
	const total = totalSteps(record);
	const out: RunEvent[] = [];
	const push = (kind: RunEventKind, every: number | undefined, nodes: string[]) => {
		if (!every || every <= 0) return;
		const n = Math.floor(total / every);
		const stride = Math.max(1, Math.ceil(n / MAX_MARKERS));
		for (let i = 1; i <= n; i += stride) out.push({ kind, step: i * every, at: (i * every) / total, nodes });
	};
	push("eval", record.job.eval_every, ["eval"]);
	push("checkpoint", record.job.checkpoint_every, ["checkpoint"]);

	// When the eval spike is what pushes memory over, the first eval is where the run dies.
	const spike = est.mem_eval_spike;
	const oom = est.p_oom;
	if (trusted(oom) && oom.point > 0.5) {
		const evalDriven = trusted(spike) && spike.point > 0 && record.job.eval_every;
		const step = evalDriven ? (record.job.eval_every as number) : 1;
		out.push({ kind: "predicted_oom", step, at: step / total, nodes: evalDriven ? ["eval"] : ["blocks"] });
	}
	return out.sort((a, b) => a.at - b.at);
}

/** Map nodes that are busy at a playhead position: the training loop, or an event in progress. */
export function activeNodesAt(record: FlightRecord, at: number): string[] {
	if (at <= 0) return [];
	const total = totalSteps(record);
	const step = Math.round(at * total);
	const near = (every?: number) => !!every && step > 0 && step % every <= Math.max(1, total / 400);
	if (near(record.job.eval_every)) return ["eval"];
	if (near(record.job.checkpoint_every)) return ["checkpoint"];
	return [];
}
