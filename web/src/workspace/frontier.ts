/**
 * The cost frontier of a model sweep: each model's sampled tasks (agent_sample.tasks, the
 * contract's AgentTask: `model`, `success`, `cost_usd`) as a success rate with its Wilson 95%
 * interval and a mean cost per task with the 95% interval of the mean, and which models no other
 * model beats on both (the Pareto frontier). Only scored tasks count toward the success rate; a
 * model with none scored has no point.
 */
import { type Proportion, wilson } from "../addons/_stats/wilson";
import type { LoadedRecord } from "./record/types";

/** Source: derived, the normal quantile at 0.975, for the 95% interval of a mean (n of 2 or more). */
const Z95 = 1.959964;

export interface FrontierPoint {
	model: string;
	/** Tasks sampled with this model. */
	n: number;
	/** Of those, tasks with a success verdict. */
	scored: number;
	success: Proportion;
	costMean: number;
	costLo: number;
	costHi: number;
	/** True when another model costs no more and succeeds at least as often, and is better on one. */
	dominated: boolean;
}

export function frontierPoints(record: LoadedRecord): FrontierPoint[] {
	const tasks = record.agent_sample?.tasks ?? [];
	const by = new Map<string, typeof tasks>();
	for (const t of tasks) {
		if (!t.model) continue;
		by.set(t.model, [...(by.get(t.model) ?? []), t]);
	}
	const pts: FrontierPoint[] = [];
	for (const [model, ts] of by) {
		const scored = ts.filter((t) => typeof t.success === "boolean");
		const costs = ts.map((t) => t.cost_usd).filter((c): c is number => typeof c === "number" && Number.isFinite(c));
		if (!scored.length || costs.length < 2) continue;
		const mean = costs.reduce((a, b) => a + b, 0) / costs.length;
		const sd = Math.sqrt(costs.reduce((a, b) => a + (b - mean) ** 2, 0) / (costs.length - 1));
		const half = (Z95 * sd) / Math.sqrt(costs.length);
		pts.push({
			model,
			n: ts.length,
			scored: scored.length,
			success: wilson(scored.filter((t) => t.success).length, scored.length),
			costMean: mean,
			costLo: Math.max(0, mean - half),
			costHi: mean + half,
			dominated: false,
		});
	}
	for (const p of pts) {
		p.dominated = pts.some((q) => q !== p && q.costMean <= p.costMean && q.success.p >= p.success.p && (q.costMean < p.costMean || q.success.p > p.success.p));
	}
	return pts.sort((a, b) => a.costMean - b.costMean);
}

/** A sweep worth drawing: at least two models, each with scored tasks and costs. */
export function hasSweep(record: LoadedRecord): boolean {
	return frontierPoints(record).length >= 2;
}
