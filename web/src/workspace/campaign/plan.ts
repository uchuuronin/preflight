/**
 * The campaign plan as B's planner returns it (python/preflight/campaign/planner.py `CampaignPlan`,
 * serialised field by field; fixtures/ hold two real outputs). The calendar only draws these
 * fields: every number is B's, with B's interval and status.
 */
import type { Estimate } from "../record/types";

export interface PlanCell {
	id: string;
	knobs: Record<string, string | number | boolean>;
	/** The engine's time.total for this run. */
	time: Estimate;
	gpu_hours: Estimate | null;
	usd: Estimate | null;
	/** Jobs this run needs under the scheduler's time limit; null when it can't be dated. */
	segments: number | null;
	lane: string | null;
	cut: boolean;
	reason: string;
	/** Asked of B (from-c.md, Oct 31): the slot inside the lane and the start, from B's schedule. */
	slot?: number | null;
	start_s?: number | null;
}

export interface CampaignPlan {
	cells: PlanCell[];
	gpu_hours: Estimate;
	usd: Estimate;
	finish_s: Estimate;
	/** p10, p50, p90 as ISO date-times; empty when the finish abstains. */
	completion: Record<string, string>;
	p_by_deadline: number | null;
	checkpoint_interval: Estimate;
	quota: { needed_bytes: Estimate; quota_bytes: number | null; fits: boolean | null; reason: string };
	asha: { rungs: { steps: number; runs: number }[]; gpu_hours: Estimate; saved_gpu_hours: Estimate; reason: string } | null;
	lanes: string[];
	cut_list: string[];
	notes: string[];
	grid_gpu_hours: Estimate | null;
	grid_usd: Estimate | null;
	card: { id: string; notes?: string | null };
}

/** B writes GPU-hours as "gpu_hours"; fmt() calls the unit "gpu_h". */
export function unitOf(e: Estimate): string {
	return e.unit === "gpu_hours" ? "gpu_h" : e.unit;
}

export interface LaneRow {
	lane: string;
	cells: PlanCell[];
}

/**
 * Runs grouped as B placed them: each lane in B's order, then the runs B didn't place (cut for the
 * budget, or with no run time to place).
 */
export function lanesOf(plan: CampaignPlan): { rows: LaneRow[]; unplaced: PlanCell[] } {
	const rows = plan.lanes.map((lane) => ({ lane, cells: plan.cells.filter((c) => c.lane === lane && !c.cut) }));
	const placed = new Set(rows.flatMap((r) => r.cells.map((c) => c.id)));
	return { rows, unplaced: plan.cells.filter((c) => !placed.has(c.id)) };
}

/** True when B gave every placed run a start, so the lanes can be drawn against time. */
export function hasSchedule(plan: CampaignPlan): boolean {
	const placed = plan.cells.filter((c) => c.lane && !c.cut);
	return placed.length > 0 && placed.every((c) => typeof c.start_s === "number");
}
