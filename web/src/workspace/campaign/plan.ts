/**
 * The campaign plan as B's planner returns it (python/preflight/campaign/planner.py `CampaignPlan`,
 * serialised field by field; fixtures/ hold two real outputs). The calendar only draws these
 * fields: every number is B's, with B's interval and status.
 */
import type { Campaign, CampaignCell, ModelCard } from "../../../../schema/ts/contract";
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
	/** Contract 0.5.0: when this run starts, seconds after the campaign starts, in its lane's slot order. */
	start_s?: Estimate | number | null;
	/** Which of its lane's slots runs it, 0-based (B's CellPlan.slot; contract 0.6.0 as requested). */
	slot?: number | null;
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
	/** When the campaign clock starts (contract 0.6.0), as an ISO date-time; null when the record doesn't say. */
	start?: string | null;
	/** Slots per lane (0.6.0), so a lane with a free slot still draws its row. */
	lane_slots?: Record<string, number> | null;
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

/** A run's start as a number of seconds, or null: the contract's Estimate (its point), or a bare number. */
export function startOf(c: PlanCell): number | null {
	const s = c.start_s;
	if (typeof s === "number") return Number.isFinite(s) ? s : null;
	return s && s.status !== "abstain" && typeof s.point === "number" ? s.point : null;
}

/** True when B gave every placed run a start and a run time, so the lanes can be drawn against time. */
export function hasSchedule(plan: CampaignPlan): boolean {
	const placed = plan.cells.filter((c) => c.lane && !c.cut);
	return placed.length > 0 && placed.every((c) => startOf(c) !== null && c.time.status !== "abstain" && typeof c.time.point === "number");
}

/**
 * A lane's runs split into rows, so two runs going at once never draw over each other. A run with
 * its slot (B's CellPlan.slot; 0.6.0 as requested) goes on that slot's row; one without (records
 * B writes today) goes on the first row where it doesn't overlap a run already there.
 */
export function slotsOf(cells: PlanCell[], minRows = 1): PlanCell[][] {
	const hasSlot = (c: PlanCell) => typeof c.slot === "number" && Number.isInteger(c.slot) && c.slot >= 0;
	const rows = new Map<number, PlanCell[]>();
	for (const c of cells.filter(hasSlot)) rows.set(c.slot as number, [...(rows.get(c.slot as number) ?? []), c]);
	const ordered = [...rows.keys()].sort((a, b) => a - b).map((k) => rows.get(k)!);
	const end = (c: PlanCell) => (startOf(c) ?? 0) + (c.time.hi ?? c.time.point ?? 0);
	const overlaps = (row: PlanCell[], c: PlanCell) => row.some((o) => (startOf(o) ?? 0) < end(c) && (startOf(c) ?? 0) < end(o));
	for (const c of cells.filter((x) => !hasSlot(x)).sort((a, b) => (startOf(a) ?? 0) - (startOf(b) ?? 0))) {
		const row = ordered.find((r) => !overlaps(r, c));
		if (row) row.push(c);
		else ordered.push([c]);
	}
	// A lane's free slots still get a row (0.6.0 lane_slots): an idle GPU is part of the plan.
	while (ordered.length < Math.max(1, minRows)) ordered.push([]);
	return ordered;
}

function abstain(why: string, unit: string, modelId: string): Estimate {
	return { point: null, lo: null, hi: null, unit, status: "abstain", model_id: modelId, source: "exact", reason: why } as Estimate;
}

/**
 * The calendar's plan from a shared record's campaign block, so a record drawn later shows what the
 * live plan showed. Every number is the record's: a run with no time is unplaceable (with the
 * record's reason or a plain one), a total the record lacks says so, and per-run costs appear only
 * as the intervals 0.6.0 adds (a bare point would read as exact).
 */
export function planFromRecord(
	c: Campaign,
	models: readonly ModelCard[] = [],
	price: (knobs: Record<string, string | number | boolean>, e: Estimate | null | undefined) => number = () => 1,
): CampaignPlan | null {
	const cells = c.cells ?? [];
	if (!cells.length) return null;
	const modelId = c.finish_s?.model_id ?? cells.find((x) => x.time)?.time?.model_id ?? "campaign";
	const card = models.find((m) => m.id === modelId);
	const missing = (unit: string) => abstain("not in this record", unit, modelId);
	// The visitor's rate, where they set one (workspace/rate.ts): each run's dollars at its own setting,
	// the totals at the record's. `price` reads each estimate's own rate (0.6.0 Estimate.rate) when it has one.
	const usdAt = (e: Estimate | null | undefined, knobs: Record<string, string | number | boolean>): Estimate | null => {
		if (!e || e.unit !== "usd") return e ?? null;
		const f = price(knobs, e);
		if (f === 1) return e;
		const m = (v: number | null | undefined) => (typeof v === "number" ? v * f : (v ?? null));
		return { ...e, point: m(e.point), lo: m(e.lo), hi: m(e.hi) } as Estimate;
	};
	const knobsOf = (x: CampaignCell) => Object.fromEntries(Object.entries(x.knobs).filter((kv): kv is [string, string | number | boolean] => kv[1] !== undefined));
	return {
		cells: cells.map((x: CampaignCell) => ({
			id: x.id,
			knobs: knobsOf(x),
			time: (x.time as Estimate | null | undefined) ?? abstain("no run time in this record", "s", modelId),
			gpu_hours: (x.gpu_hours_estimate as Estimate | null | undefined) ?? null,
			usd: usdAt(x.usd_estimate as Estimate | null | undefined, knobsOf(x)),
			segments: x.segments ?? null,
			lane: x.lane ?? null,
			cut: x.cut ?? false,
			reason: typeof x.reason === "string" ? x.reason : "",
			start_s: (x.start_s as Estimate | null | undefined) ?? null,
			slot: typeof x.slot === "number" ? x.slot : null,
		})),
		gpu_hours: (c.gpu_hours as Estimate | null | undefined) ?? missing("gpu_hours"),
		usd: usdAt(c.usd as Estimate | null | undefined, {}) ?? missing("usd"),
		finish_s: (c.finish_s as Estimate | null | undefined) ?? missing("s"),
		completion: Object.fromEntries(Object.entries(c.completion_quantiles ?? {}).filter((kv): kv is [string, string] => typeof kv[1] === "string")),
		p_by_deadline: c.p_by_deadline ?? null,
		checkpoint_interval: (c.checkpoint_interval as Estimate | null | undefined) ?? missing("s"),
		quota: c.quota
			? { needed_bytes: (c.quota.needed_bytes as Estimate | undefined) ?? missing("bytes"), quota_bytes: c.quota.quota_bytes ?? null, fits: c.quota.fits ?? null, reason: c.quota.reason ?? "" }
			: { needed_bytes: missing("bytes"), quota_bytes: null, fits: null, reason: "" },
		asha: c.asha
			? { rungs: (c.asha.rungs ?? []).map((r) => ({ steps: r.steps, runs: r.runs })), gpu_hours: (c.asha.gpu_hours as Estimate | null | undefined) ?? missing("gpu_hours"), saved_gpu_hours: (c.asha.saved_gpu_hours as Estimate | null | undefined) ?? missing("gpu_hours"), reason: c.asha.reason ?? "" }
			: null,
		lanes: c.lanes ?? [],
		cut_list: c.cut_list ?? [],
		notes: c.notes ?? [],
		grid_gpu_hours: (c.grid_gpu_hours as Estimate | null | undefined) ?? null,
		grid_usd: usdAt(c.grid_usd as Estimate | null | undefined, {}),
		card: { id: modelId, notes: card?.notes ?? null },
		start: typeof c.start === "string" ? c.start : null,
		lane_slots: c.lane_slots ? Object.fromEntries(Object.entries(c.lane_slots).filter((kv): kv is [string, number] => typeof kv[1] === "number")) : null,
	};
}

/**
 * When the last run ends, on the calendar's own axis (seconds after the campaign starts): B's p10,
 * p50 and p90 dates less the start (the record's own from 0.6.0, else derived). Null unless all of it holds.
 */
export function finishBand(plan: CampaignPlan): { p10: number; p50: number; p90: number } | null {
	const f = plan.finish_s;
	if (!f || f.status === "abstain" || typeof f.point !== "number") return null;
	const d = (k: string) => Date.parse(plan.completion[k] ?? "");
	const [p10, p50, p90] = [d("p10"), d("p50"), d("p90")];
	if (![p10, p50, p90].every(Number.isFinite) || !(p10 <= p50 && p50 <= p90)) return null;
	// 0.6.0 records carry the start; older ones give it as p50 less finish_s.point (B dates p50 that way).
	const stated = plan.start ? Date.parse(plan.start) : Number.NaN;
	const start = Number.isFinite(stated) ? stated : p50 - f.point * 1000;
	return { p10: (p10 - start) / 1000, p50: (p50 - start) / 1000, p90: (p90 - start) / 1000 };
}
