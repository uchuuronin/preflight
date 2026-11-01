/**
 * The visitor's limits on the guide: hard limits both tours obey and the server enforces.
 *
 * - `allowed`: a knob may only take these values (allowed GPUs, from the constraints panel).
 * - `excluded`: values the visitor turned down with "Not this", for this session.
 * - `declined`: knobs the visitor turned down a suggestion on; the tours offer another lever
 *   next instead of the same knob at its next value.
 * - `budgetUsd`, `deadlineS`, `gpuHours`: a move may not take the run's predicted cost, wall
 *   time or GPU-hours past these. A move that lowers a number already past its limit is allowed
 *   (it helps), so a run that starts over budget can still be brought back.
 * - `frozen`: "don't change" knobs. Unlike a pin (which the visitor, or the guide when asked, can
 *   release), a frozen knob is never moved, never offered, and can't be unpinned by the guide.
 * - Number limits are checked against the upper end of the readout's interval when it has one:
 *   "under $40" means the run's predicted range stays under $40, not just its middle.
 *
 * Value limits are checked again on the server (routes/guide.py), which has no provider; the
 * number limits are checked here, against the same readouts the dials show, like the fit rule.
 */
import type { PredictionProvider, SensitivityOptions } from "../workspace/provider/PredictionProvider";
import { targetFor } from "../workspace/record/model";
import type { KnobValue, LoadedRecord, Prediction } from "../workspace/record/types";

export interface Limits {
	allowed: Readonly<Record<string, readonly KnobValue[]>>;
	excluded: Readonly<Record<string, readonly KnobValue[]>>;
	declined: readonly string[];
	frozen: readonly string[];
	budgetUsd: number | null;
	deadlineS: number | null;
	gpuHours: number | null;
}

export const NO_LIMITS: Limits = { allowed: {}, excluded: {}, declined: [], frozen: [], budgetUsd: null, deadlineS: null, gpuHours: null };

export type LimitRefusal = "outside_limits" | "over_budget" | "past_deadline" | "over_quota";

const same = (a: KnobValue | undefined, b: KnobValue | undefined) => a === b || (typeof a === "number" && typeof b === "number" && Math.abs(a - b) <= 1e-12 * Math.max(1, Math.abs(a)));

/** True when the visitor's limits let this knob take this value. */
export function valueAllowed(limits: Limits, knobId: string, value: KnobValue): boolean {
	if (limits.frozen.includes(knobId)) return false;
	const allow = limits.allowed[knobId];
	if (allow && allow.length && !allow.some((v) => same(v, value))) return false;
	return !(limits.excluded[knobId] ?? []).some((v) => same(v, value));
}

/** The upper end of a readout's interval (its point when it has no interval), or null. */
function upper(p: Prediction, id: string | undefined): number | null {
	const e = id ? p[id] : undefined;
	if (!e || e.status === "abstain" || typeof e.point !== "number") return null;
	return typeof e.hi === "number" ? Math.max(e.hi, e.point) : e.point;
}

/** GPUs the run uses at a setting: the gpus knob, else the record's own count. */
function gpuCount(record: LoadedRecord, knobs: Readonly<Record<string, KnobValue>>): number {
	const g = knobs.gpus;
	return typeof g === "number" && g > 0 ? g : (record.env?.gpu?.count ?? 1);
}

/** The run's GPU-hours at a setting: wall time x GPUs (derived; no record carries it as a target yet). */
export function gpuHoursAt(record: LoadedRecord, p: Prediction, knobs: Readonly<Record<string, KnobValue>>): number | null {
	const s = upper(p, targetFor(record, "time")?.id);
	return s === null ? null : (s / 3600) * gpuCount(record, knobs);
}

/** Why a move breaks a limit, or null. `before` is the prediction at the current setting. */
export function breaksLimit(
	record: LoadedRecord,
	limits: Limits,
	knobId: string,
	value: KnobValue,
	after: Prediction,
	before: Prediction,
	knobs: Readonly<Record<string, KnobValue>>,
): LimitRefusal | null {
	if (!valueAllowed(limits, knobId, value)) return "outside_limits";
	const next = { ...knobs, [knobId]: value };
	const checks: Array<[number | null, number | null, number | null, LimitRefusal]> = [
		[limits.budgetUsd, upper(after, targetFor(record, "cost")?.id), upper(before, targetFor(record, "cost")?.id), "over_budget"],
		[limits.deadlineS, upper(after, targetFor(record, "time")?.id), upper(before, targetFor(record, "time")?.id), "past_deadline"],
		[limits.gpuHours, gpuHoursAt(record, after, next), gpuHoursAt(record, before, knobs), "over_quota"],
	];
	for (const [cap, a, b, why] of checks) {
		if (cap === null || a === null) continue;
		// Past the limit, and not an improvement on a setting that was already past it.
		if (a > cap && !(b !== null && b > cap && a < b)) return why;
	}
	return null;
}

/** Provider options that keep every move inside the limits (and off declined knobs, if asked). */
export function withinLimits(
	record: LoadedRecord,
	provider: PredictionProvider,
	limits: Limits,
	knobs: Readonly<Record<string, KnobValue>>,
	opts: SensitivityOptions,
	skipDeclined = true,
): SensitivityOptions {
	const before = provider.predict(record, knobs);
	const pinned = new Set([...(opts.pinned ?? []), ...limits.frozen, ...(skipDeclined ? limits.declined : [])]);
	return {
		...opts,
		pinned,
		knobs,
		allow: (knobId, value, after) => breaksLimit(record, limits, knobId, value, after, before, knobs) === null,
	};
}

/** Limits as the server checks them (routes/guide.py Setting.limits). */
export function limitsForServer(limits: Limits): Record<string, unknown> {
	return {
		allowed: limits.allowed,
		excluded: limits.excluded,
		declined: limits.declined,
		frozen: limits.frozen,
		...(limits.budgetUsd !== null ? { budget_usd: limits.budgetUsd } : {}),
		...(limits.deadlineS !== null ? { deadline_s: limits.deadlineS } : {}),
		...(limits.gpuHours !== null ? { gpu_hours: limits.gpuHours } : {}),
	};
}

/** The server's copy of this session's limits on a record (api session_limits.py). */
function fromServer(raw: Record<string, unknown>): Limits {
	const num = (v: unknown) => (typeof v === "number" && v > 0 ? v : null);
	return {
		allowed: (raw.allowed as Limits["allowed"]) ?? {},
		excluded: (raw.excluded as Limits["excluded"]) ?? {},
		declined: (raw.declined as string[]) ?? [],
		frozen: (raw.frozen as string[]) ?? [],
		budgetUsd: num(raw.budget_usd),
		deadlineS: num(raw.deadline_s),
		gpuHours: num(raw.gpu_hours),
	};
}

const saving = new Map<string, Promise<boolean>>();

/** Keep the server's copy in step (the fix route reads only that). Resolves true once saved. */
export function saveLimits(record: string, limits: Limits, put: typeof fetch = fetch): Promise<boolean> {
	const p = put("/v1/guide/limits/state", {
		method: "PUT",
		credentials: "same-origin",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify({ record, limits: limitsForServer(limits) }),
	})
		.then((r) => r.ok)
		.catch(() => false);
	saving.set(record, p);
	return p;
}

/** Wait for the last save of this record's limits (before a request the server judges by them). */
export function limitsSaved(record: string): Promise<boolean> {
	return saving.get(record) ?? Promise.resolve(true);
}

/** This session's limits on a record from the server, so a reload keeps them; null if none. */
export async function loadLimits(record: string, get: typeof fetch = fetch): Promise<Limits | null> {
	try {
		const r = await get(`/v1/guide/limits/state?record=${encodeURIComponent(record)}`, { credentials: "same-origin" });
		if (!r.ok) return null;
		const body = (await r.json()) as { limits?: Record<string, unknown> | null };
		return body.limits ? fromServer(body.limits) : null;
	} catch {
		return null;
	}
}
