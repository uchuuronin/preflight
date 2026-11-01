/**
 * The visitor's own rate for a job's cost readout: $/GPU-hour for a GPU job, or $/M input and $/M
 * output tokens for an API job. The engine priced every dollar at B's table rate (record/prices.json,
 * generated from engine/prices); with the visitor's rate, every dollar estimate at a setting is
 * rescaled by visitor / table, so the dials, the timeline, the guide's cost moves, the limits and
 * the campaign totals all read the same number.
 *
 * Exact where the cost is a quantity times one rate: a GPU job's cost is GPU-hours (redone work
 * included) times the hourly rate; an API job's is input and output tokens at their rates, mixed
 * as the record's sampled tasks mixed them. The override applies to the GPU or model it was set
 * for (an assigned university L4 at $0 says nothing about an H100's price); other settings keep
 * the table's rate. $0 is a valid rate.
 */
import table from "./record/prices.json";
import type { Move, PredictionProvider, SensitivityOptions } from "./provider/PredictionProvider";
import { failureTarget, targetFor } from "./record/model";
import { FAILS_AT } from "./record/terms";
import { chosenGpu, resolveGpu } from "./record/terms";
import type { Estimate, KnobValue, LoadedRecord, Objective, Prediction } from "./record/types";

export interface GpuRow {
	id: string;
	gpu: string;
	provider: string;
	region: string;
	kind: string;
	usd_per_hour: number;
	as_of: string;
	source: string;
}
export interface TokenRow {
	id: string;
	model: string;
	provider: string;
	input_per_m: number;
	output_per_m: number;
	as_of: string;
	source: string;
}

export type RateOverride = { kind: "gpu"; gpu: string; usdPerHour: number } | { kind: "tokens"; model: string; inputPerM: number; outputPerM: number };

const GPU_ROWS = table.gpu as GpuRow[];
const TOKEN_ROWS = table.tokens as TokenRow[];
export const PRICE_TABLE_VERSION = table.version;

type Knobs = Readonly<Record<string, KnobValue>>;

/** What the record's cost is made of: only a pure GPU or a pure token cost can be rescaled exactly. */
export function costKind(record: LoadedRecord): "gpu" | "tokens" | null {
	const lines = record.job_cost?.lines ?? [];
	if (lines.length) {
		if (lines.every((l) => l.kind === "gpu")) return "gpu";
		if (lines.every((l) => l.kind === "tokens")) return "tokens";
		return null;
	}
	// No breakdown in the record: an API run is priced in tokens, any other job in GPU-hours.
	return record.job.lens === "agent" || record.env?.backend === "api" ? "tokens" : "gpu";
}

function provider(record: LoadedRecord): string {
	const p = record.cost_basis?.provider ?? record.env?.provider;
	return p && p !== "api" && p !== "local" ? p : "modal";
}

/** B's table row for the GPU at this setting (on demand, the record's provider), as the engine resolves it. */
export function gpuRow(record: LoadedRecord, knobs: Knobs): GpuRow | null {
	const gpu = resolveGpu(chosenGpu(record, knobs));
	if (!gpu) return null;
	const rows = GPU_ROWS.filter((r) => r.gpu === gpu && r.kind === "on_demand");
	return rows.find((r) => r.provider === provider(record)) ?? rows[0] ?? null;
}

/** The model the knobs choose, else the one the sampled tasks ran. */
export function chosenModel(record: LoadedRecord, knobs: Knobs): string | null {
	const v = knobs.model;
	if (typeof v === "string") return v;
	return record.agent_sample?.tasks?.find((t) => t.model)?.model ?? null;
}

/** B's token row for a model: its exact id, or the most specific id it starts with (engine/prices token_rate). */
export function tokenRow(model: string | null): TokenRow | null {
	if (!model) return null;
	const m = model.toLowerCase();
	const rows = TOKEN_ROWS.filter((r) => r.model === m || m.startsWith(`${r.model}-`));
	return rows.sort((a, b) => b.model.length - a.model.length)[0] ?? null;
}

/** Input and output tokens as the record's sampled tasks mixed them (reasoning is inside output). */
export function tokenMix(record: LoadedRecord): { input: number; output: number } | null {
	const tasks = record.agent_sample?.tasks ?? [];
	const input = tasks.reduce((a, t) => a + (t.input_tokens ?? 0), 0);
	const output = tasks.reduce((a, t) => a + (t.output_tokens ?? 0), 0);
	return input + output > 0 ? { input, output } : null;
}

/** Dollars per token for a mix at a pair of $/M rates. */
function blended(mix: { input: number; output: number }, inputPerM: number, outputPerM: number): number {
	return (mix.input * inputPerM + mix.output * outputPerM) / (mix.input + mix.output) / 1e6;
}

/** The table's rate at a setting, for the cost line: $/GPU-hour, or the mix's $/token. */
export function tableRate(record: LoadedRecord, knobs: Knobs): { kind: "gpu"; row: GpuRow; usdPerHour: number } | { kind: "tokens"; row: TokenRow; usdPerToken: number } | null {
	const kind = costKind(record);
	if (kind === "gpu") {
		const row = gpuRow(record, knobs);
		return row ? { kind, row, usdPerHour: row.usd_per_hour } : null;
	}
	if (kind === "tokens") {
		const row = tokenRow(chosenModel(record, knobs));
		const mix = tokenMix(record);
		if (!row || !mix) return null;
		// The engine's own blended rate where it priced this row (it counts cached input at the cached rate).
		const line = record.job_cost?.lines?.find((l) => l.kind === "tokens" && l.price_row === row.id && typeof l.rate_usd === "number" && l.rate_usd > 0);
		return { kind, row, usdPerToken: line?.rate_usd ?? blended(mix, row.input_per_m, row.output_per_m) };
	}
	return null;
}

/** True when the visitor's rate is for the GPU or model at this setting. */
export function appliesHere(record: LoadedRecord, knobs: Knobs, o: RateOverride | null): boolean {
	const tr = o ? tableRate(record, knobs) : null;
	if (!o || !tr) return false;
	if (o.kind === "gpu" && tr.kind === "gpu") return tr.row.gpu === resolveGpu(o.gpu);
	if (o.kind === "tokens" && tr.kind === "tokens") return tokenRow(o.model)?.id === tr.row.id;
	return false;
}

/**
 * Visitor / table at a setting: 1 where the override doesn't apply (no override, another GPU or
 * model, no table row), so those dollars stay the engine's own.
 */
export function rateFactor(record: LoadedRecord, knobs: Knobs, o: RateOverride | null): number {
	if (!o) return 1;
	const tr = tableRate(record, knobs);
	if (!tr) return 1;
	if (o.kind === "gpu" && tr.kind === "gpu") {
		if (tr.row.gpu !== resolveGpu(o.gpu) || !(tr.usdPerHour > 0)) return 1;
		return o.usdPerHour / tr.usdPerHour;
	}
	if (o.kind === "tokens" && tr.kind === "tokens") {
		if (tokenRow(o.model)?.id !== tr.row.id) return 1;
		const mix = tokenMix(record)!;
		// Both sides mixed the same way, so the ratio holds whatever the engine did with cached input.
		const table = blended(mix, tr.row.input_per_m, tr.row.output_per_m);
		return table > 0 ? blended(mix, o.inputPerM, o.outputPerM) / table : 1;
	}
	return 1;
}

function scaled(e: Estimate | undefined, f: number): Estimate | undefined {
	if (!e || e.unit !== "usd" || f === 1) return e;
	const m = (v: number | null | undefined) => (typeof v === "number" ? v * f : v);
	return { ...e, point: m(e.point) ?? null, lo: m(e.lo) ?? null, hi: m(e.hi) ?? null } as Estimate;
}

/** Every dollar estimate in a prediction at the visitor's rate. */
export function priced(record: LoadedRecord, knobs: Knobs, p: Prediction, o: RateOverride | null): Prediction {
	const f = rateFactor(record, knobs, o);
	if (f === 1) return p;
	const out: Prediction = {};
	for (const [id, e] of Object.entries(p)) out[id] = scaled(e, f);
	return out;
}

/**
 * A provider whose dollars are at the visitor's rate. Cost moves are re-measured at that rate (a
 * GPU at $0 offers no saving), and the limits see the same dollars through `allow`.
 */
export function withRate(base: PredictionProvider, o: RateOverride | null): PredictionProvider {
	if (!o) return base;
	const wrap = (record: LoadedRecord, opts: SensitivityOptions | undefined): SensitivityOptions | undefined => {
		if (!opts?.allow) return opts;
		const knobs = opts.knobs ?? {};
		const allow = opts.allow;
		return { ...opts, allow: (id, v, after) => allow(id, v, priced(record, { ...knobs, [id]: v }, after, o)) };
	};
	return {
		name: base.name,
		ready: () => base.ready(),
		predict: (record, knobs) => priced(record, knobs, base.predict(record, knobs), o),
		sensitivity: (record, objective: Objective, opts) => base.sensitivity(record, objective, wrap(record, opts)),
		moves: (record, objective: Objective, opts) => {
			const ms = base.moves(record, objective, wrap(record, opts));
			const costId = targetFor(record, "cost")?.id;
			if (!costId || !ms.some((m) => m.targetId === costId)) return ms;
			const knobs = opts?.knobs ?? {};
			const before = priced(record, knobs, base.predict(record, knobs), o)[costId]?.point;
			// On a run predicted to fail, moves that clear the failure or lower its risk stay, in the
			// engine's order and first, whatever they cost: a run that would crash needs them. Their cost
			// change is quoted at the visitor's rate. Review, Nov 1: filtering them dropped the OOM fix.
			const fail = failureTarget(record);
			const riskNow = fail ? base.predict(record, knobs)[fail]?.point : null;
			const failing = typeof riskNow === "number" && riskNow >= FAILS_AT;
			const keeps = (m: Move) => m.clearsFailure || (failing && m.risk !== null && m.risk < (riskNow as number));
			const repriced = (m: Move): Move | null => {
				if (m.targetId !== costId || typeof before !== "number") return m;
				const next = { ...knobs, [m.knobId]: m.to };
				const after = priced(record, next, base.predict(record, next), o)[costId]?.point;
				if (typeof after !== "number") return null;
				const delta = after - before;
				return { ...m, delta, relative: before !== 0 ? delta / Math.abs(before) : null };
			};
			const clearing = ms.filter(keeps).map((m) => repriced(m) ?? m);
			const rest: Move[] = [];
			for (const m of ms) {
				if (keeps(m)) continue;
				const r = repriced(m);
				// At the visitor's rate a "saving" that saves nothing isn't offered.
				if (!r || (r.targetId === costId && r.delta >= 0)) continue;
				rest.push(r);
			}
			const costMoves = rest.filter((m) => m.targetId === costId).sort((a, b) => a.delta - b.delta);
			return [...clearing, ...costMoves, ...rest.filter((m) => m.targetId !== costId)];
		},
	};
}

const KEY = "pf.rate.v1";

/** The visitor's rate for a job, from this browser (per job: keyed by the record's reference). */
export function loadRate(ref: string): RateOverride | null {
	try {
		const all = JSON.parse(localStorage.getItem(KEY) ?? "{}") as Record<string, RateOverride>;
		const o = all[ref];
		if (!o || typeof o !== "object") return null;
		if (o.kind === "gpu" && typeof o.gpu === "string" && Number.isFinite(o.usdPerHour) && o.usdPerHour >= 0) return o;
		if (o.kind === "tokens" && typeof o.model === "string" && Number.isFinite(o.inputPerM) && Number.isFinite(o.outputPerM) && o.inputPerM >= 0 && o.outputPerM >= 0) return o;
		return null;
	} catch {
		return null;
	}
}

export function saveRate(ref: string, o: RateOverride | null): void {
	try {
		const all = JSON.parse(localStorage.getItem(KEY) ?? "{}") as Record<string, RateOverride>;
		if (o) all[ref] = o;
		else delete all[ref];
		localStorage.setItem(KEY, JSON.stringify(all));
	} catch {
		// Private windows and blocked storage: the rate holds for this visit only.
	}
}
