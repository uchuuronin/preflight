/**
 * The engine's own prediction where the record's grids hold no value (Chiya, Nov 2).
 *
 * GridProvider reads predictions off the record's grids, instantly, and a grid holds only the knob
 * values it was built over: off them it abstained ("off grid", "between measured points"), though the
 * engine predicts there. On the Nov 2 LoRA record the engine gives lr 2e-4 the same 1.709 s step as
 * 1e-4, and the A10 1.236 s (extrapolated across GPUs), where the cockpit said nothing. So for a
 * setting the grids don't hold, the API's `POST /v1/records/{ref}/predict` (the engine,
 * `preflight.engine.api.predict`) is asked once, and its answer is cached and read through this
 * provider like any grid value: a target the grids abstained on takes the engine's estimate with
 * the engine's own status (predicted, or extrapolated across GPUs).
 *
 * Only settings the visitor lands on are asked (state.tsx), and every read of `predict` (dials,
 * timeline, the guide's readouts) uses what is cached and never waits. The guide's ranked moves and
 * the fit map stay on the grids: they read hundreds of settings a frame. Behind `c.engine_fallback`.
 */
import { flag } from "../../lib/flags";
import type { KnobValue, LoadedRecord, Prediction } from "../record/types";
import type { PredictionProvider } from "./PredictionProvider";

/** Abstentions that mean "no grid value here", not "the engine can't say". */
const OFF_GRID = new Set(["off_grid", "between_measured"]);
/** The API's own cap per request (api/preflight_api/routes/predict.py MAX_CELLS). */
const MAX_CELLS = 16;
/** A ref the API can't resolve: a live file read from the visitor's own disk. */
const LOCAL_REFS = new Set(["live", ""]);

type Entry = Prediction | "pending" | "failed";
// Per loaded record, so a record revised by a later probe starts afresh, and a record no longer open
// takes its answers with it.
let caches = new WeakMap<LoadedRecord, Map<string, Entry>>();
function cacheOf(record: LoadedRecord): Map<string, Entry> {
	let c = caches.get(record);
	if (!c) caches.set(record, (c = new Map()));
	return c;
}
const listeners = new Set<() => void>();
let version = 0;

function bump(): void {
	version += 1;
	for (const fn of listeners) fn();
}

export function subscribeEngine(fn: () => void): () => void {
	listeners.add(fn);
	return () => listeners.delete(fn);
}

export function engineVersion(): number {
	return version;
}

/** Tests only. */
export function resetEngineCache(): void {
	caches = new WeakMap();
	bump();
}

/** A record read from the visitor's own disk: the server has never seen it. */
export function isLocalRef(ref: string | null | undefined): boolean {
	return typeof ref !== "string" || LOCAL_REFS.has(ref);
}

export function engineOn(ref: string | null | undefined): ref is string {
	return flag("c.engine_fallback") && typeof ref === "string" && !LOCAL_REFS.has(ref);
}

/** The setting as the API takes it: every record knob, in a fixed order, so equal settings share a key. */
function settingOf(record: LoadedRecord, knobs: Readonly<Record<string, KnobValue>>): Record<string, KnobValue> {
	const out: Record<string, KnobValue> = {};
	for (const k of record.knobs) {
		const v = k.id in knobs ? knobs[k.id] : k.current;
		if (v !== undefined && v !== null) out[k.id] = v as KnobValue;
	}
	return out;
}

const keyOf = (ref: string, setting: Record<string, KnobValue>) => `${ref}|${JSON.stringify(setting)}`;

/** Does any target abstain only for want of a grid value? */
export function needsEngine(p: Prediction): boolean {
	return Object.values(p).some((e) => e?.status === "abstain" && OFF_GRID.has(String(e.reason ?? "")));
}

/**
 * The grid's prediction, with the engine's estimate for each target the grids abstained on. Never
 * "measured": the engine matches a probe ignoring knobs that don't change cost (lr, epochs), so at lr
 * 2e-4 it calls the lr 1e-4 probe's out-of-memory measured; here "measured" means a probe ran this
 * very setting (GridProvider), so the engine's reads as predicted.
 */
export function mergeEngine(grid: Prediction, engine: Prediction): Prediction {
	const out: Prediction = { ...grid };
	for (const [id, e] of Object.entries(grid)) {
		if (e?.status !== "abstain" || !OFF_GRID.has(String(e.reason ?? ""))) continue;
		const got = engine[id];
		if (!got || got.status === "abstain" || typeof got.point !== "number" || got.unit !== e.unit) continue;
		out[id] = got.status === "measured" ? { ...got, status: "predicted" } : got;
	}
	return out;
}

/** The base provider, reading the engine's cached answers where the grids abstained. Never fetches. */
export function withEngine(base: PredictionProvider, ref: string | null | undefined): PredictionProvider {
	if (!engineOn(ref)) return base;
	return {
		name: base.name,
		ready: () => base.ready(),
		predict(record, knobs) {
			const p = base.predict(record, knobs);
			if (!needsEngine(p)) return p;
			const hit = cacheOf(record).get(keyOf(ref, settingOf(record, knobs)));
			return hit && hit !== "pending" && hit !== "failed" ? mergeEngine(p, hit) : p;
		},
		sensitivity: (record, objective, opts) => base.sensitivity(record, objective, opts),
		moves: (record, objective, opts) => base.moves(record, objective, opts),
	};
}

/**
 * Ask the engine for these settings (those not asked before), in pages the API takes. A failure is
 * remembered, so a setting the engine can't read isn't asked again; the grids' abstention stands.
 */
export async function askEngine(ref: string, record: LoadedRecord, settings: readonly Readonly<Record<string, KnobValue>>[]): Promise<void> {
	if (!engineOn(ref)) return;
	const cache = cacheOf(record);
	const todo: { key: string; setting: Record<string, KnobValue> }[] = [];
	for (const s of settings) {
		const setting = settingOf(record, s);
		const key = keyOf(ref, setting);
		if (cache.has(key) || todo.some((x) => x.key === key)) continue;
		cache.set(key, "pending");
		todo.push({ key, setting });
	}
	// Only an answer re-renders: a busy or offline reply clears its settings without a bump, so they
	// are asked again when the visitor next lands on them, never in a loop of renders and retries.
	let answered = false;
	for (let i = 0; i < todo.length; i += MAX_CELLS) {
		const page = todo.slice(i, i + MAX_CELLS);
		try {
			const res = await fetch(`/v1/records/${encodeURIComponent(ref)}/predict`, {
				method: "POST",
				credentials: "same-origin",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ cells: page.map((x) => x.setting) }),
			});
			if (res.status === 429 || res.status >= 500) {
				page.forEach((x) => cache.delete(x.key));
				continue;
			}
			const json = res.ok ? ((await res.json()) as { cells?: Prediction[] }) : null;
			page.forEach((x, j) => cache.set(x.key, json?.cells?.[j] ?? "failed"));
			answered = true;
		} catch {
			page.forEach((x) => cache.delete(x.key));
		}
	}
	if (answered) bump();
}
