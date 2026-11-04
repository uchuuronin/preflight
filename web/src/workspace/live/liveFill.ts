/**
 * A followed run's own readings where the record has no prediction (Chiya, Nov 4: on `#/cockpit/live`
 * the dials said "Not measured here" while the live panel was measuring right under them). At the live
 * run's own settings only, for a target the record abstains on:
 *
 * - time to finish: the projected total (time so far plus the remaining steps at the recent pace);
 * - run cost: that time at the record's GPU rate (the visitor's own rate re-prices it, as any cost);
 * - peak memory: the run's peak so far.
 *
 * Each is an estimate with status "measured" and model id LIVE_MODEL, so every reader shows the number
 * and the provenance word reads "Measured live". No range: it is one reading, not a prediction. Where
 * the run has no reading either, the record's abstention stands.
 */
import { currentKnobs, targetFor } from "../record/model";
import type { Estimate, FlightRecord, KnobValue, LoadedRecord, Prediction } from "../record/types";
import { tableRate } from "../rate";
import type { PredictionProvider } from "../provider/PredictionProvider";
import { readLive } from "./liveCheck";

export const LIVE_MODEL = "live";

export function isLiveEstimate(e: Estimate | null | undefined): boolean {
	return !!e && e.model_id === LIVE_MODEL;
}

/** The run is still going: its peak is a peak so far. */
export function liveStillRunning(live: FlightRecord): boolean {
	return readLive(live)?.status === "running";
}

function same(a: KnobValue | undefined, b: KnobValue | undefined): boolean {
	if (typeof a === "number" && typeof b === "number") return Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(a));
	return a === b;
}

function atOwnSettings(record: LoadedRecord, knobs: Readonly<Record<string, KnobValue>>): boolean {
	const own = currentKnobs(record);
	return record.knobs.every((k) => !(k.id in knobs) || same(knobs[k.id], own[k.id]));
}

function reading(point: number, unit: string): Estimate {
	return { point, lo: point, hi: point, unit, status: "measured", model_id: LIVE_MODEL, source: "exact" };
}

export function liveFill(record: LoadedRecord, knobs: Readonly<Record<string, KnobValue>>, p: Prediction, live: FlightRecord): Prediction {
	if (!atOwnSettings(record, knobs)) return p;
	const r = readLive(live);
	if (!r) return p;
	const out: Prediction = { ...p };
	const missing = (id: string) => !p[id] || p[id]?.status === "abstain";
	const time = targetFor(record, "time");
	if (time && time.unit === "s" && missing(time.id) && r.projectedS !== null) out[time.id] = reading(r.projectedS, "s");
	const cost = targetFor(record, "cost");
	if (cost && cost.unit === "usd" && missing(cost.id) && r.projectedS !== null) {
		const rate = tableRate(record, knobs);
		if (rate?.kind === "gpu") {
			const hours = r.projectedS / 3600;
			out[cost.id] = {
				...reading(hours * rate.usdPerHour, "usd"),
				rate: { quantity: "cost.gpu_hours", unit: "gpu_hours", usd_per_unit: rate.usdPerHour, price_row: rate.row.id },
			};
		}
	}
	const mem = targetFor(record, "memory");
	if (mem && mem.unit === "bytes" && missing(mem.id) && r.peakBytes !== null) out[mem.id] = reading(r.peakBytes, "bytes");
	return out;
}

/** The base provider, with the live run's readings where it abstains at the run's own settings. */
export function withLive(base: PredictionProvider, live: FlightRecord | null): PredictionProvider {
	if (!live) return base;
	return {
		name: base.name,
		ready: () => base.ready(),
		predict: (record, knobs) => liveFill(record, knobs, base.predict(record, knobs), live),
		sensitivity: (record, objective, opts) => base.sensitivity(record, objective, opts),
		moves: (record, objective, opts) => base.moves(record, objective, opts),
	};
}
