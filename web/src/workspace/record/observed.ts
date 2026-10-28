/**
 * Which readouts a probe actually observed. A number is "measured" only when a successful probe
 * ran at exactly these settings AND that probe saw this target itself: its memory peak, a term
 * the allocator attributed, a phase share, a throughput. Run totals (time.total, cost.total)
 * and risks are always computed from what was observed, so they are never "measured".
 */
import type { KnobValue, LoadedRecord, Probe } from "./types";

type Obj = Record<string, unknown>;

function num(o: unknown, key: string): boolean {
	const v = o && typeof o === "object" ? (o as Obj)[key] : undefined;
	return typeof v === "number" && Number.isFinite(v);
}

function sub(o: unknown, key: string): unknown {
	return o && typeof o === "object" ? (o as Obj)[key] : undefined;
}

/** Probe blocks whose numeric fields map one to one onto targets `<block>.<field>`. */
const DIRECT_BLOCKS = ["throughput", "generation", "eval", "power", "checkpoint"] as const;

/** Targets a probe observed itself (not derived from what it observed). */
export function probeObserves(probe: Probe, targetId: string): boolean {
	if (probe.status !== "ok" && probe.status !== "cached") return false;
	const [head, ...rest] = targetId.split(".");
	const tail = rest.join(".");
	switch (head) {
		case "memory": {
			const mem = probe.memory;
			if (!mem) return false;
			// A memory peak needs at least two optimizer steps (the contract's rule for a valid peak).
			if (tail === "peak") return (probe.steps ?? 0) >= 2 && (num(mem, "peak_allocated_bytes") || num(mem, "peak_reserved_bytes"));
			const term = tail.startsWith("term.") ? tail.slice(5) : tail;
			return num(mem.by_term, term);
		}
		case "time": {
			const st = probe.step_time;
			if (!st) return false;
			if (tail.startsWith("phase.")) return num(st.by_phase, tail.slice(6));
			if (tail === "step" || tail === "step_mean") return num(st, "mean_s");
			if (tail === "step_p95") return num(st, "p95_s");
			return false;
		}
		default:
			if ((DIRECT_BLOCKS as readonly string[]).includes(head ?? "")) return num(sub(probe, head ?? ""), tail);
			return false;
	}
}

function same(a: KnobValue | undefined, b: KnobValue | undefined): boolean {
	if (typeof a === "number" && typeof b === "number") return Math.abs(a - b) <= 1e-9 * Math.max(1, Math.abs(a));
	return a === b;
}

/** Successful probes that ran at exactly these settings (a probe holds unlisted knobs at current). */
export function probesAt(record: LoadedRecord, knobs: Readonly<Record<string, KnobValue>>): Probe[] {
	return record.probes.filter(
		(p) => (p.status === "ok" || p.status === "cached") && record.knobs.every((k) => same(knobs[k.id], p.knobs[k.id] ?? k.current)),
	);
}

/** True when a probe at exactly these settings observed this target. */
export function observedAt(record: LoadedRecord, knobs: Readonly<Record<string, KnobValue>>, targetId: string): boolean {
	return probesAt(record, knobs).some((p) => probeObserves(p, targetId));
}
