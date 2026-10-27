/**
 * The orbit layout: a training step is one lap. Pure function of the record, the current
 * knobs and the provider's estimates; no prediction happens here.
 *
 * - Phase arcs (outer band) are as long as each phase's share of a step: from `time.phase.*`
 *   predictions when the record has them, else from the nearest probe's measured shares
 *   (and the caption says which probe), else equal and labelled "not measured yet".
 * - Loop nodes sit at the middle of their phase's arc; a phase with no graph node (backward,
 *   usually) gets a phase node of its own.
 * - Branch nodes (an eval pass, checkpoints) hang outside the loop on tethers.
 * - The memory ring is drawn against GPU capacity: full circle = capacity, overflow past it
 *   is a dashed outer arc.
 */
import { nearestProbe } from "../provider/metrics";
import { capacityBytes, targetFor } from "../record/model";
import { type MemorySegment, type MemoryStack, memorySegments } from "../record/terms";
import type { GraphNode, KnobValue, LoadedRecord, Prediction } from "../record/types";
import { trusted } from "../status";

export const FRAME = { w: 600, h: 560, cx: 300, cy: 280 };
export const R_ORBIT = 180;
export const R_BAND = 216;
export const R_MEM = 98;
export const R_OVER = 116;
export type { MemorySegment };
export const R_BRANCH = 262;

const LOOP_PHASES = ["data", "forward", "loss", "backward", "optimizer", "rollout", "reference", "reward", "update", "weight_sync"];
const BRANCH_PHASES = new Set(["eval", "checkpoint"]);

export type NodeShape = "circle" | "diamond" | "hexagon";

export interface OrbitNode {
	id: string;
	label: string;
	/** Graph node id, or null for a synthetic phase node. */
	graphId: string | null;
	phase: string | null;
	x: number;
	y: number;
	/** Diameter in frame units. */
	size: number;
	shape: NodeShape;
	share: number | null;
	repeat: number;
	branch: boolean;
}

export interface Arc {
	phase: string;
	share: number;
	startDeg: number;
	sweepDeg: number;
	midDeg: number;
}

export interface OrbitLayout {
	arcs: Arc[];
	shareSource: { kind: "prediction" | "probe" | "none"; probeId?: string; probeKnobs?: Record<string, KnobValue | undefined> };
	nodes: OrbitNode[];
	tethers: Array<{ from: { x: number; y: number }; to: { x: number; y: number }; nodeId: string }>;
	memory: Pick<MemoryStack, "segments" | "breakdown" | "untrusted" | "abstaining" | "partsBytes"> & { total: number | null; capacity: number | null };
	tokensPerSecond: number | null;
}

export function polar(deg: number, r: number): { x: number; y: number } {
	const rad = (deg * Math.PI) / 180;
	return { x: FRAME.cx + r * Math.cos(rad), y: FRAME.cy + r * Math.sin(rad) };
}

function clamp(v: number, lo: number, hi: number): number {
	return Math.max(lo, Math.min(hi, v));
}

function phaseShares(record: LoadedRecord, knobs: Readonly<Record<string, KnobValue>>, est: Prediction): OrbitLayout["shareSource"] & { shares: Record<string, number> } {
	const predicted: Record<string, number> = {};
	for (const p of LOOP_PHASES) {
		const e = est[`time.phase.${p}`];
		if (trusted(e) && e.point > 0) predicted[p] = e.point;
	}
	if (Object.keys(predicted).length >= 2) return { kind: "prediction", shares: normalise(predicted) };
	const probe = nearestProbe(record, knobs);
	const byPhase = probe?.step_time?.by_phase;
	if (probe && byPhase) {
		const measured: Record<string, number> = {};
		for (const p of LOOP_PHASES) {
			const v = byPhase[p];
			if (typeof v === "number" && v > 0) measured[p] = v;
		}
		if (Object.keys(measured).length >= 2) return { kind: "probe", probeId: probe.id, probeKnobs: probe.knobs, shares: normalise(measured) };
	}
	return { kind: "none", shares: {} };
}

function normalise(v: Record<string, number>): Record<string, number> {
	const total = Object.values(v).reduce((a, b) => a + b, 0);
	const out: Record<string, number> = {};
	for (const [k, x] of Object.entries(v)) out[k] = x / total;
	return out;
}

/** Which phase a top-level graph node belongs to, by its kind and id. */
function phaseOf(n: GraphNode): string | null {
	if (n.kind === "data") return "data";
	if (n.kind === "loss") return "loss";
	if (n.kind === "optimizer") return "optimizer";
	if (n.kind === "group" || n.kind === "module" || n.kind === "model_call") return "forward";
	if (n.kind === "phase") return BRANCH_PHASES.has(n.id) || /eval/i.test(n.id) ? "eval" : n.id;
	if (n.kind === "io" || n.kind === "store") return "checkpoint";
	return null;
}

function isBranch(n: GraphNode): boolean {
	const p = phaseOf(n);
	return p === "eval" || p === "checkpoint";
}

const GAP_DEG = 4;
const NODE_GAP_PX = 10;

/**
 * Nudge nodes apart round the loop so none overlap, keeping loop order and staying as close
 * to their own arc as space allows. Small phases (data, loss, optimiser) often sit side by side.
 */
function spread(angles: number[], sizes: number[], radius: number): number[] {
	const n = angles.length;
	if (n < 2) return angles;
	const toDeg = 180 / Math.PI;
	let half = sizes.map((s) => ((s / 2 + NODE_GAP_PX / 2) / radius) * toDeg);
	const need = half.reduce((a, b) => a + 2 * b, 0);
	if (need > 360) half = half.map((h) => (h * 360) / need);
	const order = angles.map((a, i) => ({ a, i })).sort((x, y) => x.a - y.a);
	const a = order.map((o) => o.a);
	const h = order.map((o) => half[o.i] ?? 0);
	const target = [...a];
	for (let iter = 0; iter < 300; iter++) {
		let moved = false;
		for (let k = 0; k < n; k++) {
			const j = (k + 1) % n;
			const gap = j === 0 ? a[j]! + 360 - a[k]! : a[j]! - a[k]!;
			const want = h[k]! + h[j]!;
			if (gap < want - 0.01) {
				const push = (want - gap) / 2;
				a[k] = a[k]! - push;
				a[j] = a[j]! + push;
				moved = true;
			}
		}
		for (let k = 0; k < n; k++) a[k] = a[k]! + (target[k]! - a[k]!) * 0.02;
		if (!moved && iter > 20) break;
	}
	const out = new Array<number>(n);
	order.forEach((o, k) => {
		out[o.i] = a[k]!;
	});
	return out;
}

function angleOf(x: number, y: number): number {
	return (Math.atan2(y - FRAME.cy, x - FRAME.cx) * 180) / Math.PI;
}

export function layoutOrbit(record: LoadedRecord, knobs: Readonly<Record<string, KnobValue>>, est: Prediction): OrbitLayout {
	const top = record.graph.nodes.filter((n) => !n.parent);
	const loopNodes = top.filter((n) => !isBranch(n));
	const branchNodes = top.filter(isBranch);
	const source = phaseShares(record, knobs, est);

	// Phases on the loop: those with a share, plus any a loop node needs, in loop order.
	const needed = new Set<string>(Object.keys(source.shares));
	for (const n of loopNodes) {
		const p = phaseOf(n);
		if (p) needed.add(p);
	}
	let phases = LOOP_PHASES.filter((p) => needed.has(p));
	// A lens with no phase vocabulary (agent turns, serving) puts its nodes round the loop in order.
	const generic = phases.length < 2;
	if (generic) phases = loopNodes.map((n) => n.id);

	const shareOf = (p: string): number => {
		if (source.kind !== "none" && !generic) return source.shares[p] ?? 0.04;
		return 1 / Math.max(1, phases.length);
	};
	const raw = phases.map(shareOf);
	const sum = raw.reduce((a, b) => a + b, 0) || 1;
	const shares = raw.map((s) => s / sum);
	const avail = 360 - phases.length * GAP_DEG;
	const arcs: Arc[] = [];
	let start = -90 - (shares[0] * avail) / 2;
	phases.forEach((p, i) => {
		const sweep = shares[i] * avail;
		arcs.push({ phase: p, share: shares[i], startDeg: start, sweepDeg: sweep, midDeg: start + sweep / 2 });
		start += sweep + GAP_DEG;
	});
	const arcByPhase = new Map(arcs.map((a) => [a.phase, a]));

	const nodes: OrbitNode[] = [];
	const nodeSize = (share: number | null) => clamp(70 + (share ?? 0.15) * 120, 86, 126);
	const usedPhases = new Set<string>();
	for (const n of loopNodes) {
		const phase = generic ? n.id : phaseOf(n);
		const arc = phase ? arcByPhase.get(phase) : undefined;
		if (!arc || !phase) continue;
		const sameArc = loopNodes.filter((m) => (generic ? m.id : phaseOf(m)) === phase);
		const idx = sameArc.indexOf(n);
		const offset = sameArc.length > 1 ? (idx - (sameArc.length - 1) / 2) * Math.min(18, arc.sweepDeg / sameArc.length) : 0;
		const p = polar(arc.midDeg + offset, R_ORBIT);
		usedPhases.add(phase);
		nodes.push({
			id: n.id,
			label: n.label,
			graphId: n.id,
			phase,
			x: p.x,
			y: p.y,
			size: nodeSize(source.kind !== "none" ? arc.share : null),
			shape: "circle",
			share: source.kind !== "none" ? arc.share : null,
			repeat: n.repeat ?? 1,
			branch: false,
		});
	}
	// Phases with time but no node (backward, usually) get a phase node.
	for (const arc of arcs) {
		if (usedPhases.has(arc.phase) || generic) continue;
		const p = polar(arc.midDeg, R_ORBIT);
		nodes.push({
			id: `phase:${arc.phase}`,
			label: arc.phase,
			graphId: null,
			phase: arc.phase,
			x: p.x,
			y: p.y,
			size: nodeSize(source.kind !== "none" ? arc.share : null),
			shape: "circle",
			share: source.kind !== "none" ? arc.share : null,
			repeat: 1,
			branch: false,
		});
	}

	// Keep loop nodes from overlapping; the arcs still show the true shares.
	{
		const loopAngles = nodes.map((m) => angleOf(m.x, m.y));
		const spreadAngles = spread(loopAngles, nodes.map((m) => m.size), R_ORBIT);
		nodes.forEach((m, k) => {
			const p = polar(spreadAngles[k] ?? loopAngles[k] ?? 0, R_ORBIT);
			m.x = p.x;
			m.y = p.y;
		});
	}

	// Branches hang off the phase their control edge leaves from, pushed outwards into free sky.
	const tethers: OrbitLayout["tethers"] = [];
	branchNodes.forEach((n, i) => {
		const kind = phaseOf(n);
		const edge = record.graph.edges.find((e) => e.target === n.id);
		const src = nodes.find((m) => m.graphId === edge?.source) ?? nodes.find((m) => m.phase === (kind === "eval" ? "loss" : "optimizer"));
		const anchorDeg = src ? angleOf(src.x, src.y) : -30 + i * 60;
		const size = kind === "eval" ? 88 : 76;
		const prefer = kind === "eval" ? -1 : 1;
		const fits = (deg: number) => {
			const p = polar(deg, R_BRANCH);
			const r = size / 2 + 6;
			if (p.x < r || p.x > FRAME.w - r || p.y < r || p.y > FRAME.h - r) return false;
			return nodes.every((m) => Math.hypot(p.x - m.x, p.y - m.y) >= (m.size + size) / 2 + (m.branch ? 16 : 8));
		};
		let deg = anchorDeg + prefer * 28;
		for (let off = 24; off <= 180; off += 4) {
			if (fits(anchorDeg + prefer * off)) {
				deg = anchorDeg + prefer * off;
				break;
			}
			if (fits(anchorDeg - prefer * off)) {
				deg = anchorDeg - prefer * off;
				break;
			}
		}
		const raw = polar(deg, R_BRANCH);
		const to = { x: clamp(raw.x, size / 2, FRAME.w - size / 2), y: clamp(raw.y, size / 2, FRAME.h - size / 2) };
		const from = src ? { x: src.x, y: src.y } : polar(anchorDeg, R_ORBIT);
		tethers.push({ from, to, nodeId: n.id });
		nodes.push({
			id: n.id,
			label: n.label,
			graphId: n.id,
			phase: kind,
			x: to.x,
			y: to.y,
			size,
			shape: kind === "eval" ? "diamond" : "hexagon",
			share: null,
			repeat: n.repeat ?? 1,
			branch: true,
		});
	});

	// Memory ring.
	const memT = targetFor(record, "memory");
	const peak = memT ? est[memT.id] : undefined;
	const stack = memorySegments(est, trusted(peak) ? peak.point : null);
	const tps = est["throughput.tokens_per_s"];
	return {
		arcs,
		shareSource: { kind: source.kind, ...(source.probeId ? { probeId: source.probeId, probeKnobs: source.probeKnobs } : {}) },
		nodes,
		tethers,
		memory: {
			segments: stack.segments,
			untrusted: stack.untrusted,
			abstaining: stack.abstaining,
			partsBytes: stack.partsBytes,
			total: trusted(peak) ? peak.point : null,
			// A fraction (an agent's context fill) has its own full circle: 100%.
			capacity: memT?.unit === "fraction" ? 1 : capacityBytes(record, knobs),
			breakdown: stack.breakdown,
		},
		tokensPerSecond: trusted(tps) ? tps.point : null,
	};
}

/** Circle dasharray for an arc of `sweepDeg` on radius r. */
export function arcDash(sweepDeg: number, r: number): string {
	const c = 2 * Math.PI * r;
	return `${((Math.max(0, sweepDeg) / 360) * c).toFixed(1)} ${c.toFixed(1)}`;
}
