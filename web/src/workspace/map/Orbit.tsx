import { type Node, type NodeProps, ReactFlow, ReactFlowProvider, useReactFlow } from "@xyflow/react";
import "@xyflow/react/dist/base.css";
import { memo, useEffect, useMemo, useRef } from "react";
import { MOVE_MS, moveEase, useTweenedText } from "../../lib/tween";
import { fmt } from "../../lib/fmt";
import { hasKey, t } from "../../lib/i18n";
import { useMapOverlays } from "../../registry";
import { targetFor } from "../record/model";
import { useWorkspace } from "../state";
import { memoryTermBytes } from "../record/terms";
import { fitOf, likelyFails, trusted, type Fit } from "../status";
import { FitMark } from "../components/StatusMark";
import { R_BAND, R_MEM, R_ORBIT, R_OVER, FRAME, NODE_BOX, arcDash, layoutOrbit, type OrbitLayout, type OrbitNode } from "./orbitLayout";
import { AddonBoundary } from "../../components/AddonBoundary";

interface BgData extends Record<string, unknown> {
	layout: OrbitLayout;
	highlightPhases: string[];
	fit: Fit;
	evalStalled: boolean;
	coreNumber: string;
	/** The number behind coreNumber, so it can travel with the ring (null: abstaining). */
	coreValue: number | null;
	coreUnit: string;
	coreCaption: string;
	coreStatus: string;
}

interface NodeData extends Record<string, unknown> {
	node: OrbitNode;
	title: string;
	meta: string;
	accent: boolean;
	dim: boolean;
	fails: boolean;
	finding: string | null;
}

function phaseLabel(phase: string): string {
	return hasKey(`phase.${phase}`) ? t(`phase.${phase}`) : phase;
}

/**
 * The memory ring's parts: each group's arc (a full ring is this GPU's memory), the hatched part
 * past the peak's angle when the parts add up to more than the peak, and how far past the GPU the
 * peak goes (the dashed outer ring). Shared by the orbit and its glance in the side column.
 */
export function memoryRings(memory: OrbitLayout["memory"], rMem: number = R_MEM, rOver: number = R_OVER) {
	// A full ring is this GPU's memory; what doesn't fit continues on the dashed outer ring.
	const scale = memory.capacity ?? memory.total ?? 1;
	const cMem = 2 * Math.PI * rMem;
	let acc = -90;
	const segs = memory.segments.flatMap((s) => {
		const room = 270 - acc;
		const deg = Math.min((s.bytes / scale) * 360, room);
		if (deg <= 0) return [];
		const gap = deg > 3 ? 2 : 0;
		const len = Math.max(0, (deg / 360) * cMem - gap);
		const rot = acc;
		acc += deg;
		return [{ ...s, dash: `${len.toFixed(1)} ${cMem.toFixed(1)}`, rot }];
	});
	// Parts that add up past the peak: hatched past the peak's angle, never squeezed to fit.
	const untrustedArc = (() => {
		const u = memory.untrusted;
		if (u?.kind !== "overshoot" || !memory.total) return null;
		const start = -90 + Math.min(1, memory.total / scale) * 360;
		const sweep = Math.min((u.bytes / scale) * 360, 270 - start);
		if (sweep <= 0) return null;
		const len = (sweep / 360) * cMem;
		const dashes: string[] = [];
		for (let d = 0; d + 2 <= len; d += 5) dashes.push(d + 5 <= len ? "2 3" : `2 ${(len - d - 2).toFixed(1)}`);
		return { rot: start, dash: `${dashes.join(" ")} 0 ${cMem.toFixed(1)}` };
	})();
	const over = memory.total && memory.capacity && memory.total > memory.capacity ? Math.min(1, (memory.total - memory.capacity) / memory.capacity) : 0;
	const cOver = 2 * Math.PI * rOver;
	return { segs, untrustedArc, over, cMem, cOver };
}

const OrbitBackground = memo(function OrbitBackground({ data }: NodeProps<Node<BgData>>) {
	const { layout, highlightPhases, fit, evalStalled } = data;
	const coreRef = useRef<HTMLSpanElement>(null);
	const coreUnit = data.coreUnit;
	const coreFormat = useMemo(() => (v: number) => fmt(v, coreUnit), [coreUnit]);
	useTweenedText(coreRef, data.coreValue, coreFormat, { ms: MOVE_MS, ease: moveEase });
	const { memory } = layout;
	const { segs, untrustedArc, over, cMem, cOver } = memoryRings(memory);
	// One lap of dots takes less time when the job moves more tokens per second.
	const lap = layout.tokensPerSecond ? Math.max(0.6, Math.min(8, 7000 / layout.tokensPerSecond)) : 3.2;
	return (
		<div className="pf-orbit-bg" style={{ width: FRAME.w, height: FRAME.h }}>
			<svg viewBox={`0 0 ${FRAME.w} ${FRAME.h}`} width={FRAME.w} height={FRAME.h} aria-hidden="true">
				{layout.arcs.map((a) => (
					<circle
						key={a.phase}
						className={`pf-arc${highlightPhases.includes(a.phase) ? " is-accent" : ""}`}
						cx={FRAME.cx}
						cy={FRAME.cy}
						r={R_BAND}
						strokeDasharray={arcDash(a.sweepDeg, R_BAND)}
						style={{ transform: `rotate(${a.startDeg.toFixed(2)}deg)` }}
					/>
				))}
				<circle className="pf-orbit-base" cx={FRAME.cx} cy={FRAME.cy} r={R_ORBIT} />
				<circle className="pf-orbit-flow" cx={FRAME.cx} cy={FRAME.cy} r={R_ORBIT} style={{ animationDuration: `${lap.toFixed(2)}s` }} />
				{layout.tethers.map((tt) => {
					const d = `M${tt.from.x.toFixed(1)} ${tt.from.y.toFixed(1)} L${tt.to.x.toFixed(1)} ${tt.to.y.toFixed(1)}`;
					return (
						<g key={tt.nodeId}>
							<path className="pf-tether" d={d} />
							<path className={`pf-tether-flow${evalStalled && tt.nodeId === "eval" ? " is-stalled" : ""}`} d={d} />
						</g>
					);
				})}
				<circle className="pf-mem-track" cx={FRAME.cx} cy={FRAME.cy} r={R_MEM} />
				{segs.map((s) => (
					<circle
						key={s.key}
						className={`pf-mring pf-mring--${s.slot}${s.key === "total" ? " is-total" : ""}`}
						cx={FRAME.cx}
						cy={FRAME.cy}
						r={R_MEM}
						strokeDasharray={s.dash}
						style={{ transform: `rotate(${s.rot.toFixed(2)}deg)` }}
					/>
				))}
				{untrustedArc ? (
					<circle
						className="pf-mring-untrusted"
						cx={FRAME.cx}
						cy={FRAME.cy}
						r={R_MEM}
						strokeDasharray={untrustedArc.dash}
						style={{ transform: `rotate(${untrustedArc.rot.toFixed(2)}deg)` }}
					/>
				) : null}
				{over > 0 ? (
					<circle
						className="pf-mem-over"
						cx={FRAME.cx}
						cy={FRAME.cy}
						r={R_OVER}
						strokeDasharray={`${(over * cOver).toFixed(1)} ${cOver.toFixed(1)}`}
						transform={`rotate(-90 ${FRAME.cx} ${FRAME.cy})`}
					/>
				) : null}
			</svg>
			<div className="pf-core">
				<span ref={coreRef} className="pf-core__num">
					{data.coreNumber}
				</span>
				<span className="pf-core__cap">{data.coreCaption}</span>
				{fit !== "unknown" ? <FitMark fit={fit} label={data.coreStatus} /> : null}
			</div>
		</div>
	);
});

const OrbitNodeView = memo(function OrbitNodeView({ data }: NodeProps<Node<NodeData>>) {
	const { node, title, meta, accent, dim, fails, finding } = data;
	const cls = [
		"pf-onode",
		`pf-onode--${node.shape}`,
		accent ? "is-accent" : "",
		dim ? "is-dim" : "",
		fails ? "is-fails" : "",
	].join(" ");
	// Outline weight carries the share of a step.
	const bw = node.share === null ? 1 : node.share > 0.4 ? 3 : node.share > 0.2 ? 2 : 1;
	return (
		<div className="pf-onode-box">
			<button
				type="button"
				className={cls}
				style={{ width: node.size, height: node.size, borderWidth: node.shape === "hexagon" ? 0 : bw }}
				aria-label={t("map.node.aria", { name: title, meta })}
			>
				<span className="pf-onode__inner">
					<b>{title}</b>
					{node.repeat > 1 ? <small>{t("map.repeat", { n: node.repeat })}</small> : null}
					<small>{meta}</small>
				</span>
				{finding ? <span className={`pf-onode__chip${fails ? " is-fails" : ""}`}>{finding}</span> : null}
			</button>
		</div>
	);
});

const NODE_TYPES = { bg: OrbitBackground, onode: OrbitNodeView };

/**
 * How far a focused node is zoomed: 1.4 times the zoom that fits the loop in this box (1.35 on
 * the old fixed 600 px stage), so focusing always zooms in, however big the stage has grown.
 */
export function focusZoom(width: number, height: number): number {
	const fitted = Math.min(width / FRAME.w, height / FRAME.h) / 1.08;
	return Math.max(1.35, Math.min(3, fitted * 1.4));
}

function OrbitInner({ compact = false }: { compact?: boolean }) {
	const { record, estimates, state, actions } = useWorkspace();
	const overlays = useMapOverlays(record.job.lens);
	const flow = useReactFlow();
	const layout = useMemo(() => layoutOrbit(record, state.knobs, estimates), [record, state.knobs, estimates]);
	const fit = fitOf(record, estimates, state.knobs);
	const failing = likelyFails(record, estimates);

	const nodes = useMemo<Node[]>(() => {
		const memT = targetFor(record, "memory");
		const peak = memT ? estimates[memT.id] : undefined;
		const cap = layout.memory.capacity;
		const bg: Node<BgData> = {
			id: "__orbit",
			type: "bg",
			position: { x: 0, y: 0 },
			data: {
				layout,
				highlightPhases: state.highlight.flatMap((h) => {
					const n = layout.nodes.find((m) => m.id === h || m.graphId === h);
					return n?.phase ? [n.phase] : [];
				}),
				fit,
				evalStalled: failing,
				coreNumber: trusted(peak) ? fmt(peak.point, memT?.unit ?? "bytes") : t("readout.abstain_short"),
				coreValue: trusted(peak) ? peak.point : null,
				coreUnit: memT?.unit ?? "bytes",
				coreCaption:
					memT && memT.unit !== "bytes"
						? t(memT.label_key)
						: cap
							? t("map.core.of_capacity", { cap: fmt(cap, "bytes") })
							: t("map.core.peak"),
				coreStatus:
					fit === "fails" && trusted(peak) && cap && peak.point > cap && memT?.unit === "bytes"
						? t("map.core.over_by", { over: fmt(peak.point - cap, "bytes") })
						: t(`fit.${fit}`),
			},
			draggable: false,
			selectable: false,
			focusable: false,
			zIndex: 0,
			// A fixed size, so React Flow never hides it waiting for a measurement that may not
			// come when its size hasn't changed (seen after quick guide moves, Oct 27).
			width: FRAME.w,
			height: FRAME.h,
			// Known size and no handles: React Flow keeps its measurement instead of re-measuring
			// every node after each knob change (a forced layout per node, measured Oct 27).
			measured: { width: FRAME.w, height: FRAME.h },
			handles: [],
			style: { pointerEvents: "none" },
		};
		const findingsByNode = new Map<string, { title: string; severity: string }>();
		for (const f of record.findings) {
			if (!f.node_id || f.severity === "info") continue;
			const prev = findingsByNode.get(f.node_id);
			const rank = { critical: 3, error: 2, warn: 1, info: 0 } as const;
			if (!prev || rank[f.severity] > rank[prev.severity as keyof typeof rank]) findingsByNode.set(f.node_id, f);
		}
		const drawn = new Set(layout.nodes.map((n) => n.graphId).filter((x): x is string => x !== null));
		const hiddenParents = new Set(
			state.highlight
				.filter((h) => !drawn.has(h))
				.map((h) => record.graph.nodes.find((g) => g.id === h)?.parent)
				.filter((x): x is string => !!x),
		);
		const pfNodes: Node<NodeData>[] = layout.nodes.map((n) => {
			const share = n.share !== null ? t("map.node.share", { share: fmt(n.share, "fraction") }) : "";
			const evalSpike = n.phase === "eval" ? memoryTermBytes(estimates, "eval_spike") : 0;
			const meta =
				evalSpike > 0
					? t("map.node.adds", { bytes: fmt(evalSpike, "bytes") })
					: n.phase === "checkpoint"
						? t("map.node.saves")
						: share;
			const finding = n.graphId ? findingsByNode.get(n.graphId) : undefined;
			const title = n.graphId === null ? phaseLabel(n.label) : n.label;
			return {
				id: n.id,
				type: "onode",
				position: { x: n.x - NODE_BOX / 2, y: n.y - NODE_BOX / 2 },
				data: {
					node: n,
					title,
					meta,
					accent:
						state.highlight.includes(n.id) ||
						(n.graphId !== null && state.highlight.includes(n.graphId)) ||
						// A highlighted node the map doesn't draw (a group's child) lights its parent.
						(n.graphId !== null && hiddenParents.has(n.graphId)),
					dim: !!state.focus && state.focus !== n.id,
					fails: n.phase === "eval" && failing,
					finding: finding ? finding.title : null,
				},
				draggable: false,
				zIndex: 2,
				// A fixed box (see NODE_BOX); only the button inside it takes clicks.
				width: NODE_BOX,
				height: NODE_BOX,
				measured: { width: NODE_BOX, height: NODE_BOX },
				handles: [],
				style: { pointerEvents: "none" },
			};
		});
		return [bg, ...pfNodes];
	}, [record, estimates, state.highlight, state.focus, layout, fit, failing]);

	const box = useRef<HTMLDivElement>(null);
	const focusRef = useRef(state.focus);
	focusRef.current = state.focus;
	const layoutRef = useRef(layout);
	layoutRef.current = layout;
	// When the focus glide ends: a resize meanwhile re-fits after it, not over it.
	const glideUntil = useRef(0);

	// Focus: glide to the node; back out to the whole loop when focus clears.
	useEffect(() => {
		const reduce = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
		const duration = reduce ? 0 : 800;
		const n = state.focus ? layout.nodes.find((m) => m.id === state.focus) : null;
		const el = box.current;
		glideUntil.current = performance.now() + duration;
		if (n) void flow.setCenter(n.x, n.y, { zoom: el ? focusZoom(el.clientWidth, el.clientHeight) : 1.35, duration });
		else void flow.fitView({ padding: 0.04, duration });
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [state.focus]);

	// The stage grows and shrinks with the window (and swaps between the stage and the side
	// column), so the loop is re-fitted to its box whenever the box changes size; React Flow only
	// fits once, on first paint. No glide: it follows the resize. A focused node keeps its centre.
	useEffect(() => {
		const el = box.current;
		if (!el || typeof ResizeObserver === "undefined") return;
		let frame = 0;
		let later = 0;
		let last = "";
		const ro = new ResizeObserver(([entry]) => {
			const { width, height } = entry!.contentRect;
			const size = `${Math.round(width)}x${Math.round(height)}`;
			if (size === last || width === 0) return;
			last = size;
			cancelAnimationFrame(frame);
			clearTimeout(later);
			const refit = () => {
				const wait = glideUntil.current - performance.now();
				if (wait > 0) {
					later = window.setTimeout(refit, wait + 16);
					return;
				}
				const f = focusRef.current;
				const n = f ? layoutRef.current.nodes.find((m) => m.id === f) : null;
				if (n) void flow.setCenter(n.x, n.y, { zoom: focusZoom(width, height), duration: 0 });
				else void flow.fitView({ padding: 0.04, duration: 0 });
			};
			frame = requestAnimationFrame(refit);
		});
		ro.observe(el);
		return () => {
			cancelAnimationFrame(frame);
			clearTimeout(later);
			ro.disconnect();
		};
		// eslint-disable-next-line react-hooks/exhaustive-deps
	}, [flow]);

	return (
		<div ref={box} className={`pf-orbit${compact ? " pf-orbit--compact" : ""}${state.focus && !compact ? " is-zoomed" : ""}`} role="region" aria-label={t("map.label")}>
			<ReactFlow
				nodes={nodes}
				edges={[]}
				nodeTypes={NODE_TYPES}
				fitView
				fitViewOptions={{ padding: 0.04 }}
				minZoom={0.3}
				maxZoom={3}
				nodesDraggable={false}
				// The node's own button takes focus (with its ring); React Flow's wrapper would take a
				// second, ringless stop (audit finding 10).
				nodesFocusable={false}
				edgesFocusable={false}
				nodesConnectable={false}
				panOnDrag={false}
				zoomOnScroll={false}
				zoomOnDoubleClick={false}
				zoomOnPinch={false}
				preventScrolling={false}
				onNodeClick={(_, n) => {
					if (n.id === "__orbit") return;
					actions.focus(state.focus === n.id ? null : n.id);
				}}
				onPaneClick={() => actions.focus(null)}
				proOptions={{ hideAttribution: true }}
			/>
			{compact ? null : (
			<span className="pf-orbit__tps">
				{layout.tokensPerSecond ? t("map.tps", { tps: fmt(layout.tokensPerSecond, "tokens/s") }) : null}
			</span>
			)}
			{compact ? null : overlays.map((o) => (
				<AddonBoundary key={o.id} id={o.id}>
					<o.component record={record} knobs={state.knobs} estimates={estimates} />
				</AddonBoundary>
			))}
		</div>
	);
}

/** `compact`: the small orbit in the side column while the fit map has the stage. */
export function Orbit({ compact = false }: { compact?: boolean } = {}) {
	return (
		<ReactFlowProvider>
			<OrbitInner compact={compact} />
		</ReactFlowProvider>
	);
}

export function useOrbitLayout(): OrbitLayout {
	const { record, estimates, state } = useWorkspace();
	return useMemo(() => layoutOrbit(record, state.knobs, estimates), [record, state.knobs, estimates]);
}
