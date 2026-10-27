import {
	Background,
	BaseEdge,
	Controls,
	getSmoothStepPath,
	Handle,
	Position,
	ReactFlow,
	type Edge,
	type EdgeProps,
	type Node,
	type NodeProps,
} from "@xyflow/react";
import "@xyflow/react/dist/base.css";
import { memo, useMemo } from "react";
import { fmt } from "../../lib/fmt";
import { hasKey, t } from "../../lib/i18n";
import { useMapOverlays } from "../../registry";
import type { Estimate, Finding, GraphNode, Prediction } from "../record/types";
import { useWorkspace } from "../state";
import { capacityBytes, trusted } from "../status";
import { MemoryStackBar } from "../views/MemoryStack";
import { activeNodesAt } from "../timeline";

const COL = 184;
const ROW = 116;
const NODE_W = 150;

/** Nodes that live on the GPU; the rest (data loading, checkpoint storage) sit outside. */
const ON_GPU = new Set(["embed", "blocks", "head", "loss", "backward", "optimizer", "eval", "attention", "mlp", "generate", "reference", "reward", "trainer"]);

interface PfNodeData extends Record<string, unknown> {
	node: GraphNode;
	label: string;
	timeShare: number | null;
	timeEst: Estimate | null;
	memShare: number | null;
	memEst: Estimate | null;
	accent: boolean;
	active: boolean;
	finding: Finding | null;
	status: Estimate["status"] | null;
}

interface GpuData extends Record<string, unknown> {
	title: string;
	width: number;
	height: number;
}

interface PfEdgeData extends Record<string, unknown> {
	speed: number | null;
	kind: "flow" | "branch" | "loop";
	label?: string;
	accent: boolean;
}

function nodeLabel(n: GraphNode): string {
	return n.label_key && hasKey(n.label_key) ? t(n.label_key) : n.label;
}

function stepTime(est: Prediction): number | null {
	let total = 0;
	let any = false;
	for (const [id, e] of Object.entries(est)) {
		if (id.startsWith("t_") && trusted(e)) {
			total += e.point;
			any = true;
		}
	}
	return any ? total : null;
}

type Side = "left" | "right" | "top" | "bottom";

/** Pick the sides that face each other, so edges never detour around a node. */
function handlesFor(
	a: { col: number; row: number } | undefined,
	b: { col: number; row: number } | undefined,
): { sourceHandle: Side; targetHandle: Side } {
	if (!a || !b) return { sourceHandle: "right", targetHandle: "left" };
	const dc = b.col - a.col;
	const dr = b.row - a.row;
	if (dr === 0) return dc >= 0 ? { sourceHandle: "right", targetHandle: "left" } : { sourceHandle: "left", targetHandle: "right" };
	if (dr > 0) return dc === 0 ? { sourceHandle: "bottom", targetHandle: "top" } : { sourceHandle: "bottom", targetHandle: dc > 0 ? "left" : "top" };
	return dc < 0 ? { sourceHandle: "left", targetHandle: "bottom" } : { sourceHandle: "top", targetHandle: "bottom" };
}

export function JobMap({ onSelect }: { onSelect: (nodeId: string | null) => void }) {
	const { record, estimates, state } = useWorkspace();
	const overlays = useMapOverlays(record.job.lens);
	const capacity = capacityBytes(record, estimates, String(state.knobs.gpu ?? ""));
	const activeAtPlayhead = activeNodesAt(record, state.playhead);

	const { nodes, edges } = useMemo(() => {
		const step = stepTime(estimates);
		const findings = new Map<string, Finding>();
		for (const f of record.findings) if (f.node && !findings.has(f.node)) findings.set(f.node, f);

		const graphNodes = record.graph.nodes.filter((n) => !n.parent);
		const flowNodes: Node[] = graphNodes.map((n) => {
			const timeEst = n.metrics?.time ? (estimates[n.metrics.time] ?? null) : null;
			const memEst = n.metrics?.memory ? (estimates[n.metrics.memory] ?? null) : null;
			const statuses = [timeEst?.status, memEst?.status].filter(Boolean) as Estimate["status"][];
			const status = statuses.includes("abstain")
				? "abstain"
				: statuses.includes("extrapolated")
					? "extrapolated"
					: statuses.includes("predicted")
						? "predicted"
						: statuses.length
							? "measured"
							: null;
			const data: PfNodeData = {
				node: n,
				label: nodeLabel(n),
				timeShare: trusted(timeEst ?? undefined) && step ? (timeEst as Estimate).point / step : null,
				timeEst,
				memShare: trusted(memEst ?? undefined) && capacity ? (memEst as Estimate).point / capacity : null,
				memEst,
				accent: state.highlight.includes(n.id),
				active: activeAtPlayhead.includes(n.id),
				finding: findings.get(n.id) ?? null,
				status,
			};
			return {
				id: n.id,
				type: "pf",
				position: { x: (n.pos?.col ?? 0) * COL + 24, y: (n.pos?.row ?? 0) * ROW + 48 },
				data,
				draggable: false,
				zIndex: 2,
			};
		});

		// The GPU container: drawn behind the nodes that run on the GPU.
		const onGpu = graphNodes.filter((n) => ON_GPU.has(n.kind));
		if (onGpu.length && record.env.gpu) {
			const cols = onGpu.map((n) => n.pos?.col ?? 0);
			const rows = onGpu.map((n) => n.pos?.row ?? 0);
			const minC = Math.min(...cols);
			const maxC = Math.max(...cols);
			const minR = Math.min(...rows);
			const maxR = Math.max(...rows);
			const gpuName = String(state.knobs.gpu ?? record.env.gpu.name);
			flowNodes.unshift({
				id: "__gpu",
				type: "gpu",
				position: { x: minC * COL + 24 - 20, y: minR * ROW + 48 - 36 },
				data: {
					title: t("map.gpu_container", { gpu: gpuName, count: record.env.gpu.count }),
					width: (maxC - minC) * COL + NODE_W + 40,
					height: (maxR - minR) * ROW + 72 + 36 + 46,
				} satisfies GpuData,
				draggable: false,
				selectable: false,
				zIndex: 0,
			});
		}

		const pos = new Map(graphNodes.map((n) => [n.id, n.pos ?? { col: 0, row: 0 }]));
		const flowEdges: Edge[] = record.graph.edges.map((e) => {
			const thr = e.throughput ? estimates[e.throughput] : undefined;
			const speed = trusted(thr) ? thr.point : null;
			const kind = e.kind ?? "flow";
			const every =
				e.target === "eval" ? record.job.eval_every : e.target === "checkpoint" ? record.job.checkpoint_every : undefined;
			const label = every ? t("map.edge.every_n_steps", { n: fmt(every, "count") }) : e.label;
			return {
				id: e.id,
				source: e.source,
				target: e.target,
				type: "pf",
				zIndex: 1,
				...handlesFor(pos.get(e.source), pos.get(e.target)),
				data: {
					speed,
					kind,
					accent: state.highlight.includes(e.source) && state.highlight.includes(e.target),
					...(label ? { label } : {}),
				} satisfies PfEdgeData,
			};
		});
		return { nodes: flowNodes, edges: flowEdges };
	}, [record, estimates, state.highlight, state.knobs.gpu, capacity, activeAtPlayhead]);

	return (
		<div className="pf-map" aria-label={t("map.label")} role="region">
			<div className="pf-map__flow">
			<ReactFlow
				nodes={nodes}
				edges={edges}
				nodeTypes={NODE_TYPES}
				edgeTypes={EDGE_TYPES}
				fitView
				fitViewOptions={{ padding: 0.12 }}
				minZoom={0.4}
				maxZoom={1.6}
				nodesDraggable={false}
				nodesConnectable={false}
				elementsSelectable
				onNodeClick={(_, n) => onSelect(n.id === "__gpu" ? null : n.id)}
				onPaneClick={() => onSelect(null)}
				proOptions={{ hideAttribution: true }}
			>
				<Background gap={24} size={1} color="var(--pf-line-hairline)" />
				<Controls showInteractive={false} position="bottom-right" />
			</ReactFlow>
			</div>
			<div className="pf-map__legend">
				<span><i className="pf-legend-edge" /> {t("map.legend.edge")}</span>
				<span><i className="pf-legend-outline" /> {t("map.legend.outline")}</span>
				<span><i className="pf-legend-dash" /> {t("map.legend.dashed")}</span>
				<span><i className="pf-legend-accent" /> {t("map.legend.accent")}</span>
			</div>
			<div className="pf-map__memory">
				<MemoryStackBar compact />
			</div>
			{overlays.map((o) => (
				<o.component key={o.id} record={record} knobs={state.knobs} estimates={estimates} />
			))}
		</div>
	);
}

const PfNode = memo(function PfNode({ data }: NodeProps<Node<PfNodeData>>) {
	const { node, label, timeShare, memShare, memEst, timeEst, accent, active, finding, status } = data;
	// Outline weight carries share of step time: 1 px at nothing, 4 px at half the step.
	const outline = timeShare === null ? 1 : 1 + Math.min(3, timeShare * 6);
	const classes = [
		"pf-node",
		accent ? "is-accent" : "",
		active ? "is-active" : "",
		status ? `is-${status}` : "",
		finding ? `has-${finding.severity}` : "",
	].join(" ");
	return (
		<div className={classes} style={{ width: NODE_W, ["--pf-outline" as string]: `${outline}px` }}>
			{SIDES.map(([id, p]) => (
				<Handle key={`t-${id}`} type="target" position={p} id={id} />
			))}
			<div className="pf-node__title">
				<span>{label}</span>
				{node.repeat ? <span className="pf-node__repeat">{t("map.repeat", { n: node.repeat })}</span> : null}
			</div>
			<div className="pf-node__metrics">
				{timeEst && trusted(timeEst) && timeShare !== null ? (
					<span>{t("map.node.time_share", { share: fmt(timeShare, "ratio") })}</span>
				) : null}
				{memEst && trusted(memEst) && memEst.point > 0 ? <span>{fmt(memEst.point, "bytes")}</span> : null}
				{status === "abstain" ? <span className="pf-muted">{t("map.node.verify")}</span> : null}
			</div>
			{memShare !== null ? (
				<div className="pf-gauge" role="meter" aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(memShare * 100)} aria-label={t("map.node.memory_gauge")}>
					<span style={{ width: `${Math.min(100, memShare * 100)}%` }} />
				</div>
			) : null}
			{finding ? (
				<div className={`pf-node__finding is-${finding.severity}`}>{t(finding.title)}</div>
			) : null}
			{SIDES.map(([id, p]) => (
				<Handle key={`s-${id}`} type="source" position={p} id={id} />
			))}
		</div>
	);
});

function GpuNode({ data }: NodeProps<Node<GpuData>>) {
	return (
		<div className="pf-gpu" style={{ width: data.width, height: data.height }}>
			<span className="pf-gpu__title">{data.title}</span>
		</div>
	);
}

function PfEdge(props: EdgeProps<Edge<PfEdgeData>>) {
	const { sourceX, sourceY, targetX, targetY, sourcePosition, targetPosition, data, id } = props;
	const loop = data?.kind === "loop";
	const [path, labelX, labelY] = getSmoothStepPath({
		sourceX,
		sourceY,
		targetX,
		targetY,
		sourcePosition,
		targetPosition,
		borderRadius: 14,
		offset: loop ? 40 : 20,
	});
	// Edge motion speed is throughput: one dash cycle takes less time when more flows.
	const speed = data?.speed ?? null;
	const duration = speed && speed > 0 ? Math.max(0.35, Math.min(6, 4000 / speed)) : null;
	const kind = data?.kind ?? "flow";
	return (
		<>
			<BaseEdge id={id} path={path} className={`pf-edge pf-edge--${kind}${data?.accent ? " is-accent" : ""}`} />
			{kind === "flow" && duration ? (
				<path
					d={path}
					className="pf-edge__flow"
					style={{ animationDuration: `${duration}s` }}
					fill="none"
				/>
			) : null}
			{data?.label ? (
				<foreignObject x={labelX - 60} y={labelY - 10} width={120} height={20} className="pf-edge__label-wrap">
					<div className="pf-edge__label">{data.label}</div>
				</foreignObject>
			) : null}
		</>
	);
}

const SIDES: Array<[Side, Position]> = [
	["left", Position.Left],
	["right", Position.Right],
	["top", Position.Top],
	["bottom", Position.Bottom],
];

const NODE_TYPES = { pf: PfNode, gpu: GpuNode };
const EDGE_TYPES = { pf: PfEdge };
