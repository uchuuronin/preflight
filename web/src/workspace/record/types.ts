/**
 * The flight record's types: generated from the Pydantic schema (schema/ts/contract.ts, owned
 * by the PM chat). Re-exported here so web/ imports one place; nothing in this file redefines
 * a contract shape.
 */
export type {
	Estimate,
	Finding,
	FlightRecord,
	Grid,
	GridAxis,
	GraphEdge,
	GraphNode,
	Job,
	JobEvent,
	JobRequest,
	Knob,
	KnobImpact,
	ModelCard,
	Objective,
	Prediction,
	Probe,
	Target,
} from "../../../../schema/ts/contract";

import type {
	Estimate,
	FlightRecord,
	Grid,
	GraphEdge,
	GraphNode,
	Job,
	Objective,
	Target,
} from "../../../../schema/ts/contract";

export type Lens = Job["lens"];
export type KnobValue = boolean | number | string;
export type EstimateStatus = "measured" | "predicted" | "extrapolated" | "abstain";
export type Readout = NonNullable<Target["readout"]>;
export type ObjectiveKind = NonNullable<Objective["kind"]>;

/** The record as the UI holds it once loaded: required arrays filled in. */
export type LoadedRecord = Omit<FlightRecord, "graph" | "knobs" | "probes" | "predictions" | "models" | "findings"> & {
	graph: { nodes: GraphNode[]; edges: GraphEdge[]; skeleton: boolean };
	knobs: NonNullable<FlightRecord["knobs"]>;
	probes: NonNullable<FlightRecord["probes"]>;
	predictions: { targets: Target[]; estimates: Record<string, Estimate | undefined>; grids: Grid[] };
	models: NonNullable<FlightRecord["models"]>;
	findings: NonNullable<FlightRecord["findings"]>;
};

/** Fill optional arrays once at load, so components never write `?? []`. */
export function normaliseRecord(r: FlightRecord): LoadedRecord {
	return {
		...r,
		graph: { nodes: r.graph?.nodes ?? [], edges: r.graph?.edges ?? [], skeleton: r.graph?.skeleton ?? false },
		knobs: r.knobs ?? [],
		probes: r.probes ?? [],
		predictions: {
			targets: r.predictions?.targets ?? [],
			estimates: r.predictions?.estimates ?? {},
			grids: r.predictions?.grids ?? [],
		},
		models: r.models ?? [],
		findings: r.findings ?? [],
	};
}
