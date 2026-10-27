/**
 * PROVISIONAL flight-record types, hand-written from contract v0.1 because the repo had no
 * schema yet. The source of truth is python/preflight/schema (Pydantic); once its generated
 * TS types land, this file becomes a re-export of them and any field differences are fixed
 * here and in record/adapt.ts only. Nothing else in web/ should depend on raw field names.
 */

export type Lens =
	| "finetune"
	| "pretrain"
	| "rl"
	| "agent"
	| "eval"
	| "serving"
	| "retrieval"
	| "gnn"
	| "vision"
	| "campaign";

export type EstimateStatus = "measured" | "predicted" | "extrapolated" | "abstain";

export interface Estimate {
	point: number;
	lo: number;
	hi: number;
	unit: string;
	status: EstimateStatus;
	model_id: string;
	reason?: string;
	source: "grid" | "exact";
}

export type Prediction = Record<string, Estimate>;

export type KnobValue = number | string | boolean;

export interface KnobDomainNumeric {
	min: number;
	max: number;
	step?: number;
	log?: boolean;
	/** Discrete allowed values (batch sizes, sequence lengths). */
	values?: number[];
}

export interface KnobDomainEnum {
	values: Array<string | boolean>;
}

export interface KnobSpec {
	id: string;
	label_key: string;
	unit: string;
	kind: "int" | "float" | "enum" | "bool";
	domain: KnobDomainNumeric | KnobDomainEnum;
	value: KnobValue;
	probed_range?: { min: number; max: number } | { values: Array<string | boolean> };
	gridded: boolean;
	needs_code: boolean;
	patch_template?: string;
}

export interface GraphNode {
	id: string;
	kind: string;
	label: string;
	label_key?: string;
	parent?: string;
	repeat?: number;
	/** Metric references point at prediction target ids. */
	metrics?: {
		time?: string;
		memory?: string;
		throughput?: string;
	};
	/** Optional layout hint, column and row on a coarse grid. */
	pos?: { col: number; row: number };
}

export interface GraphEdge {
	id: string;
	source: string;
	target: string;
	kind?: "flow" | "branch" | "loop";
	/** Throughput target that sets edge speed. */
	throughput?: string;
	/** Volume target that sets edge thickness. */
	volume?: string;
	label?: string;
}

export interface Grid {
	id: string;
	target_id: string;
	axes: Array<{ knob_id: string; values: Array<number | string | boolean> }>;
	/** base64 float16, row-major, last axis fastest. */
	point: string;
	lo: string;
	hi: string;
	/** base64 uint8, 1 = abstain. */
	abstain: string;
	model_id: string;
	/** Knob values this grid assumes for knobs not on its axes. */
	fixed: Record<string, KnobValue>;
	/**
	 * Not in v0.1 (requested): stored values times scale = values in the target's unit.
	 * float16 tops out at 65,504, so bytes travel as GiB and long durations as hours.
	 */
	scale?: number;
}

export interface TargetSpec {
	id: string;
	unit: string;
	label_key: string;
	/** "higher" if a bigger number is better for this target. */
	better?: "higher" | "lower";
}

export interface Finding {
	id: string;
	severity: "info" | "warn" | "critical";
	node?: string;
	title: string;
	evidence: string;
	source: "rule" | "spark_open_triage" | "spark_diagnosis" | "model";
	fix_template?: string;
	knob?: string;
	citations?: Array<{ title: string; url: string }>;
}

export interface ModelCard {
	id: string;
	kind: string;
	version: string;
	trained_on: string;
	heldout_error?: number;
	coverage?: number;
}

export interface Probe {
	id: string;
	knobs: Record<string, KnobValue>;
	steps: number;
	status: "ok" | "oom" | "error" | "timeout";
	memory?: Record<string, number>;
	step_time?: Record<string, number>;
	throughput?: number;
	mfu?: number;
	cost_usd?: number;
}

export interface FlightRecord {
	schema_version: string;
	record_id: string;
	parent_id?: string | null;
	created_at: string;
	job: {
		lens: Lens;
		title?: string;
		source: { kind: "sample" | "repo" | "upload" | "notebook" | "template"; ref?: string; commit?: string; hash?: string };
		framework?: string;
		objective: "cost" | "time" | "quality" | "headroom";
		constraints: {
			budget_usd?: number;
			deadline?: string;
			gpus?: string[];
			quota_bytes?: number;
			wall_time_cap_s?: number;
			preemption_rule?: string;
		};
		total_steps?: number;
		eval_every?: number;
		checkpoint_every?: number;
	};
	env: {
		backend: "cuda" | "rocm" | "api";
		provider?: string;
		gpu?: { name: string; count: number; memory_bytes: number };
		libs?: Record<string, string>;
		seed?: number;
	};
	graph: { nodes: GraphNode[]; edges: GraphEdge[] };
	knobs: KnobSpec[];
	probes: Probe[];
	predictions: { targets: TargetSpec[]; grids: Grid[] };
	models: ModelCard[];
	findings: Finding[];
	patches?: unknown[];
	campaign?: unknown;
	provenance?: Array<Record<string, unknown>>;
	costs?: { gpu_usd?: number; llm_usd?: number };
	/** Not in v0.1: set on development fixtures so the UI can label them. */
	fixture?: boolean;
}

export interface KnobImpact {
	knob_id: string;
	/** Change in the objective target for the best single move of this knob. */
	delta: number;
	best_value: KnobValue;
	target_id: string;
}

export type Objective = "cost" | "time" | "quality" | "headroom";
