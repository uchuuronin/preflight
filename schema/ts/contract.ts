/* Generated from python/preflight/schema by `uv run python -m preflight.schema.generate`. Do not edit. */

/**
 * Preflight contract v0.1.0. Generated; do not edit.
 */
export interface PreflightContract {
	FlightRecord?: FlightRecord;
	JobEvent?: JobEvent;
	JobRequest?: JobRequest;
	ScriptMap?: ScriptMap;
	Prediction?: Prediction;
	Estimate?: Estimate | undefined;
	Grid?: Grid;
	Target?: Target;
	Objective?: Objective;
	KnobImpact?: KnobImpact;
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "FlightRecord".
 */
export interface FlightRecord {
	schema_version?: string;
	/**
	 * ULID, internal only
	 */
	record_id?: string;
	/**
	 * Random token for read-only links
	 */
	share_token?: string;
	/**
	 * Record this one derives from (verify, what-if)
	 */
	parent_id?: string | null;
	created_at?: string;
	job: Job;
	env?: Env | null;
	graph?: Graph;
	knobs?: Knob[];
	probes?: Probe[];
	agent_sample?: AgentSample | null;
	predictions?: Predictions;
	models?: ModelCard[];
	findings?: Finding[];
	patches?: Patch[];
	campaign?: Campaign | null;
	provenance?: ModelCall[];
	costs?: Costs;
}
/**
 * Written by A.
 *
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "Job".
 */
export interface Job {
	lens: "finetune" | "pretrain" | "rl" | "agent" | "eval" | "serving" | "retrieval" | "gnn" | "vision";
	source: Source;
	framework?:
		| (
				| "hf_trainer"
				| "trl"
				| "lightning"
				| "torchtitan"
				| "plain_pytorch"
				| "vllm"
				| "hf_generate"
				| "sentence_transformers"
				| "pyg"
				| "dgl"
				| "timm"
				| "openai_client"
				| "other"
		  )
		| null;
	title?: string | null;
	script_map?: ScriptMap | null;
	objective?: Objective;
	constraints?: Constraints;
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "Source".
 */
export interface Source {
	kind: "sample" | "repo" | "upload" | "notebook" | "template" | "cli" | "inflight";
	/**
	 * sample://name, https://github.com/..., upload://<hash>
	 */
	uri?: string | null;
	commit?: string | null;
	/**
	 * sha256 of the submitted code
	 */
	content_hash?: string | null;
	entrypoint?: string | null;
	args?: string[];
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "ScriptMap".
 */
export interface ScriptMap {
	lens: "finetune" | "pretrain" | "rl" | "agent" | "eval" | "serving" | "retrieval" | "gnn" | "vision";
	framework:
		| "hf_trainer"
		| "trl"
		| "lightning"
		| "torchtitan"
		| "plain_pytorch"
		| "vllm"
		| "hf_generate"
		| "sentence_transformers"
		| "pyg"
		| "dgl"
		| "timm"
		| "openai_client"
		| "other";
	entrypoint: string;
	/**
	 * Command line that runs the script
	 */
	launch?: string | null;
	model?: ModelInfo | null;
	data?: DataInfo | null;
	training?: TrainInfo | null;
	loop?: LoopInfo | null;
	knobs?: ScriptKnob[];
	blocks?: Block[];
	agent?: AgentInfo | null;
	rl?: RLInfo | null;
	/**
	 * Problems noticed while reading, in plain words
	 */
	risks?: string[];
	produced_by?: "muse_spark" | "adapter" | "manual";
	confidence?: number;
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "ModelInfo".
 */
export interface ModelInfo {
	name: string;
	hf_id?: string | null;
	family?: "decoder" | "encoder" | "encoder_decoder" | "cnn" | "vit" | "gnn" | "mlp" | "api" | "other";
	param_count?: number | null;
	n_layers?: number | null;
	d_model?: number | null;
	n_heads?: number | null;
	n_kv_heads?: number | null;
	d_ff?: number | null;
	vocab_size?: number | null;
	dtype?: string | null;
	/**
	 * e.g. 4bit-nf4, 8bit, none
	 */
	quantization?: string | null;
	lora?: LoraInfo | null;
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "LoraInfo".
 */
export interface LoraInfo {
	r: number;
	alpha?: number | null;
	target_modules?: string[];
	dropout?: number | null;
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "DataInfo".
 */
export interface DataInfo {
	dataset?: string | null;
	batch_size?: number | null;
	eval_batch_size?: number | null;
	seq_len?: number | null;
	num_workers?: number | null;
	packing?: boolean | null;
	/**
	 * longest, max_length, none
	 */
	padding?: string | null;
	n_examples?: number | null;
	/**
	 * Column or file holding per-document quality
	 */
	quality_scores?: string | null;
	languages?: string[];
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "TrainInfo".
 */
export interface TrainInfo {
	optimizer?: string | null;
	lr?: number | null;
	warmup_steps?: number | null;
	grad_accum?: number | null;
	precision?: "fp32" | "fp16" | "bf16" | "fp8" | "mixed" | "unknown";
	max_steps?: number | null;
	epochs?: number | null;
	checkpoint_every?: number | null;
	eval_every?: number | null;
	grad_checkpointing?: boolean | null;
	eval_under_no_grad?: boolean | null;
	distributed?: "none" | "ddp" | "fsdp" | "deepspeed" | "tp" | "other";
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "LoopInfo".
 */
export interface LoopInfo {
	kind:
		| "hf_trainer"
		| "trl_trainer"
		| "lightning"
		| "torchtitan"
		| "plain"
		| "vllm"
		| "agent_loop"
		| "eval_loop"
		| "other";
	location?: CodeLocation | null;
	/**
	 * The call that advances one step, e.g. optimizer.step()
	 */
	step_call?: string | null;
	model_var?: string | null;
	optimizer_var?: string | null;
	dataloader_var?: string | null;
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "CodeLocation".
 */
export interface CodeLocation {
	file: string;
	line?: number | null;
	/**
	 * The literal expression that sets the value, if any
	 */
	expr?: string | null;
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "ScriptKnob".
 */
export interface ScriptKnob {
	id: string;
	label: string;
	kind: "int" | "float" | "categorical" | "bool";
	current?: boolean | number | string | null;
	unit?: string | null;
	choices?: (boolean | number | string)[];
	location?: CodeLocation | null;
	/**
	 * True when changing it means editing code, not a flag
	 */
	needs_code?: boolean;
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "Block".
 */
export interface Block {
	id: string;
	label: string;
	kind?: "module" | "data" | "loss" | "optimizer" | "io" | "phase" | "model_call" | "tool" | "other";
	parent?: string | null;
	repeat?: number;
	params?: number | null;
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "AgentInfo".
 */
export interface AgentInfo {
	/**
	 * openai, anthropic, meta, litellm, ...
	 */
	client: string;
	base_url_var?: string | null;
	model?: string | null;
	tools?: string[];
	max_turns?: number | null;
	task_source?: string | null;
	n_tasks?: number | null;
	prompt_caching?: boolean | null;
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "RLInfo".
 */
export interface RLInfo {
	/**
	 * grpo, ppo, dpo, rloo, ...
	 */
	algorithm: string;
	group_size?: number | null;
	max_completion_len?: number | null;
	reward_functions?: string[];
	reference_model?: boolean | null;
	generation_backend?: ("hf" | "vllm" | "sglang" | "other") | null;
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "Objective".
 */
export interface Objective {
	kind?:
		"min_time" | "min_cost" | "max_throughput" | "fit_memory" | "min_risk" | "max_quality_per_usd" | "custom";
	/**
	 * For custom objectives: the target to optimise
	 */
	target_id?: string | null;
	/**
	 * target_id -> weight, for blends
	 */
	weights?: {
		[k: string]: number | undefined;
	};
	/**
	 * The user's use case in their own words
	 */
	description?: string | null;
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "Constraints".
 */
export interface Constraints {
	budget_usd?: number | null;
	deadline?: string | null;
	/**
	 * Allowed GPU names; empty means any
	 */
	gpu_types?: string[];
	max_gpus?: number | null;
	quota?: Quota | null;
	/**
	 * Scheduler limit per job, e.g. Slurm
	 */
	wall_time_cap_s?: number | null;
	preemption?: PreemptionRule;
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "Quota".
 */
export interface Quota {
	disk_bytes?: number | null;
	gpu_hours?: number | null;
	api_usd?: number | null;
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "PreemptionRule".
 */
export interface PreemptionRule {
	kind?: "none" | "poisson" | "policy";
	/**
	 * Mean time to interrupt, for kind=poisson
	 */
	mtti_s?: number | null;
	/**
	 * The cluster's rule in words, for kind=policy
	 */
	policy?: string | null;
	grace_s?: number | null;
}
/**
 * Written by A.
 *
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "Env".
 */
export interface Env {
	backend: "cuda" | "rocm" | "api" | "cpu" | "mps";
	/**
	 * modal, amd_devcloud, local, colab, kaggle, api
	 */
	provider: string;
	gpu?: GPU | null;
	nodes?: number;
	/**
	 * package -> version
	 */
	libraries?: {
		[k: string]: string | undefined;
	};
	python?: string | null;
	seed?: number | null;
	region?: string | null;
	/**
	 * Container image digest the probe ran in
	 */
	image?: string | null;
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "GPU".
 */
export interface GPU {
	/**
	 * Canonical name from the spec table, e.g. H100-SXM-80GB, MI300X, L4
	 */
	name: string;
	count?: number;
	memory_bytes?: number | null;
	vendor?: "nvidia" | "amd" | "apple" | "none";
	compute_capability?: string | null;
	driver?: string | null;
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "Graph".
 */
export interface Graph {
	nodes?: GraphNode[];
	edges?: GraphEdge[];
	/**
	 * True until probes have measured it
	 */
	skeleton?: boolean;
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "GraphNode".
 */
export interface GraphNode {
	id: string;
	label: string;
	kind?:
		| "module"
		| "data"
		| "loss"
		| "optimizer"
		| "io"
		| "phase"
		| "model_call"
		| "tool"
		| "store"
		| "queue"
		| "group"
		| "other";
	/**
	 * Enclosing group, for collapsed hierarchy
	 */
	parent?: string | null;
	/**
	 * Shown once with ×N when > 1
	 */
	repeat?: number;
	params?: number | null;
	/**
	 * target ids or probe metric keys this node animates
	 */
	metrics?: string[];
	attrs?: {
		[k: string]: (boolean | number | string) | undefined;
	};
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "GraphEdge".
 */
export interface GraphEdge {
	source: string;
	target: string;
	kind?: "data" | "grad" | "control" | "tokens" | "weights";
	/**
	 * target id or metric key whose rate animates the edge
	 */
	metric?: string | null;
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "Knob".
 */
export interface Knob {
	id: string;
	/**
	 * i18n key, e.g. knob.batch_size
	 */
	label_key: string;
	unit?: string | null;
	kind: "int" | "float" | "categorical" | "bool";
	domain?: Domain;
	current: boolean | number | string;
	/**
	 * Filled by B from the probes
	 */
	probed_range?: Range | null;
	gridded?: boolean;
	needs_code?: boolean;
	/**
	 * Patch template id for code knobs
	 */
	patch_template?: string | null;
	/**
	 * memory, speed, data, model, rl, agent, cluster
	 */
	group?: string | null;
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "Domain".
 */
export interface Domain {
	min?: number | null;
	max?: number | null;
	/**
	 * Allowed values for categorical knobs
	 */
	values?: (boolean | number | string)[];
	step?: number | null;
	log?: boolean;
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "Range".
 */
export interface Range {
	min: number;
	max: number;
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "Probe".
 */
export interface Probe {
	id: string;
	/**
	 * Knob values this probe ran at (only those it changed or read)
	 */
	knobs: {
		[k: string]: (boolean | number | string) | undefined;
	};
	status: "queued" | "running" | "ok" | "oom" | "error" | "timeout" | "cancelled" | "cached";
	/**
	 * Optimizer steps completed; at least 2 for a valid memory peak
	 */
	steps: number;
	started_at?: string | null;
	duration_s?: number | null;
	error?: string | null;
	/**
	 * Stable code for the failure; the UI shows t('error.<code>') and its hint
	 */
	error_code?: string | null;
	memory?: Memory | null;
	step_time?: StepTime | null;
	throughput?: Throughput | null;
	stability?: Stability | null;
	data?: DataStats | null;
	rl?: RLSignals | null;
	power?: Power | null;
	eval?: EvalStats | null;
	checkpoint?: CheckpointStats | null;
	generation?: GenerationStats | null;
	/**
	 * Lens-specific measurements
	 */
	extras?: {
		[k: string]: number | undefined;
	};
	cost?: ProbeCost;
	/**
	 * sha256 of script, env and knob values
	 */
	cache_key?: string | null;
	trace?: string | null;
	log?: string | null;
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "Memory".
 */
export interface Memory {
	peak_allocated_bytes?: number | null;
	peak_reserved_bytes?: number | null;
	device_total_bytes?: number | null;
	/**
	 * weights, gradients, optimizer, activations, logits, context, kv_cache, other
	 */
	by_term?: {
		[k: string]: number | undefined;
	};
	/**
	 * Step where the peak occurred
	 */
	peak_step?: number | null;
	timeline?: string | null;
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "StepTime".
 */
export interface StepTime {
	n_steps: number;
	mean_s?: number | null;
	p50_s?: number | null;
	p95_s?: number | null;
	/**
	 * Includes compile and warm-up
	 */
	first_step_s?: number | null;
	/**
	 * Fractions of a step: data, forward, backward, optimizer, eval, rollout, reference, reward, update, weight_sync, idle
	 */
	by_phase?: {
		[k: string]: number | undefined;
	};
	/**
	 * Timings come from steps without the profiler attached
	 */
	unprofiled?: boolean;
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "Throughput".
 */
export interface Throughput {
	tokens_per_s?: number | null;
	samples_per_s?: number | null;
	flops_per_step?: number | null;
	mfu?: number | null;
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "Stability".
 */
export interface Stability {
	/**
	 * Per step: loss, grad_norm, grad_norm_ma, update_to_weight, max_attn_logit, output_logit_norm, loss_scale, lr
	 */
	series?: {
		[k: string]: (number | null)[] | undefined;
	};
	nan_count?: number;
	inf_count?: number;
	loss_scale_events?: number;
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "DataStats".
 */
export interface DataStats {
	padding_fraction?: number | null;
	packing?: boolean | null;
	tokens_per_sample_mean?: number | null;
	tokens_per_sample_p95?: number | null;
	loader_wait_fraction?: number | null;
	num_workers?: number | null;
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "RLSignals".
 */
export interface RLSignals {
	response_length?: LengthStats | null;
	truncation_rate?: number | null;
	zero_spread_fraction?: number | null;
	pass_rate_per_prompt?: number[];
	group_size?: number | null;
	entropy?: number[];
	kl?: number[];
	clip_fraction?: number[];
	reward_mean?: number[];
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "LengthStats".
 */
export interface LengthStats {
	mean: number;
	p95: number;
	max: number;
	n: number;
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "Power".
 */
export interface Power {
	mean_w: number;
	max_w: number;
	samples: number;
}
/**
 * One eval pass, pulled inside the probe window and capped at a few batches.
 *
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "EvalStats".
 */
export interface EvalStats {
	batches: number;
	batch_size?: number | null;
	seconds_per_batch?: number | null;
	peak_allocated_bytes?: number | null;
	/**
	 * True when eval ran with autograd on (no no_grad)
	 */
	grad_enabled?: boolean | null;
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "CheckpointStats".
 */
export interface CheckpointStats {
	saves: number;
	seconds_mean?: number | null;
	bytes_mean?: number | null;
}
/**
 * Generation inside the job: RL rollouts, eval sweeps, serving probes.
 *
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "GenerationStats".
 */
export interface GenerationStats {
	calls: number;
	seconds_total?: number;
	sequences?: number;
	new_tokens_mean?: number | null;
	new_tokens_p95?: number | null;
	new_tokens_max?: number | null;
	/**
	 * The cap the script set, if seen
	 */
	max_new_tokens?: number | null;
	/**
	 * Share of sequences that hit the cap
	 */
	truncation_rate?: number | null;
	backend?: ("hf" | "vllm" | "api" | "other") | null;
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "ProbeCost".
 */
export interface ProbeCost {
	gpu_seconds?: number;
	usd?: number;
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "AgentSample".
 */
export interface AgentSample {
	tasks?: AgentTask[];
	/**
	 * Tasks in the full run
	 */
	population?: number | null;
	/**
	 * How the sample was chosen, e.g. 30–70% pass-rate band
	 */
	selection?: string | null;
	rate_limit_rpm?: number | null;
	rate_limit_tpm?: number | null;
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "AgentTask".
 */
export interface AgentTask {
	task_id: string;
	model?: string | null;
	turns: number;
	input_tokens: number;
	output_tokens: number;
	reasoning_tokens?: number;
	cached_input_tokens?: number;
	cache_creation_tokens?: number;
	tool_calls?: number;
	errors?: number;
	latency_s: number;
	success?: boolean | null;
	context_tokens_by_turn?: number[];
	cost_usd?: number | null;
	stop_reason?: ("done" | "max_turns" | "context_overflow" | "error" | "timeout" | "loop") | null;
	trace?: string | null;
}
/**
 * Written by B.
 *
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "Predictions".
 */
export interface Predictions {
	targets?: Target[];
	/**
	 * Estimates at the record's current knob values
	 */
	estimates?: {
		[k: string]: Estimate | undefined;
	};
	grids?: Grid[];
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "Target".
 */
export interface Target {
	/**
	 * Dotted id, e.g. memory.peak, time.total, cost.total, risk.oom
	 */
	id: string;
	label_key: string;
	unit: string;
	/**
	 * Which of the four readouts shows it, if any
	 */
	readout?: ("memory" | "time" | "cost" | "risk") | null;
}
/**
 * One number on screen: a point with an interval, or an honest abstention.
 *
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "Estimate".
 */
export interface Estimate {
	point: number | null;
	lo: number | null;
	hi: number | null;
	unit: string;
	status: "measured" | "predicted" | "extrapolated" | "abstain";
	model_id: string;
	source: "grid" | "exact";
	/**
	 * Why the engine abstained or extrapolated
	 */
	reason?: string | null;
	/**
	 * Nominal coverage of [lo, hi]
	 */
	level?: number;
}
/**
 * Predictions precomputed over a knob grid, row-major over `axes`.
 *
 * point, lo and hi are base64 little-endian float16 holding real / scale;
 * abstain is a base64 LSB-first bit mask. Grids over MAX_GRID_CELLS cells go
 * to a sidecar blob. Use encoding.choose_scale() when writing one.
 *
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "Grid".
 */
export interface Grid {
	id: string;
	target_id: string;
	unit: string;
	/**
	 * @minItems 1
	 * @maxItems 4
	 */
	axes:
		| [GridAxis]
		| [GridAxis, GridAxis]
		| [GridAxis, GridAxis, GridAxis]
		| [GridAxis, GridAxis, GridAxis, GridAxis];
	point?: string;
	lo?: string;
	hi?: string;
	abstain?: string;
	model_id: string;
	/**
	 * Power of two; real value = decoded float16 x scale
	 */
	scale?: number;
	/**
	 * Values every knob not on an axis is held at
	 */
	fixed?: {
		[k: string]: (boolean | number | string) | undefined;
	};
	/**
	 * Set when the grid is too large to inline
	 */
	sidecar?: string | null;
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "GridAxis".
 */
export interface GridAxis {
	knob_id: string;
	/**
	 * @minItems 1
	 */
	values: [boolean | number | number | string, ...(boolean | number | number | string)[]];
	scale?: "linear" | "log" | "categorical";
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "ModelCard".
 */
export interface ModelCard {
	/**
	 * name@version, pinned by every grid and estimate that used it
	 */
	id: string;
	kind:
		| "analytical"
		| "gbdt"
		| "gnn"
		| "classifier"
		| "conformal"
		| "bootstrap"
		| "simulation"
		| "ensemble"
		| "rule";
	version: string;
	training_data?: string | null;
	/**
	 * metric -> value, e.g. mape, auroc
	 */
	heldout_error?: {
		[k: string]: number | undefined;
	};
	coverage?: number | null;
	per_gpu_coverage?: {
		[k: string]: number | undefined;
	};
	notes?: string | null;
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "Finding".
 */
export interface Finding {
	id: string;
	severity: "info" | "warn" | "error" | "critical";
	/**
	 * Stable rule id, e.g. eval.no_grad, data.num_workers_zero
	 */
	code: string;
	title: string;
	detail?: string | null;
	/**
	 * Job-map node it attaches to
	 */
	node_id?: string | null;
	evidence?: Evidence[];
	source: "rule" | "spark_open_triage" | "spark_diagnosis" | "model";
	fix_template?: string | null;
	knob_id?: string | null;
	citations?: Citation[];
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "Evidence".
 */
export interface Evidence {
	metric: string;
	value?: number | string | null;
	threshold?: number | null;
	probe_id?: string | null;
	step?: number | null;
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "Citation".
 */
export interface Citation {
	title: string;
	url?: string | null;
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "Patch".
 */
export interface Patch {
	id: string;
	kind: "template" | "open_ended";
	template_id?: string | null;
	title: string;
	diff: string;
	verify?: VerifyResult | null;
	session_log?: string | null;
	finding_id?: string | null;
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "VerifyResult".
 */
export interface VerifyResult {
	status: "pending" | "passed" | "failed" | "error";
	metric?: string | null;
	before?: number | null;
	after?: number | null;
	fixed_seed_loss_delta?: number | null;
	effective_batch_unchanged?: boolean | null;
	probe_id?: string | null;
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "Campaign".
 */
export interface Campaign {
	cells?: CampaignCell[];
	lanes?: string[];
	calendar?: string | null;
	cut_list?: string[];
	/**
	 * p10, p50, p90 ...
	 */
	completion_quantiles?: {
		[k: string]: string | undefined;
	};
	checkpoint_interval_s?: number | null;
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "CampaignCell".
 */
export interface CampaignCell {
	id: string;
	knobs: {
		[k: string]: (boolean | number | string) | undefined;
	};
	gpu_hours?: number | null;
	usd?: number | null;
	wall_time_s?: number | null;
	/**
	 * Wall-time-capped segments the run splits into
	 */
	segments?: number | null;
	lane?: string | null;
	cut?: boolean;
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "ModelCall".
 */
export interface ModelCall {
	/**
	 * meta, baseten, fireworks, local
	 */
	provider: string;
	model: string;
	/**
	 * script_map, triage, diagnosis, guide, patch, label, command
	 */
	purpose: string;
	input_tokens?: number;
	output_tokens?: number;
	cached_input_tokens?: number;
	cost_usd?: number;
	request_id?: string | null;
	at: string;
	latency_s?: number | null;
	ok?: boolean;
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "Costs".
 */
export interface Costs {
	gpu_usd?: number;
	llm_usd?: number;
	breakdown?: {
		[k: string]: number | undefined;
	};
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "JobEvent".
 */
export interface JobEvent {
	job_id: string;
	record_id?: string | null;
	seq: number;
	at: string;
	stage:
		| "queued"
		| "admitted"
		| "fetching"
		| "mapping"
		| "skeleton"
		| "harnessing"
		| "probing"
		| "triage"
		| "predicting"
		| "done"
		| "failed"
		| "cancelled";
	/**
	 * Within the current stage
	 */
	progress?: number | null;
	queue_position?: number | null;
	/**
	 * i18n key or plain log line
	 */
	message?: string | null;
	level?: "debug" | "info" | "warn" | "error";
	probe_id?: string | null;
	finding?: Finding | null;
	/**
	 * Bumped when the stored record changed; refetch it
	 */
	record_rev?: number | null;
	/**
	 * On failed: whether resubmitting may succeed
	 */
	retryable?: boolean | null;
}
/**
 * Body of POST /v1/jobs (validated by C, executed by A).
 *
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "JobRequest".
 */
export interface JobRequest {
	/**
	 * A Source: sample, repo, upload, notebook or template
	 */
	source: {
		[k: string]: unknown | undefined;
	};
	/**
	 * Optional hint; the script map decides
	 */
	lens?: string | null;
	objective?: {
		[k: string]: unknown | undefined;
	} | null;
	constraints?: {
		[k: string]: unknown | undefined;
	} | null;
	/**
	 * Requested GPU class; admission control may change it
	 */
	gpu?: string | null;
	/**
	 * The visitor's use case, for the guide
	 */
	description?: string | null;
}
/**
 * target_id -> Estimate, for one knob setting.
 *
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "Prediction".
 */
export interface Prediction {
	[k: string]: Estimate | undefined;
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "KnobImpact".
 */
export interface KnobImpact {
	knob_id: string;
	/**
	 * Change in the objective across the knob's range, normalised to [0, 1]
	 */
	impact: number;
	direction: "increase" | "decrease" | "either";
	best_value?: boolean | number | string | null;
	reason?: string | null;
}

export const SCHEMA_VERSION = "0.1.0";
