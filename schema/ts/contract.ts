/* Generated from python/preflight/schema by `uv run python -m preflight.schema.generate`. Do not edit. */

/**
 * Preflight contract v0.6.4. Generated; do not edit.
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
	/**
	 * Where the run cost's rates come from (B)
	 */
	cost_basis?: CostBasis | null;
	/**
	 * The run cost breakdown, written by B
	 */
	job_cost?: JobCost | null;
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
	/**
	 * 0.6.3 adds command: a console tool run by name (args holds the whole command; uri is command://<tool>), with runner.command_checkin on
	 */
	kind: "sample" | "repo" | "upload" | "notebook" | "template" | "cli" | "inflight" | "record" | "command";
	/**
	 * sample://name, https://github.com/..., upload://<hash>
	 */
	uri?: string | null;
	/**
	 * kind=record: a share token or sample:<id>; the job reuses that record's script, env and data, and writes a new record whose parent_id is it
	 */
	record_ref?: string | null;
	/**
	 * For repos: the full commit the code was read at
	 */
	commit?: string | null;
	/**
	 * sha256 of the submitted code
	 */
	content_hash?: string | null;
	entrypoint?: string | null;
	args?: string[];
	/**
	 * 0.6.3: what was read. repo (or absent): the whole source, or its entry point's imports when it's too big to read whole (runner.read_closure). entrypoint (needs entrypoint): only that file, the source's own modules it imports (transitively) and its requirements files, at any size
	 */
	scope?: ("repo" | "entrypoint") | null;
	/**
	 * 0.6.3: the job's extra packages (JobRequest.extra_packages), as PyPI names on the list
	 */
	extra_packages?:
		| []
		| [string]
		| [string, string]
		| [string, string, string]
		| [string, string, string, string]
		| [string, string, string, string, string]
		| [string, string, string, string, string, string]
		| [string, string, string, string, string, string, string]
		| [string, string, string, string, string, string, string, string]
		| [string, string, string, string, string, string, string, string, string]
		| [string, string, string, string, string, string, string, string, string, string]
		| null;
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
	head_dim?: number | null;
	/**
	 * MoE experts per layer
	 */
	n_experts?: number | null;
	/**
	 * MoE experts routed per token
	 */
	experts_per_token?: number | null;
	/**
	 * Longest context the model was built for
	 */
	max_positions?: number | null;
	/**
	 * SwiGLU/GeGLU-style MLP with a gate projection
	 */
	gated_mlp?: boolean | null;
	/**
	 * Output head shares the embedding matrix
	 */
	tie_embeddings?: boolean | null;
	norm?: ("rmsnorm" | "layernorm") | null;
	/**
	 * eager, sdpa, flash_attention_2, ...
	 */
	attn_impl?: string | null;
	dtype?: string | null;
	/**
	 * Weight format, as engine.memory keys it: nf4, nf4_dq, fp4, fp4_dq, int8, gptq_4bit, gptq_8bit, awq_4bit; none or absent for unquantised
	 */
	quantization?: string | null;
	/**
	 * Where the shape fields came from: the code, or the running model
	 */
	shape_source?: ("code" | "hf_config" | "modules") | null;
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
	/**
	 * Job-map node ids the knob acts on
	 *
	 * @maxItems 16
	 */
	affects?:
		| []
		| [string]
		| [string, string]
		| [string, string, string]
		| [string, string, string, string]
		| [string, string, string, string, string]
		| [string, string, string, string, string, string]
		| [string, string, string, string, string, string, string]
		| [string, string, string, string, string, string, string, string]
		| [string, string, string, string, string, string, string, string, string]
		| [string, string, string, string, string, string, string, string, string, string]
		| [string, string, string, string, string, string, string, string, string, string, string]
		| [string, string, string, string, string, string, string, string, string, string, string, string]
		| [string, string, string, string, string, string, string, string, string, string, string, string, string]
		| [
				string,
				string,
				string,
				string,
				string,
				string,
				string,
				string,
				string,
				string,
				string,
				string,
				string,
				string
		  ]
		| [
				string,
				string,
				string,
				string,
				string,
				string,
				string,
				string,
				string,
				string,
				string,
				string,
				string,
				string,
				string
		  ]
		| [
				string,
				string,
				string,
				string,
				string,
				string,
				string,
				string,
				string,
				string,
				string,
				string,
				string,
				string,
				string,
				string
		  ];
	/**
	 * True if moving it changes what the model learns (offered only to clear a failure, never as a saving); false if it changes only speed, memory or cost; null if unknown
	 */
	changes_results?: boolean | null;
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
	/**
	 * 0.6.3: on a missing_package failure, the import the script couldn't find (as Python named it)
	 */
	missing_module?: string | null;
	/**
	 * 0.6.3: the PyPI name that provides missing_module when it's on Preflight's list (an extra package the visitor can add); absent when it isn't, or for a submodule (`a.b`)
	 */
	missing_package?: string | null;
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
	/**
	 * Every optimizer step the probe ran, in order
	 *
	 * @maxItems 256
	 */
	step_log?: StepSample[];
	/**
	 * Per-kernel table, current-settings probe only (profiled window)
	 */
	kernels?: KernelTable | null;
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
	/**
	 * Batches the full eval loader would run, so the pass can be extrapolated
	 */
	loader_batches?: number | null;
	/**
	 * Allocated bytes after each measured eval batch: growth across batches (outputs kept for metrics) is what makes an eval OOM late in the pass
	 *
	 * @maxItems 64
	 */
	memory_by_batch?: number[];
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
 * One optimizer step as the probe saw it, so B fits on every step, not a few summary points.
 *
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "StepSample".
 */
export interface StepSample {
	index: number;
	/**
	 * warmup (discarded), timed (no recorder sync points, CUDA events at the step boundary only), phase (synchronised, for the phase split), profiled (profiler attached)
	 */
	kind: "warmup" | "timed" | "phase" | "profiled";
	/**
	 * Step boundary to step boundary, minus eval and checkpoint time
	 */
	wall_s: number;
	/**
	 * Tokens in the step's batches, padding included
	 */
	tokens?: number | null;
	/**
	 * Tokens the attention mask keeps
	 */
	real_tokens?: number | null;
	samples?: number | null;
	/**
	 * Padded sequence length of the step's batches
	 */
	padded_len?: number | null;
	micro_steps?: number | null;
	peak_allocated_bytes?: number | null;
	peak_reserved_bytes?: number | null;
	/**
	 * Whole-device use at the boundary (cudaMemGetInfo): PyTorch plus everything else
	 */
	device_used_bytes?: number | null;
	/**
	 * 0.6.1: the allocator's reserved memory read at the same instant as device_used_bytes, so device_used_bytes - reserved_bytes is the CUDA or ROCm context plus other processes
	 */
	reserved_bytes?: number | null;
}
/**
 * Top KERNEL_ROWS ops by device time, then the tail per class (probe-plan K: 100% within 20%).
 *
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "KernelTable".
 */
export interface KernelTable {
	/**
	 * @maxItems 96
	 */
	rows: KernelStat[];
	/**
	 * Profiled steps the rows average over
	 */
	steps: number;
	backend: "cuda" | "rocm" | "cpu";
	/**
	 * fp32 matmuls ran as TF32, so B picks the TF32 peak
	 */
	allow_tf32: boolean;
	/**
	 * Which classifier set kernel_class
	 */
	classified_by?: string | null;
}
/**
 * One row of the profiled probe's kernel table, per optimizer step; B scales it to other GPUs.
 *
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "KernelStat".
 */
export interface KernelStat {
	/**
	 * aten op; '(tail) <class> (<n> ops)' rolls up the rest of a class; '(unattributed device time)' is GPU time no op claimed
	 */
	name: string;
	/**
	 * engine.steptime.classify_kernel class, e.g. gemm, attention
	 */
	kernel_class?: string | null;
	/**
	 * Calls per step
	 */
	calls: number;
	/**
	 * Self device time per step (self CPU time on CPU probes)
	 */
	device_s: number;
	/**
	 * Per step; counted by PyTorch or estimated
	 */
	flops?: number | null;
	/**
	 * Read plus write traffic per step, estimated
	 */
	bytes?: number | null;
	/**
	 * bf16, fp16, fp32, ...
	 */
	dtype: string;
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
	/**
	 * 0.6.0: how the sampled run ended; stalled = no model call for the stall time (a tool stuck)
	 */
	ended?: ("finished" | "stalled" | "cancelled" | "timeout") | null;
	/**
	 * 0.6.1: tasks run at once (the script's workers); in a stalled run at most this many can be stuck
	 */
	concurrency?: number | null;
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
	/**
	 * All generated tokens, reasoning included (as OpenAI and Anthropic bill them)
	 */
	output_tokens: number;
	/**
	 * The part of output_tokens spent reasoning; not added again
	 */
	reasoning_tokens?: number;
	cached_input_tokens?: number;
	cache_creation_tokens?: number;
	tool_calls?: number;
	errors?: number;
	latency_s: number;
	success?: boolean | null;
	context_tokens_by_turn?: number[];
	cost_usd?: number | null;
	/**
	 * cut (0.6.0): stopped with the whole run by a cancel or the run's time limit; never on a stall, where tasks in flight keep no stop reason (0.6.1)
	 */
	stop_reason?: ("done" | "max_turns" | "context_overflow" | "error" | "timeout" | "loop" | "cut") | null;
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
	/**
	 * 0.6.0: cost estimates only, quantity x rate
	 */
	rate?: EstimateRate | null;
}
/**
 * 0.6.0 (B, Nov 1): how a cost estimate splits into a quantity times a rate, so a client can
 * re-price it at the visitor's own rate without parsing the reason.
 *
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "EstimateRate".
 */
export interface EstimateRate {
	/**
	 * The target holding the quantity, e.g. cost.gpu_hours
	 */
	quantity: string;
	usd_per_unit: number;
	/**
	 * The quantity's unit, e.g. gpu_hours
	 */
	unit: string;
	/**
	 * The price-table row id the rate came from
	 */
	price_row?: string | null;
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
	 * 0.6.4: knobs this target doesn't depend on (the learning rate, say): the grid answers at any value of them, so a reader holds them nowhere
	 */
	independent_of?: string[] | null;
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
	/**
	 * Illustrative numbers from a fixture, not a fitted model; the UI says so
	 */
	fixture?: boolean;
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
	gpu_hours?: Estimate | undefined | null;
	usd?: Estimate | undefined | null;
	/**
	 * Seconds from start until the last run ends
	 */
	finish_s?: Estimate | undefined | null;
	grid_gpu_hours?: Estimate | undefined | null;
	grid_usd?: Estimate | undefined | null;
	quota?: CampaignQuota | null;
	asha?: CampaignAsha | null;
	notes?: string[];
	/**
	 * 0.6.0: when the campaign clock starts; start_s and finish_s count from it
	 */
	start?: string | null;
	/**
	 * 0.6.0: probability every run finishes by job.constraints.deadline
	 */
	p_by_deadline?: number | null;
	/**
	 * 0.6.0: the checkpoint interval with its interval; checkpoint_interval_s stays the point
	 */
	checkpoint_interval?: Estimate | undefined | null;
	/**
	 * 0.6.0: slots per lane
	 */
	lane_slots?: {
		[k: string]: number | undefined;
	} | null;
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
	/**
	 * 0.5.0: this run's wall time, with its interval
	 */
	time?: Estimate | undefined | null;
	/**
	 * 0.5.0: when this run starts, in seconds after the campaign starts, in its lane's slot order
	 */
	start_s?: Estimate | undefined | null;
	/**
	 * 0.6.0: which of its lane's slots runs it, 0-based
	 */
	slot?: number | null;
	/**
	 * 0.6.0: why the run is cut or undated, in words
	 */
	reason?: string | null;
	/**
	 * 0.6.0: this run's GPU-hours with its interval; gpu_hours stays the point
	 */
	gpu_hours_estimate?: Estimate | undefined | null;
	/**
	 * 0.6.0: this run's dollars with its interval; usd stays the point
	 */
	usd_estimate?: Estimate | undefined | null;
}
/**
 * 0.5.0: whether the campaign's checkpoints and outputs fit the storage quota.
 *
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "CampaignQuota".
 */
export interface CampaignQuota {
	needed_bytes: Estimate | undefined;
	quota_bytes?: number | null;
	/**
	 * None when the need straddles the quota
	 */
	fits?: boolean | null;
	reason?: string | null;
}
/**
 * 0.5.0: an ASHA pruning plan and the GPU-hours it saves.
 *
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "CampaignAsha".
 */
export interface CampaignAsha {
	rungs?: AshaRung[];
	gpu_hours?: Estimate | undefined | null;
	saved_gpu_hours?: Estimate | undefined | null;
	reason?: string | null;
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "AshaRung".
 */
export interface AshaRung {
	steps: number;
	runs: number;
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
 * What Preflight itself spent making this record (probes, verify, model calls); never part of the run cost
 */
export interface Costs {
	gpu_usd?: number;
	llm_usd?: number;
	breakdown?: {
		[k: string]: number | undefined;
	};
}
/**
 * Where the run cost's rates come from, so every cost readout can say what it is based on.
 *
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "CostBasis".
 */
export interface CostBasis {
	/**
	 * Version of B's price table, e.g. 'prices@2026-10-27'
	 */
	price_table: string;
	/**
	 * Date the rates were checked
	 */
	as_of: string;
	/**
	 * Provider the user picked, e.g. 'modal', 'unity', 'amd-devcloud'
	 */
	provider: string;
	region?: string | null;
	billing?: "on_demand" | "spot" | "own_cluster";
	/**
	 * Rates are USD; fmt() converts for display
	 */
	currency?: "USD";
	/**
	 * Line kinds counted in the total
	 */
	includes?: ("gpu" | "tokens" | "storage")[];
}
/**
 * What the job itself would cost to run in full. Preflight's own spend is `costs`, never here.
 *
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "JobCost".
 */
export interface JobCost {
	lines?: CostLine[];
	/**
	 * Sum of billed lines, lows and highs added line by line; None on own cluster
	 */
	total_usd?: Estimate | undefined | null;
	/**
	 * Shown first on the user's own cluster
	 */
	gpu_hours?: Estimate | undefined | null;
	/**
	 * On-demand price of the same GPU-hours, shown grey on an own cluster
	 */
	market_equivalent_usd?: Estimate | undefined | null;
	/**
	 * Money at risk as GPU-hours, shown first on an own cluster (at_risk_usd is then the grey market figure)
	 */
	at_risk_gpu_hours?: Estimate | undefined | null;
	/**
	 * Chance of failure x cost up to the failure point; shown beside the total, never in it
	 */
	at_risk_usd?: Estimate | undefined | null;
}
/**
 * One line of the run cost: a predicted quantity times a rate from the price table.
 *
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "CostLine".
 */
export interface CostLine {
	kind: "gpu" | "tokens" | "storage";
	/**
	 * t() key for the line's label
	 */
	label_key: string;
	quantity: Estimate1;
	/**
	 * USD per unit; None when not billed
	 */
	rate_usd?: number | null;
	/**
	 * quantity x rate; None when not billed in dollars
	 */
	usd?: Estimate | undefined | null;
	/**
	 * Id of the price-table row the rate came from
	 */
	price_row?: string | null;
}
/**
 * One number on screen: a point with an interval, or an honest abstention.
 */
export interface Estimate1 {
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
	/**
	 * 0.6.0: cost estimates only, quantity x rate
	 */
	rate?: EstimateRate | null;
}
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "JobEvent".
 */
export interface JobEvent {
	job_id: string;
	record_id?: string | null;
	/**
	 * Where the record is read (GET /v1/records/{share_token}) once a skeleton exists
	 */
	share_token?: string | null;
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
	/**
	 * On failed: a stable code in Probe.error_code's families; the UI shows t('error.<code>')
	 */
	error_code?: string | null;
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
	constraints?: Constraints | null;
	/**
	 * Requested GPU class; admission control may change it
	 */
	gpu?: string | null;
	/**
	 * The visitor's use case, for the guide
	 */
	description?: string | null;
	/**
	 * plan: map and probe a script (default). probe: one probe at `knobs`. verify: a short fixed-seed run of `knobs` against the record's current settings. sample_more: `tasks` more agent tasks. All but plan take a record source.
	 */
	mode?: "plan" | "probe" | "verify" | "sample_more";
	/**
	 * probe and verify: the full setting to run at
	 */
	knobs?: {
		[k: string]: (boolean | number | string) | undefined;
	} | null;
	/**
	 * sample_more: how many more tasks
	 */
	tasks?: number | null;
	/**
	 * Who asked, for the logbook, e.g. guide.run_probe
	 */
	purpose?: string | null;
	/**
	 * 0.5.0: a unified diff applied to the record's code before a probe or verify job (C's fix loop); code fixes can't be passed as knob overrides
	 */
	patch?: string | null;
	/**
	 * 0.6.2: the visitor chose private mode; with harness.glimmer_private on, triage runs on Muse Glimmer on Preflight's own GPU instead of Muse Spark Open on Baseten
	 */
	private?: boolean;
	/**
	 * 0.6.3: PyPI names (no versions or URLs) the probe installs after the code's own requirements; only names on Preflight's list (recorder.requirements.known_packages), wheels only, the image's pinned stack wins. A name off the list fails the job with missing_package.not_allowed (.no_wheels for a package PyPI has no wheels for). Kept on the record's source, so later probes of it install them too
	 */
	extra_packages?:
		| []
		| [string]
		| [string, string]
		| [string, string, string]
		| [string, string, string, string]
		| [string, string, string, string, string]
		| [string, string, string, string, string, string]
		| [string, string, string, string, string, string, string]
		| [string, string, string, string, string, string, string, string]
		| [string, string, string, string, string, string, string, string, string]
		| [string, string, string, string, string, string, string, string, string, string]
		| null;
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
/**
 * This interface was referenced by `PreflightContract`'s JSON-Schema
 * via the `definition` "Costs".
 */
export interface Costs1 {
	gpu_usd?: number;
	llm_usd?: number;
	breakdown?: {
		[k: string]: number | undefined;
	};
}

export const SCHEMA_VERSION = "0.6.4";
