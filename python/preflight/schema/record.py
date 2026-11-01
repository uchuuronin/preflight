"""The flight record: one per job, under 1 MB gzipped.

Section ownership (who writes it) follows the Contracts tab; each class says
who in its docstring. Raw traces and logs go to the object store and are
referenced by blob:// keys, never inlined.
"""

from datetime import UTC, date, datetime
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, model_validator

from .base import BlobRef, KnobValue, Lens, Strict, Unit
from .ids import new_share_token, new_ulid
from .prediction import Estimate, Objective, Predictions
from .script_map import Framework, ScriptMap

SCHEMA_VERSION = "0.6.2"


def _now() -> datetime:
	return datetime.now(UTC).replace(microsecond=0)


# ---------------------------------------------------------------- job (A)


RECORD_REF = r"^(sample:[a-z0-9][a-z0-9_-]{0,39}|[A-Za-z0-9_-]{16,64})$"


class Source(Strict):
	kind: Literal["sample", "repo", "upload", "notebook", "template", "cli", "inflight", "record"]
	uri: str | None = Field(
		default=None, description="sample://name, https://github.com/..., upload://<hash>"
	)
	record_ref: str | None = Field(
		default=None,
		pattern=RECORD_REF,
		description="kind=record: a share token or sample:<id>; the job reuses that record's script, "
		"env and data, and writes a new record whose parent_id is it",
	)
	commit: str | None = Field(default=None, description="For repos: the full commit the code was read at")
	content_hash: str | None = Field(default=None, description="sha256 of the submitted code")
	entrypoint: str | None = None
	args: list[str] = Field(default_factory=list)


class PreemptionRule(Strict):
	kind: Literal["none", "poisson", "policy"] = "none"
	mtti_s: float | None = Field(default=None, description="Mean time to interrupt, for kind=poisson")
	policy: str | None = Field(default=None, description="The cluster's rule in words, for kind=policy")
	grace_s: float | None = None


class Quota(Strict):
	disk_bytes: int | None = None
	gpu_hours: float | None = None
	api_usd: float | None = None


# visitor-supplied GPU names reach regex-based resolution (engine.gpus.resolve): keep them short
GpuName = Annotated[str, Field(min_length=1, max_length=64)]


class Constraints(Strict):
	budget_usd: float | None = Field(default=None, ge=0)
	deadline: datetime | None = None
	gpu_types: list[GpuName] = Field(default_factory=list, description="Allowed GPU names; empty means any")
	max_gpus: int | None = Field(default=None, ge=1)
	quota: Quota | None = None
	wall_time_cap_s: float | None = Field(
		default=None, gt=0, description="Scheduler limit per job, e.g. Slurm"
	)
	preemption: PreemptionRule = Field(default_factory=PreemptionRule)


class Job(Strict):
	"""Written by A."""

	lens: Lens
	source: Source
	framework: Framework | None = None
	title: str | None = None
	script_map: ScriptMap | None = None
	objective: Objective = Field(default_factory=Objective)
	constraints: Constraints = Field(default_factory=Constraints)


# ---------------------------------------------------------------- env (A)


class GPU(Strict):
	name: str = Field(description="Canonical name from the spec table, e.g. H100-SXM-80GB, MI300X, L4")
	count: int = Field(default=1, ge=0)
	memory_bytes: int | None = None
	vendor: Literal["nvidia", "amd", "apple", "none"] = "nvidia"
	compute_capability: str | None = None
	driver: str | None = None


class Env(Strict):
	"""Written by A."""

	backend: Literal["cuda", "rocm", "api", "cpu", "mps"]
	provider: str = Field(description="modal, amd_devcloud, local, colab, kaggle, api")
	gpu: GPU | None = None
	nodes: int = Field(default=1, ge=1)
	libraries: dict[str, str] = Field(default_factory=dict, description="package -> version")
	python: str | None = None
	seed: int | None = None
	region: str | None = None
	image: str | None = Field(default=None, description="Container image digest the probe ran in")


# ---------------------------------------------------------------- graph (A builds, C reads)

NodeKind = Literal[
	"module",
	"data",
	"loss",
	"optimizer",
	"io",
	"phase",
	"model_call",
	"tool",
	"store",
	"queue",
	"group",
	"other",
]


class GraphNode(Strict):
	id: str
	label: str
	kind: NodeKind = "module"
	parent: str | None = Field(default=None, description="Enclosing group, for collapsed hierarchy")
	repeat: int = Field(default=1, ge=1, description="Shown once with ×N when > 1")
	params: int | None = None
	metrics: list[str] = Field(
		default_factory=list, description="target ids or probe metric keys this node animates"
	)
	attrs: dict[str, KnobValue] = Field(default_factory=dict)


class GraphEdge(Strict):
	source: str
	target: str
	kind: Literal["data", "grad", "control", "tokens", "weights"] = "data"
	metric: str | None = Field(
		default=None, description="target id or metric key whose rate animates the edge"
	)


class Graph(Strict):
	nodes: list[GraphNode] = Field(default_factory=list)
	edges: list[GraphEdge] = Field(default_factory=list)
	skeleton: bool = Field(default=True, description="True until probes have measured it")

	@model_validator(mode="after")
	def _check(self):
		ids = [n.id for n in self.nodes]
		if len(set(ids)) != len(ids):
			raise ValueError("graph node ids must be unique")
		known = set(ids)
		for n in self.nodes:
			if n.parent is not None and n.parent not in known:
				raise ValueError(f"node {n.id} has unknown parent {n.parent}")
		for e in self.edges:
			if e.source not in known or e.target not in known:
				raise ValueError(f"edge {e.source}->{e.target} references an unknown node")
		return self


# ---------------------------------------------------------------- knobs (A declares, B fills ranges)


class Domain(Strict):
	min: float | None = None
	max: float | None = None
	values: list[KnobValue] = Field(default_factory=list, description="Allowed values for categorical knobs")
	step: float | None = None
	log: bool = False


class Range(Strict):
	min: float
	max: float


class Knob(Strict):
	id: str = Field(pattern=r"^[a-z][a-z0-9_.]*$")
	label_key: str = Field(description="i18n key, e.g. knob.batch_size")
	unit: Unit | None = None
	kind: Literal["int", "float", "categorical", "bool"]
	domain: Domain = Field(default_factory=Domain)
	current: KnobValue
	probed_range: Range | None = Field(default=None, description="Filled by B from the probes")
	gridded: bool = False
	needs_code: bool = False
	patch_template: str | None = Field(default=None, description="Patch template id for code knobs")
	group: str | None = Field(default=None, description="memory, speed, data, model, rl, agent, cluster")
	affects: list[str] = Field(
		default_factory=list, max_length=16, description="Job-map node ids the knob acts on"
	)
	changes_results: bool | None = Field(
		default=None,
		description="True if moving it changes what the model learns (offered only to clear a failure, "
		"never as a saving); false if it changes only speed, memory or cost; null if unknown",
	)

	@model_validator(mode="after")
	def _check(self):
		v = self.current
		if self.kind in ("int", "float"):
			if isinstance(v, bool) or not isinstance(v, int | float):
				raise ValueError(f"knob {self.id}: numeric knob has non-numeric value {v!r}")
			if self.kind == "int" and not float(v).is_integer():
				raise ValueError(f"knob {self.id}: int knob has value {v!r}")
			if self.domain.min is not None and v < self.domain.min:
				raise ValueError(f"knob {self.id}: {v} below domain min {self.domain.min}")
			if self.domain.max is not None and v > self.domain.max:
				raise ValueError(f"knob {self.id}: {v} above domain max {self.domain.max}")
		elif self.kind == "bool":
			if not isinstance(v, bool):
				raise ValueError(f"knob {self.id}: bool knob has value {v!r}")
		elif self.domain.values and v not in self.domain.values:
			raise ValueError(f"knob {self.id}: {v!r} not among {self.domain.values}")
		return self


# ---------------------------------------------------------------- probes (A)

ProbeStatus = Literal["queued", "running", "ok", "oom", "error", "timeout", "cancelled", "cached"]


class Memory(Strict):
	peak_allocated_bytes: int | None = None
	peak_reserved_bytes: int | None = None
	device_total_bytes: int | None = None
	by_term: dict[str, int] = Field(
		default_factory=dict,
		description="weights, gradients, optimizer, activations, logits, context, kv_cache, other",
	)
	peak_step: int | None = Field(default=None, description="Step where the peak occurred")
	timeline: BlobRef | None = None


class StepTime(Strict):
	n_steps: int = Field(ge=0)
	mean_s: float | None = None
	p50_s: float | None = None
	p95_s: float | None = None
	first_step_s: float | None = Field(default=None, description="Includes compile and warm-up")
	by_phase: dict[str, float] = Field(
		default_factory=dict,
		description="Fractions of a step: data, forward, backward, optimizer, eval, rollout, reference, reward, update, weight_sync, idle",
	)
	unprofiled: bool = Field(
		default=True, description="Timings come from steps without the profiler attached"
	)


StepKind = Literal["warmup", "timed", "phase", "profiled"]


class StepSample(Strict):
	"""One optimizer step as the probe saw it, so B fits on every step, not a few summary points."""

	model_config = ConfigDict(allow_inf_nan=False)

	index: int = Field(ge=1)
	kind: StepKind = Field(
		description="warmup (discarded), timed (no recorder sync points, CUDA events at the step boundary "
		"only), phase (synchronised, for the phase split), profiled (profiler attached)"
	)
	wall_s: float = Field(ge=0, description="Step boundary to step boundary, minus eval and checkpoint time")
	tokens: int | None = Field(
		default=None, ge=0, description="Tokens in the step's batches, padding included"
	)
	real_tokens: int | None = Field(default=None, ge=0, description="Tokens the attention mask keeps")
	samples: int | None = Field(default=None, ge=0)
	padded_len: int | None = Field(
		default=None, ge=0, description="Padded sequence length of the step's batches"
	)
	micro_steps: int | None = Field(default=None, ge=0)
	peak_allocated_bytes: int | None = Field(default=None, ge=0)
	peak_reserved_bytes: int | None = Field(default=None, ge=0)
	device_used_bytes: int | None = Field(
		default=None,
		ge=0,
		description="Whole-device use at the boundary (cudaMemGetInfo): PyTorch plus everything else",
	)
	reserved_bytes: int | None = Field(
		default=None,
		ge=0,
		description="0.6.1: the allocator's reserved memory read at the same instant as device_used_bytes, so "
		"device_used_bytes - reserved_bytes is the CUDA or ROCm context plus other processes",
	)


class Throughput(Strict):
	tokens_per_s: float | None = None
	samples_per_s: float | None = None
	flops_per_step: float | None = None
	mfu: float | None = Field(default=None, ge=0, le=1.5)


class Stability(Strict):
	series: dict[str, list[float | None]] = Field(
		default_factory=dict,
		description="Per step: loss, grad_norm, grad_norm_ma, update_to_weight, max_attn_logit, output_logit_norm, loss_scale, lr",
	)
	nan_count: int = 0
	inf_count: int = 0
	loss_scale_events: int = 0


class DataStats(Strict):
	padding_fraction: float | None = Field(default=None, ge=0, le=1)
	packing: bool | None = None
	tokens_per_sample_mean: float | None = None
	tokens_per_sample_p95: float | None = None
	loader_wait_fraction: float | None = Field(default=None, ge=0, le=1)
	num_workers: int | None = None


class LengthStats(Strict):
	mean: float
	p95: float
	max: float
	n: int


class RLSignals(Strict):
	response_length: LengthStats | None = None
	truncation_rate: float | None = Field(default=None, ge=0, le=1)
	zero_spread_fraction: float | None = Field(default=None, ge=0, le=1)
	pass_rate_per_prompt: list[float] = Field(default_factory=list)
	group_size: int | None = None
	entropy: list[float] = Field(default_factory=list)
	kl: list[float] = Field(default_factory=list)
	clip_fraction: list[float] = Field(default_factory=list)
	reward_mean: list[float] = Field(default_factory=list)


class Power(Strict):
	mean_w: float
	max_w: float
	samples: int


class EvalStats(Strict):
	"""One eval pass, pulled inside the probe window and capped at a few batches."""

	batches: int = Field(ge=0)
	batch_size: int | None = None
	seconds_per_batch: float | None = None
	peak_allocated_bytes: int | None = None
	grad_enabled: bool | None = Field(
		default=None, description="True when eval ran with autograd on (no no_grad)"
	)
	loader_batches: int | None = Field(
		default=None,
		ge=0,
		description="Batches the full eval loader would run, so the pass can be extrapolated",
	)
	memory_by_batch: list[int] = Field(
		default_factory=list,
		max_length=64,
		description="Allocated bytes after each measured eval batch: growth across batches (outputs kept "
		"for metrics) is what makes an eval OOM late in the pass",
	)


class CheckpointStats(Strict):
	saves: int = Field(ge=0)
	seconds_mean: float | None = None
	bytes_mean: int | None = None


class GenerationStats(Strict):
	"""Generation inside the job: RL rollouts, eval sweeps, serving probes."""

	calls: int = Field(ge=0)
	seconds_total: float = 0
	sequences: int = 0
	new_tokens_mean: float | None = None
	new_tokens_p95: float | None = None
	new_tokens_max: float | None = None
	max_new_tokens: int | None = Field(default=None, description="The cap the script set, if seen")
	truncation_rate: float | None = Field(
		default=None, ge=0, le=1, description="Share of sequences that hit the cap"
	)
	backend: Literal["hf", "vllm", "api", "other"] | None = None


# Families: oom, oom.eval, oom.host, missing_package, download_blocked, gated_model, file_not_found,
# cuda_error, nccl, dtype_unsupported, shape_mismatch, dataloader_worker, syntax_error, script_exit,
# too_few_steps, timeout, killed, recorder, user_error, nan_divergence
ErrorCode = Annotated[str, Field(pattern=r"^[a-z_]+(\.[a-z_]+)?$")]


class ProbeCost(Strict):
	gpu_seconds: float = 0
	usd: float = 0


KERNEL_TABLE_ROWS = 96  # top 64, one tail row per kernel class (16), the unattributed row, and room
KERNEL_NAME_LEN = 200  # a visitor's record_function label can be any length; the recorder cuts it


class KernelStat(Strict):
	"""One row of the profiled probe's kernel table, per optimizer step; B scales it to other GPUs."""

	model_config = ConfigDict(allow_inf_nan=False)

	name: str = Field(
		max_length=KERNEL_NAME_LEN,
		description="aten op; '(tail) <class> (<n> ops)' rolls up the rest of a class; "
		"'(unattributed device time)' is GPU time no op claimed",
	)
	kernel_class: str | None = Field(
		default=None, max_length=64, description="engine.steptime.classify_kernel class, e.g. gemm, attention"
	)
	# strict: numbers only (no "3", no true); bounded well past anything a real step reaches
	calls: float = Field(ge=0, le=1e9, strict=True, description="Calls per step")
	device_s: float = Field(
		ge=0, le=86_400, strict=True, description="Self device time per step (self CPU time on CPU probes)"
	)
	flops: float | None = Field(
		default=None, ge=0, le=1e20, strict=True, description="Per step; counted by PyTorch or estimated"
	)
	bytes: float | None = Field(
		default=None, ge=0, le=1e18, strict=True, description="Read plus write traffic per step, estimated"
	)
	dtype: str = Field(max_length=8, description="bf16, fp16, fp32, ...")


class KernelTable(Strict):
	"""Top KERNEL_ROWS ops by device time, then the tail per class (probe-plan K: 100% within 20%)."""

	rows: list[KernelStat] = Field(max_length=KERNEL_TABLE_ROWS)
	steps: int = Field(ge=1, le=1_000, strict=True, description="Profiled steps the rows average over")
	backend: Literal["cuda", "rocm", "cpu"]
	allow_tf32: bool = Field(description="fp32 matmuls ran as TF32, so B picks the TF32 peak")
	classified_by: str | None = Field(
		default=None, max_length=64, description="Which classifier set kernel_class"
	)


class Probe(Strict):
	id: str
	knobs: dict[str, KnobValue] = Field(
		description="Knob values this probe ran at (only those it changed or read)"
	)
	status: ProbeStatus
	steps: int = Field(ge=0, description="Optimizer steps completed; at least 2 for a valid memory peak")
	started_at: datetime | None = None
	duration_s: float | None = None
	error: str | None = None
	error_code: ErrorCode | None = Field(
		default=None, description="Stable code for the failure; the UI shows t('error.<code>') and its hint"
	)
	memory: Memory | None = None
	step_time: StepTime | None = None
	throughput: Throughput | None = None
	stability: Stability | None = None
	data: DataStats | None = None
	rl: RLSignals | None = None
	power: Power | None = None
	eval: EvalStats | None = None
	checkpoint: CheckpointStats | None = None
	generation: GenerationStats | None = None
	extras: dict[str, float] = Field(default_factory=dict, description="Lens-specific measurements")
	step_log: list[StepSample] = Field(
		default_factory=list, max_length=256, description="Every optimizer step the probe ran, in order"
	)
	kernels: KernelTable | None = Field(
		default=None, description="Per-kernel table, current-settings probe only (profiled window)"
	)
	cost: ProbeCost = Field(default_factory=ProbeCost)
	cache_key: str | None = Field(default=None, description="sha256 of script, env and knob values")
	trace: BlobRef | None = None
	log: BlobRef | None = None


# ---------------------------------------------------------------- agent sample (A)


class AgentTask(Strict):
	task_id: str
	model: str | None = None
	turns: int = Field(ge=0)
	input_tokens: int = Field(ge=0)
	output_tokens: int = Field(
		ge=0, description="All generated tokens, reasoning included (as OpenAI and Anthropic bill them)"
	)
	reasoning_tokens: int = Field(
		default=0, ge=0, description="The part of output_tokens spent reasoning; not added again"
	)
	cached_input_tokens: int = Field(default=0, ge=0)
	cache_creation_tokens: int = Field(default=0, ge=0)
	tool_calls: int = Field(default=0, ge=0)
	errors: int = Field(default=0, ge=0)
	latency_s: float = Field(ge=0)
	success: bool | None = None
	context_tokens_by_turn: list[int] = Field(default_factory=list)
	cost_usd: float | None = None
	stop_reason: (
		Literal["done", "max_turns", "context_overflow", "error", "timeout", "loop", "cut"] | None
	) = Field(
		default=None,
		description="cut (0.6.0): stopped with the whole run by a cancel or the run's time limit; never on a stall, where tasks in flight keep no stop reason (0.6.1)",
	)
	trace: BlobRef | None = None


class AgentSample(Strict):
	tasks: list[AgentTask] = Field(default_factory=list)
	population: int | None = Field(default=None, description="Tasks in the full run")
	selection: str | None = Field(
		default=None, description="How the sample was chosen, e.g. 30–70% pass-rate band"
	)
	rate_limit_rpm: float | None = None
	rate_limit_tpm: float | None = None
	ended: Literal["finished", "stalled", "cancelled", "timeout"] | None = Field(
		default=None,
		description="0.6.0: how the sampled run ended; stalled = no model call for the stall time (a tool stuck)",
	)
	concurrency: int | None = Field(
		default=None,
		ge=1,
		description="0.6.1: tasks run at once (the script's workers); in a stalled run at most this many can be stuck",
	)


# ---------------------------------------------------------------- models (B)


class ModelCard(Strict):
	id: str = Field(description="name@version, pinned by every grid and estimate that used it")
	kind: Literal[
		"analytical", "gbdt", "gnn", "classifier", "conformal", "bootstrap", "simulation", "ensemble", "rule"
	]
	version: str
	training_data: str | None = None
	heldout_error: dict[str, float] = Field(
		default_factory=dict, description="metric -> value, e.g. mape, auroc"
	)
	coverage: float | None = Field(default=None, ge=0, le=1)
	per_gpu_coverage: dict[str, float] = Field(default_factory=dict)
	notes: str | None = None
	fixture: bool = Field(
		default=False, description="Illustrative numbers from a fixture, not a fitted model; the UI says so"
	)


# ---------------------------------------------------------------- findings (A and B write, C shows)


class Evidence(Strict):
	metric: str
	value: float | str | None = None
	threshold: float | None = None
	probe_id: str | None = None
	step: int | None = None


class Citation(Strict):
	title: str
	url: str | None = None


class Finding(Strict):
	id: str
	severity: Literal["info", "warn", "error", "critical"]
	code: str = Field(description="Stable rule id, e.g. eval.no_grad, data.num_workers_zero")
	title: str
	detail: str | None = None
	node_id: str | None = Field(default=None, description="Job-map node it attaches to")
	evidence: list[Evidence] = Field(default_factory=list)
	source: Literal["rule", "spark_open_triage", "spark_diagnosis", "model"]
	fix_template: str | None = None
	knob_id: str | None = None
	citations: list[Citation] = Field(default_factory=list)


# ---------------------------------------------------------------- patches (C)


class VerifyResult(Strict):
	status: Literal["pending", "passed", "failed", "error"]
	metric: str | None = None
	before: float | None = None
	after: float | None = None
	fixed_seed_loss_delta: float | None = None
	effective_batch_unchanged: bool | None = None
	probe_id: str | None = None


class Patch(Strict):
	id: str
	kind: Literal["template", "open_ended"]
	template_id: str | None = None
	title: str
	diff: BlobRef
	verify: VerifyResult | None = None
	session_log: BlobRef | None = None
	finding_id: str | None = None


# ---------------------------------------------------------------- campaign (B)


class CampaignCell(Strict):
	id: str
	knobs: dict[str, KnobValue]
	gpu_hours: float | None = None
	usd: float | None = None
	wall_time_s: float | None = None
	segments: int | None = Field(default=None, description="Wall-time-capped segments the run splits into")
	lane: str | None = None
	cut: bool = False
	time: Estimate | None = Field(default=None, description="0.5.0: this run's wall time, with its interval")
	start_s: Estimate | None = Field(
		default=None,
		description="0.5.0: when this run starts, in seconds after the campaign starts, in its lane's slot order",
	)
	# 0.6.0 (B and C, Nov 1): what a shared record needs to draw the same calendar as the live plan.
	slot: int | None = Field(
		default=None, ge=0, description="0.6.0: which of its lane's slots runs it, 0-based"
	)
	reason: str | None = Field(default=None, description="0.6.0: why the run is cut or undated, in words")
	gpu_hours_estimate: Estimate | None = Field(
		default=None, description="0.6.0: this run's GPU-hours with its interval; gpu_hours stays the point"
	)
	usd_estimate: Estimate | None = Field(
		default=None, description="0.6.0: this run's dollars with its interval; usd stays the point"
	)


class CampaignQuota(Strict):
	"""0.5.0: whether the campaign's checkpoints and outputs fit the storage quota."""

	needed_bytes: Estimate
	quota_bytes: float | None = None
	fits: bool | None = Field(default=None, description="None when the need straddles the quota")
	reason: str | None = None


class AshaRung(Strict):
	steps: int = Field(ge=0)
	runs: int = Field(ge=0)


class CampaignAsha(Strict):
	"""0.5.0: an ASHA pruning plan and the GPU-hours it saves."""

	rungs: list[AshaRung] = Field(default_factory=list)
	gpu_hours: Estimate | None = None
	saved_gpu_hours: Estimate | None = None
	reason: str | None = None


class Campaign(Strict):
	cells: list[CampaignCell] = Field(default_factory=list)
	lanes: list[str] = Field(default_factory=list)
	calendar: BlobRef | None = None
	cut_list: list[str] = Field(default_factory=list)
	completion_quantiles: dict[str, datetime] = Field(default_factory=dict, description="p10, p50, p90 ...")
	checkpoint_interval_s: float | None = None
	# 0.5.0 (B's request, Oct 31): the plan's intervals. gpu_hours and usd cover the runs that go ahead;
	# grid_gpu_hours and grid_usd the whole grid before any budget cut.
	gpu_hours: Estimate | None = None
	usd: Estimate | None = None
	finish_s: Estimate | None = Field(default=None, description="Seconds from start until the last run ends")
	grid_gpu_hours: Estimate | None = None
	grid_usd: Estimate | None = None
	quota: CampaignQuota | None = None
	asha: CampaignAsha | None = None
	notes: list[str] = Field(default_factory=list)
	# 0.6.0 (B and C, Nov 1)
	start: datetime | None = Field(
		default=None, description="0.6.0: when the campaign clock starts; start_s and finish_s count from it"
	)
	p_by_deadline: float | None = Field(
		default=None,
		ge=0,
		le=1,
		description="0.6.0: probability every run finishes by job.constraints.deadline",
	)
	checkpoint_interval: Estimate | None = Field(
		default=None,
		description="0.6.0: the checkpoint interval with its interval; checkpoint_interval_s stays the point",
	)
	lane_slots: dict[str, int] | None = Field(default=None, description="0.6.0: slots per lane")


# ---------------------------------------------------------------- provenance and costs


class ModelCall(Strict):
	provider: str = Field(description="meta, baseten, fireworks, local")
	model: str
	purpose: str = Field(description="script_map, triage, diagnosis, guide, patch, label, command")
	input_tokens: int = 0
	output_tokens: int = 0
	cached_input_tokens: int = 0
	cost_usd: float = 0
	request_id: str | None = None
	at: datetime
	latency_s: float | None = None
	ok: bool = True


class Costs(Strict):
	gpu_usd: float = 0
	llm_usd: float = 0
	breakdown: dict[str, float] = Field(default_factory=dict)

	@property
	def total_usd(self) -> float:
		return self.gpu_usd + self.llm_usd


# ------------------------------------------------------------- run cost (0.2.0)

CostLineKind = Literal["gpu", "tokens", "storage"]
BillingMode = Literal["on_demand", "spot", "own_cluster"]


class CostBasis(Strict):
	"""Where the run cost's rates come from, so every cost readout can say what it is based on."""

	price_table: str = Field(description="Version of B's price table, e.g. 'prices@2026-10-27'")
	as_of: date = Field(description="Date the rates were checked")
	provider: str = Field(description="Provider the user picked, e.g. 'modal', 'unity', 'amd-devcloud'")
	region: str | None = None
	billing: BillingMode = "on_demand"
	currency: Literal["USD"] = Field(default="USD", description="Rates are USD; fmt() converts for display")
	includes: list[CostLineKind] = Field(
		default_factory=lambda: ["gpu", "tokens"], description="Line kinds counted in the total"
	)


class CostLine(Strict):
	"""One line of the run cost: a predicted quantity times a rate from the price table."""

	kind: CostLineKind
	label_key: str = Field(description="t() key for the line's label")
	quantity: Estimate = Field(description="GPU-hours, tokens or GB, with its interval")
	rate_usd: float | None = Field(default=None, ge=0, description="USD per unit; None when not billed")
	usd: Estimate | None = Field(default=None, description="quantity x rate; None when not billed in dollars")
	price_row: str | None = Field(default=None, description="Id of the price-table row the rate came from")


class JobCost(Strict):
	"""What the job itself would cost to run in full. Preflight's own spend is `costs`, never here."""

	lines: list[CostLine] = Field(default_factory=list)
	total_usd: Estimate | None = Field(
		default=None,
		description="Sum of billed lines, lows and highs added line by line; None on own cluster",
	)
	gpu_hours: Estimate | None = Field(default=None, description="Shown first on the user's own cluster")
	market_equivalent_usd: Estimate | None = Field(
		default=None, description="On-demand price of the same GPU-hours, shown grey on an own cluster"
	)
	at_risk_gpu_hours: Estimate | None = Field(
		default=None,
		description="Money at risk as GPU-hours, shown first on an own cluster (at_risk_usd is then the grey market figure)",
	)
	at_risk_usd: Estimate | None = Field(
		default=None,
		description="Chance of failure x cost up to the failure point; shown beside the total, never in it",
	)


# ---------------------------------------------------------------- the record


class FlightRecord(Strict):
	schema_version: str = Field(default=SCHEMA_VERSION, pattern=r"^\d+\.\d+\.\d+$")
	record_id: str = Field(default_factory=new_ulid, description="ULID, internal only")
	share_token: str = Field(default_factory=new_share_token, description="Random token for read-only links")
	parent_id: str | None = Field(default=None, description="Record this one derives from (verify, what-if)")
	created_at: datetime = Field(default_factory=_now)
	job: Job
	env: Env | None = None
	graph: Graph = Field(default_factory=Graph)
	knobs: list[Knob] = Field(default_factory=list)
	probes: list[Probe] = Field(default_factory=list)
	agent_sample: AgentSample | None = None
	predictions: Predictions = Field(default_factory=Predictions)
	models: list[ModelCard] = Field(default_factory=list)
	findings: list[Finding] = Field(default_factory=list)
	patches: list[Patch] = Field(default_factory=list)
	campaign: Campaign | None = None
	provenance: list[ModelCall] = Field(default_factory=list)
	costs: Costs = Field(
		default_factory=Costs,
		description="What Preflight itself spent making this record (probes, verify, model calls); never part of the run cost",
	)
	cost_basis: CostBasis | None = Field(default=None, description="Where the run cost's rates come from (B)")
	job_cost: JobCost | None = Field(default=None, description="The run cost breakdown, written by B")

	@model_validator(mode="after")
	def _check(self):
		ids = [k.id for k in self.knobs]
		if len(set(ids)) != len(ids):
			raise ValueError("knob ids must be unique")
		known = set(ids)
		for p in self.probes:
			extra = set(p.knobs) - known
			if extra:
				raise ValueError(f"probe {p.id} sets undeclared knobs {sorted(extra)}")
		node_ids = {n.id for n in self.graph.nodes}
		for k in self.knobs:
			missing = [n for n in k.affects if node_ids and n not in node_ids]
			if missing:
				raise ValueError(f"knob {k.id} affects unknown nodes {missing}")
		for f in self.findings:
			if f.node_id is not None and node_ids and f.node_id not in node_ids:
				raise ValueError(f"finding {f.id} points at unknown node {f.node_id}")
		return self

	def knob(self, knob_id: str) -> Knob:
		for k in self.knobs:
			if k.id == knob_id:
				return k
		raise KeyError(knob_id)


def new_record(job: Job, **kw) -> FlightRecord:
	return FlightRecord(job=job, **kw)
