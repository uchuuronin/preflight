"""The script map: Muse Spark's structured reading of a user's script.

A's harness generator turns it into a probe harness, the job-map skeleton and
the declared knobs. The adapters (HF Trainer, TRL, Lightning, torchtitan, plain
loops) produce the same shape without a model call, so the rest of the system
never knows which path ran.
"""

from typing import Literal

from pydantic import Field

from .base import KnobValue, Lens, Strict

Framework = Literal[
	"hf_trainer",
	"trl",
	"lightning",
	"torchtitan",
	"plain_pytorch",
	"vllm",
	"hf_generate",
	"sentence_transformers",
	"pyg",
	"dgl",
	"timm",
	"openai_client",
	"other",
]


class CodeLocation(Strict):
	file: str
	line: int | None = None
	expr: str | None = Field(default=None, description="The literal expression that sets the value, if any")


class LoraInfo(Strict):
	r: int
	alpha: float | None = None
	target_modules: list[str] = Field(default_factory=list)
	dropout: float | None = None


class ModelInfo(Strict):
	name: str
	hf_id: str | None = None
	family: Literal["decoder", "encoder", "encoder_decoder", "cnn", "vit", "gnn", "mlp", "api", "other"] = (
		"other"
	)
	param_count: int | None = None
	n_layers: int | None = None
	d_model: int | None = None
	n_heads: int | None = None
	n_kv_heads: int | None = None
	d_ff: int | None = None
	vocab_size: int | None = None
	head_dim: int | None = None
	n_experts: int | None = Field(default=None, description="MoE experts per layer")
	experts_per_token: int | None = Field(default=None, description="MoE experts routed per token")
	max_positions: int | None = Field(default=None, description="Longest context the model was built for")
	gated_mlp: bool | None = Field(default=None, description="SwiGLU/GeGLU-style MLP with a gate projection")
	tie_embeddings: bool | None = Field(default=None, description="Output head shares the embedding matrix")
	norm: Literal["rmsnorm", "layernorm"] | None = None
	attn_impl: str | None = Field(default=None, description="eager, sdpa, flash_attention_2, ...")
	dtype: str | None = None
	quantization: str | None = Field(
		default=None,
		description="Weight format, as engine.memory keys it: nf4, nf4_dq, fp4, fp4_dq, int8, "
		"gptq_4bit, gptq_8bit, awq_4bit; none or absent for unquantised",
	)
	shape_source: Literal["code", "hf_config", "modules"] | None = Field(
		default=None, description="Where the shape fields came from: the code, or the running model"
	)
	lora: LoraInfo | None = None


class DataInfo(Strict):
	dataset: str | None = None
	batch_size: int | None = None
	eval_batch_size: int | None = None
	seq_len: int | None = None
	num_workers: int | None = None
	packing: bool | None = None
	padding: str | None = Field(default=None, description="longest, max_length, none")
	n_examples: int | None = None
	quality_scores: str | None = Field(
		default=None, description="Column or file holding per-document quality"
	)
	languages: list[str] = Field(default_factory=list)


class TrainInfo(Strict):
	optimizer: str | None = None
	lr: float | None = None
	warmup_steps: int | None = None
	grad_accum: int | None = None
	precision: Literal["fp32", "fp16", "bf16", "fp8", "mixed", "unknown"] = "unknown"
	max_steps: int | None = None
	epochs: float | None = None
	checkpoint_every: int | None = None
	eval_every: int | None = None
	grad_checkpointing: bool | None = None
	eval_under_no_grad: bool | None = None
	distributed: Literal["none", "ddp", "fsdp", "deepspeed", "tp", "other"] = "none"


class LoopInfo(Strict):
	kind: Literal[
		"hf_trainer",
		"trl_trainer",
		"lightning",
		"torchtitan",
		"plain",
		"vllm",
		"agent_loop",
		"eval_loop",
		"other",
	]
	location: CodeLocation | None = None
	step_call: str | None = Field(
		default=None, description="The call that advances one step, e.g. optimizer.step()"
	)
	model_var: str | None = None
	optimizer_var: str | None = None
	dataloader_var: str | None = None


class ScriptKnob(Strict):
	id: str = Field(pattern=r"^[a-z][a-z0-9_.]*$")
	label: str
	kind: Literal["int", "float", "categorical", "bool"]
	current: KnobValue | None = None
	unit: str | None = None
	choices: list[KnobValue] = Field(default_factory=list)
	location: CodeLocation | None = None
	needs_code: bool = Field(
		default=False, description="True when changing it means editing code, not a flag"
	)


class Block(Strict):
	id: str
	label: str
	kind: Literal["module", "data", "loss", "optimizer", "io", "phase", "model_call", "tool", "other"] = (
		"module"
	)
	parent: str | None = None
	repeat: int = 1
	params: int | None = None


class AgentInfo(Strict):
	client: str = Field(description="openai, anthropic, meta, litellm, ...")
	base_url_var: str | None = None
	model: str | None = None
	tools: list[str] = Field(default_factory=list)
	max_turns: int | None = None
	task_source: str | None = None
	n_tasks: int | None = None
	prompt_caching: bool | None = None


class RLInfo(Strict):
	algorithm: str = Field(description="grpo, ppo, dpo, rloo, ...")
	group_size: int | None = None
	max_completion_len: int | None = None
	reward_functions: list[str] = Field(default_factory=list)
	reference_model: bool | None = None
	generation_backend: Literal["hf", "vllm", "sglang", "other"] | None = None


class ScriptMap(Strict):
	lens: Lens
	framework: Framework
	entrypoint: str
	launch: str | None = Field(default=None, description="Command line that runs the script")
	model: ModelInfo | None = None
	data: DataInfo | None = None
	training: TrainInfo | None = None
	loop: LoopInfo | None = None
	knobs: list[ScriptKnob] = Field(default_factory=list)
	blocks: list[Block] = Field(default_factory=list)
	agent: AgentInfo | None = None
	rl: RLInfo | None = None
	risks: list[str] = Field(
		default_factory=list, description="Problems noticed while reading, in plain words"
	)
	produced_by: Literal["muse_spark", "adapter", "manual"] = "adapter"
	confidence: float = Field(default=1.0, ge=0, le=1)
