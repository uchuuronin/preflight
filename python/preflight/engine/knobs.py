"""What each knob does in the engine, and how a script map becomes engine inputs.

KNOBS covers every knob id A's harness declares (harness/skeleton.py) plus the
ones B needs: what it maps to, which readouts it moves, and whether the engine
models that yet. sensitivity() ranks only knobs it models; the UI says so for
the rest instead of inventing a number.

ARCH_HINTS fills in what the script map can't say about a model family
(gated MLP, tied embeddings, norm, learned positions, MoE). PRECISION maps the
precision knob and the model's load dtype to storage and compute dtypes.
WORK_UNITS gives each lens's calibration axis.

Pure Python, standard library only. Safe in Pyodide.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from typing import Literal

from preflight.engine.memory import Generation, GenericArch, Parallel, TrainingSetup, TransformerArch

Readout = Literal["memory", "time", "cost", "risk", "quality", "storage"]


@dataclass(frozen=True)
class KnobSpec:
	id: str
	maps_to: str  # engine input it sets, e.g. shape.batch, setup.optimizer, generation.new_tokens
	affects: frozenset[str]
	modelled: frozenset[str] = frozenset()  # readouts the engine computes today; the rest say "not modelled"
	note: str = ""


def _k(kid: str, maps_to: str, affects: str, modelled: str | None = None, note: str = "") -> KnobSpec:
	aff = frozenset(affects.split())
	mod = frozenset((affects if modelled is None else modelled).split())
	return KnobSpec(kid, maps_to, aff, mod & aff, note)


# fmt: off
_SPECS = [
	# shape of the work
	_k("batch_size", "shape.batch", "memory time cost risk", "memory time cost"),
	_k("eval_batch_size", "setup.eval_batch", "memory time cost"),
	_k("encode_batch_size", "shape.encode_batch", "memory time cost", note="Retrieval corpus encoding."),
	_k("seq_len", "shape.seq", "memory time cost"),
	_k("image_size", "shape.image", "memory time cost"),
	_k("grad_accum", "setup.grad_accum", "memory time cost risk", "memory time cost", "Effective batch = batch x accum x GPUs."),
	_k("max_steps", "run.steps", "time cost"),
	_k("epochs", "run.epochs", "time cost quality", "time cost"),
	_k("checkpoint_every", "run.checkpoint_every", "time cost risk", note="Save time, and work lost on preemption."),
	_k("eval_every", "run.eval_every", "time cost memory"),
	_k("log_every", "run.log_every", "time", "", "Logging cost is folded into step time; not modelled on its own."),
	_k("warmup_steps", "stability.warmup", "risk quality", "", "Feeds the divergence classifier (G2)."),
	# data loading
	_k("num_workers", "data.num_workers", "time"),
	_k("prefetch_factor", "data.prefetch", "time"),
	_k("packing", "data.packing", "time cost", "",
		"Removes padding, so real tokens per step rise; the probe's padding_fraction bounds the gain."),
	_k("group_by_length", "data.group_by_length", "time cost", "",
		"Batches similar lengths, so less padding; the peak is unchanged because one batch of the "
		"longest examples still happens. The probe's padding_fraction bounds the gain."),
	# model shape
	_k("n_layer", "arch.n_layers", "memory time cost quality", "memory time cost"),
	_k("d_model", "arch.d_model", "memory time cost quality", "memory time cost"),
	_k("n_head", "arch.n_heads", "memory time quality", "memory time"),
	_k("hidden_channels", "arch.hidden", "memory time cost quality", "memory time cost", "GNN width; calibrated from probes."),
	_k("gnn_layers", "arch.n_layers", "memory time cost quality", "memory time cost"),
	_k("lora_r", "setup.lora_rank", "memory time quality", "memory time"),
	_k("lora_alpha", "quality.lora_alpha", "quality", "", "Scales the update; no memory or time effect."),
	_k("lora_dropout", "quality.lora_dropout", "quality", "", "No memory or time effect worth modelling."),
	# precision and memory savers
	_k("precision", "setup.precision", "memory time cost risk", "memory time cost"),
	_k("dtype", "setup.precision", "memory time cost", note="Alias of precision."),
	_k("torch_dtype", "setup.precision", "memory time cost", note="Alias of precision."),
	_k("ptdtype", "setup.precision", "memory time cost", note="Alias of precision."),
	_k("quantization", "setup.quantization", "memory time quality", "memory time"),
	_k("optimizer", "setup.optimizer", "memory time risk quality", "memory time"),
	_k("gradient_checkpointing", "setup.grad_checkpointing", "memory time cost"),
	_k("eval_no_grad", "setup.eval_no_grad", "memory time"),
	_k("attn", "setup.flash_attention", "memory time"),
	_k("compile", "speed.compile", "time cost", "", "Speed-up measured by a verify run, not predicted."),
	_k("channels_last", "speed.channels_last", "time cost", "", "Speed-up measured by a verify run, not predicted."),
	# optimisation hyperparameters
	_k("lr", "stability.lr", "risk quality", "", "Feeds the LR range test and divergence classifier (G2)."),
	_k("lr_schedule", "stability.lr_schedule", "risk quality", ""),
	_k("weight_decay", "quality.weight_decay", "quality", ""),
	_k("grad_clip", "stability.grad_clip", "risk quality", ""),
	# hardware
	_k("gpu_type", "gpu", "memory time cost risk", "memory time cost risk"),
	_k("gpus", "parallel.world_size", "memory time cost"),
	_k("tensor_parallel", "parallel.tensor", "memory time cost"),
	_k("pipeline_parallel", "parallel.pipeline", "memory time cost"),
	# RL
	_k("group_size", "rl.group_size", "memory time cost risk", "memory time cost risk", "Zero-spread fraction p^G + (1-p)^G."),
	_k("max_completion_len", "generation.new_tokens", "memory time cost risk", "memory time cost risk", "Truncation share from the length tail."),
	_k("max_prompt_len", "generation.prompt_len", "memory time cost"),
	_k("rl_iterations", "rl.iterations", "time cost quality", "time cost"),
	_k("use_vllm", "generation.engine", "memory time cost"),
	_k("loss_type", "rl.loss_type", "risk quality", "", "DAPO/Dr. GRPO change length bias, not cost."),
	_k("kl_beta", "rl.kl_beta", "risk quality", ""),
	_k("clip_low", "rl.clip_low", "risk quality", ""),
	_k("clip_high", "rl.clip_high", "risk quality", ""),
	_k("temperature", "generation.temperature", "risk quality", "", "Changes pass rates; re-probe to measure."),
	# generation and serving
	_k("max_new_tokens", "generation.new_tokens", "memory time cost risk", "memory time cost risk"),
	_k("max_model_len", "serving.max_model_len", "memory time cost"),
	_k("max_num_seqs", "serving.max_num_seqs", "memory time cost"),
	_k("max_batched_tokens", "serving.max_batched_tokens", "memory time"),
	_k("gpu_memory_utilization", "generation.colocated_fraction", "memory time"),
	_k("kv_cache_dtype", "generation.kv_dtype", "memory time quality", "memory time"),
	# retrieval
	_k("index_m", "retrieval.index_m", "memory time quality", "memory time"),
	_k("index_nlist", "retrieval.index_nlist", "time quality", "time"),
	# agents
	_k("model", "agent.model", "time cost risk quality", "time cost risk"),
	_k("prompt_caching", "agent.prompt_caching", "cost time", "", "Cache pricing comes from the API price table."),
	_k("concurrency", "agent.concurrency", "time risk", "time risk", "Rate limits cap the gain."),
	_k("max_turns", "agent.max_turns", "cost time risk quality", "cost time risk"),
	_k("n_tasks", "agent.n_tasks", "cost time"),
	_k("tool_retry_cap", "agent.tool_retry_cap", "cost time risk", "", "Retry loops are counted, not yet simulated."),
	_k("reasoning_effort", "agent.reasoning_effort", "cost time quality", "cost time"),
	# distributed and kernels
	_k("fsdp_mode", "parallel.strategy", "memory time cost", note="full_shard, shard_grad_op, hybrid_shard."),
	_k("deepspeed_config", "parallel.strategy", "memory time cost", note="ZeRO stage and offload read from the config."),
	_k("tf32", "speed.tf32", "time cost", note="Picks the fp32 matmul peak for time and MFU."),
	_k("pin_memory", "data.pin_memory", "time", "", "Host-to-device copy speed; folded into the data phase."),
	_k("prefix_caching", "serving.prefix_caching", "time cost", "", "Serving model (G2) uses the measured hit rate."),
	# storage (eval sweeps that dump activations, checkpoints)
	_k("dump_hidden_states", "storage.hidden_states", "storage time", "", "Disk per row x layers x positions; storage model lands with campaign mode (G2)."),
	_k("dump_attentions", "storage.attentions", "storage time memory", "", "Attention maps are b x heads x s^2 per layer; storage model lands with campaign mode (G2)."),
	_k("context_limit", "agent.context_limit", "cost risk quality", "cost risk", "Context overflow at step k; summarisation trades cost."),
]
# fmt: on

KNOBS: dict[str, KnobSpec] = {s.id: s for s in _SPECS}


def knob_spec(knob_id: str) -> KnobSpec:
	"""The spec for a knob id; fanout.<layer> knobs share one; unknown ids are honest no-ops."""
	if knob_id in KNOBS:
		return KNOBS[knob_id]
	if knob_id.startswith("fanout."):
		return KnobSpec(
			knob_id,
			"gnn.fanout",
			frozenset({"memory", "time", "cost", "quality"}),
			frozenset({"memory", "time", "cost"}),
		)
	return KnobSpec(knob_id, "unknown", frozenset(), frozenset(), "Preflight doesn't model this knob yet.")


def modelled(knob_id: str, readout: str) -> bool:
	return readout in knob_spec(knob_id).modelled


# ---------------------------------------------------------------- precision


@dataclass(frozen=True)
class Precision:
	param_dtype: str
	compute_dtype: str
	master_weights: bool
	grad_scaler: bool = False


def precision_setup(precision: str | None, model_dtype: str | None = None) -> Precision:
	"""Storage and compute dtypes from the precision knob and the dtype the model loads in.

	AMP (HF bf16=True with an fp32 model) keeps fp32 weights and runs bf16 kernels; loading
	the model in bf16 makes it pure bf16 with bf16 optimizer states.
	"""
	loaded = (model_dtype or "fp32").lower().replace("torch.", "")
	loaded = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32", "auto": "bf16"}.get(loaded, loaded)
	p = (precision or "unknown").lower().replace("torch.", "")
	p = {"bfloat16": "bf16", "float16": "fp16", "half": "fp16", "float32": "fp32", "mixed": "bf16_amp"}.get(
		p, p
	)
	if p == "unknown":
		# nothing says how it trains: it runs in the dtype it loads in
		p = loaded if loaded in ("fp32", "bf16", "fp16") else "fp32"
	if p == "fp32":
		return Precision("fp32", "fp32", False)
	if p in ("bf16", "fp16", "bf16_amp"):
		compute = "fp16" if p == "fp16" else "bf16"
		if loaded in ("bf16", "fp16") and p != "bf16_amp":
			return Precision(loaded, compute, False, grad_scaler=compute == "fp16")
		return Precision("fp32", compute, False, grad_scaler=compute == "fp16")
	if p == "fp8":
		return Precision(
			"bf16", "bf16", True
		)  # FP8 GEMMs keep bf16 weights, an fp32 master and mostly bf16 activations
	raise ValueError(f"unknown precision {precision!r}")


# ---------------------------------------------------------------- model families


@dataclass(frozen=True)
class ArchHint:
	gated_mlp: bool
	norm: Literal["layernorm", "rmsnorm"]
	bias: bool
	tie_embeddings: bool | None  # None: tied below 4B parameters (Qwen-style)
	learned_positions: int = 0
	n_experts: int = 1
	experts_per_token: int = 1


# Most specific first, matched against the model name or Hugging Face id.
ARCH_HINTS: list[tuple[str, ArchHint]] = [
	(r"mixtral", ArchHint(True, "rmsnorm", False, False, n_experts=8, experts_per_token=2)),
	(r"qwen.*moe|qwen3.*a\d+b", ArchHint(True, "rmsnorm", False, False, n_experts=64, experts_per_token=8)),
	# MoE DeepSeek models only; the R1 distills are dense Qwen/Llama models and match below
	(
		r"deepseek-(v2|v3|moe|coder-v2)|deepseek-r1(?!-distill)",
		ArchHint(True, "rmsnorm", False, False, n_experts=64, experts_per_token=6),
	),
	(r"olmoe", ArchHint(True, "rmsnorm", False, False, n_experts=64, experts_per_token=8)),
	(
		r"llama-?3\.2-(1|3)b",
		ArchHint(True, "rmsnorm", False, True),
	),  # the small Llama 3.2 models tie embeddings
	(r"llama|mistral|olmo|smollm|tinyllama|vicuna|yi-", ArchHint(True, "rmsnorm", False, False)),
	(r"qwen", ArchHint(True, "rmsnorm", False, None)),
	(r"gemma", ArchHint(True, "rmsnorm", False, True)),
	(r"phi-?3|phi-?4", ArchHint(True, "rmsnorm", False, False)),
	(r"phi", ArchHint(False, "layernorm", True, False)),
	(r"gpt-?2|distilgpt2|nanogpt", ArchHint(False, "layernorm", True, True, learned_positions=1024)),
	(r"gpt-?neox|pythia", ArchHint(False, "layernorm", True, False)),
	(r"opt-", ArchHint(False, "layernorm", True, True, learned_positions=2048)),
	(
		r"bert|roberta|minilm|mpnet|e5-|bge-|gte-",
		ArchHint(False, "layernorm", True, True, learned_positions=512),
	),
	(r"t5|flan", ArchHint(True, "rmsnorm", False, False)),
	(r"vit|deit|clip|siglip", ArchHint(False, "layernorm", True, True)),
	(r"muse", ArchHint(True, "rmsnorm", False, False)),
]
DEFAULT_HINT = ArchHint(True, "rmsnorm", False, False)


def arch_hint(name: str | None) -> ArchHint:
	s = (name or "").lower()
	for pat, hint in ARCH_HINTS:
		if re.search(pat, s):
			return hint
	return DEFAULT_HINT


def arch_from_model_info(info) -> TransformerArch | GenericArch:
	"""A schema ModelInfo (or anything with the same attributes) to an engine architecture.

	Falls back to GenericArch when the transformer shape is incomplete; the probes then
	carry activations through calibration.
	"""
	family = getattr(info, "family", "other")
	need = ("n_layers", "d_model", "n_heads", "vocab_size")
	if family in ("decoder", "encoder", "encoder_decoder", "vit") and all(
		getattr(info, k, None) for k in need
	):
		hint = _measured(info, arch_hint(getattr(info, "hf_id", None) or getattr(info, "name", None)))
		pc = getattr(info, "param_count", None)
		measured_moe = getattr(info, "n_experts", None) is not None
		if not measured_moe and hint.n_experts > 1 and pc and _hint_params(info, hint) > 2 * pc:
			# the name looked MoE but the stated size says dense: trust the size
			hint = ArchHint(hint.gated_mlp, hint.norm, hint.bias, hint.tie_embeddings, hint.learned_positions)
		tie = hint.tie_embeddings if hint.tie_embeddings is not None else (pc is None or pc < 4e9)
		d_ff = getattr(info, "d_ff", None) or _solve_d_ff(info, hint, tie, pc)
		arch = TransformerArch(
			n_layers=info.n_layers,
			d_model=info.d_model,
			n_heads=info.n_heads,
			n_kv_heads=getattr(info, "n_kv_heads", None),
			d_ff=d_ff,
			vocab_size=info.vocab_size,
			gated_mlp=hint.gated_mlp,
			norm=hint.norm,
			bias=hint.bias,
			tie_embeddings=tie,
			learned_positions=hint.learned_positions,
			n_experts=hint.n_experts,
			experts_per_token=hint.experts_per_token,
			head_dim_override=_head_dim(info),
		)
		if family == "encoder_decoder":
			# decoder layers add cross-attention: roughly 1.33x an encoder layer; n_layers counts both stacks
			arch = TransformerArch(**{**arch.__dict__, "n_layers": math.ceil(arch.n_layers * 1.17)})
		return arch
	pc = getattr(info, "param_count", None)
	if not pc:
		raise ValueError("the script map has neither a full transformer shape nor a parameter count")
	return GenericArch(
		n_params=pc,
		vocab_size=getattr(info, "vocab_size", None) or 0,
		n_layers=getattr(info, "n_layers", None) or 1,
	)


def _measured(info, hint: ArchHint) -> ArchHint:
	"""Shape fields A read from the running model (or the code) beat what the name suggests."""
	n_exp = getattr(info, "n_experts", None)
	per_tok = getattr(info, "experts_per_token", None)
	return replace(
		hint,
		gated_mlp=hint.gated_mlp if getattr(info, "gated_mlp", None) is None else info.gated_mlp,
		norm=getattr(info, "norm", None) or hint.norm,
		tie_embeddings=hint.tie_embeddings
		if getattr(info, "tie_embeddings", None) is None
		else info.tie_embeddings,
		n_experts=hint.n_experts if n_exp is None else max(1, n_exp),
		experts_per_token=_per_token(n_exp, per_tok, hint),
	)


def _per_token(n_exp: int | None, per_tok: int | None, hint: ArchHint) -> int:
	"""Experts routed per token: measured, else the name's hint, never more than the experts there are."""
	if n_exp is None:
		return hint.experts_per_token
	if n_exp <= 1:
		return 1
	return max(1, min(n_exp, per_tok or hint.experts_per_token or 1))


def _head_dim(info) -> int | None:
	hd = getattr(info, "head_dim", None)
	return hd if hd and hd != info.d_model // info.n_heads else None


def _hint_params(info, hint: ArchHint) -> int:
	return TransformerArch(
		n_layers=info.n_layers, d_model=info.d_model, n_heads=info.n_heads,
		n_kv_heads=getattr(info, "n_kv_heads", None), vocab_size=info.vocab_size,
		d_ff=getattr(info, "d_ff", None) or 4 * info.d_model, gated_mlp=hint.gated_mlp,
		n_experts=hint.n_experts, experts_per_token=hint.experts_per_token, head_dim_override=_head_dim(info),
	).param_count()  # fmt: skip


def _solve_d_ff(info, hint: ArchHint, tie: bool, pc: int | None) -> int | None:
	"""The MLP width that makes the parameter count add up, when the script map gives the count but not d_ff."""
	if not pc:
		return None
	probe = TransformerArch(
		n_layers=info.n_layers,
		d_model=info.d_model,
		n_heads=info.n_heads,
		n_kv_heads=getattr(info, "n_kv_heads", None),
		vocab_size=info.vocab_size,
		d_ff=1,
		gated_mlp=hint.gated_mlp,
		norm=hint.norm,
		bias=hint.bias,
		tie_embeddings=tie,
		learned_positions=hint.learned_positions,
		n_experts=hint.n_experts,
		head_dim_override=_head_dim(info),
	)
	fixed = probe.param_count()
	mats = (3 if hint.gated_mlp else 2) * hint.n_experts
	per_ff = info.n_layers * mats * info.d_model + (info.n_layers * hint.n_experts * (1 if hint.bias else 0))
	ff = (pc - fixed) / per_ff + 1 if per_ff else 0
	if ff <= info.d_model / 4:
		return None  # the count doesn't fit this family's shape; keep the 4h default
	return int(round(ff / 64) * 64) if ff > 256 else round(ff)


# ---------------------------------------------------------------- work units


def work_units(lens: str, values: dict) -> float:
	"""The calibration axis per lens: what memory and step time grow with."""
	b = float(values.get("batch_size") or values.get("encode_batch_size") or 1)
	if lens in ("finetune", "pretrain", "rl", "eval", "serving", "retrieval"):
		s = values.get("seq_len") or values.get("max_model_len")
		if lens == "rl":
			s = (values.get("max_prompt_len") or 0) + (values.get("max_completion_len") or 0) or s
		return b * float(s or 1)
	if lens == "vision":
		img = float(values.get("image_size") or 224)
		return b * img * img
	if lens == "gnn":
		fan = [float(v) for k, v in values.items() if k.startswith("fanout.") and v]
		return b * (math.prod(fan) if fan else 1.0)
	return b


# ---------------------------------------------------------------- engine inputs from knob values


_GC = {True: "full", False: "none", "full": "full", "selective": "selective", "none": "none"}
_ATTN_FLASH = {
	"eager": False,
	"math": False,
	"sdpa": True,
	"flash_attention_2": True,
	"flash_attention_3": True,
	"flex_attention": True,
	"flash": True,
	"flash_attention": True,
	"mem_efficient": True,
	"efficient": True,
	"xformers": True,
	"paged_attention": True,
	"sdpa_math": False,
}


@dataclass(frozen=True)
class Shape:
	batch: int = 1
	seq: int = 1
	image: int | None = None
	values: dict = field(default_factory=dict)


def setup_from(script_map, values: dict) -> tuple[TransformerArch | GenericArch, TrainingSetup, Shape]:
	"""Engine inputs for one knob setting. `values` are knob id -> value, already merged with the record's current values."""
	model = getattr(script_map, "model", None)
	training = getattr(script_map, "training", None)
	rl = getattr(script_map, "rl", None)
	if model is None:
		raise ValueError("the script map has no model")
	arch = arch_from_model_info(model)
	if isinstance(arch, TransformerArch):
		over = {
			k: values[v]
			for v, k in (("n_layer", "n_layers"), ("d_model", "d_model"), ("n_head", "n_heads"))
			if v in values
		}
		if over:
			arch = TransformerArch(**{**arch.__dict__, **over})

	prec_val = next(
		(values[k] for k in ("precision", "dtype", "torch_dtype", "ptdtype") if k in values), None
	)
	prec = precision_setup(prec_val or (training.precision if training else None), model.dtype)
	quant = values.get("quantization", model.quantization)
	lora = model.lora
	method = "full"
	if lora is not None or "lora_r" in values:
		method = "qlora" if quant and quant.lower() not in ("none", "") else "lora"
	targets = tuple(_lora_target(t) for t in (lora.target_modules if lora else [])) or ("q", "v")

	gc = values.get("gradient_checkpointing", training.grad_checkpointing if training else False)
	attn = values.get("attn", getattr(model, "attn_impl", None))
	dist = (training.distributed if training else "none") or "none"
	strategy = _strategy(dist, values.get("fsdp_mode"), values.get("deepspeed_config"))
	world = int(values.get("gpus", 1) or 1)
	parallel = Parallel(
		strategy=strategy if world > 1 else "none",
		world_size=world,
		tensor=int(values.get("tensor_parallel", 1) or 1),
		pipeline=int(values.get("pipeline_parallel", 1) or 1),
	)

	generation = None
	new_tokens = values.get("max_completion_len") or values.get("max_new_tokens")
	if new_tokens and (rl is not None or "max_new_tokens" in values):
		seqs = int(values.get("batch_size", 1) or 1) * int(values.get("group_size", 1) or 1 if rl else 1)
		engine = "vllm" if values.get("use_vllm") or (rl and rl.generation_backend == "vllm") else "hf"
		generation = Generation(
			sequences=seqs,
			prompt_len=int(values.get("max_prompt_len") or values.get("seq_len") or 512),
			new_tokens=int(new_tokens),
			kv_dtype="fp8"
			if str(values.get("kv_cache_dtype", "")).startswith("fp8")
			else (prec.compute_dtype),
			engine=engine,
			colocated_fraction=values.get("gpu_memory_utilization") if engine == "vllm" else None,
		)

	setup = TrainingSetup(
		param_dtype=prec.param_dtype,
		compute_dtype=prec.compute_dtype,
		quantization=quant,  # frozen quantised models (eval, serving) count it too
		method=method if (training is not None or lora is not None) else "frozen",
		lora_rank=int(values.get("lora_r", lora.r if lora else 0) or 0),
		lora_targets=targets,
		optimizer=str(
			values.get("optimizer", training.optimizer if training and training.optimizer else "adamw")
		),
		master_weights=prec.master_weights,
		grad_checkpointing=_GC.get(gc, "none"),
		flash_attention=_ATTN_FLASH.get(str(attn).lower(), True) if attn is not None else True,
		grad_accum=int(
			values.get("grad_accum", training.grad_accum if training and training.grad_accum else 1) or 1
		),
		parallel=parallel,
		eval_batch=values.get("eval_batch_size"),
		eval_no_grad=bool(
			values.get(
				"eval_no_grad",
				training.eval_under_no_grad if training and training.eval_under_no_grad is not None else True,
			)
		),
		generation=generation,
		reference_model=bool(rl and rl.reference_model),
	)
	shape = Shape(
		batch=int(values.get("batch_size") or values.get("encode_batch_size") or 1),
		seq=int(
			values.get("seq_len") or (generation.prompt_len + generation.new_tokens if generation else 1)
		),
		image=values.get("image_size"),
		values=dict(values),
	)
	return arch, setup, shape


_FSDP_MODES = {
	"full_shard": "fsdp",
	"shard_grad_op": "fsdp_grad_op",
	"hybrid_shard": "hsdp",
	"no_shard": "ddp",
}


def _strategy(dist: str, fsdp_mode, deepspeed_config) -> str:
	"""Distributed strategy from the script map plus the fsdp_mode / deepspeed_config knobs."""
	if fsdp_mode:
		return _FSDP_MODES.get(str(fsdp_mode).lower(), "fsdp")
	if dist == "deepspeed" or deepspeed_config is not None:
		m = re.search(r"([0-3])", str(deepspeed_config)) if deepspeed_config is not None else None
		stage = (
			int(m.group(1)) if m else 2
		)  # DeepSpeed's own default is ZeRO-0, but scripts that use it set a stage
		return {0: "ddp", 1: "zero1", 2: "zero2", 3: "zero3"}[stage]
	return {"ddp": "ddp", "fsdp": "fsdp", "tp": "none", "none": "none"}.get(dist, "none")


_LORA_NAMES = {
	"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o", "out_proj": "o",
	"gate_proj": "gate", "up_proj": "up", "down_proj": "down",
	"c_attn": "q", "c_proj": "o", "c_fc": "up", "query": "q", "key": "k", "value": "v", "dense": "o",
	"w1": "gate", "w3": "up", "w2": "down", "fc1": "up", "fc2": "down",
	"all-linear": "all-linear", "all_linear": "all-linear",
}  # fmt: skip


def _lora_target(name: str) -> str:
	return _LORA_NAMES.get(name.split(".")[-1].lower(), name.lower())
