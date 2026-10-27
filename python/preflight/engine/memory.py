"""Analytical GPU memory model, calibrated on probe peaks, with P(OOM).

Method (research map, section 1; settings from docs/decisions/engine-settings.md):
- static terms: weights by dtype and quantisation, master weights, gradients,
  optimizer states (absent until the first optimizer step), each sharded per
  the distributed strategy (DDP, FSDP, ZeRO-1/2/3, tensor and pipeline
  parallel), with DDP's gradient buckets and FSDP's all-gather buffers
- activations: Korthikanti et al. (MLSys 2023), itemised per layer; reduces to
  s*b*h*(34 + 5*a*s/h) for GPT layers with dropout and no FlashAttention.
  FlashAttention drops the s^2 terms; selective and full recompute as in the
  paper; tensor parallel splits the parts Megatron splits
- logits: b*s*V in the compute dtype plus the fp32 upcast and saved
  log-softmax that HF-style losses keep, often the hidden OOM at long sequences
- moments: training step, eval pass (eval batch, with or without no_grad),
  generation (KV cache for RL rollouts and eval sweeps; vLLM's reserved
  fraction when it shares the GPU); the peak is the worst moment
- RL extras: a frozen reference model (none under LoRA, where the reference is
  the base with adapters off) and a PPO value model with its own optimizer
- calibration (S1, chosen E3): residual = measured reserved - analytical,
  least squares in tokens over the probes
- band (S2a, chosen R5): 90% half-width in log space of 0.107, widened to 1.5x
  the leave-one-probe-out error; 0.107 with two probes and 0.160 with one (S2c);
  0.241 with no probe; abstain outside the probed range or past +-50%
- words (S2b): fits below P(OOM) 1%, fails at 80% or more, tight between

Pure Python: imports only the standard library and the GPU table. Safe in Pyodide.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Literal, Protocol

from preflight.engine import gpus

MODEL_ID = "memory-analytical@0.2.0"
DEFAULT_CONTEXT_BYTES = int(0.75 * 2**30)  # CUDA/ROCm context, typically 0.5-1 GB
Z90 = 1.6448536269514722
HALF_WIDTH_CALIBRATED = 0.107  # S2a R3: q90 |log error| with three probes
HALF_WIDTH_ONE_PROBE = 0.160  # S2c: one usable probe (88.1% coverage); two probes keep 0.107 (91%)
HALF_WIDTH_PRIOR = 0.241  # analytical only, no probe
LOO_WIDEN = 1.5
ABSTAIN_HALF_WIDTH = math.log(1.5)  # band wider than +-50%
EXTRAPOLATE_FACTOR = 2.0  # past 2x the largest probed work units
FITS_BELOW = 0.01
FAILS_AT = 0.80

# ---------------------------------------------------------------- dtypes and quantisation

_BYTES = {"fp32": 4.0, "tf32": 4.0, "bf16": 2.0, "fp16": 2.0, "fp8": 1.0}
# Quantised weight formats: bytes per weight, constants included.
# NF4/FP4 (bitsandbytes, QLoRA): 4 bits + one fp32 absmax per 64 weights; double quant cuts that to 0.127 bits.
# LLM.int8: 1 byte + one fp32 scale per row (negligible). GPTQ/AWQ 4-bit: 4 bits + fp16 scale and zero per 128-group.
QUANT_BYTES = {
	"nf4": 0.5 + 32 / 64 / 8,
	"fp4": 0.5 + 32 / 64 / 8,
	"nf4_dq": 0.5 + 0.127 / 8,
	"fp4_dq": 0.5 + 0.127 / 8,
	"int8": 1.0,
	"llm_int8": 1.0,
	"fp8": 1.0,
	"gptq_4bit": 0.5 + 32 / 128 / 8,
	"awq_4bit": 0.5 + 32 / 128 / 8,
	"gptq_8bit": 1.0 + 32 / 128 / 8,
	"hqq_4bit": 0.5 + 32 / 64 / 8,
}
_QUANT_ALIASES = {
	"4bit-nf4": "nf4",
	"4bit_nf4": "nf4",
	"bnb_4bit": "nf4",
	"4bit": "nf4",
	"nf4-dq": "nf4_dq",
	"4bit-fp4": "fp4",
	"8bit": "int8",
	"bnb_8bit": "int8",
	"int8": "int8",
	"gptq": "gptq_4bit",
	"awq": "awq_4bit",
}


def quant_key(q: str | None) -> str | None:
	if q is None:
		return None
	k = q.lower().strip()
	if k in ("none", "", "no", "fp16", "bf16", "fp32"):
		return None
	k = _QUANT_ALIASES.get(k, k)
	if k not in QUANT_BYTES:
		raise ValueError(f"unknown quantisation {q!r}")
	return k


def dtype_bytes(dtype: str, double_quant: bool = False) -> float:
	d = dtype.lower().replace("torch.", "")
	if d in ("nf4", "fp4", "int4") and double_quant:
		return QUANT_BYTES[f"{'nf4' if d == 'int4' else d}_dq"]
	q = _QUANT_ALIASES.get(d, d)
	if q in QUANT_BYTES and q not in _BYTES:
		return QUANT_BYTES[q]
	if d == "int4":
		return QUANT_BYTES["nf4"]
	if d in _BYTES:
		return _BYTES[d]
	return _BYTES[gpus.normalise_dtype(d)]


# ---------------------------------------------------------------- optimizers

# id -> (states per trainable parameter, fixed bytes per state or None for the state dtype)
# State dtype is fp32 under master weights or fp32 params, otherwise the param dtype (torch's zeros_like).
OPTIMIZERS: dict[str, tuple[int, float | None]] = {
	"adamw": (2, None),
	"adam": (2, None),
	"adamw_torch": (2, None),
	"adamw_torch_fused": (2, None),
	"adamw_apex_fused": (2, None),
	"adamw_hf": (2, None),
	"paged_adamw_32bit": (2, 4.0),
	"adamw_8bit": (2, 1.0),
	"adamw_bnb_8bit": (2, 1.0),
	"paged_adamw_8bit": (2, 1.0),
	"lion": (1, None),
	"lion_32bit": (1, 4.0),
	"lion_8bit": (1, 1.0),
	"paged_lion_8bit": (1, 1.0),
	"sgd": (0, None),
	"sgd_momentum": (1, None),
	"rmsprop": (1, None),
	"adagrad": (1, None),
	"muon": (1, None),
	"adafactor": (0, None),  # factored second moment: see _adafactor_bytes
}
_OPTIMIZER_ALIASES = {
	"adamw_torch_4bit": "adamw_8bit",
	"adamw_anyprecision": "adamw",
	"galore_adamw": "adamw",
}


def optimizer_key(name: str) -> tuple[str, bool]:
	"""(table key, known). Unknown optimizers fall back to AdamW, the conservative choice."""
	k = name.lower().strip()
	k = _OPTIMIZER_ALIASES.get(k, k)
	return (k, True) if k in OPTIMIZERS else ("adamw", False)


# ---------------------------------------------------------------- architectures


class Arch(Protocol):
	n_layers: int
	vocab_size: int

	def param_count(self) -> int: ...
	def param_breakdown(self) -> dict[str, int]: ...


@dataclass(frozen=True)
class TransformerArch:
	n_layers: int
	d_model: int
	n_heads: int
	vocab_size: int
	d_ff: int | None = None
	n_kv_heads: int | None = None
	tie_embeddings: bool = True
	gated_mlp: bool = False
	bias: bool = False
	norm: Literal["layernorm", "rmsnorm"] = "rmsnorm"
	learned_positions: int = 0
	head_dim_override: int | None = None
	n_experts: int = 1  # MoE: experts per layer
	experts_per_token: int = 1

	@property
	def ff(self) -> int:
		return self.d_ff if self.d_ff is not None else 4 * self.d_model

	@property
	def kv_heads(self) -> int:
		return self.n_kv_heads if self.n_kv_heads is not None else self.n_heads

	@property
	def head_dim(self) -> int:
		return self.head_dim_override or self.d_model // self.n_heads

	def linear_shapes(self) -> dict[str, tuple[int, int]]:
		"""(in, out) for each projection in one layer (one expert for MoE MLPs)."""
		h, q, kv = self.d_model, self.n_heads * self.head_dim, self.kv_heads * self.head_dim
		shapes = {"q": (h, q), "k": (h, kv), "v": (h, kv), "o": (q, h)}
		if self.gated_mlp:
			shapes |= {"gate": (h, self.ff), "up": (h, self.ff), "down": (self.ff, h)}
		else:
			shapes |= {"up": (h, self.ff), "down": (self.ff, h)}
		return shapes

	def param_breakdown(self) -> dict[str, int]:
		h, v = self.d_model, self.vocab_size
		attn = mlp = 0
		for name, (din, dout) in self.linear_shapes().items():
			n = din * dout + (dout if self.bias else 0)
			if name in ("q", "k", "v", "o"):
				attn += n
			else:
				mlp += n
		router = h * self.n_experts if self.n_experts > 1 else 0
		norm_size = 2 * h if self.norm == "layernorm" else h
		return {
			"embedding": v * h + self.learned_positions * h,
			"head": 0 if self.tie_embeddings else v * h,
			"linear": self.n_layers * (attn + mlp * self.n_experts + router),
			"norm": (2 * self.n_layers + 1) * norm_size,
		}

	def param_count(self) -> int:
		return sum(self.param_breakdown().values())

	def active_linear_params(self) -> int:
		"""Linear parameters a token touches (MoE: only its experts)."""
		pb = self.param_breakdown()
		if self.n_experts <= 1:
			return pb["linear"]
		mlp_one = sum(
			din * dout for n, (din, dout) in self.linear_shapes().items() if n not in ("q", "k", "v", "o")
		)
		dense = pb["linear"] - self.n_layers * mlp_one * self.n_experts
		return dense + self.n_layers * mlp_one * self.experts_per_token

	def largest_layer_params(self) -> int:
		return self.param_breakdown()["linear"] // max(self.n_layers, 1)


@dataclass(frozen=True)
class GenericArch:
	"""A model known only by its parameter counts (CNN, GNN, custom); activations come from probes."""

	n_params: int
	vocab_size: int = 0
	n_layers: int = 1

	def param_breakdown(self) -> dict[str, int]:
		return {"embedding": 0, "head": 0, "linear": self.n_params, "norm": 0}

	def param_count(self) -> int:
		return self.n_params

	def largest_layer_params(self) -> int:
		return self.n_params // max(self.n_layers, 1)


# ---------------------------------------------------------------- training setup

Strategy = Literal["none", "ddp", "fsdp", "fsdp_grad_op", "hsdp", "zero1", "zero2", "zero3"]


@dataclass(frozen=True)
class Parallel:
	strategy: Strategy = "none"
	world_size: int = 1  # data-parallel GPUs
	tensor: int = 1
	pipeline: int = 1
	offload_optimizer: bool = False
	offload_params: bool = False
	ddp_bucket_view: bool = False  # gradient_as_bucket_view: DDP buckets alias the grads
	shard_group: int | None = None  # HSDP: GPUs per shard group (a node); replicated across groups

	def shards(self) -> tuple[int, int, int]:
		"""How many ways (params, grads, optimizer states) are split across data-parallel ranks."""
		w = max(self.world_size, 1)
		s = self.strategy
		if s == "hsdp":
			g = min(self.shard_group or min(w, 8), w)
			return g, g, g
		if s in ("fsdp", "zero3"):
			return w, w, w
		if s in ("fsdp_grad_op", "zero2"):
			return 1, w, w
		if s == "zero1":
			return 1, 1, w
		return 1, 1, 1


@dataclass(frozen=True)
class Generation:
	"""Generation inside the job: RL rollouts, eval sweeps, serving probes."""

	sequences: int  # in flight at once
	prompt_len: int
	new_tokens: int
	kv_dtype: str = "bf16"
	engine: Literal["hf", "vllm", "sglang", "api", "other"] = "hf"
	colocated_fraction: float | None = None  # vLLM gpu_memory_utilization when it shares the training GPU


@dataclass(frozen=True)
class TrainingSetup:
	param_dtype: str = "bf16"  # storage dtype of base weights
	compute_dtype: str = "bf16"  # activation dtype
	quantization: str | None = None  # nf4, int8, gptq_4bit ... for frozen base weights
	method: Literal["full", "lora", "qlora", "frozen"] = "full"
	lora_rank: int = 0
	lora_targets: tuple[str, ...] = ("q", "v")
	adapter_dtype: str = "fp32"
	optimizer: str = "adamw"
	master_weights: bool = False  # fp32 master copy (mixed precision with low-precision params)
	grad_checkpointing: Literal["none", "selective", "full"] = "none"
	flash_attention: bool = True
	dropout: bool = False
	double_quant: bool = False
	grad_accum: int = 1
	parallel: Parallel = field(default_factory=Parallel)
	eval_batch: int | None = None
	eval_no_grad: bool = True
	generation: Generation | None = None
	reference_model: bool = False  # RL: a frozen reference policy
	value_model: bool = False  # PPO: a value head model with its own optimizer
	logits_fp32_upcast: bool = True  # HF losses upcast logits to fp32


def adapter_params(arch: Arch, setup: TrainingSetup) -> int:
	if setup.method not in ("lora", "qlora") or not isinstance(arch, TransformerArch):
		return 0
	shapes = arch.linear_shapes()
	targets = setup.lora_targets
	if "all-linear" in targets or "all_linear" in targets:
		targets = tuple(shapes)
	per_layer = 0
	for t in targets:
		if t in shapes:
			din, dout = shapes[t]
			experts = arch.n_experts if t in ("gate", "up", "down") else 1
			per_layer += setup.lora_rank * (din + dout) * experts
	return arch.n_layers * per_layer


def trainable_params(arch: Arch, setup: TrainingSetup) -> int:
	if setup.method == "frozen":
		return 0
	if setup.method in ("lora", "qlora"):
		return adapter_params(arch, setup)
	return arch.param_count()


def _adafactor_bytes(arch: Arch) -> float:
	"""Row and column second-moment factors in fp32, plus full state for vectors."""
	if not isinstance(arch, TransformerArch):
		return 0.02 * arch.param_count()
	pb = arch.param_breakdown()
	per_layer = sum(4.0 * (din + dout) for din, dout in arch.linear_shapes().values())
	return arch.n_layers * per_layer + 4.0 * (arch.vocab_size + arch.d_model) + 4.0 * pb["norm"]


def _weights_bytes(arch: Arch, setup: TrainingSetup) -> float:
	n = arch.param_count()
	q = quant_key(setup.quantization) or (
		quant_key(setup.param_dtype)
		if setup.param_dtype.lower() in QUANT_BYTES or setup.param_dtype.lower() in _QUANT_ALIASES
		else None
	)
	if setup.method == "qlora" and q is None:
		q = "nf4_dq" if setup.double_quant else "nf4"
	if q is not None:
		quantised = arch.param_breakdown()["linear"]
		qb = QUANT_BYTES[f"{q}_dq"] if setup.double_quant and f"{q}_dq" in QUANT_BYTES else QUANT_BYTES[q]
		return quantised * qb + (n - quantised) * dtype_bytes(setup.compute_dtype)
	return n * dtype_bytes(setup.param_dtype)


def static_terms(arch: Arch, setup: TrainingSetup, after_optimizer_step: bool = True) -> dict[str, float]:
	"""Per-GPU bytes for weights, master weights, gradients, optimizer, frozen copies and comm buffers."""
	par = setup.parallel
	p_shard, g_shard, o_shard = par.shards()
	mp = max(par.tensor, 1) * max(par.pipeline, 1)  # model-parallel split of everything
	trainable = trainable_params(arch, setup)
	adapters = adapter_params(arch, setup)
	adapter_b = dtype_bytes(setup.adapter_dtype)

	base = _weights_bytes(arch, setup)
	weights = base / (p_shard * mp) + adapters * adapter_b / mp
	if par.offload_params:
		weights = adapters * adapter_b / mp

	lora_like = setup.method in ("lora", "qlora")
	trainable_b = adapter_b if lora_like else dtype_bytes(setup.param_dtype)
	master = 4.0 * trainable / (o_shard * mp) if setup.master_weights and trainable_b < 4 else 0.0

	grads = trainable * trainable_b / (g_shard * mp)

	key, _ = optimizer_key(setup.optimizer)
	n_states, fixed = OPTIMIZERS[key]
	state_b = (
		fixed if fixed is not None else (4.0 if (setup.master_weights or trainable_b >= 4) else trainable_b)
	)
	if key == "adafactor":
		optimizer = _adafactor_bytes(arch) if not lora_like else 4.0 * adapters * 0.05
	else:
		optimizer = n_states * state_b * trainable
	optimizer /= o_shard * mp
	if par.offload_optimizer:
		optimizer, master = 0.0, 0.0
	if not after_optimizer_step:
		optimizer = 0.0

	comm = 0.0
	if par.strategy == "ddp" and par.world_size > 1 and not par.ddp_bucket_view:
		comm += trainable * trainable_b / mp  # DDP's bucket copy of the gradients
	if p_shard > 1:
		# FSDP/ZeRO-3 gather one unit's full params, with one more in flight from prefetch
		comm += 2 * arch.largest_layer_params() * dtype_bytes(setup.param_dtype) / mp

	frozen = 0.0
	if setup.reference_model and not lora_like:
		frozen += base / (p_shard * mp)  # under LoRA the reference is the base with adapters off: no copy
	if setup.value_model:
		vt = arch.param_count()
		frozen += vt * dtype_bytes(setup.param_dtype) / (p_shard * mp)
		frozen += vt * (dtype_bytes(setup.param_dtype) + n_states * state_b) / (o_shard * mp)

	return {
		"weights": weights,
		"master_weights": master,
		"gradients": grads,
		"optimizer": optimizer,
		"frozen_models": frozen,
		"comm_buffers": comm,
	}


# ---------------------------------------------------------------- activations


def _act_bytes(setup: TrainingSetup) -> float:
	return dtype_bytes(setup.compute_dtype) if setup.compute_dtype.lower() not in QUANT_BYTES else 2.0


def activation_bytes_per_layer(arch: Arch, setup: TrainingSetup, batch: int, seq: int) -> float:
	"""Bytes one transformer layer saves for backward without recompute (per GPU under TP)."""
	if not isinstance(arch, TransformerArch):
		return 0.0
	e = _act_bytes(setup)
	t = max(setup.parallel.tensor, 1)
	b, s, h, a = batch, seq, arch.d_model, arch.n_heads
	sbh = s * b * h
	kv_frac = arch.kv_heads / arch.n_heads
	mask = 1.0 if setup.dropout else 0.0

	norms = 2 * e * sbh  # not split by tensor parallel
	# qkv input stays whole under TP (Korthikanti); q, k, v and the out-proj input split
	attn_linear = e * sbh + (e * sbh * (1 + 2 * kv_frac + 1) + mask * sbh) / t
	if setup.flash_attention:
		core = (e * sbh + 4.0 * b * a * s) / t  # attention output and fp32 logsumexp
	elif setup.grad_checkpointing == "selective":
		core = 0.0  # selective recompute drops the s^2 terms
	else:
		core = e * a * s * s * b / t  # softmax output
		if setup.dropout:
			core += (1 + e) * a * s * s * b / t  # dropout mask and dropout output
	ffr = arch.ff / h * max(1, arch.experts_per_token)
	mlp_in = e * sbh
	mlp = (e * sbh * (4 * ffr if arch.gated_mlp else 2 * ffr) + mask * sbh) / t
	return norms + attn_linear + core + mlp_in + mlp


def activation_bytes(arch: Arch, setup: TrainingSetup, batch: int, seq: int) -> float:
	"""Activations held at the end of the forward pass (per GPU)."""
	if not isinstance(arch, TransformerArch):
		return 0.0
	e = _act_bytes(setup)
	sbh = seq * batch * arch.d_model
	per_layer = activation_bytes_per_layer(arch, setup, batch, seq)
	extras = 2 * e * sbh  # embedding output, final norm input
	pp = max(setup.parallel.pipeline, 1)
	layers = arch.n_layers / pp
	# 1F1B: the first stage keeps `pp` micro-batches in flight, so its activations don't shrink with pp
	in_flight = pp
	if setup.grad_checkpointing == "full":
		return in_flight * layers * e * sbh + per_layer + extras
	return in_flight * layers * per_layer + extras


def logits_bytes(arch: Arch, setup: TrainingSetup, batch: int, seq: int) -> float:
	"""Logits held for the loss: compute dtype, fp32 upcast, saved fp32 log-softmax (HF style)."""
	v = getattr(arch, "vocab_size", 0)
	if not v:
		return 0.0
	e = _act_bytes(setup)
	upcast = 4.0 if (e < 4 and setup.logits_fp32_upcast) else 0.0
	return batch * seq * v * (e + upcast + 4.0) / max(setup.parallel.tensor, 1)


def logits_grad_bytes(arch: Arch, setup: TrainingSetup, batch: int, seq: int) -> float:
	v = getattr(arch, "vocab_size", 0)
	return 4.0 * batch * seq * v / max(setup.parallel.tensor, 1) if v else 0.0


def kv_cache_bytes(
	arch: Arch, sequences: int, tokens_per_seq: int, kv_dtype: str = "bf16", tensor: int = 1
) -> float:
	"""2 (K and V) x layers x kv heads x head dim x bytes x tokens."""
	if not isinstance(arch, TransformerArch):
		return 0.0
	b = dtype_bytes(kv_dtype)
	return (
		2.0 * arch.n_layers * arch.kv_heads * arch.head_dim * b * sequences * tokens_per_seq / max(tensor, 1)
	)


# ---------------------------------------------------------------- breakdown


@dataclass(frozen=True)
class MemoryBreakdown:
	terms: dict[str, float]  # full size of each term
	stack: dict[str, float]  # each term's share at the peak moment; sums to total
	allocator_peak: float  # what the caching allocator holds at the peak moment
	total: float  # allocator peak plus context
	moments: dict[str, float] = field(default_factory=dict)  # allocator bytes at each moment
	peak_moment: str = "train"


def _train_moment(arch, setup, batch, seq, st) -> tuple[float, dict[str, float]]:
	act = activation_bytes(arch, setup, batch, seq)
	logits = logits_bytes(arch, setup, batch, seq)
	lgrad = logits_grad_bytes(arch, setup, batch, seq)
	n_layers = max(getattr(arch, "n_layers", 1), 1)
	resident = (
		st["weights"] + st["master_weights"] + st["optimizer"] + st["frozen_models"] + st["comm_buffers"]
	)
	# micro-batches after the first keep the accumulated grads alive through the forward pass
	grads_in_fwd = st["gradients"] if setup.grad_accum > 1 else 0.0
	at_loss = act + logits + lgrad + grads_in_fwd
	at_backward_end = st["gradients"] + act / n_layers
	if at_loss >= at_backward_end:
		stack = {"activations": act, "logits": logits + lgrad, "gradients": grads_in_fwd}
	else:
		stack = {"activations": act / n_layers, "logits": 0.0, "gradients": st["gradients"]}
	return resident + max(at_loss, at_backward_end), stack


def _eval_moment(arch, setup, seq, st) -> tuple[float, dict[str, float]]:
	b = setup.eval_batch or 0
	act = (
		activation_bytes_per_layer(arch, setup, b, seq)
		if setup.eval_no_grad
		else activation_bytes(arch, setup, b, seq)
	)
	logits = logits_bytes(arch, setup, b, seq)
	resident = (
		st["weights"] + st["master_weights"] + st["optimizer"] + st["frozen_models"] + st["comm_buffers"]
	)
	return resident + act + logits, {"activations": act, "logits": logits, "gradients": 0.0}


def _generation_moment(arch, setup, st, capacity_bytes) -> tuple[float, dict[str, float]]:
	g = setup.generation
	resident = (
		st["weights"] + st["master_weights"] + st["optimizer"] + st["frozen_models"] + st["comm_buffers"]
	)
	kv = kv_cache_bytes(arch, g.sequences, g.prompt_len + g.new_tokens, g.kv_dtype, setup.parallel.tensor)
	step_logits = g.sequences * getattr(arch, "vocab_size", 0) * 4.0  # one decode step's logits, fp32
	stack = {"activations": 0.0, "logits": step_logits, "gradients": 0.0, "kv_cache": kv}
	if g.engine in ("vllm", "sglang") and g.colocated_fraction:
		# the engine reserves its fraction up front and holds weights and KV inside it
		reserved = g.colocated_fraction * (capacity_bytes or 0)
		stack = {
			"activations": 0.0,
			"logits": 0.0,
			"gradients": 0.0,
			"kv_cache": 0.0,
			"generation_engine": reserved,
		}
		return resident + reserved, stack
	return resident + kv + step_logits, stack


def breakdown(
	arch: Arch,
	setup: TrainingSetup,
	batch: int,
	seq: int,
	after_optimizer_step: bool = True,
	context_bytes: float = DEFAULT_CONTEXT_BYTES,
	capacity_bytes: float | None = None,
) -> MemoryBreakdown:
	"""Per-GPU memory at each moment of a step (train, eval, generate) and at the worst one."""
	st = static_terms(arch, setup, after_optimizer_step)
	moments: dict[str, tuple[float, dict[str, float]]] = {"train": _train_moment(arch, setup, batch, seq, st)}
	if setup.eval_batch:
		moments["eval"] = _eval_moment(arch, setup, seq, st)
	if setup.generation is not None:
		moments["generate"] = _generation_moment(arch, setup, st, capacity_bytes)
	peak_name = max(moments, key=lambda k: moments[k][0])
	peak, dyn = moments[peak_name]
	resident = {k: st[k] for k in ("weights", "master_weights", "optimizer", "frozen_models", "comm_buffers")}
	stack = {**resident, **dyn, "context": float(context_bytes)}
	terms = {
		**st,
		"activations": activation_bytes(arch, setup, batch, seq),
		"logits": logits_bytes(arch, setup, batch, seq) + logits_grad_bytes(arch, setup, batch, seq),
		"context": float(context_bytes),
	}
	for name in ("kv_cache", "generation_engine"):
		if name in dyn:
			terms[name] = dyn[name]
	if "eval" in moments:
		terms["eval_spike"] = moments["eval"][0] - moments["train"][0]
	return MemoryBreakdown(
		terms, stack, peak, peak + context_bytes, {k: v[0] for k, v in moments.items()}, peak_name
	)


def eval_breakdown(
	arch: Arch,
	setup: TrainingSetup,
	eval_batch: int,
	seq: int,
	no_grad: bool = True,
	context_bytes: float = DEFAULT_CONTEXT_BYTES,
) -> MemoryBreakdown:
	"""Peak during an eval pass mid-training: weights and optimizer stay resident."""
	s = replace(setup, eval_batch=eval_batch, eval_no_grad=no_grad)
	st = static_terms(arch, s, after_optimizer_step=True)
	peak, dyn = _eval_moment(arch, s, seq, st)
	resident = {k: st[k] for k in ("weights", "master_weights", "optimizer", "frozen_models", "comm_buffers")}
	stack = {**resident, **dyn, "context": float(context_bytes)}
	return MemoryBreakdown(
		{**st, **dyn, "context": float(context_bytes)},
		stack,
		peak,
		peak + context_bytes,
		{"eval": peak},
		"eval",
	)


# ---------------------------------------------------------------- calibration


@dataclass(frozen=True)
class MemoryProbe:
	tokens: float  # work units: batch*seq for LMs, batch*image^2 for vision, sampled edges for GNNs
	measured_bytes: float  # peak reserved after >= 2 optimizer steps
	analytical_bytes: float  # breakdown(...).allocator_peak at the probe's knobs


def _line(xs: list[float], rs: list[float]) -> tuple[float, float]:
	if len(set(xs)) <= 1:
		return sum(rs) / len(rs), 0.0
	mx, mr = sum(xs) / len(xs), sum(rs) / len(rs)
	sxx = sum((x - mx) ** 2 for x in xs)
	slope = sum((x - mx) * (r - mr) for x, r in zip(xs, rs, strict=True)) / sxx
	return mr - slope * mx, slope


@dataclass(frozen=True)
class Calibration:
	"""S1/E3: residual (measured - analytical) linear in work units, least squares over the probes."""

	intercept: float
	slope: float
	max_tokens: float
	min_tokens: float
	n_probes: int
	loo_log_error: float | None  # max leave-one-probe-out |log error|; None with fewer than 3 probes
	probe_tokens: tuple[float, ...] = ()

	@classmethod
	def fit(cls, probes: list[MemoryProbe]) -> Calibration:
		if not probes:
			raise ValueError("calibration needs at least one probe")
		xs = [p.tokens for p in probes]
		rs = [p.measured_bytes - p.analytical_bytes for p in probes]
		intercept, slope = _line(xs, rs)
		loo = None
		if len(probes) >= 3 and len(set(xs)) >= 2:
			errs = []
			for i, p in enumerate(probes):
				c0, c1 = _line(xs[:i] + xs[i + 1 :], rs[:i] + rs[i + 1 :])
				pred = p.analytical_bytes + c0 + c1 * p.tokens
				if pred > 0 and p.measured_bytes > 0:
					errs.append(abs(math.log(p.measured_bytes / pred)))
			loo = max(errs) if errs else None
		return cls(intercept, slope, max(xs), min(xs), len(probes), loo, tuple(xs))

	def correct(self, tokens: float, analytical_bytes: float) -> float:
		return analytical_bytes + self.intercept + self.slope * tokens

	def range(self) -> tuple[float, float]:
		"""Work units the probes vouch for: half the smallest to twice the largest (A's D2 query range)."""
		return self.min_tokens / EXTRAPOLATE_FACTOR, self.max_tokens * EXTRAPOLATE_FACTOR

	def in_range(self, tokens: float) -> bool:
		lo, hi = self.range()
		return lo <= tokens <= hi

	def is_extrapolated(self, tokens: float) -> bool:
		return not self.in_range(tokens)

	def half_width(self, tokens: float | None = None) -> float:
		"""90% half-width in log space (S2a R5; S2c for one or two probes). Out-of-range queries abstain instead."""
		if self.n_probes >= 3 and self.loo_log_error is not None:
			return max(HALF_WIDTH_CALIBRATED, LOO_WIDEN * self.loo_log_error)
		return HALF_WIDTH_CALIBRATED if self.n_probes == 2 else HALF_WIDTH_ONE_PROBE

	def sigma_log(self, tokens: float) -> float:
		return self.half_width(tokens) / Z90


# ---------------------------------------------------------------- OOM


def p_oom(point_bytes: float, sigma_log: float, capacity_bytes: float, margin_bytes: float = 0.0) -> float:
	"""P(peak > usable capacity) under a log-normal error around the point estimate."""
	usable = capacity_bytes - margin_bytes
	if usable <= 0:
		return 1.0
	if point_bytes <= 0:
		return 0.0
	if sigma_log <= 0:
		return 1.0 if point_bytes > usable else 0.0
	z = (math.log(usable) - math.log(point_bytes)) / sigma_log
	return 0.5 * math.erfc(z / math.sqrt(2))


def oom_status(p: float) -> Literal["fits", "tight", "fails"]:
	"""S2b: fits below 1%, fails at 80% or more."""
	if p < FITS_BELOW:
		return "fits"
	if p < FAILS_AT:
		return "tight"
	return "fails"


@dataclass(frozen=True)
class MemoryEstimate:
	point: float | None
	lo: float | None
	hi: float | None
	unit: str
	status: Literal["measured", "predicted", "extrapolated", "abstain"]
	model_id: str
	p_oom: float | None
	oom_status: str | None
	sigma_log: float
	breakdown: MemoryBreakdown
	reason: str = ""


def predict_peak(
	arch: Arch,
	setup: TrainingSetup,
	batch: int,
	seq: int,
	probes: list[MemoryProbe] | None = None,
	gpu: str | None = None,
	capacity_bytes: float | None = None,
	context_bytes: float = DEFAULT_CONTEXT_BYTES,
	margin_bytes: float = 0.0,
	work_units: float | None = None,
	measured: bool = False,
) -> MemoryEstimate:
	"""Peak per-GPU memory with a 90% band and P(OOM) against the GPU's capacity.

	`measured` is the caller's claim that every knob equals a probe's; work units alone can't
	say so (another optimizer or precision at the same batch x seq is not a measurement).
	Calibrated queries outside the probed range abstain, as the contract requires.
	"""
	if capacity_bytes is None:
		if gpu is None:
			raise ValueError("pass gpu or capacity_bytes")
		capacity_bytes = gpus.get_gpu(gpu).capacity_bytes()
	bd = breakdown(arch, setup, batch, seq, context_bytes=context_bytes, capacity_bytes=capacity_bytes)
	tokens = work_units if work_units is not None else batch * seq
	reason = ""
	if probes:
		cal = Calibration.fit(probes)
		alloc = cal.correct(tokens, bd.allocator_peak)
		h = cal.half_width(tokens)
		if measured:
			status = "measured"
			h = min(h, HALF_WIDTH_CALIBRATED)
		elif not cal.in_range(tokens):
			lo_r, hi_r = cal.range()
			why = f"{tokens:.0f} work units is outside the probed range ({lo_r:.0f} to {hi_r:.0f}); run a probe here"
			return MemoryEstimate(
				None, None, None, "bytes", "abstain", MODEL_ID, None, None, h / Z90, bd, why
			)
		else:
			status = "predicted"
	else:
		alloc, h, status = bd.allocator_peak, HALF_WIDTH_PRIOR, "predicted"
		reason = "analytical only, not calibrated by a probe yet"
	sigma = h / Z90
	if h > ABSTAIN_HALF_WIDTH or alloc <= 0:
		why = f"the band would be wider than +-50% ({reason or 'too far from the probes'}); run a probe here"
		return MemoryEstimate(None, None, None, "bytes", "abstain", MODEL_ID, None, None, sigma, bd, why)
	point = alloc + context_bytes
	p = p_oom(point, sigma, capacity_bytes, margin_bytes)
	return MemoryEstimate(
		point=point,
		lo=point * math.exp(-h),
		hi=point * math.exp(h),
		unit="bytes",
		status=status,
		model_id=MODEL_ID,
		p_oom=p,
		oom_status=oom_status(p),
		sigma_log=sigma,
		breakdown=bd,
		reason=reason,
	)
