"""Step-time model: per-phase fits on the probed GPU, roofline replay to others.

Method (research map, section 2; settings from docs/decisions/engine-settings.md S4):
- same GPU: each phase fitted as t = a + b * work units from the probes,
  non-negative; phases that don't grow with the batch (optimizer, weight sync,
  checkpoint, idle) are held constant. Probe phases arrive as fractions of the
  median step (A's StepTime.by_phase with p50_s) and are converted here.
- other GPUs (K1): each profiled kernel keeps its measured efficiency against
  the source GPU's roofline and is replayed on the target's roofline; host time
  doesn't scale. Kernels without a FLOP count fall back to K4: their class,
  read from the kernel name (CUDA, HIP, Triton or aten), says whether they
  scale with peak FLOPs, bandwidth, the GPU link, or not at all.
- recompute: full activation checkpointing adds one forward pass.
- multi-GPU: ring all-reduce of the gradients over the GPU link, with the
  share not hidden behind backward exposed (analytical only, wide band).
- MFU (PaLM appendix B): 6N + 12*L*h*s FLOPs per trained token, active
  parameters only for MoE.

Pure Python, standard library only. Safe in Pyodide.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Literal

from preflight.engine import gpus
from preflight.engine.memory import TransformerArch

MODEL_ID_SAME = "steptime-phases@0.2.0"
MODEL_ID_CROSS = "steptime-roofline@0.2.0"
MODEL_ID_COMM = "steptime-comm-analytical@0.1.0"
Z90 = 1.6448536269514722
SIGMA_SAME_GPU = 0.08  # A's D1: median of 5 timed steps lands within 15% in 99.3% of trials
SIGMA_SAME_VENDOR = 0.12  # S4/K1 within 20% on an unmeasured GPU
SIGMA_CROSS_VENDOR = 0.25  # hipBLASLt/CK kernels don't map 1:1 to cuBLAS
SIGMA_SECONDARY_SPECS = 1.5  # multiplier when the target GPU's specs come from a spec database
SIGMA_COMM = 0.35  # analytical-only multi-GPU extrapolation
LATENCY_KERNEL_S = 20e-6
EXPOSED_COMM_SHARE = 0.3  # share of DDP/FSDP gradient comm not overlapped with backward

Mode = Literal["forward", "train", "train_frozen_base"]


# ---------------------------------------------------------------- FLOPs and MFU


def flops_per_token(arch: TransformerArch, seq: int, mode: Mode = "train", recompute: bool = False) -> float:
	"""Matmul FLOPs per token. 'train_frozen_base' is LoRA/QLoRA: no base weight gradients."""
	n_mm = arch.active_linear_params() + arch.d_model * arch.vocab_size  # projections plus the LM head
	attn_fwd = 4.0 * arch.n_layers * arch.n_heads * arch.head_dim * seq  # QK^T and AV
	fwd = 2.0 * n_mm + attn_fwd
	if mode == "forward":
		return fwd
	if mode == "train":
		total = 6.0 * n_mm + 3.0 * attn_fwd
	elif mode == "train_frozen_base":
		total = 4.0 * n_mm + 3.0 * attn_fwd
	else:
		raise ValueError(f"unknown mode {mode!r}")
	return total + (fwd if recompute else 0.0)


def mfu(
	model_flops: float,
	seconds: float,
	gpu: str,
	dtype: str = "bf16",
	n_gpus: int = 1,
	allow_tf32: bool = True,
) -> float:
	peak = gpus.get_gpu(gpu).peak_flops(dtype, allow_tf32)
	if peak is None:
		raise ValueError(f"{gpu} has no fast path for {dtype}")
	return model_flops / seconds / (peak * n_gpus)


# ---------------------------------------------------------------- phases

Scaling = Literal["work", "generation", "eval", "constant", "comm"]

# Every phase name the recorder and schema use (StepTime.by_phase), and what it grows with.
PHASES: dict[str, Scaling] = {
	"data": "work",
	"forward": "work",
	"backward": "work",
	"optimizer": "constant",  # grows with trainable params, not the batch
	"eval": "eval",
	"rollout": "generation",
	"generate": "generation",
	"reference": "work",
	"reward": "work",
	"update": "work",
	"weight_sync": "constant",
	"checkpoint": "constant",
	"comm": "comm",
	"idle": "constant",
	"other": "constant",
}
CONSTANT_PHASES = frozenset(k for k, v in PHASES.items() if v == "constant")


def phase_scaling(name: str) -> Scaling:
	return PHASES.get(name, "work")


def phases_seconds(by_phase: dict[str, float], step_s: float) -> dict[str, float]:
	"""A's StepTime.by_phase holds fractions of a step; turn them into seconds of the median step."""
	total = sum(v for v in by_phase.values() if v and v > 0)
	if total <= 0:
		return {}
	return {k: step_s * v / total for k, v in by_phase.items() if v and v > 0}


def with_recompute(phases: dict[str, float]) -> dict[str, float]:
	"""Full activation checkpointing replays the forward pass inside backward."""
	out = dict(phases)
	out["backward"] = out.get("backward", 0.0) + out.get("forward", 0.0)
	return out


# ---------------------------------------------------------------- estimates


@dataclass(frozen=True)
class StepEstimate:
	point: float | None
	lo: float | None
	hi: float | None
	unit: str
	status: Literal["measured", "predicted", "extrapolated", "abstain"]
	model_id: str
	sigma_log: float
	breakdown: dict[str, float]
	reason: str = ""


def _band(point: float, sigma: float) -> tuple[float, float]:
	return point * math.exp(-Z90 * sigma), point * math.exp(Z90 * sigma)


def _abstain(model_id: str, sigma: float, reason: str) -> StepEstimate:
	return StepEstimate(None, None, None, "s", "abstain", model_id, sigma, {}, reason)


RANGE_FACTOR = 2.0  # past twice the largest probe: abstain, as memory does
BELOW_GROWTH_LOG = 0.1  # S4b T4: below the smallest probe the 90% log half-width grows 0.1 per halving
ABSTAIN_HALF_WIDTH = math.log(1.5)  # wider than +-50%: abstain, as memory does


# ---------------------------------------------------------------- same GPU


@dataclass(frozen=True)
class PhaseFit:
	intercept: float
	slope: float

	@classmethod
	def fit(cls, points: list[tuple[float, float]], constant: bool = False) -> PhaseFit:
		if not points:
			raise ValueError("need at least one point")
		xs = [p[0] for p in points]
		ys = [p[1] for p in points]
		mean_y = sum(ys) / len(ys)
		if constant:
			return cls(mean_y, 0.0)
		if len(set(xs)) == 1:
			return cls(0.0, mean_y / xs[0]) if xs[0] else cls(mean_y, 0.0)
		mx = sum(xs) / len(xs)
		sxx = sum((x - mx) ** 2 for x in xs)
		slope = sum((x - mx) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / sxx
		intercept = mean_y - slope * mx
		if slope < 0:
			return cls(mean_y, 0.0)
		if intercept < 0:
			return cls(0.0, sum(x * y for x, y in zip(xs, ys, strict=True)) / sum(x * x for x in xs))
		return cls(intercept, slope)

	def predict(self, x: float) -> float:
		return self.intercept + self.slope * x


@dataclass(frozen=True)
class StepProbe:
	tokens: float  # work units of the probe
	phases: dict[str, float]  # seconds per step, from unprofiled steps
	id: str = ""

	@classmethod
	def from_fractions(
		cls, tokens: float, p50_s: float, by_phase: dict[str, float], id: str = ""
	) -> StepProbe:
		return cls(tokens, phases_seconds(by_phase, p50_s), id)


@dataclass(frozen=True)
class SameGpuModel:
	fits: dict[str, PhaseFit]
	probe_tokens: tuple[float, ...]
	sigma: float = SIGMA_SAME_GPU
	_measured: dict[float, dict[str, float]] = field(default_factory=dict)

	@classmethod
	def fit(
		cls, probes: list[StepProbe], constant_phases: set[str] | frozenset[str] = CONSTANT_PHASES
	) -> SameGpuModel:
		names = sorted({k for p in probes for k in p.phases})
		fits = {
			n: PhaseFit.fit(
				[(p.tokens, p.phases[n]) for p in probes if n in p.phases], constant=n in constant_phases
			)
			for n in names
		}
		sigma = SIGMA_SAME_GPU * (1.5 if len({p.tokens for p in probes}) == 1 else 1.0)
		measured = {(p.id or f"probe{i}"): dict(p.phases) for i, p in enumerate(probes)}
		return cls(fits, tuple(p.tokens for p in probes), sigma, measured)

	def range(self) -> tuple[float, float]:
		"""Predicted between the smallest probe and twice the largest; below it, extrapolated (S4b)."""
		return min(self.probe_tokens), max(self.probe_tokens) * RANGE_FACTOR

	def predict(
		self, tokens: float, recompute: bool = False, measured_probe: str | None = None
	) -> StepEstimate:
		"""Step time at `tokens`. `measured_probe` is the caller's claim that every knob equals that probe's."""
		reason = ""
		if measured_probe is not None and measured_probe in self._measured and not recompute:
			bd = dict(self._measured[measured_probe])
			status, sigma = "measured", self.sigma / 2
		else:
			lo_r, hi_r = self.range()
			sigma, status = self.sigma, "predicted"
			if 0 < tokens < lo_r:
				# S4b T4: small batches under-use the GPU, so the line drifts; the band grows per halving
				h = Z90 * self.sigma + BELOW_GROWTH_LOG * math.log2(lo_r / tokens)
				if h > ABSTAIN_HALF_WIDTH:
					why = f"{tokens:.0f} work units per step is too far below the smallest probe ({lo_r:.0f}); run a probe here"
					return _abstain(MODEL_ID_SAME, h / Z90, why)
				sigma, status = h / Z90, "extrapolated"
				reason = f"below the smallest probed size ({lo_r:.0f} work units), so the range is wider; run a probe here for a measured number"
			elif not lo_r <= tokens <= hi_r:
				why = f"{tokens:.0f} work units per step is outside the probed range ({lo_r:.0f} to {hi_r:.0f}); run a probe here"
				return _abstain(MODEL_ID_SAME, self.sigma, why)
			bd = {n: f.predict(tokens) for n, f in self.fits.items()}
			if recompute:
				bd = with_recompute(bd)
		point = sum(bd.values())
		lo, hi = _band(point, sigma)
		return StepEstimate(point, lo, hi, "s", status, MODEL_ID_SAME, sigma, bd, reason)


# ---------------------------------------------------------------- kernel classes

KernelClass = Literal[
	"gemm", "conv", "attention", "norm", "softmax", "elementwise", "reduction", "embedding",
	"index", "loss", "optimizer", "copy", "memset", "comm", "sort", "host", "other",
]  # fmt: skip
KERNEL_CLASSES: frozenset[str] = frozenset(KernelClass.__args__)
Bound = Literal["compute", "memory", "link", "latency"]

# (regex, class), most specific first. Covers CUDA (cuBLAS/cuBLASLt/CUTLASS/nvjet, cuDNN, FlashAttention),
# ROCm (Tensile Cijk_, hipBLASLt, Composable Kernel, MIOpen, AITER, RCCL), Triton (torch.compile) and aten ops.
_KERNEL_PATTERNS: list[tuple[re.Pattern[str], KernelClass]] = [
	(re.compile(p, re.I), c)
	for p, c in [
		# launches nothing on a GPU: views, metadata and host syncs; their time is host overhead
		(
			r"aten::(view|_?reshape_alias|reshape|as_strided|expand|expand_as|t|permute|transpose|unsqueeze|squeeze"
			r"|select|slice|narrow|detach|alias|unbind|split|split_with_sizes|chunk|unfold|view_as|_unsafe_view"
			r"|broadcast_tensors|broadcast_to|numel|size|stride|is_nonzero|item|_local_scalar_dense"
			r"|resolve_conj|resolve_neg|lift_fresh|empty|empty_like|empty_strided|new_empty|new_empty_strided"
			r"|set_|result_type)_?\b",
			"host",
		),
		# communication
		(
			r"nccl|rccl|c10d::|record_param_comms|all_?reduce|all_?gather|all_to_all|reduce_?scatter|broadcast|sendrecv|ncclDevKernel",
			"comm",
		),
		# attention (before gemm: flash kernels contain "gemm" in some builds)
		(
			r"flash_?(fwd|bwd|attn|attention)|fmha|FmhaFwd|FmhaBwd|cutlass.*attention|scaled_dot_product|efficient_attention|_cudnn_attention|aiter.*attn|mha_(fwd|bwd)|paged_attention|flex_attention",
			"attention",
		),
		# conv
		(
			r"implicit_?convolve|cudnn.*conv|conv_?transpose\dd|conv\dd|conv_tbc|convolution|miopen.*(conv|igemm)|naive_conv|igemm_|xmma_fprop|xmma_dgrad|xmma_wgrad|winograd|fft_conv",
			"conv",
		),
		# gemm
		(
			r"gemm|Cijk_|hipblaslt|rocblas|cublas|cutlass|nvjet|sm\d+_xmma|ampere_.*(sgemm|hgemm|bf16)|volta_.*gemm|turing_.*gemm|ck::.*gemm|device_gemm|kernel_gemm|triton_tem|aten::(mm|addmm|addbmm|addmv|mv|dot|vdot|bmm|baddbmm|matmul|linear|_scaled_mm|_int_mm|_weight_int4pack_mm|_weight_int8pack_mm|_convert_weight_to_int4pack|_cudnn_rnn|lstm|gru|rnn_tanh|rnn_relu)_?\b|splitK",
			"gemm",
		),
		# normalisation
		(
			r"layer_?norm|rms_?norm|group_?norm|batch_?norm|instance_?norm|aten::native_(layer|batch|group)_norm|_fused_rms_norm|bn_fw|bn_bw",
			"norm",
		),
		(r"softmax|log_softmax", "softmax"),
		(r"cross_?entropy|nll_?loss|aten::(nll_loss|cross_entropy|mse_loss|binary_cross)", "loss"),
		(r"embedding|EmbeddingBag", "embedding"),
		(
			r"index_(select|add|put)|gather|scatter|aten::(index|_unsafe_index|_index_put_impl|take|nonzero|nonzero_static|masked_select|masked_scatter"
			r"|unique|_unique2|unique_consecutive|bincount|histc|searchsorted|one_hot)|indexing_backward",
			"index",
		),
		(
			r"multi_tensor_apply|_foreach_|fused_adam|_fused_adamw|adam_|lamb_|aten::_fused_sgd|lion",
			"optimizer",
		),
		(r"sort|topk|radix|argsort|cub::", "sort"),
		(
			r"reduce_kernel|reduction|linalg_vector_norm|aten::(sum|mean|norm|amax|amin|max|min|var|std|var_mean|std_mean|prod|all|any|argmax|argmin|cumsum|cumprod|logsumexp|count_nonzero|max_pool\dd|max_pool\dd_with_indices|avg_pool\dd|adaptive_avg_pool\dd|adaptive_max_pool\dd|_adaptive_avg_pool\dd)\w*?_?\b"
			r"|triton_(red|per)_|block_reduce|pool",
			"reduction",
		),
		(
			r"memcpy|copy_?buffer|__amd_rocclr_copy|aten::(copy_|_to_copy|to|contiguous|clone|cat|stack|hstack|vstack|constant_pad_nd|pad|reflection_pad\dd|replication_pad\dd|slice_backward|select_backward|narrow_copy|repeat|repeat_interleave|flip|roll|_copy_from|_copy_from_and_resize|[a-z_]+_copy|upsample_\w+|_upsample_\w+|pixel_shuffle|pixel_unshuffle)_?\b"
			r"|CatArrayBatchedCopy|transpose|upsample",
			"copy",
		),
		# fills and generators: write-only, so bandwidth-bound
		(
			r"memset|(?<!masked_)fill_|zero_|aten::(zero_|fill_|zeros|zeros_like|new_zeros|ones|ones_like|new_ones|full|full_like|new_full|scalar_tensor"
			r"|arange|linspace|logspace|eye|rand|randn|randint|rand_like|randn_like|random_|normal_|uniform_|bernoulli_|bernoulli"
			r"|exponential_|multinomial|randperm)_?\b|philox|curand",
			"memset",
		),
		(
			r"elementwise|vectorized|unrolled_elementwise|triton_poi_|aten::(add|mul|sub|div|gelu|silu|relu|sigmoid|tanh|exp|pow|where|dropout|glu|_fused_dropout|hardswish|hardsigmoid)(_backward)?_?\b|act_and_mul|rotary|rope"
			r"|aten::(neg|abs|sqrt|rsqrt|reciprocal|cos|sin|log|log1p|log2|erf|erfinv|sign|sgn|square|floor|ceil|round|trunc|frac"
			r"|clamp|clamp_min|clamp_max|clip|maximum|minimum|fmax|fmin|eq|ne|gt|ge|lt|le|isclose"
			r"|bitwise_and|bitwise_or|bitwise_xor|bitwise_not|logical_and|logical_or|logical_xor|logical_not"
			r"|masked_fill|masked_scale|_masked_scale|lerp|addcmul|addcdiv|rsub|threshold|threshold_backward|hardtanh|leaky_relu|elu|softplus|mish"
			r"|native_dropout|native_dropout_backward|isfinite|isinf|isnan|isneginf|isposinf|signbit|nan_to_num"
			r"|atan2|hypot|fmod|remainder|copysign|xlogy|tril|triu|type_as|_amp_update_scale)_?\b",
			"elementwise",
		),
	]
]
CLASS_BOUND: dict[str, Bound] = {
	"gemm": "compute",
	"conv": "compute",
	"attention": "compute",
	"norm": "memory",
	"softmax": "memory",
	"elementwise": "memory",
	"reduction": "memory",
	"embedding": "memory",
	"index": "memory",
	"loss": "memory",
	"optimizer": "memory",
	"copy": "memory",
	"memset": "memory",
	"sort": "memory",
	"host": "latency",  # launches nothing on the GPU, so it doesn't scale with it
	"comm": "link",
	"other": "memory",
}


_TAIL_ROW = re.compile(r"^\(tail\) ([a-z]+) \(\d+ ops?\)$")


MAX_KERNEL_NAME = 200  # the schema's KERNEL_NAME_LEN; longer names are cut before any regex sees them


def classify_kernel(name: str) -> KernelClass:
	"""Kernel or aten op name -> class. A's roll-up rows, `(tail) <class> (<n> ops)`, keep their class."""
	name = name[:MAX_KERNEL_NAME]
	if name.startswith("("):
		m = _TAIL_ROW.match(name)
		return m.group(1) if m and m.group(1) in KERNEL_CLASSES else "other"  # type: ignore[return-value]
	for pat, cls in _KERNEL_PATTERNS:
		if pat.search(name):
			return cls
	return "other"


# ---------------------------------------------------------------- cross GPU


@dataclass(frozen=True)
class KernelRow:
	name: str
	seconds: float  # this kernel's time per step on the source GPU
	flops: float | None = None
	bytes: float | None = None
	dtype: str = "bf16"
	klass: str | None = None  # set when the recorder already classified it
	allow_tf32: bool = False  # fp32 kernels: was TF32 on for this run
	calls: float | None = None  # launches per step; latency is judged per launch, not per row

	def per_launch_s(self) -> float:
		return self.seconds / self.calls if self.calls and self.calls > 0 else self.seconds


def kernel_rows(table) -> list[KernelRow]:
	"""A probe's `Probe.kernels` (schema KernelTable, or its dict) as engine rows, per optimizer step.

	The recorder's class is kept when it is one this classifier knows, else the name is classified
	again; zero FLOPs or bytes mean "not counted", so those rows fall back to their class (K4).
	"""
	if isinstance(table, dict):
		from preflight.schema.record import KernelTable

		table = KernelTable.model_validate(table)  # the same bounds as a record: finite, non-negative, capped
	tf32 = bool(table.allow_tf32)
	out = []
	for r in table.rows:
		cls = r.kernel_class
		out.append(
			KernelRow(
				name=r.name,
				seconds=r.device_s,
				flops=r.flops or None,
				bytes=r.bytes or None,
				dtype=r.dtype or "bf16",
				klass=cls if cls in KERNEL_CLASSES else None,
				allow_tf32=tf32,
				calls=r.calls or None,
			)
		)
	return out


def predict_cross_gpu_from_probe(table, measured_step_s: float, src: str, tgt: str) -> StepEstimate:
	"""K1 from a recorded kernel table; abstains when the table can't be scaled to a GPU."""
	sigma = cross_sigma(src, tgt)
	if table is None:
		return _abstain(MODEL_ID_CROSS, sigma, "no kernel table on the probe")
	try:
		rows = kernel_rows(table)
	except ValueError as exc:  # pydantic's ValidationError is a ValueError
		return _abstain(
			MODEL_ID_CROSS, sigma, f"the probe's kernel table is malformed ({str(exc).splitlines()[0]})"
		)
	backend = table.get("backend") if isinstance(table, dict) else table.backend
	if not rows:
		return _abstain(MODEL_ID_CROSS, sigma, "no kernel table on the probe")
	if backend == "cpu":
		return _abstain(
			MODEL_ID_CROSS, sigma, "the kernel table came from a CPU probe; its times don't scale to a GPU"
		)
	return predict_cross_gpu(rows, measured_step_s, src, tgt)


def _peak(spec: gpus.GpuSpec, dtype: str, allow_tf32: bool = False) -> float:
	"""Peak for the dtype; callers check spec.supports(dtype) first, so this never silently swaps dtypes."""
	peak = spec.peak_flops(dtype, allow_tf32)
	if peak is None:
		raise ValueError(f"{spec.name} has no fast path for {dtype}")
	return peak


def _roofline_s(k: KernelRow, gpu: gpus.GpuSpec) -> tuple[float, float]:
	"""(compute time, memory time) at peak on this GPU."""
	tc = k.flops / _peak(gpu, k.dtype, k.allow_tf32) if k.flops else 0.0
	tm = (k.bytes or 0.0) / gpu.bandwidth_bytes()
	return tc, tm


def bound(k: KernelRow, gpu: str) -> Bound | Literal["unknown"]:
	spec = gpus.get_gpu(gpu)
	cls = k.klass or classify_kernel(k.name)
	if cls == "comm":
		return "link"
	if cls == "host":
		return "latency"
	if not k.flops and not k.bytes:
		if k.calls and k.per_launch_s() < LATENCY_KERNEL_S and CLASS_BOUND.get(cls) == "memory":
			return "latency"  # many tiny launches: launch overhead, not bandwidth
		return CLASS_BOUND.get(cls, "memory") if cls != "other" else "unknown"
	tc, tm = _roofline_s(k, spec)
	if k.per_launch_s() < LATENCY_KERNEL_S and max(tc, tm) < 0.1 * k.seconds:
		return "latency"
	return "compute" if tc >= tm else "memory"


def scale_kernel(k: KernelRow, src: str, tgt: str) -> float:
	s, t = gpus.get_gpu(src), gpus.get_gpu(tgt)
	kind = bound(k, src)
	if kind == "latency":
		return k.seconds
	if kind == "link":
		return k.seconds * s.link_gbps / t.link_gbps if t.link_gbps else k.seconds
	if not k.flops and not k.bytes:
		# K4 fallback: no costs recorded, so the kernel's class decides what it scales with
		if kind == "compute":
			return k.seconds * _peak(s, k.dtype, k.allow_tf32) / _peak(t, k.dtype, k.allow_tf32)
		return k.seconds * s.bandwidth_bytes() / t.bandwidth_bytes()
	# keep the kernel's measured efficiency: scale by the ratio of rooflines. Same as roof_tgt / efficiency,
	# but stays exact on the same GPU when an estimated byte count overshoots what the GPU could move
	roof_src = max(_roofline_s(k, s))
	if roof_src <= 0:
		return k.seconds
	return k.seconds * max(_roofline_s(k, t)) / roof_src


def cross_sigma(src: str, tgt: str) -> float:
	s, t = gpus.get_gpu(src), gpus.get_gpu(tgt)
	sigma = SIGMA_SAME_VENDOR if s.vendor == t.vendor else SIGMA_CROSS_VENDOR
	if "secondary" in (s.confidence, t.confidence):
		sigma *= SIGMA_SECONDARY_SPECS
	return sigma


def unsupported_dtypes(dtypes: set[str], tgt: str) -> set[str]:
	t = gpus.get_gpu(tgt)
	return {d for d in dtypes if not t.supports(d)}


def predict_cross_gpu(kernels: list[KernelRow], measured_step_s: float, src: str, tgt: str) -> StepEstimate:
	if (
		not math.isfinite(measured_step_s)
		or measured_step_s <= 0
		or any(not math.isfinite(k.seconds) or k.seconds < 0 for k in kernels)
	):
		return _abstain(
			MODEL_ID_CROSS, cross_sigma(src, tgt), "the measured step or a kernel time isn't a usable number"
		)
	missing = unsupported_dtypes({k.dtype for k in kernels if k.flops or bound(k, src) == "compute"}, tgt)
	if missing:
		why = f"{gpus.resolve(tgt)} has no fast path for {', '.join(sorted(missing))}; the job would fail there or need another precision"
		return _abstain(MODEL_ID_CROSS, cross_sigma(src, tgt), why)
	host = max(0.0, measured_step_s - sum(k.seconds for k in kernels))
	bd = {"compute": 0.0, "memory": 0.0, "link": 0.0, "latency": 0.0, "host": host}
	for k in kernels:
		kind = bound(k, src)
		bd["memory" if kind == "unknown" else kind] += scale_kernel(k, src, tgt)
	sigma = cross_sigma(src, tgt)
	point = sum(bd.values())
	lo, hi = _band(point, sigma)
	reasons = []
	if gpus.get_gpu(src).vendor != gpus.get_gpu(tgt).vendor:
		reasons.append("cross-vendor: kernel libraries differ, so the band is wider")
	if gpus.get_gpu(tgt).confidence == "secondary":
		reasons.append(f"{gpus.resolve(tgt)}'s specs come from a spec database, not a datasheet")
	return StepEstimate(point, lo, hi, "s", "predicted", MODEL_ID_CROSS, sigma, bd, "; ".join(reasons))


def scale_phases_by_class(
	phases: dict[str, float], src: str, tgt: str, compute_share: float = 0.7, dtype: str = "bf16"
) -> dict[str, float]:
	"""Phase-level K4 when no kernel table is available: forward/backward split compute vs memory."""
	s, t = gpus.get_gpu(src), gpus.get_gpu(tgt)
	pr = _peak(s, dtype) / _peak(t, dtype)
	br = s.bandwidth_bytes() / t.bandwidth_bytes()
	out = {}
	for name, sec in phases.items():
		sc = phase_scaling(name)
		if name in ("forward", "backward", "reference", "update", "eval"):
			out[name] = sec * (compute_share * pr + (1 - compute_share) * br)
		elif sc == "generation":
			out[name] = sec * br  # decoding is bandwidth-bound
		elif name == "optimizer":
			out[name] = sec * br
		else:
			out[name] = sec  # data loading, host time, checkpoint I/O don't follow the GPU
	return out


# ---------------------------------------------------------------- multi-GPU communication


def allreduce_seconds(
	nbytes: float, world: int, gpu: str, efficiency: float = 0.7, hop_latency_s: float = 10e-6
) -> float:
	"""Ring all-reduce: 2(w-1)/w of the bytes cross each GPU's link, plus a per-hop latency."""
	if world <= 1 or nbytes <= 0:
		return 0.0
	spec = gpus.get_gpu(gpu)
	if spec.link_gbps <= 0:
		raise ValueError(f"{spec.name} has no GPU-to-GPU link (a MIG slice can't join a data-parallel group)")
	bw = spec.link_bytes_one_way() * efficiency
	return 2 * (world - 1) / world * nbytes / bw + 2 * (world - 1) * hop_latency_s


def multi_gpu_step(
	single_gpu_s: float,
	grad_bytes: float,
	world: int,
	gpu: str,
	strategy: str = "ddp",
	exposed_share: float = EXPOSED_COMM_SHARE,
) -> StepEstimate:
	"""Data-parallel step time from a measured single-GPU step: compute splits, gradient comm is added.

	FSDP/ZeRO-3 also all-gather the parameters forward and backward: about 1.5x DDP's traffic.
	"""
	traffic = grad_bytes * (1.5 if strategy in ("fsdp", "zero3", "hsdp") else 1.0)
	try:
		comm = allreduce_seconds(traffic, world, gpu)
	except ValueError as exc:
		return _abstain(MODEL_ID_COMM, SIGMA_COMM, str(exc))
	point = single_gpu_s + exposed_share * comm
	lo, hi = _band(point, SIGMA_COMM)
	return StepEstimate(
		point, lo, hi, "s", "extrapolated", MODEL_ID_COMM, SIGMA_COMM,
		{"compute": single_gpu_s, "comm_exposed": exposed_share * comm, "comm_total": comm},
		"analytical only: multi-GPU communication is extrapolated, not measured",
	)  # fmt: skip
