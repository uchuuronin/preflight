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


RANGE_FACTOR = 2.0  # same rule as memory: half the smallest to twice the largest probe


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
		return min(self.probe_tokens) / RANGE_FACTOR, max(self.probe_tokens) * RANGE_FACTOR

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
			if not lo_r <= tokens <= hi_r:
				why = f"{tokens:.0f} work units per step is outside the probed range ({lo_r:.0f} to {hi_r:.0f}); run a probe here"
				return _abstain(MODEL_ID_SAME, self.sigma, why)
			bd = {n: f.predict(tokens) for n, f in self.fits.items()}
			if recompute:
				bd = with_recompute(bd)
			sigma, status = self.sigma, "predicted"
		point = sum(bd.values())
		lo, hi = _band(point, sigma)
		return StepEstimate(point, lo, hi, "s", status, MODEL_ID_SAME, sigma, bd, reason)


# ---------------------------------------------------------------- kernel classes

KernelClass = Literal[
	"gemm", "conv", "attention", "norm", "softmax", "elementwise", "reduction", "embedding",
	"index", "loss", "optimizer", "copy", "memset", "comm", "sort", "other",
]  # fmt: skip
Bound = Literal["compute", "memory", "link", "latency"]

# (regex, class), most specific first. Covers CUDA (cuBLAS/cuBLASLt/CUTLASS/nvjet, cuDNN, FlashAttention),
# ROCm (Tensile Cijk_, hipBLASLt, Composable Kernel, MIOpen, AITER, RCCL), Triton (torch.compile) and aten ops.
_KERNEL_PATTERNS: list[tuple[re.Pattern[str], KernelClass]] = [
	(re.compile(p, re.I), c)
	for p, c in [
		# communication
		(
			r"nccl|rccl|c10d::|all_?reduce|all_?gather|reduce_?scatter|broadcast|sendrecv|ncclDevKernel",
			"comm",
		),
		# attention (before gemm: flash kernels contain "gemm" in some builds)
		(
			r"flash_?(fwd|bwd|attn)|fmha|FmhaFwd|FmhaBwd|cutlass.*attention|_scaled_dot_product|efficient_attention|aiter.*attn|mha_(fwd|bwd)|paged_attention|flex_attention",
			"attention",
		),
		# conv
		(
			r"implicit_?convolve|cudnn.*conv|conv2d|convolution|miopen.*(conv|igemm)|naive_conv|igemm_|xmma_fprop|xmma_dgrad|xmma_wgrad|winograd|fft_conv",
			"conv",
		),
		# gemm
		(
			r"gemm|Cijk_|hipblaslt|rocblas|cublas|cutlass|nvjet|sm\d+_xmma|ampere_.*(sgemm|hgemm|bf16)|volta_.*gemm|turing_.*gemm|ck::.*gemm|device_gemm|kernel_gemm|triton_tem|aten::(mm|addmm|bmm|baddbmm|matmul|linear|_scaled_mm)\b|splitK",
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
		(r"index_(select|add|put)|gather|scatter|aten::(index|take)|indexing_backward", "index"),
		(
			r"multi_tensor_apply|_foreach_|fused_adam|_fused_adamw|adam_|lamb_|aten::_fused_sgd|lion",
			"optimizer",
		),
		(r"sort|topk|radix|argsort|cub::", "sort"),
		(
			r"reduce_kernel|reduction|aten::(sum|mean|norm|amax|max|min|var|std)\b|triton_(red|per)_|block_reduce",
			"reduction",
		),
		(
			r"memcpy|copy_?buffer|__amd_rocclr_copy|aten::(copy_|_to_copy|to|contiguous|clone)\b|CatArrayBatchedCopy|transpose",
			"copy",
		),
		(r"memset|fill_|zero_|aten::(zero_|fill_|zeros|ones|empty)", "memset"),
		(
			r"elementwise|vectorized|unrolled_elementwise|triton_poi_|aten::(add|mul|sub|div|gelu|silu|relu|sigmoid|tanh|exp|pow|where|dropout|_softmax_backward)|act_and_mul|rotary|rope",
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
	"comm": "link",
	"other": "memory",
}


def classify_kernel(name: str) -> KernelClass:
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


def _peak(spec: gpus.GpuSpec, dtype: str, allow_tf32: bool = False) -> float:
	"""Peak for the dtype; callers check spec.supports(dtype) first, so this never silently swaps dtypes."""
	peak = spec.peak_flops(dtype, allow_tf32)
	if peak is None:
		raise ValueError(f"{spec.name} has no fast path for {dtype}")
	return peak


def _roofline_s(k: KernelRow, gpu: gpus.GpuSpec) -> tuple[float, float]:
	"""(compute time, memory time) at peak on this GPU."""
	tc = (k.flops or 0.0) / _peak(gpu, k.dtype, k.allow_tf32)
	tm = (k.bytes or 0.0) / gpu.bandwidth_bytes()
	return tc, tm


def bound(k: KernelRow, gpu: str) -> Bound | Literal["unknown"]:
	spec = gpus.get_gpu(gpu)
	cls = k.klass or classify_kernel(k.name)
	if cls == "comm":
		return "link"
	if k.flops is None and k.bytes is None:
		return CLASS_BOUND.get(cls, "memory") if cls != "other" else "unknown"
	tc, tm = _roofline_s(k, spec)
	if k.seconds < LATENCY_KERNEL_S and max(tc, tm) < 0.1 * k.seconds:
		return "latency"
	return "compute" if tc >= tm else "memory"


def scale_kernel(k: KernelRow, src: str, tgt: str) -> float:
	s, t = gpus.get_gpu(src), gpus.get_gpu(tgt)
	kind = bound(k, src)
	if kind == "latency":
		return k.seconds
	if kind == "link":
		return k.seconds * s.link_gbps / t.link_gbps if t.link_gbps else k.seconds
	if k.flops is None and k.bytes is None:
		# K4 fallback: no costs recorded, so the kernel's class decides what it scales with
		if kind == "compute":
			return k.seconds * _peak(s, k.dtype, k.allow_tf32) / _peak(t, k.dtype, k.allow_tf32)
		return k.seconds * s.bandwidth_bytes() / t.bandwidth_bytes()
	roof_src = max(_roofline_s(k, s))
	efficiency = min(1.0, roof_src / k.seconds) if k.seconds > 0 else 1.0
	return max(_roofline_s(k, t)) / max(efficiency, 1e-3)


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
