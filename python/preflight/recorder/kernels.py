"""The per-kernel table B's cross-GPU step-time model reads (docs/requests.md, B -> A).

Built from the current-settings probe's profiled window (3 active steps, with
FLOPs and shapes recorded). Rows are aten ops with the device time of the
kernels each launched (self device time, so nothing is counted twice), FLOPs
where PyTorch counts them (matmul, conv, attention), and bytes estimated from
input shapes. Times, calls, FLOPs and bytes are per step.

Size (scripts/mc_kernels_gpu.py, K): the top KERNEL_ROWS ops by time, then the
rest rolled into one row per kernel class. Dropping the tail sends its time to
B's unscaled host bucket: the top 64 alone keeps 89% of cross-GPU estimates
within 20%; with the tail rolled up per class it is 100%, in about 78 rows.
Device time no op claims (kernels launched outside aten, e.g. some custom
extensions) gets its own row so the table still adds up.
"""

import math
import re
from collections.abc import Callable

KERNEL_ROWS = 64  # simulated: probe-plan K (plus the per-class tail); B's ask was 64
_MATMUL = (
	"aten::mm",
	"aten::addmm",
	"aten::bmm",
	"aten::baddbmm",
	"aten::matmul",
	"aten::linear",
	"aten::_scaled_mm",
	"aten::addmm_",
	"aten::baddbmm_",
	"aten::addbmm",
	"aten::addbmm_",
	"aten::addmv",
	"aten::addmv_",
	"aten::mv",
	"aten::dot",
	"aten::_int_mm",
)
DTYPE_BYTES = {"fp32": 4, "tf32": 4, "bf16": 2, "fp16": 2, "fp8": 1}
# record_function ranges the profiler or frameworks add around real ops: their time belongs to the ops inside
_RANGES = (
	"ProfilerStep",
	"Optimizer.",
	"enumerate(DataLoader)",
	"## ",
	"[pl][",
	"Trainer.",
	"autograd::engine",
)
# On CPU every op has self time, including work that is host bookkeeping on any device: autograd
# nodes (MmBackward0, AccumulateGrad), view and metadata ops, and host syncs. B's model scales
# table rows with the target GPU and leaves the rest of the step as unscaled host time, which is
# where these belong. On a GPU they launch nothing, so they carry no device time and drop out
# anyway; a custom backward that launches a kernel directly keeps its row there.
_AUTOGRAD_NODE = re.compile(r"(Backward\d*|^torch::autograd::.*|^detach_?)$")
_HOST_OPS = frozenset(
	f"aten::{n}"
	for n in (
		"view", "_unsafe_view", "_reshape_alias", "reshape", "t", "transpose", "permute", "as_strided", "expand", "expand_as",
		"slice", "select", "narrow", "unsqueeze", "squeeze", "flatten", "unflatten", "split", "chunk",
		"unbind", "alias", "detach", "detach_", "lift_fresh", "view_as", "contiguous", "empty", "empty_like",
		"empty_strided", "new_empty", "new_empty_strided", "resolve_conj", "resolve_neg", "resize_", "set_", "item", "_local_scalar_dense", "is_nonzero", "result_type",
		"size", "stride", "numel", "dim",
	)
)  # fmt: skip


def _classifier() -> Callable[[str], str] | None:
	try:
		from preflight.engine.steptime import classify_kernel
	except ImportError:
		return None
	return classify_kernel


def _numel(shape) -> int:
	try:
		return math.prod(int(d) for d in shape) if shape else 0
	except (TypeError, ValueError):
		return 0


def _bytes(name: str, shapes, elt: int) -> float | None:
	"""Read plus write traffic estimated from input shapes; matmuls add their output."""
	if not shapes:
		return None
	tensors = [s for s in shapes if isinstance(s, list | tuple) and s and all(isinstance(d, int) for d in s)]
	if not tensors:
		return None
	read = sum(_numel(s) for s in tensors)
	if name in _MATMUL and len(tensors) >= 2:
		if name == "aten::linear":  # (input, weight[out, in], bias?)
			x, w = tensors[0], tensors[1]
			out = _numel(x[:-1]) * (w[0] if len(w) == 2 else 1)
		else:
			# addmm/baddbmm put the bias first; _scaled_mm puts scales and an optional bias last
			a, b = (tensors[0], tensors[1]) if name == "aten::_scaled_mm" else (tensors[-2], tensors[-1])
			out = _numel(a[:-1]) * (b[-1] if len(b) >= 2 else 1)
		return float((read + out) * elt)
	return float((read + max(_numel(s) for s in tensors)) * elt)  # elementwise-like: write one output


# ops whose q, k, v arrive as [batch, seq, heads, dim] (SDPA transposes before calling them);
# the _scaled_dot_product_* wrappers and the CPU math path take [batch, heads, seq, dim]
_BLHD = ("aten::_flash_attention", "aten::_efficient_attention")
_ATTENTION = ("scaled_dot_product", "flash_attention", "efficient_attention", "cudnn_attention")


def _matmul_flops(name: str, shapes) -> float | None:
	"""2·M·N·K from the operand shapes, for matmul ops PyTorch's counter skips (in-place, mv, dot, int8).

	derived: one multiply and one add per inner-product term.
	"""
	# linear and matmul are composites: their FLOPs are counted on the mm/addmm/bmm they call
	if name not in _MATMUL or name in ("aten::linear", "aten::matmul") or not shapes:
		return None
	t = [s for s in shapes if isinstance(s, list | tuple) and s and all(isinstance(d, int) for d in s)]
	if name in ("aten::dot",) and len(t) >= 2:
		return 2.0 * t[0][0]
	if name in ("aten::mv", "aten::addmv", "aten::addmv_") and len(t) >= 2:
		m = t[-2]  # (bias?, mat[M, K], vec[K])
		return 2.0 * m[0] * m[1] if len(m) == 2 else None
	a, b = (
		(t[0], t[1])
		if name in ("aten::_scaled_mm", "aten::_int_mm")
		else (t[-2], t[-1])
		if len(t) >= 2
		else (None, None)
	)
	if a is None or len(a) < 2 or len(b) < 2:
		return None
	batch = math.prod(a[:-2]) if len(a) > 2 else 1  # bmm/baddbmm/addbmm: [B, M, K] @ [B, K, N]
	return 2.0 * batch * a[-2] * a[-1] * b[-1]


def _attention_flops(name: str, shapes) -> float | None:
	"""PyTorch counts no FLOPs for fused attention; estimate from q and k: forward 4·B·H·L·S·D, backward 2.5×.

	Read by position (q, k are the first two inputs; backward ops take grad_out
	first), never by filtering for 4-d shapes, so varlen packing (3-d q) gives
	None instead of a wrong count. Causal masking is not visible in shapes, so
	this is the dense count. For B's roofline the FLOPs cancel out of a
	compute-bound kernel's scaling; they only decide which side of the roofline
	the kernel sits on.
	"""
	if not any(a in name for a in _ATTENTION) or not shapes:
		return None
	backward = "backward" in name
	off = 1 if backward else 0
	if len(shapes) < off + 2:
		return None
	q, k = shapes[off], shapes[off + 1]
	if not all(
		isinstance(t, list | tuple) and len(t) == 4 and all(isinstance(d, int) for d in t) for t in (q, k)
	):
		return None
	if name.startswith(_BLHD):
		b, lq, h, d = q
		ls = k[1]
	else:
		b, h, lq, d = q
		ls = k[2]
	fwd = 4.0 * b * h * lq * ls * d
	return 2.5 * fwd if backward else fwd


def _attr(e, *names, default=0.0):
	for n in names:
		v = getattr(e, n, None)
		if v is not None:
			return v
	return default


def kernel_table(
	prof,
	*,
	steps: int,
	device: str,
	dtype: str,
	allow_tf32: bool,
	classify: Callable[[str], str] | None = None,
	rows: int = KERNEL_ROWS,
) -> dict:
	"""prof: a finished torch.profiler.profile. device: "cuda", "rocm" or "cpu"."""
	from torch.autograd import DeviceType

	classify = classify if classify is not None else _classifier()
	elt = DTYPE_BYTES.get(dtype, 4)
	gpu = device in ("cuda", "rocm")
	agg: dict[str, dict] = {}
	device_total = 0.0
	for e in prof.key_averages(group_by_input_shape=True):
		dt = getattr(e, "device_type", DeviceType.CPU)
		if getattr(e, "is_user_annotation", False):
			# record_function ranges (ProfilerStep#, Optimizer.step, the visitor's own labels) and their
			# mirrors on the GPU timeline span the kernels inside them; counting them would count twice
			continue
		if gpu and dt != DeviceType.CPU:
			device_total += _attr(e, "self_device_time_total", "self_cuda_time_total")
			continue
		if e.key.startswith(_RANGES):
			continue
		if not gpu and (e.key in _HOST_OPS or _AUTOGRAD_NODE.search(e.key)):
			continue
		t_us = (
			_attr(e, "self_device_time_total", "self_cuda_time_total")
			if gpu
			else _attr(e, "self_cpu_time_total")
		)
		if t_us <= 0:
			continue
		row = agg.setdefault(
			e.key, {"name": e.key, "t_us": 0.0, "calls": 0, "flops": 0.0, "bytes": 0.0, "has_b": False}
		)
		row["t_us"] += t_us
		row["calls"] += int(e.count)
		counted = float(_attr(e, "flops", default=0) or 0)
		if not counted:
			shapes = getattr(e, "input_shapes", None)
			est = _attention_flops(e.key, shapes) or _matmul_flops(e.key, shapes)
			counted = est * e.count if est else 0.0
		row["flops"] += counted
		b = _bytes(e.key, getattr(e, "input_shapes", None), elt)
		if b is not None:
			row["bytes"] += b * e.count
			row["has_b"] = True

	def out_row(name, t_us, calls, flops, nbytes, cls):
		return {
			"name": name,
			"kernel_class": cls,
			"calls": calls / steps,
			"device_s": t_us / 1e6 / steps,
			"flops": flops / steps if flops else None,
			"bytes": nbytes / steps if nbytes else None,
			"dtype": dtype,
		}

	ordered = sorted(agg.values(), key=lambda r: -r["t_us"])
	table = []
	for r in ordered[:rows]:
		cls = classify(r["name"]) if classify else None
		table.append(
			out_row(r["name"], r["t_us"], r["calls"], r["flops"], r["bytes"] if r["has_b"] else 0, cls)
		)
	tail: dict[str, dict] = {}
	for r in ordered[rows:]:
		cls = classify(r["name"]) if classify else "other"
		t = tail.setdefault(cls, {"t_us": 0.0, "calls": 0, "flops": 0.0, "bytes": 0.0, "n": 0})
		t["t_us"] += r["t_us"]
		t["calls"] += r["calls"]
		t["flops"] += r["flops"]
		t["bytes"] += r["bytes"]
		t["n"] += 1
	for cls, t in sorted(tail.items(), key=lambda kv: -kv[1]["t_us"]):
		table.append(
			out_row(f"(tail) {cls} ({t['n']} ops)", t["t_us"], t["calls"], t["flops"], t["bytes"], cls)
		)
	attributed = sum(r["t_us"] for r in agg.values())
	unclaimed_us = max(0.0, device_total - attributed) if gpu else 0.0
	if unclaimed_us > 0.01 * max(device_total, 1e-9):
		table.append(out_row("(unattributed device time)", unclaimed_us, 0, 0, 0, "other"))
	return {
		"rows": table,
		"steps": steps,
		"backend": device,
		"allow_tf32": allow_tf32,
		"classified_by": "engine.steptime" if classify else None,
	}
