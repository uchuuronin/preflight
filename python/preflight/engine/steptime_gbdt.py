"""A gradient-boosted residual on K1's cross-GPU step time (main plan: "plus a LightGBM residual on kernel features").

K1 replays each kernel of the source GPU's table on the target by its roofline ratio. What K1 can't see (cache sizes,
launch overheads, how well a library tunes for a GPU) is left in log(actual / K1). The residual model learns that
from the corpus, where every config ran on several GPUs. The fitted trees are a generated literal
(engine/_steptime_gbdt.py, written by scripts/fit_steptime_gbdt.py), walked here with plain Python: no scikit-learn
in the engine, so it runs in Pyodide.

It is used only if it beats K1 alone on GPUs it never saw (leave-one-GPU-out), which the generated file records.
Until then `available()` is False and K1 stands alone.
"""

from __future__ import annotations

import math

from preflight.engine import gpus, steptime

FEATURES = (
	"log_flops_ratio",  # target over source dense peak at the kernels' dtype
	"log_bw_ratio",  # target over source memory bandwidth
	"log_sm_ratio",  # target over source SM (CU) count
	"share_compute",  # source step time in compute-bound kernels
	"share_memory",
	"share_latency",
	"share_host",  # source step time outside the kernel table
	"log_step_s",  # source step time
	"log_k1_ratio",  # K1's target over the source step
	"log_launch_rate",  # launches per second on the source
	"is_fp32",
	"is_half",
)


def features(
	kernels: list[steptime.KernelRow], step_s: float, src: str, tgt: str, breakdown: dict
) -> list[float]:
	s, t = gpus.get_gpu(src), gpus.get_gpu(tgt)
	dtype = kernels[0].dtype if kernels else "bf16"
	tf32 = bool(kernels and kernels[0].allow_tf32)
	ps = steptime._peak(s, dtype, tf32)
	pt = steptime._peak(t, dtype, tf32)
	share = {"compute": 0.0, "memory": 0.0, "latency": 0.0, "link": 0.0}
	for k in kernels:
		b = steptime.bound(k, src)
		share["memory" if b == "unknown" else b] += k.seconds
	total = max(step_s, 1e-12)
	host = max(0.0, step_s - sum(k.seconds for k in kernels))
	k1 = sum(breakdown.values()) if breakdown else step_s
	calls = sum(k.calls or 1.0 for k in kernels)
	d = gpus.normalise_dtype(dtype)
	return [
		math.log(pt / ps),
		math.log(t.bandwidth_bytes() / s.bandwidth_bytes()),
		math.log(t.compute_units / s.compute_units),
		share["compute"] / total,
		share["memory"] / total,
		share["latency"] / total,
		host / total,
		math.log(total),
		math.log(max(k1, 1e-12) / total),
		math.log(max(calls, 1.0) / total),
		1.0 if d == "fp32" else 0.0,
		1.0 if d in ("bf16", "fp16") else 0.0,
	]


def _walk(tree: dict, x: list[float]) -> float:
	node = 0
	left, right = tree["left"], tree["right"]
	while left[node] != -1:
		node = left[node] if x[tree["feature"][node]] <= tree["threshold"][node] else right[node]
	return tree["value"][node]


def _load() -> dict | None:
	try:
		import preflight.engine._steptime_gbdt as mod  # generated; absent until the residual beats K1 held out
	except ModuleNotFoundError:
		return None
	return getattr(mod, "DATA", None)


DATA = _load()


def available() -> bool:
	"""True only when the installed trees beat K1 alone on held-out GPUs."""
	if not DATA:
		return False
	h = DATA.get("held_out") or {}
	k1, gb = h.get("k1"), h.get("gbdt")
	return bool(k1 and gb and gb[1] and gb[0] > k1[0])


def residual(x: list[float]) -> float:
	"""log(actual / K1) the trees predict for these features."""
	if not DATA:
		return 0.0
	out = DATA["init"]
	for t in DATA["trees"]:
		out += DATA["learning_rate"] * _walk(t, x)
	return out
