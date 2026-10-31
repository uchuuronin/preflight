"""Cross-GPU cases from the corpus: every config two GPUs both ran, with K1's replay of one onto the other.

Shared by corpus/calibrate.py (the cross-GPU conformal band) and scripts/fit_steptime_gbdt.py (the residual model).
"""

from __future__ import annotations

from collections import defaultdict

from preflight.corpus import calibrate as cal
from preflight.engine import conformal, steptime
from preflight.engine import steptime_gbdt as G

TARGET = "time.step.cross"


def kernels_of(row: dict) -> list[steptime.KernelRow]:
	"""The aten ops of a corpus row as K1 rows. The worker's op table holds three views of the same device time:
	profiler ranges (ProfilerStep*, Optimizer.step), aten ops, and the CUDA kernels under them, each with its own
	self device time, so summing all of it counts the step about 2.6 times (measured on the L4 rows). The aten ops
	carry the FLOPs and byte estimates K1 needs, and add up to about 0.9 of the step on their own."""
	dtype = row["config"]["params"].get("dtype") or "fp32"
	return [
		steptime.KernelRow(
			name=o["name"],
			seconds=float(o["device_s"]),
			flops=o.get("flops") or None,
			bytes=o.get("bytes") or None,
			dtype=dtype,
			klass=o.get("class") if o.get("class") in steptime.KERNEL_CLASSES else None,
			allow_tf32=bool(row.get("tf32")),
			calls=o.get("count") or None,
		)
		for o in row.get("ops") or []
		if o.get("device_s") is not None and str(o.get("name", "")).startswith("aten::")
	]


def cases(rows: list[dict]) -> list[dict]:
	"""One case per (config, source GPU, target GPU) where both GPUs ran the config and timed a step."""
	by_key: dict[str, dict[str, dict]] = defaultdict(dict)
	for r in rows:
		if r.get("status") != "ok" or cal.held_out(r.get("gpu")) or not (r.get("timing") or {}).get("step_s"):
			continue
		if not r.get("ops"):
			continue
		by_key[r["key"]][r["gpu"]] = r
	out = []
	for key, per in by_key.items():
		for src, s in per.items():
			ks = kernels_of(s)
			for tgt, t in per.items():
				if src == tgt:
					continue
				est = steptime.predict_cross_gpu(ks, float(s["timing"]["step_s"]), src, tgt)
				if est.status == "abstain" or not est.point:
					continue
				feats = G.features(ks, float(s["timing"]["step_s"]), src, tgt, est.breakdown or {})
				out.append(
					{
						"key": key,
						"src": src,
						"tgt": tgt,
						"group": cal.arch_key(s["config"]),
						"k1": est.point,
						"actual": float(t["timing"]["step_s"]),
						"x": feats,
					}
				)
	return out


def scores(cs: list[dict]) -> list[conformal.Score]:
	"""One conformal score per case: K1's prediction for the target GPU against what it measured. Grouped by
	architecture, so coverage never calibrates and tests on one architecture."""
	return [conformal.Score(TARGET, c["tgt"], "", c["group"], c["k1"], c["actual"]) for c in cs]


def gpu_out_coverage(cs: list[dict], level: float = conformal.LEVEL) -> tuple[int, int]:
	"""Leave-one-GPU-out: for each target GPU, the band from pairs that touch neither it as source nor as target,
	tested on the pairs that target it. This is the claim the band makes for a GPU the corpus never ran."""
	import math

	hits, n = 0, 0
	for g in sorted({c["tgt"] for c in cs}):
		train = [abs(math.log(c["actual"] / c["k1"])) for c in cs if c["src"] != g and c["tgt"] != g]
		h = conformal.conformal_half_width(train, level)
		if h is None:
			continue
		test = [abs(math.log(c["actual"] / c["k1"])) for c in cs if c["tgt"] == g]
		hits += sum(v <= h for v in test)
		n += len(test)
	return hits, n
