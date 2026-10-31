"""Cross-GPU step time: K1's per-kernel replay, and a gradient-boosted residual on top of it, scored leave-one-GPU-out.

Every corpus config ran on several GPUs, so each ordered pair of GPUs that both ran it gives one case: replay the
source GPU's kernel table onto the target with K1, and compare with what the target measured. The residual model
learns log(actual / K1) from features of the pair and the kernel mix. Its error is reported only on GPUs it never
saw: for each GPU, train on pairs that don't touch it, test on pairs that target it. The A100 boards are the
cross-GPU holdout and never enter (calibrate.held_out).

	uv run --extra probe --with scikit-learn python scripts/fit_steptime_gbdt.py <store> [--out python/preflight/engine/_steptime_gbdt.py]

The engine never imports scikit-learn: --out writes the fitted trees as a plain literal that engine/steptime_gbdt.py
walks with numpy, so it runs in Pyodide.
"""

from __future__ import annotations

import argparse
import pprint
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from preflight.corpus import calibrate as cal
from preflight.engine import conformal, steptime
from preflight.engine import steptime_gbdt as G

TOL = 0.20  # cited: main plan, step time on an unmeasured GPU within 20%


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


def within(pred: np.ndarray, actual: np.ndarray) -> tuple[int, int]:
	ok = np.abs(pred / actual - 1) <= TOL
	return int(ok.sum()), int(ok.size)


def fit(x: np.ndarray, y: np.ndarray):
	from sklearn.ensemble import GradientBoostingRegressor

	# guess: shallow trees, strong shrinkage, few hundred rows per GPU pair; fixed before looking at test GPUs
	m = GradientBoostingRegressor(
		n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, min_samples_leaf=20, random_state=0
	)
	return m.fit(x, y)


def export(model) -> dict:
	trees = []
	for est in model.estimators_[:, 0]:
		t = est.tree_
		trees.append(
			{
				"feature": [int(v) for v in t.feature],
				"threshold": [float(v) for v in t.threshold],
				"left": [int(v) for v in t.children_left],
				"right": [int(v) for v in t.children_right],
				"value": [float(v[0][0]) for v in t.value],
			}
		)
	return {
		"version": "0.1.0",
		"features": list(G.FEATURES),
		"init": float(model.init_.constant_[0][0])
		if hasattr(model.init_, "constant_")
		else float(model._raw_predict_init(np.zeros((1, len(G.FEATURES))))[0][0]),
		"learning_rate": float(model.learning_rate),
		"trees": trees,
	}


def main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument("stores", nargs="+", type=Path)
	ap.add_argument("--out", type=Path)
	args = ap.parse_args(argv)
	rows = [r for p in args.stores for r in cal.load_rows(p)]
	cs = cases(rows)
	if not cs:
		print("no cross-GPU cases")
		return 1
	x = np.asarray([c["x"] for c in cs], dtype=float)
	y = np.log(np.asarray([c["actual"] for c in cs]) / np.asarray([c["k1"] for c in cs]))
	tgts = sorted({c["tgt"] for c in cs})
	print(f"{len(cs)} cross-GPU cases over {len({c['key'] for c in cs})} configs; GPUs {tgts}")
	tot_k1, tot_gb, n_all = 0, 0, 0
	for g in tgts:
		train = np.array([c["src"] != g and c["tgt"] != g for c in cs])
		test = np.array([c["tgt"] == g for c in cs])
		if train.sum() < 50 or test.sum() == 0:
			print(f"  {g}: too few cases to hold out")
			continue
		m = fit(x[train], y[train])
		actual = np.asarray([c["actual"] for c in cs])[test]
		k1 = np.asarray([c["k1"] for c in cs])[test]
		gb = k1 * np.exp(m.predict(x[test]))
		h1, n = within(k1, actual)
		h2, _ = within(gb, actual)
		tot_k1, tot_gb, n_all = tot_k1 + h1, tot_gb + h2, n_all + n
		w1, w2 = conformal.wilson(h1, n), conformal.wilson(h2, n)
		print(
			f"  target {g}: n={n}; K1 within 20% {h1}/{n} ({w1[0]:.0%} to {w1[1]:.0%}), median |log| "
			f"{np.median(np.abs(np.log(k1 / actual))):.3f}; K1+GBDT {h2}/{n} ({w2[0]:.0%} to {w2[1]:.0%}), median |log| "
			f"{np.median(np.abs(np.log(gb / actual))):.3f}"
		)
	w1, w2 = conformal.wilson(tot_k1, n_all), conformal.wilson(tot_gb, n_all)
	print(
		f"leave-one-GPU-out: K1 {tot_k1}/{n_all} ({w1[0]:.0%} to {w1[1]:.0%}); K1+GBDT {tot_gb}/{n_all} ({w2[0]:.0%} to {w2[1]:.0%})"
	)
	if args.out:
		m = fit(x, y)
		data = export(m)
		data["held_out"] = {"k1": [tot_k1, n_all], "gbdt": [tot_gb, n_all], "tolerance": TOL}
		data["built_from"] = (
			f"{len(cs)} cases from {', '.join(str(p) for p in args.stores)}; A100 boards withheld"
		)
		body = pprint.pformat(data, indent=1, width=110, sort_dicts=True, compact=True)
		args.out.write_text(
			'"""Generated by scripts/fit_steptime_gbdt.py; do not edit. Cross-GPU step-time residual trees."""\n\n'
			f"DATA = {body}\n"
		)
		print(f"wrote {args.out}")
	return 0


if __name__ == "__main__":
	sys.exit(main())
