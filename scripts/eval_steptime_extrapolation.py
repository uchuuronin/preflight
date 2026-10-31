"""Score same-GPU step-time extrapolation, the fitted line against the anchored model, on corpus groups.

Each group is fitted on its three probes and scored on its two held-out queries, as calibrate.py does. Queries are
split by where they fall: above the largest probe, below the smallest, or inside. The anchored model's elasticities
were fitted on the L4 HF jobs (steptime.ABOVE_ELASTICITY, BELOW_ELASTICITY), so those rows are the design set and
are reported apart; everything else is held out. Proportions carry Wilson 95% intervals.

	uv run python scripts/eval_steptime_extrapolation.py <store> [<store> ...]
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

from preflight.corpus import calibrate as cal
from preflight.engine import conformal, steptime

TOL = 0.15  # cited: Project Goals, step time within 15% on the same GPU


def scored(rows, mode: str):
	steptime.EXTRAPOLATION = mode
	groups = defaultdict(list)
	for r in rows:
		g = r.get("config", {}).get("params", {}).get("group")
		if g and cal._ok(r) and (r.get("timing") or {}).get("step_s"):
			groups[(g, r["gpu"])].append(r)
	out = []
	for (_g, gpu), rs in groups.items():
		probes = [r for r in rs if r["config"]["params"].get("role") == "probe"]
		queries = [r for r in rs if r["config"]["params"].get("role") == "query"]
		if len(probes) < 2:
			continue
		model = steptime.SameGpuModel.fit(
			[steptime.StepProbe(float(r["work_units"]), cal._phases(r), str(i)) for i, r in enumerate(probes)]
		)
		lo, hi = min(model.probe_tokens), max(model.probe_tokens)
		for q in queries:
			x = float(q["work_units"])
			e = model.predict(x)
			where = "above" if x > hi else "below" if x < lo else "inside"
			design = q["config"]["family"] == "hf_lora" and gpu == "L4"
			m = float(q["timing"]["step_s"])
			out.append((gpu, q["config"]["family"], where, design, e, m))
	return out


def summary(rows, label):
	n = len(rows)
	if not n:
		return f"{label}: none"
	scored_ = [(e, m) for e, m in rows if e.status != "abstain" and e.point]
	hits = sum(abs(e.point / m - 1) <= TOL for e, m in scored_)
	inside = sum(e.lo <= m <= e.hi for e, m in scored_)
	k = len(scored_)
	med = sorted(abs(math.log(e.point / m)) for e, m in scored_)[k // 2] if k else float("nan")
	w1, w2 = conformal.wilson(hits, k), conformal.wilson(inside, k)
	return (
		f"{label}: n={n}, abstained {n - k}; within 15% {hits}/{k} ({w1[0]:.0%} to {w1[1]:.0%}); "
		f"inside band {inside}/{k} ({w2[0]:.0%} to {w2[1]:.0%}); median |log error| {med:.3f}"
	)


def main(paths: list[str]) -> int:
	rows = [r for p in paths for r in cal.load_rows(Path(p))]
	for mode in ("line", "anchored"):
		res = scored(rows, mode)
		print(f"== {mode}")
		for design in (True, False):
			name = "design set (L4 HF)" if design else "held out"
			for where in ("above", "below", "inside"):
				sel = [(e, m) for _, _, w, d, e, m in res if d == design and w == where]
				print("  " + summary(sel, f"{name}, {where}"))
	steptime.EXTRAPOLATION = "line"
	return 0


if __name__ == "__main__":
	raise SystemExit(main(sys.argv[1:]))
