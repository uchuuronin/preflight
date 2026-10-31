"""Why does held-out memory miss? Diagnosed on non-A100 rows only (the A100 holdout is never re-scored or tuned on).

For each held-out query, as calibrate.scores_from_rows makes it, this splits the miss log(actual / predicted) by what
could grow with the card:
- the allocator's reserved-over-allocated gap (fragmentation and block rounding), on the query against its probes;
- the fixed part of reserved memory that doesn't scale with batch or sequence (cuBLAS/cuDNN workspaces, the CUDA
  context), measured as the intercept of reserved against work units;
- the GPU and its capacity, and the family (which kernels get picked);
- where the query sits against the probes (above the largest, below the smallest).

	uv run --extra probe python scripts/diagnose_memory_misses.py <store>
"""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from preflight.corpus import calibrate as cal
from preflight.engine import memory


def main(store: str) -> int:
	rows = [r for r in cal.load_rows(Path(store)) if not cal.held_out(r.get("gpu"))]
	groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
	for r in rows:
		g = r.get("config", {}).get("params", {}).get("group")
		if g and cal._ok(r) and r["config"]["family"] != "lora":
			groups[(g, r["gpu"])].append(r)
	out = []
	for (_g, gpu), rs in groups.items():
		probes = [r for r in rs if r["config"]["params"].get("role") == "probe"]
		queries = [r for r in rs if r["config"]["params"].get("role") == "query"]
		if len(probes) < 2 or not queries:
			continue
		an = {id(r): cal._analytical(r) or 0.0 for r in rs}
		mp = [memory.MemoryProbe(float(r["work_units"]), float(r["memory"]["peak_reserved"]), an[id(r)]) for r in probes]
		c = memory.Calibration.fit(mp)
		xs = [float(r["work_units"]) for r in probes]
		reserved = [float(r["memory"]["peak_reserved"]) for r in probes]
		slope, icept = np.polyfit(xs, reserved, 1) if len(set(xs)) > 1 else (0.0, reserved[0])
		gap_p = np.mean([math.log(r["memory"]["peak_reserved"] / max(r["memory"]["peak_allocated"], 1)) for r in probes])
		for q in queries:
			x = float(q["work_units"])
			pred = c.correct(x, an[id(q)])
			if pred <= 0:
				continue
			act = float(q["memory"]["peak_reserved"])
			gap_q = math.log(act / max(q["memory"]["peak_allocated"], 1))
			alloc_pred = pred * math.exp(-gap_p)  # what the fit implies for allocated, at the probes' gap
			out.append(
				{
					"gpu": gpu,
					"family": q["config"]["family"],
					"res": math.log(act / pred),
					"gap_shift": gap_q - gap_p,  # how much more of reserved is unused at the query than at the probes
					"alloc_res": math.log(max(q["memory"]["peak_allocated"], 1) / alloc_pred),
					"fixed_share": max(icept, 0.0) / act,  # reserved memory that doesn't grow with work
					"where": "above" if x > max(xs) else ("below" if x < min(xs) else "inside"),
					"cap_gib": (q.get("env") or {}).get("total_memory", 0) / 2**30,
				}
			)
	res = np.array([o["res"] for o in out])
	print(f"{len(out)} held-out memory queries on {sorted({o['gpu'] for o in out})}")
	print(f"under-predicted (actual above prediction by >10%): {int((res > math.log(1.1)).sum())}; over by >10%: {int((res < -math.log(1.1)).sum())}")

	def table(key):
		print(f"  by {key}:")
		for k in sorted({o[key] for o in out}):
			r = np.array([o["res"] for o in out if o[key] == k])
			under = (r > math.log(1.1)).mean()
			print(f"    {k!s:>10}: n={len(r):3d} median {np.median(r):+.3f}  p95 |res| {np.quantile(np.abs(r), 0.95):.3f}  under>10% {under:.0%}")

	for k in ("gpu", "family", "where"):
		table(k)
	gs = np.array([o["gap_shift"] for o in out])
	ar = np.array([o["alloc_res"] for o in out])
	fs = np.array([o["fixed_share"] for o in out])
	print(f"  corr(miss, allocator gap shift) {np.corrcoef(res, gs)[0, 1]:+.2f}; miss on allocated alone: median {np.median(ar):+.3f}, p95 |.| {np.quantile(np.abs(ar), 0.95):.3f}")
	big = [o for o in out if o["res"] > math.log(1.1)]
	if big:
		print(f"  the {len(big)} underestimates: median gap shift {np.median([o['gap_shift'] for o in big]):+.3f} "
			f"(all queries {np.median(gs):+.3f}); median fixed share {np.median([o['fixed_share'] for o in big]):.2f} (all {np.median(fs):.2f})")
		for o in sorted(big, key=lambda o: -o["res"])[:12]:
			print(f"    {o['gpu']:>5} {o['family']:>8} {o['where']:>6} miss {o['res']:+.3f} gap shift {o['gap_shift']:+.3f} allocated miss {o['alloc_res']:+.3f} fixed {o['fixed_share']:.2f}")
	return 0


if __name__ == "__main__":
	sys.exit(main(sys.argv[1]))
