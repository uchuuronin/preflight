"""NVIDIA to MI300X step time on the MI300X corpus slice, scored once (Chiya, Nov 2, 9:01 PM item 2). No refit.

	uv run --extra probe python scripts/score_amd_cross.py <nvidia store> <amd store> <out.json>

Each case is a config both an NVIDIA GPU and the MI300X ran: K1 from the NVIDIA row's op table and measured step
(`steptime.predict_cross_gpu`, as `corpus/crossgpu.cases` does within NVIDIA) against the MI300X's measured step.
Rows pair on (family, kind, params), not the key, since the key holds the torch version (2.8.0 on the NVIDIA rows,
2.5.1+rocm6.2 on the MI300X). Reports within 20% and K1's own 90% band coverage, each with a Wilson interval and a
bootstrap over architecture groups (pairs share configs, and configs share groups, so cases aren't independent).
"""

from __future__ import annotations

import collections
import json
import math
import random
import sys
from pathlib import Path

from preflight.corpus import calibrate as cal
from preflight.corpus import crossgpu
from preflight.engine import gpus, steptime

BOOTSTRAP = 4000  # derived: as overhead.py; the interval's ends move under a point between seeds
SEED = 20261102


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
	p = k / n
	d = 1 + z * z / n
	c = (p + z * z / (2 * n)) / d
	h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
	return [round(c - h, 4), round(c + h, 4)]


def ident(r: dict) -> str:
	c = r.get("config") or {}
	return json.dumps([c.get("family"), c.get("kind"), c.get("params")], sort_keys=True)


def score(rows_nv: list[dict], rows_amd: list[dict]) -> dict:
	for r in rows_nv + rows_amd:
		r["key"] = ident(r)
	by_key = {(r["key"], r["gpu"]): r for r in rows_nv}
	cases = []
	for c in crossgpu.cases(rows_nv + rows_amd):
		if gpus.get_gpu(c["tgt"]).vendor != "amd" or gpus.get_gpu(c["src"]).vendor == "amd":
			continue
		s = by_key[(c["key"], c["src"])]
		est = steptime.predict_cross_gpu(
			crossgpu.kernels_of(s), float(s["timing"]["step_s"]), c["src"], c["tgt"]
		)
		cfg = json.loads(c["key"])
		cases.append({**c, "within": abs(c["k1"] / c["actual"] - 1) <= 0.20,
			"inside": est.lo <= c["actual"] <= est.hi, "family": cfg[0], "dtype": cfg[2].get("dtype"),
			"group": cfg[2].get("group")})  # fmt: skip
	out: dict = {"amd_rows": len(rows_amd), "amd_status": dict(collections.Counter(r.get("status") for r in rows_amd)),
		"cases": len(cases)}  # fmt: skip
	if not cases:
		return out
	n = len(cases)
	w = sum(c["within"] for c in cases)
	inside = sum(c["inside"] for c in cases)
	logs = sorted(math.log(c["k1"] / c["actual"]) for c in cases)
	groups: dict = collections.defaultdict(list)
	for c in cases:
		groups[c["group"]].append(c)
	rng = random.Random(SEED)
	names = list(groups)
	bw, bc = [], []
	for _ in range(BOOTSTRAP):
		pick = [c for g in rng.choices(names, k=len(names)) for c in groups[g]]
		bw.append(sum(c["within"] for c in pick) / len(pick))
		bc.append(sum(c["inside"] for c in pick) / len(pick))
	bw.sort()
	bc.sort()
	lo, hi = int(0.025 * BOOTSTRAP), int(0.975 * BOOTSTRAP) - 1

	def split(field: str) -> dict:
		t: dict = collections.defaultdict(lambda: {"within_20": 0, "covered": 0, "n": 0})
		for c in cases:
			t[c[field]]["within_20"] += c["within"]
			t[c[field]]["covered"] += c["inside"]
			t[c[field]]["n"] += 1
		return dict(sorted(t.items()))

	out.update({
		"configs": len({c["key"] for c in cases}),
		"groups": len(names),
		"within_20": [w, n, wilson(w, n)],
		"within_20_group_bootstrap_95": [round(bw[lo], 4), round(bw[hi], 4)],
		"band_covers": [inside, n, wilson(inside, n)],
		"band_covers_group_bootstrap_95": [round(bc[lo], 4), round(bc[hi], 4)],
		"band_sigma": steptime.SIGMA_CROSS_VENDOR,
		"band_90_factor": round(math.exp(steptime.Z90 * steptime.SIGMA_CROSS_VENDOR), 3),
		"median_ratio_k1_over_actual": round(math.exp(logs[n // 2]), 3),
		"q05_q95_ratio": [round(math.exp(logs[int(0.05 * (n - 1))]), 3), round(math.exp(logs[int(0.95 * (n - 1))]), 3)],
		"by_source": split("src"),
		"by_family": split("family"),
		"by_dtype": split("dtype"),
		"torch": {"nvidia": sorted({str((r["config"].get("versions") or {}).get("torch")) for r in rows_nv}),
			"mi300x": sorted({str((r["config"].get("versions") or {}).get("torch")) for r in rows_amd})},
	})  # fmt: skip
	return out


def main(argv: list[str]) -> int:
	nv, amd, dest = Path(argv[0]), Path(argv[1]), Path(argv[2])
	res = score(cal.load_rows(nv), cal.load_rows(amd))
	dest.write_text(json.dumps(res, indent=1))
	print(json.dumps({k: v for k, v in res.items() if not k.startswith("by_")}, indent=1))
	return 0


if __name__ == "__main__":
	sys.exit(main(sys.argv[1:]))
