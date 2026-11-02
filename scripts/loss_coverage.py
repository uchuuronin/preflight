"""Held-out coverage of the loss forecast's 90% band: the power law, LC-PFN and their ensemble (Nov 2 overnight list,
item 6: LC-PFN goes into the loss forecast only where its held-out coverage meets the 90% band).

The curves are the divergence sweep's runs that didn't diverge (720 short runs on L4 and A10, Nov 1; 44 diverged):
real training losses that neither half of the forecast was fitted or tuned on. Each run is seen for its first
`--seen` steps, the way a probe sees a job, and the forecast is for the loss at the run's last step, taken as the mean
of its last 10 steps (one step's loss is noisy; the forecast is of the curve).

	uv run python scripts/loss_coverage.py prepare <corpus rows .jsonl.gz ...> --out curves.json.gz
	modal run --env main python/preflight/corpus/lcpfn_app.py::coverage --curves curves.json.gz   # LC-PFN's half
	uv run python scripts/loss_coverage.py score curves.json.gz <lcpfn_coverage.json> --out report.json

`score` runs the power law and the ensemble here (plain numbers) and reads LC-PFN's quantiles from the Modal run.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path

SEEN = (60, 150)  # guess: a probe sees 10% and 25% of these 600-step runs
TAIL = 10  # guess: the truth is the mean of the last 10 steps' losses
TARGET = 0.90  # cited: the band is a 90% interval


def prepare(paths: list[Path], out: Path) -> dict:
	curves = []
	skipped = {"diverged": 0, "short or non-finite": 0}
	for p in paths:
		with gzip.open(p, "rt") as f:
			for line in f:
				r = json.loads(line)
				if (r.get("label") or {}).get("diverged"):
					skipped["diverged"] += 1
					continue
				loss = (r.get("series") or {}).get("loss") or []
				if len(loss) < max(SEEN) + TAIL or any(v is None or not math.isfinite(v) for v in loss):
					skipped["short or non-finite"] += 1
					continue
				curves.append(
					{
						"key": r["key"],
						"gpu": r.get("gpu"),
						"precision": (r.get("config") or {}).get("params", {}).get("precision"),
						"steps": len(loss),
						"seen": [round(float(v), 5) for v in loss[: max(SEEN)]],
						"truth": round(sum(loss[-TAIL:]) / TAIL, 6),
					}
				)
	doc = {"seen_steps": list(SEEN), "tail": TAIL, "skipped": skipped, "curves": curves}
	with gzip.open(out, "wt") as f:
		json.dump(doc, f)
	return {"curves": len(curves), "skipped": skipped}


def _wilson(k: int, n: int) -> tuple[float, float]:
	from preflight.engine import conformal

	return conformal.wilson(k, n)


def score(curves_path: Path, pfn_path: Path) -> dict:
	from preflight.engine import lossforecast as lf

	with gzip.open(curves_path, "rt") as f:
		doc = json.load(f)
	pfn = json.loads(pfn_path.read_text())
	pfn_by = {(r["key"], r["seen"]): r for r in pfn["rows"]}
	report: dict = {"curves": len(doc["curves"]), "tail": doc["tail"], "by_seen": {}}
	for seen in doc["seen_steps"]:
		tally = {
			m: {"covered": 0, "scored": 0, "abstained": 0, "log_err": []}
			for m in ("power", "lcpfn", "ensemble")
		}
		for c in doc["curves"]:
			truth, last = c["truth"], c["steps"]
			power = lf.forecast(c["seen"][:seen], last)
			q = pfn_by.get((c["key"], seen))
			qs = None if q is None or q.get("q") is None else tuple(q["q"])
			ens = lf.ensemble(power, qs, "LC-PFN posterior")
			got = {
				"power": None if power.point is None else (power.lo, power.point, power.hi),
				"lcpfn": qs and (qs[0], qs[1], qs[2]),
				"ensemble": None if ens.point is None else (ens.lo, ens.point, ens.hi),
			}
			for m, v in got.items():
				t = tally[m]
				if v is None:
					t["abstained"] += 1
					continue
				lo, mid, hi = v
				t["scored"] += 1
				t["covered"] += lo <= truth <= hi
				if mid > 0:
					t["log_err"].append(abs(math.log(mid / truth)))
		row = {}
		for m, t in tally.items():
			n, k = t["scored"], t["covered"]
			lo, hi = _wilson(k, n) if n else (None, None)
			errs = sorted(t["log_err"])
			row[m] = {
				"covered": k,
				"scored": n,
				"abstained": t["abstained"],
				"coverage": k / n if n else None,
				"wilson95": [lo, hi],
				"median_abs_log_err": errs[len(errs) // 2] if errs else None,
				"meets_90": bool(n) and k / n >= TARGET,
			}
		report["by_seen"][str(seen)] = row
	report["lcpfn_wired"] = all(r["ensemble"]["meets_90"] for r in report["by_seen"].values())
	return report


def main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser()
	sub = ap.add_subparsers(dest="cmd", required=True)
	a = sub.add_parser("prepare")
	a.add_argument("rows", type=Path, nargs="+")
	a.add_argument("--out", type=Path, required=True)
	b = sub.add_parser("score")
	b.add_argument("curves", type=Path)
	b.add_argument("pfn", type=Path)
	b.add_argument("--out", type=Path)
	args = ap.parse_args(argv)
	if args.cmd == "prepare":
		print(json.dumps(prepare(args.rows, args.out)))
		return 0
	rep = score(args.curves, args.pfn)
	text = json.dumps(rep, indent=1)
	if args.out:
		args.out.write_text(text + "\n")
	print(text)
	return 0


if __name__ == "__main__":
	sys.exit(main())
