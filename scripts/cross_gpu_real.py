"""The cross-GPU step-time row's real-model check (Chiya's review, Nov 2): the engine predicts a GPU no probe ran on,
from a real record, against cells measured there. The validation row itself is simulated on corpus pairs; this keeps
one pair of real runs beside it.

	uv run python scripts/cross_gpu_real.py tests/fixtures/validation/cross_gpu_real_nov2.json [--out report.json]

Prints, per cell, the prediction, its 90% band, the measured p50 step, the ratio and whether it lands within 20% (the
row's target) and inside the band.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(doc: dict) -> dict:
	from preflight.engine import api
	from preflight.schema import load_record

	rec = load_record((ROOT / doc["record"]).read_text())
	rows = []
	for c in doc["cells"]:
		e = api.predict(rec, {"gpu_type": doc["to_gpu"], "batch_size": c["batch_size"], "lr": c["lr"]}).root[
			"time.step"
		]
		ok = e.status != "abstain" and e.point is not None
		rows.append(
			{
				**c,
				"status": e.status,
				"predicted_s": e.point,
				"band": [e.lo, e.hi],
				"ratio": e.point / c["step_s"] if ok else None,
				"within_20": ok and abs(e.point / c["step_s"] - 1) <= 0.20,
				"inside": ok and e.lo <= c["step_s"] <= e.hi,
			}
		)
	n = len(rows)
	return {
		"record": doc["record"],
		"from_gpu": doc["from_gpu"],
		"to_gpu": doc["to_gpu"],
		"cells": rows,
		"within_20": sum(bool(r["within_20"]) for r in rows),
		"inside": sum(bool(r["inside"]) for r in rows),
		"n": n,
	}


def main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument("pairs", type=Path)
	ap.add_argument("--out", type=Path)
	args = ap.parse_args(argv)
	rep = check(json.loads(args.pairs.read_text()))
	for r in rep["cells"]:
		print(
			f"batch {r['batch_size']}, lr {r['lr']:g}: predicted {r['predicted_s']:.3f} s "
			f"({r['band'][0]:.3f} to {r['band'][1]:.3f}), measured {r['step_s']:.2f} s, ratio {r['ratio']:.2f}"
		)
	print(f"within 20%: {rep['within_20']} of {rep['n']}; inside the 90% band: {rep['inside']} of {rep['n']}")
	if args.out:
		args.out.write_text(json.dumps(rep, indent=1) + "\n")
	return 0


if __name__ == "__main__":
	sys.exit(main())
