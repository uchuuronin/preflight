"""The divergence classifier judged on runs at the learning rates people train at (Chiya's review, Nov 2).

	uv run --extra probe python scripts/eval_divergence_real.py --train <Nov 1 store> --test <divergence_real store> \\
		[--report PATH]

The Nov 1 sweep's 44 divergences all came at lr 0.07 or more, so its held-out AUROC of 0.66 measured the classifier on
runs nobody trains. Here the Nov 1 step model is fitted once on all 720 Nov 1 runs, with the Nov 1 warning line (10%
false alarms on stable runs it didn't train on, from inner folds), and scored once on every run of the realistic sweep
(lr 1e-4 to 3e-2, no clipping, no warm-up, fp16 without a scaler for half), which it never saw: the same AUROC, lead
and bootstrap over settings families as the plan's row (AUROC 0.85, 100 steps early). Fixed before the sweep's rows
landed. It reports how many runs diverged and at what learning rates, since a sweep with few divergences can't test
a warning either.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_divergence as F

from preflight.corpus import calibrate as cal


def evaluate(train: list[dict], test: list[dict]) -> list[dict]:
	ftr = [F.samples(r) for r in train]
	idx = list(range(len(train)))
	m = F._fit_model(train, ftr, idx)
	if m is None:
		raise SystemExit("the training runs hold no divergence to fit on")
	tau = F.warning_line(train, ftr, idx)
	oof = []
	for r in test:
		ts, xs = F.samples(r)
		ps = [m.prob(f) for f in xs]
		lab = r["label"]
		if lab["diverged"]:
			cut = lab["diverged_step"] - F.LEAD
			early = [p for t, p in zip(ts, ps, strict=True) if t <= cut] or ps[:1]
			first = next((t for t, p in zip(ts, ps, strict=True) if p >= tau), None)
			lead = lab["diverged_step"] - first if first is not None else 0
		else:
			early, lead = ps, None
		oof.append({"family": r["family"], "y": lab["diverged"], "score": max(early, default=0.0), "lead": lead,
			"p_max": max(ps, default=0.0), "alarm": max(ps, default=0.0) >= tau})  # fmt: skip
	return oof


def main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument("--train", type=Path, nargs="+", required=True)
	ap.add_argument("--test", type=Path, nargs="+", required=True)
	ap.add_argument("--report", type=Path)
	args = ap.parse_args(argv)
	train = F.runs_of([r for p in args.train for r in cal.load_rows(p)])
	test = F.runs_of([r for p in args.test for r in cal.load_rows(p)], kinds=("divergence_real",))
	if not test:
		raise SystemExit("no realistic-sweep runs in --test")
	oof = evaluate(train, test)
	d = [r for r in test if r["label"]["diverged"]]
	rep = {"train_runs": len(train), "test_runs": len(test), "test_diverged": len(d),
		"diverged_lrs": sorted(r["cfg"]["lr"] for r in d),
		"diverged_by_precision": {k: sum(r["cfg"]["precision"] == k for r in d) for k in ("fp16", "bf16", "fp32")},
		"runs_by_precision": {k: sum(r["cfg"]["precision"] == k for r in test) for k in ("fp16", "bf16", "fp32")}}  # fmt: skip
	if len(d) and len(d) < len(test):
		rep.update(F.summarise(oof))
		rep["meets_row"] = rep["auroc"]["point"] >= 0.85 and rep["lead_steps"]["point"] >= F.LEAD
	else:
		rep["verdict"] = "no divergence (or no stable run) to score: the sweep can't test the warning"
	text = json.dumps(rep, indent=1, default=float)
	if args.report:
		args.report.write_text(text + "\n")
	print(text[:3000])
	return 0


if __name__ == "__main__":
	sys.exit(main())
