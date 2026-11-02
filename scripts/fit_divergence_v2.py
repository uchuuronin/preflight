"""Refit of the divergence warning aimed at the early fp16 blow-ups (PM, Nov 2, item 7), scored held out once.

	uv run --extra probe python scripts/fit_divergence_v2.py <store> [<store> ...] [--report PATH]

The Nov 1 classifier (scripts/fit_divergence.py) reads a 25-step window, so the 15 of 44 divergences that come in the
first 30 steps (37 of 44 are fp16 blow-ups) give it almost nothing to read. This adds a prior from the settings alone,
which are known before the run starts: a logistic model on the same settings features plus the fp16 interactions the
sweep's blow-ups suggest (fp16 x learning rate, fp16 x no warmup, fp16 x no qk-norm, fp16 x no clipping, no clipping
x learning rate), one row per run. A run's score at each step is the larger of the prior and the step model; the prior
counts from step 0, so a run it flags is warned before its first step.

The design was fixed before any fold was scored, and it is scored once, with the Nov 1 script's own folds (by
settings family), warning line (10% false alarms on held-out stable runs, from inner folds), AUROC, lead and
bootstrap. It ships only if it meets the plan's row (AUROC 0.85 and a median lead of 100 steps) on runs that can test
it: the report also gives the learning rate alone on the same folds, and how many runs at a usual learning rate
diverged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from preflight.corpus import calibrate as cal
from preflight.engine import divergence as D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fit_divergence as F


def prior_features(cfg: dict) -> list[float | None]:
	"""The settings features and five fp16 interactions: fp16 x learning rate, x no warmup, x no qk-norm, x no
	clipping, and no clipping x learning rate."""
	base = D.config_features(cfg)
	log_lr, no_warm, _init, fp16, _bf16, qk, _z, _b2, no_clip = base

	def x(a, b):
		return None if a is None or b is None else a * b

	return [*base, x(fp16, log_lr), x(fp16, no_warm), x(fp16, None if qk is None else 1 - qk), x(fp16, no_clip),
		x(no_clip, log_lr)]  # fmt: skip


def _fit_prior(runs, idx):
	xm = F._matrix([prior_features(runs[i]["cfg"]) for i in idx])
	y = np.asarray([float(runs[i]["label"]["diverged"]) for i in idx])
	if len(set(y)) < 2:
		return None
	mean, scale, coef, b = D.fit(xm, y)
	return D.Model(tuple(mean), tuple(scale), tuple(coef), b)


USUAL_LR = (
	0.03  # guess: well above the usual 1e-4 to 3e-3 for these models; still no divergence at or under it
)


def _lr_only_auroc(runs) -> float:
	"""The learning rate alone, the same folds: how much of the score the sweep's learning-rate range carries."""
	oof = []
	for tr, te in F._folds(runs, list(range(len(runs))), F.FOLDS, F.SEED):
		x = F._matrix([[D.config_features(runs[i]["cfg"])[0]] for i in tr])
		y = np.asarray([float(runs[i]["label"]["diverged"]) for i in tr])
		mean, scale, coef, b = D.fit(x, y)
		m = D.Model(tuple(mean), tuple(scale), tuple(coef), b)
		oof += [(runs[i]["label"]["diverged"], m.prob([D.config_features(runs[i]["cfg"])[0]])) for i in te]
	return F.auroc([p for y, p in oof if y], [p for y, p in oof if not y])


def _line(runs, feats, tr):
	"""The warning line on held-out stable runs from inner folds inside `tr`, for the combined score."""
	stable = []
	for tr2, te2 in F._folds(runs, tr, F.FOLDS - 1, F.SEED + 1):
		pr, st = _fit_prior(runs, tr2), F._fit_model(runs, feats, tr2)
		if st is None:
			continue
		for i in te2:
			if runs[i]["label"]["diverged"]:
				continue
			p0 = pr.prob(prior_features(runs[i]["cfg"])) if pr is not None else 0.0
			stable.append(max([p0, *(max(p0, st.prob(f)) for f in feats[i][1])]))
	return float(np.quantile(stable, 1 - F.FALSE_ALARM)) if stable else 0.5


def evaluate(runs):
	feats = [F.samples(r) for r in runs]
	cfgs = {i: r["cfg"] for i, r in enumerate(runs)}
	oof = []

	def scores(prior, step, i):
		p0 = prior.prob(prior_features(cfgs[i])) if prior is not None else 0.0
		ts, xs = feats[i]
		return [0, *ts], [p0, *(max(p0, step.prob(f)) for f in xs)]

	def line(tr):
		return _line(runs, feats, tr)

	for tr, te in F._folds(runs, list(range(len(runs))), F.FOLDS, F.SEED):
		pr, st = _fit_prior(runs, tr), F._fit_model(runs, feats, tr)
		if st is None or not te:
			continue
		tau = line(tr)
		for i in te:
			ts, ps = scores(pr, st, i)
			lab = runs[i]["label"]
			if lab["diverged"]:
				cut = lab["diverged_step"] - F.LEAD
				early = [p for t, p in zip(ts, ps, strict=True) if t <= cut] or ps[:1]
				first = next((t for t, p in zip(ts, ps, strict=True) if p >= tau), None)
				lead = lab["diverged_step"] - first if first is not None else 0
			else:
				early, lead = ps, None
			oof.append({"family": runs[i]["family"], "y": lab["diverged"], "score": max(early), "lead": lead,
				"p_max": max(ps), "alarm": max(ps) >= tau, "early_onset": (lab.get("diverged_step") or 0) < 30})  # fmt: skip
	return oof


def main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument("stores", nargs="+", type=Path)
	ap.add_argument("--report", type=Path)
	ap.add_argument(
		"--out-model", type=Path, help="write engine/_divergence_model.py, only if the row is met"
	)
	args = ap.parse_args(argv)
	runs = F.runs_of([r for p in args.stores for r in cal.load_rows(p)])
	oof = evaluate(runs)
	rep = F.summarise(oof)
	early = [r for r in oof if r["y"] and r["early_onset"]]
	rep["early_onset_warned"] = {"warned": sum(r["alarm"] for r in early), "of": len(early)}
	met = rep["auroc"]["point"] >= 0.85 and rep["lead_steps"]["point"] >= F.LEAD
	# review, Nov 2: what the row measures on this sweep. Every divergence comes at a learning rate the sweep drew far
	# above where people train, so the learning rate alone ranks the runs nearly as well, and no run at a usual rate
	# diverged at all: the warning can't be checked where users train
	lr_only = _lr_only_auroc(runs)
	d_lrs = sorted(r["cfg"]["lr"] for r in runs if r["label"]["diverged"])
	usual = [r for r in runs if r["cfg"]["lr"] <= USUAL_LR]
	rep["learning_rate_alone_auroc"] = lr_only
	rep["lowest_diverged_lr"] = d_lrs[0] if d_lrs else None
	rep["at_usual_lr"] = {
		"lr_at_most": USUAL_LR,
		"runs": len(usual),
		"diverged": sum(r["label"]["diverged"] for r in usual),
	}
	testable = rep["at_usual_lr"]["diverged"] > 0
	rep["verdict"] = (
		"meets the row on the sweep, but not shipped: no run at a usual learning rate diverged, so the warning "
		"can't be checked where users train"
		if met and not testable
		else "meets the row: may ship"
		if met
		else "misses the row: not shipped"
	)
	rep["design"] = "prior on settings (+5 fp16 interactions) max the Nov 1 step model; scored once"
	text = json.dumps(rep, indent=1, default=float)
	if args.report:
		args.report.write_text(text + "\n")
	a, s, fa = rep["auroc"], rep["lead_steps"], rep["false_alarms"]
	print(f"{rep['n']} runs, {rep['diverged']} diverged, {rep['families']} families")
	print(
		f"held out: AUROC {a['point']:.3f} ({a['lo']:.3f} to {a['hi']:.3f}); lead {s['point']:.0f} ({s['lo']:.0f} to {s['hi']:.0f})"
	)
	print(
		f"false alarms {fa['alarms']} of {fa['stable']}; early onsets warned {rep['early_onset_warned']}; {rep['verdict']}"
	)
	return 0


if __name__ == "__main__":
	sys.exit(main())
