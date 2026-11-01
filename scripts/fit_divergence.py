"""Fit the divergence classifier on the corpus's divergence sweep and score it on held-out runs.

	uv run --extra probe python scripts/fit_divergence.py <store> [<store> ...] [--out-model PATH] [--report PATH]

Folds are by settings family (precision x qk-norm x init), so no family is trained and tested on: every run in the
sweep is the same small GPT, so "grouped by architecture" can only mean grouped by these. Reported on held-out runs:
- AUROC: a diverged run's score is the highest P it reached at least LEAD steps before its onset (or at the first
  step it could score, when the onset comes sooner); a stable run's score is the highest P over the whole run.
- Lead: the warning line is set on the training folds for at most 10% false alarms on stable runs; a diverged
  run's lead is onset minus the first step at or above it (0 when it never warns). Median over diverged runs.
Both with a 95% bootstrap interval over families. Hyperparameters are fixed here, before any fold is scored.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

from preflight.corpus import calibrate as cal
from preflight.engine import divergence as D

LEAD = 100  # cited: plan, "100+ steps early"
STRIDE = 10  # guess: a feature row every 10 steps keeps the fit small and the warning step to within 10
FOLDS = 5
N_BOOT = 2000
SEED = 0
FALSE_ALARM = 0.10  # decision: the warning line allows 10% false alarms on stable training runs
BIN_EDGES = (0.0, 0.05, 0.2, 0.5, 0.8, 0.95, 1.0001)
MIN_RUNS = 40  # guess: below this many labelled runs the held-out numbers are noise; the report says so


def runs_of(rows: list[dict]) -> list[dict]:
	out = []
	for r in rows:
		if r.get("kind") not in (None, "divergence") or r.get("status") != "ok":
			continue
		res = r.get("result") or r
		series, label = res.get("series"), res.get("label")
		if not series or not label or label.get("reason") == "too short to label":
			continue
		p = r["config"]["params"]
		fam = (p.get("precision"), bool(p.get("qk_norm")), p.get("init_std"))
		out.append({"key": r.get("key"), "series": series, "label": label, "cfg": p, "family": fam})
	return out


def samples(run: dict) -> tuple[list[int], list[list[float]]]:
	n = len(run["series"].get("loss") or [])
	end = run["label"]["diverged_step"] if run["label"]["diverged"] else n
	ts, xs = [], []
	ctx = D.Context(run["series"])
	for t in range(D.WINDOW - 1, max(end, D.WINDOW), STRIDE):
		f = D.features_at(ctx, t, run["cfg"])
		if f is not None:
			ts.append(t)
			xs.append(f)
	return ts, xs


def auroc(pos: list[float], neg: list[float]) -> float:
	if not pos or not neg:
		return math.nan
	wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
	return wins / (len(pos) * len(neg))


def evaluate(runs: list[dict]) -> dict:
	fams = sorted({r["family"] for r in runs}, key=str)
	rng = np.random.default_rng(SEED)
	order = list(rng.permutation(len(fams)))
	fold_of = {fams[i]: k % FOLDS for k, i in enumerate(order)}
	feats = [samples(r) for r in runs]
	oof: list[dict] = []
	for k in range(FOLDS):
		tr = [i for i, r in enumerate(runs) if fold_of[r["family"]] != k]
		te = [i for i, r in enumerate(runs) if fold_of[r["family"]] == k]
		if not te or not tr:
			continue
		x = np.asarray([f for i in tr for f in feats[i][1]])
		y = np.asarray([float(runs[i]["label"]["diverged"]) for i in tr for _ in feats[i][1]])
		if len(set(y)) < 2:
			continue
		mean, scale, coef, b = D.fit(x, y)
		m = D.Model(tuple(mean), tuple(scale), tuple(coef), b)
		stable_max = [
			max((m.prob(f) for f in feats[i][1]), default=0.0) for i in tr if not runs[i]["label"]["diverged"]
		]
		tau = float(np.quantile(stable_max, 1 - FALSE_ALARM)) if stable_max else 0.5
		for i in te:
			ts, xs = feats[i]
			ps = [m.prob(f) for f in xs]
			lab = runs[i]["label"]
			if lab["diverged"]:
				cut = lab["diverged_step"] - LEAD
				early = [p for t, p in zip(ts, ps, strict=True) if t <= cut] or ps[:1]
				first = next((t for t, p in zip(ts, ps, strict=True) if p >= tau), None)
				lead = lab["diverged_step"] - first if first is not None else 0
			else:
				early, lead = ps, None
			oof.append(
				{
					"family": runs[i]["family"],
					"y": lab["diverged"],
					"score": max(early, default=0.0),
					"lead": lead,
					"p_max": max(ps, default=0.0),
				}
			)
	return oof


def summarise(oof: list[dict]) -> dict:
	def stats(rows):
		pos = [r["score"] for r in rows if r["y"]]
		neg = [r["score"] for r in rows if not r["y"]]
		leads = [r["lead"] for r in rows if r["y"]]
		return auroc(pos, neg), (float(np.median(leads)) if leads else math.nan)

	a, lead = stats(oof)
	fams = sorted({r["family"] for r in oof}, key=str)
	by = {f: [r for r in oof if r["family"] == f] for f in fams}
	rng = np.random.default_rng(SEED)
	boots = []
	for _ in range(N_BOOT):
		pick = [r for f in rng.choice(len(fams), len(fams)) for r in by[fams[f]]]
		boots.append(stats(pick))
	bs = np.asarray([b for b in boots if all(math.isfinite(v) for v in b)])
	lo_a, hi_a = np.quantile(bs[:, 0], [0.025, 0.975]) if len(bs) else (math.nan, math.nan)
	lo_l, hi_l = np.quantile(bs[:, 1], [0.025, 0.975]) if len(bs) else (math.nan, math.nan)
	bins = []
	for lo, hi in itertools.pairwise(BIN_EDGES):
		inb = [r for r in oof if lo <= r["p_max"] < hi]
		if inb:
			bins.append(
				(
					lo,
					min(hi, 1.0),
					len(inb),
					float(np.mean([r["p_max"] for r in inb])),
					float(np.mean([r["y"] for r in inb])),
				)
			)
	return {
		"n": len(oof),
		"diverged": sum(r["y"] for r in oof),
		"families": len(fams),
		"auroc": {"point": a, "lo": float(lo_a), "hi": float(hi_a)},
		"lead_steps": {"point": lead, "lo": float(lo_l), "hi": float(hi_l)},
		"bins": bins,
		"enough": len(oof) >= MIN_RUNS,
	}


def write_model(model: D.Model, out: Path) -> None:
	from preflight.corpus.calibrate import ruff_formatted

	text = (
		'"""Generated by scripts/fit_divergence.py; do not edit. The divergence classifier, fitted on the corpus."""\n\n'
		"from preflight.engine.divergence import Model\n\n"
		f"MODEL = {model!r}\n"
	)
	out.write_text(ruff_formatted(text, out))


def main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument("stores", nargs="+", type=Path)
	ap.add_argument("--out-model", type=Path)
	ap.add_argument("--report", type=Path)
	args = ap.parse_args(argv)
	runs = runs_of([r for p in args.stores for r in cal.load_rows(p)])
	print(f"{len(runs)} labelled runs, {sum(r['label']['diverged'] for r in runs)} diverged")
	if len({r["label"]["diverged"] for r in runs}) < 2:
		print("both outcomes are needed to fit; nothing written")
		return 1
	rep = summarise(evaluate(runs))
	a, s = rep["auroc"], rep["lead_steps"]
	print(
		f"held out: AUROC {a['point']:.3f} ({a['lo']:.3f} to {a['hi']:.3f}); lead {s['point']:.0f} steps ({s['lo']:.0f} to {s['hi']:.0f}); {rep['families']} families"
	)
	if not rep["enough"]:
		print(f"only {rep['n']} runs: below {MIN_RUNS}, the numbers are noise")
	feats = [samples(r) for r in runs]
	x = np.asarray([f for _, xs in feats for f in xs])
	y = np.asarray(
		[float(r["label"]["diverged"]) for r, (_, xs) in zip(runs, feats, strict=True) for _ in xs]
	)
	mean, scale, coef, b = D.fit(x, y)
	m0 = D.Model(tuple(mean), tuple(scale), tuple(coef), b)
	stable = [
		max((m0.prob(f) for f in xs), default=0.0)
		for r, (_, xs) in zip(runs, feats, strict=True)
		if not r["label"]["diverged"]
	]
	tau = float(np.quantile(stable, 1 - FALSE_ALARM)) if stable else 0.5
	note = f"held out: AUROC {a['point']:.2f}, lead {s['point']:.0f} steps, {rep['n']} runs, {rep['families']} families"
	model = D.Model(
		tuple(float(v) for v in mean), tuple(float(v) for v in scale), tuple(float(v) for v in coef), float(b),
		tuple(tuple(x) for x in rep["bins"]), tau, note,
	)  # fmt: skip
	if args.out_model:
		write_model(model, args.out_model)
	if args.report:
		args.report.write_text(json.dumps(rep, indent=1, default=str))
	return 0


if __name__ == "__main__":
	sys.exit(main())
