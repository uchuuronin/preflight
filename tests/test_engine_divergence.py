"""The divergence classifier (engine/divergence.py, scripts/fit_divergence.py) on synthetic sweeps."""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from preflight.corpus.worker import label_divergence
from preflight.engine import divergence as D

ROOT = Path(__file__).resolve().parents[1]


def _fit_script():
	spec = importlib.util.spec_from_file_location("fit_divergence", ROOT / "scripts" / "fit_divergence.py")
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)
	return mod


def _run(i: int, rng, diverge: bool, steps: int = 400) -> dict:
	"""A loss curve that decays; a diverging one has its attention logit grow from step 60, then blows up later."""
	onset = int(rng.integers(220, 330))
	t = np.arange(steps)
	loss = 2.5 + 3.0 * np.exp(-t / 60) + rng.normal(0, 0.03, steps)
	attn = 5 + rng.normal(0, 0.2, steps)
	if diverge:
		attn = attn + np.clip(t - 60, 0, None) * 0.08
		loss = np.where(t > onset, loss + (t - onset) * 0.05, loss)
	series = {
		"loss": [float(x) for x in loss],
		"grad_norm": [float(x) for x in 1 + rng.normal(0, 0.05, steps)],
		"max_attn_logit": [float(x) for x in attn],
		"output_logit_norm": [float(x) for x in 10 + rng.normal(0, 0.1, steps)],
		"update_ratio": [1e-3] * steps,
		"loss_scale": [None] * steps,
	}
	cfg = {
		"lr": 0.05 if diverge else 0.005,
		"warmup": 20,
		"init_std": [0.02, 0.05, 0.1, 0.2][i % 4],
		"precision": ["fp32", "bf16", "fp16"][i % 3],
		"qk_norm": i % 2 == 0,
		"z_loss": 0.0,
		"beta2": 0.95,
		"grad_clip": 1.0,
	}
	return {"key": f"r{i}", "series": series, "label": label_divergence(series["loss"], 256), "cfg": cfg,
		"family": (cfg["precision"], cfg["qk_norm"], cfg["init_std"])}  # fmt: skip


@pytest.fixture(scope="module")
def sweep():
	rng = np.random.default_rng(1)
	return [_run(i, rng, diverge=i % 3 == 0) for i in range(120)]


def test_the_label_and_the_synthetic_runs_agree(sweep):
	assert sum(r["label"]["diverged"] for r in sweep) == 40


def test_features_see_only_the_past(sweep):
	r = sweep[0]
	cut = {k: v[:100] for k, v in r["series"].items()}
	assert D.step_features(r["series"], 99, r["cfg"]) == D.step_features(cut, 99, r["cfg"])
	assert D.step_features(r["series"], D.WINDOW - 2, r["cfg"]) is None
	assert len(D.step_features(r["series"], 50, r["cfg"])) == len(D.FEATURES)


def test_held_out_scores_on_a_learnable_sweep(sweep):
	fd = _fit_script()
	rep = fd.summarise(fd.evaluate(sweep))
	assert rep["auroc"]["point"] > 0.85 and rep["auroc"]["lo"] <= rep["auroc"]["point"] <= rep["auroc"]["hi"]
	assert rep["lead_steps"]["point"] >= 100
	assert rep["families"] == 12 and rep["enough"]


def test_the_fitted_model_round_trips_and_warns(sweep, tmp_path):
	fd = _fit_script()
	store = tmp_path / "store"
	store.mkdir()
	import gzip
	import json

	with gzip.open(store / "L4.jsonl.gz", "wt") as fh:
		for r in sweep:
			fh.write(json.dumps({"kind": "divergence", "status": "ok", "key": r["key"], "config": {"params": r["cfg"]},
				"series": r["series"], "label": r["label"]}) + "\n")  # fmt: skip
	out, rep = tmp_path / "_divergence_model.py", tmp_path / "rep.json"
	assert fd.main([str(store), "--out-model", str(out), "--report", str(rep)]) == 0
	spec = importlib.util.spec_from_file_location("m", out)
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)
	m = mod.MODEL
	bad = next(r for r in sweep if r["label"]["diverged"])
	good = next(r for r in sweep if not r["label"]["diverged"])
	head = {k: v[:200] for k, v in bad["series"].items()}
	w = D.warn(m, head, bad["cfg"])
	assert w["status"] == "abstain" or w["point"] >= m.threshold
	assert D.warn(m, {k: v[:10] for k, v in good["series"].items()}, good["cfg"])["status"] == "abstain"
	assert json.loads(rep.read_text())["auroc"]["point"] > 0.85


def test_without_a_model_or_a_verified_bin_it_abstains():
	assert D.warn(None, {"loss": [1.0] * 50}, {})["status"] == "abstain"
	m = D.Model(
		(0.0,) * len(D.FEATURES),
		(1.0,) * len(D.FEATURES),
		(0.0,) * len(D.FEATURES),
		0.0,
		bins=((0.0, 1.0, 3, 0.5, 0.5),),
	)
	w = D.warn(m, {"loss": [1.0] * 50}, {})
	assert w["status"] == "abstain" and w["reason"].startswith("uncertain")


def test_logistic_fit_recovers_a_known_direction():
	rng = np.random.default_rng(0)
	x = rng.normal(size=(2000, 2))
	y = (x[:, 0] + 0.1 * rng.normal(size=2000) > 0).astype(float)
	_, _, coef, _ = D.fit(x, y, l2=0.1)
	assert coef[0] > 3 and abs(coef[1]) < 0.5
	assert math.isfinite(D.fit(x, y)[3])


def test_the_engine_target_is_behind_its_flag(monkeypatch):
	from preflight.engine import api
	from preflight.schema import load_record

	ft = load_record((ROOT / "schema" / "examples" / "finetune.json").read_text())
	assert "risk.divergence" not in api.predict(ft).root
	monkeypatch.setitem(api.FEATURES, "divergence", True)
	e = api.predict(ft).root["risk.divergence"]
	assert e.status == "abstain" and e.unit == "probability"  # no fitted model yet, or no loss series
	(card,) = api._cards({D.MODEL_ID})
	assert card.kind == "classifier"


def test_user_knob_ids_reach_the_settings_and_unknowns_sit_at_the_average():
	# review, Nov 1: inference once read none of the user's knob ids, so every probe looked unwarmed and unclipped
	assert D.config_features({"warmup_steps": 100})[1] == 0.0
	assert D.config_features({"dtype": "bf16"})[4] == 1.0
	assert all(v is None for v in D.config_features({}))
	n = len(D.FEATURES)
	m = D.Model(tuple(float(i) for i in range(n)), (1.0,) * n, (1.0,) * n, 0.0)
	assert m.prob([None] * n) == 0.5  # every unknown at its mean: no push either way


def test_the_report_gives_held_out_false_alarms(sweep):
	fd = _fit_script()
	rep = fd.summarise(fd.evaluate(sweep))
	fa = rep["false_alarms"]
	assert fa["stable"] == 80 and 0 <= fa["alarms"] <= fa["stable"]
