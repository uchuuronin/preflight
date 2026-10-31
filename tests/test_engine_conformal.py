import math

import numpy as np
import pytest

from preflight.engine import conformal as cf


def test_conformal_quantile_is_the_ceil_n_plus_1_rank():
	scores = [i / 100 for i in range(1, 20)]  # n = 19: ceil(20 * 0.9) = 18th smallest
	assert cf.conformal_half_width(scores) == pytest.approx(0.18)
	assert cf.conformal_half_width(scores[:8]) is None  # n = 8 can't give a 90% quantile
	assert cf.conformal_half_width(scores[:9]) == pytest.approx(0.09)


def test_wilson_interval():
	lo, hi = cf.wilson(90, 100)
	assert lo == pytest.approx(0.8256, abs=1e-3) and hi == pytest.approx(0.9448, abs=1e-3)
	assert cf.wilson(0, 0) == (0.0, 1.0)


def _scores(n_groups=30, per=5, sigma=0.1, gpu="L4", lens="finetune", seed=0):
	g = np.random.default_rng(seed)
	out = []
	for i in range(n_groups):
		bias = g.normal(0, sigma / 2)  # architectures differ: errors correlate within a group
		for _ in range(per):
			out.append(
				cf.Score("memory.peak", gpu, lens, f"arch{i}", 1e9, 1e9 * math.exp(bias + g.normal(0, sigma)))
			)
	return out


def test_leave_one_architecture_out_coverage_is_near_nominal():
	cov = cf.evaluate(_scores(60))["memory.peak"]
	assert 0.85 <= cov.rate <= 0.96 and cov.lo < cov.rate < cov.hi and cov.n == 300


def test_lookup_falls_back_from_gpu_to_lens_to_target():
	t = cf.calibrate(_scores(4, gpu="L4") + _scores(1, gpu="H100-SXM-80GB", seed=1))
	assert t.lookup("memory.peak", "L4", "finetune").key == ("memory.peak", "L4", "finetune")
	assert t.lookup("memory.peak", "H100-SXM-80GB", "finetune").key == (
		"memory.peak",
		"",
		"finetune",
	)  # 5 runs < 9
	assert t.lookup("memory.peak", "L4", "pretrain").key == ("memory.peak", "", "")
	assert t.lookup("time.step", "L4", "finetune") is None
	back = cf.Table.from_dict(t.to_dict())
	assert back.cells.keys() == t.cells.keys()
	for k, c in t.cells.items():
		assert back.cells[k].half_width == c.half_width and back.cells[k].residuals == pytest.approx(
			c.residuals, abs=1e-5
		)


def test_half_width_falls_back_to_the_simulated_constant():
	h, source, cell = cf.half_width("memory.peak", "L4", "finetune", 0.107, table=cf.Table())
	assert (h, source, cell) == (0.107, "simulated", None)


def test_the_engine_uses_a_held_out_cell_when_there_is_one(monkeypatch):
	from pathlib import Path

	from preflight.engine import api
	from preflight.schema import load_record

	rec = load_record((Path(__file__).resolve().parents[1] / "schema/examples/pretrain.json").read_text())
	before = api.predict(rec, {"batch_size": 10}).root["memory.peak"]
	table = cf.calibrate(_scores(4, sigma=0.15, gpu="A100-SXM-40GB", lens="pretrain"))
	monkeypatch.setattr(cf, "TABLE", table)
	after = api.predict(rec, {"batch_size": 10}).root["memory.peak"]
	h = table.lookup("memory.peak", "A100-SXM-40GB", "pretrain").half_width
	assert math.log(after.hi / after.point) == pytest.approx(h, rel=1e-6)
	assert math.log(after.hi / after.point) > math.log(before.hi / before.point)
	card = next(c for c in api.fill_predictions(rec).models if c.id.startswith("memory"))
	assert card.notes.startswith("calibration: held_out")


def test_cells_say_whether_they_were_merged():
	t = cf.calibrate(_scores(4, gpu="L4") + _scores(1, gpu="H100-SXM-80GB", seed=1))
	assert t.lookup("memory.peak", "L4", "finetune").label() == "L4, finetune (n = 20)"
	merged = t.lookup("memory.peak", "H100-SXM-80GB", "finetune")
	assert merged.label() == "finetune, merged across GPUs (n = 25)"
	assert t.lookup("memory.peak", "L4", "pretrain").label() == "merged across GPUs and job types (n = 25)"
	text = t.describe("memory.peak")
	assert (
		"L4, finetune (n = 20)" in text
		and "fewer than 9" in text
		and "H100" not in text.split("Own cells:")[1].split(".")[0]
	)


def test_holdout_and_coverage_round_trip():
	t = cf.calibrate(_scores(12))
	t = cf.Table(
		t.cells, "x", coverage={"memory.peak": (54, 60)}, holdout={"memory.peak": {"A100-SXM-80GB": (17, 20)}}
	)
	back = cf.Table.from_dict(t.to_dict())
	assert back.coverage == {"memory.peak": (54, 60)}
	assert back.holdout == {"memory.peak": {"A100-SXM-80GB": (17, 20)}}
	assert "A100-SXM-80GB 17 of 20" in back.describe("memory.peak")


def test_engine_cards_and_reasons_name_the_cell(monkeypatch):
	from pathlib import Path

	from preflight.engine import api
	from preflight.schema import load_record

	rec = load_record((Path(__file__).resolve().parents[1] / "schema/examples/pretrain.json").read_text())
	table = cf.calibrate(_scores(4, sigma=0.15, gpu="L4", lens="pretrain"))
	table = cf.Table(table.cells, "corpus test", coverage={"memory.peak": (18, 20)})
	monkeypatch.setattr(cf, "TABLE", table)
	peak = api.predict(rec, {"batch_size": 10}).root["memory.peak"]
	assert (
		"band from held-out runs: pretrain, merged across GPUs (n = 20)" in peak.reason
	)  # probe GPU is an A100
	card = next(c for c in api.fill_predictions(rec).models if c.id.startswith("memory"))
	assert card.training_data == "corpus test" and card.coverage == pytest.approx(0.9)
	assert "merged across GPUs" in card.notes


def test_notes_follow_the_band_that_was_used(monkeypatch):
	from pathlib import Path

	from preflight.engine import api
	from preflight.schema import load_record

	root = Path(__file__).resolve().parents[1] / "schema/examples"
	ft = load_record((root / "finetune.json").read_text())
	table = cf.calibrate(_scores(4, gpu="L4", lens="finetune") + [
		cf.Score("time.step", "L4", "finetune", f"a{i}", 1.0, 1.0 + 0.01 * i) for i in range(12)
	])  # fmt: skip
	monkeypatch.setattr(cf, "TABLE", table)
	no_probes = ft.model_copy(update={"probes": []})
	peak = api.predict(no_probes).root["memory.peak"]
	assert "band from held-out runs" not in (peak.reason or "")  # the prior set it, not the cell
	cross = api.predict(ft, {"gpu_type": "A100-SXM-80GB"}).root
	if cross["time.step"].status != "abstain":
		assert "band from held-out runs: L4, finetune" in cross["time.step"].reason
		assert "band from held-out runs" in cross["time.total"].reason
		phases = [v for k, v in cross.items() if k.startswith("time.phase.") and v.status != "abstain"]
		assert phases and all("band from held-out runs" in v.reason for v in phases)
	cards = {c.id: c for c in api.fill_predictions(ft).models}
	roof = next((c for i, c in cards.items() if i.startswith("steptime-roofline")), None)
	if roof is not None:
		assert roof.coverage is None and not roof.per_gpu_coverage


def test_step_time_bands_cover_only_the_approved_lenses():
	# Chiya, Oct 31: vision, retrieval and pretrain get held-out step-time bands; HF fine-tune stays simulated
	from preflight.engine import api, conformal, steptime

	t = conformal.TABLE
	for lens in ("vision", "retrieval", "pretrain"):
		assert t.lookup("time.step", "L4", lens) is not None
	for lens in ("finetune", "rl", "serving", ""):
		assert t.lookup("time.step", "L4", lens) is None  # no all-lens cell to borrow
	assert t.lookup("memory.peak", "L4", "finetune") is not None
	(card,) = api._cards({steptime.MODEL_ID_SAME})
	assert "Calibrated on corpus models, not users' scripts; checked on unseen scripts Nov 2." in card.notes
	assert "keeps the simulated band" in card.notes


def test_lens_limits_parse_and_refuse_nonsense():
	import pytest

	from preflight.corpus.calibrate import parse_lenses

	assert parse_lenses("time.step=pretrain+vision; memory.peak=vision") == {
		"time.step": frozenset({"pretrain", "vision"}),
		"memory.peak": frozenset({"vision"}),
	}
	assert parse_lenses("") == {}
	with pytest.raises(SystemExit):
		parse_lenses("time.step=")
