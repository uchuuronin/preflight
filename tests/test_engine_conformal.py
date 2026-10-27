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
