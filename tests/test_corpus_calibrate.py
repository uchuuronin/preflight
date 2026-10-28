import math

import pytest

from preflight.corpus import calibrate, configs


def _row(cfg, reserved, x, gpu="L4"):
	"""Phases linear in work units, the optimizer constant: the shape the engine's same-GPU fit assumes."""
	phases = {"forward": 0.004 + x * 1e-6, "backward": 0.006 + x * 1.5e-6, "optimizer": 0.005}
	return {
		"key": cfg.key, "gpu": gpu, "config": cfg.to_dict(), "status": "ok",
		"work_units": x, "memory": {"peak_reserved": reserved},
		"timing": {"step_s": sum(phases.values()), "phases": phases},
	}  # fmt: skip


def test_grouped_rows_become_one_score_per_query_and_target():
	cfgs = [c for c in configs.grouped_configs(5) if c.family == "gpt"]
	rows = []
	for c in cfgs:
		x = c.params["batch"] * c.params["seq"]
		an = calibrate._analytical({"config": c.to_dict()})
		rows.append(_row(c, an * 1.1 + 2e8, x))
	scores = calibrate.scores_from_rows(rows)
	assert {s.target for s in scores} == {"memory.peak", "time.step"}
	assert len(scores) == 2 * 2  # two queries, two targets
	assert all(s.lens == "pretrain" and s.gpu == "L4" for s in scores)
	# a truth of the model's own form is recovered almost exactly (this checks the plumbing, not accuracy)
	assert max(s.value for s in scores) < 0.03


def test_rows_without_memory_or_probes_are_skipped():
	cfgs = [c for c in configs.grouped_configs(5) if c.family == "gpt"]
	rows = [_row(c, 1e9, c.params["batch"] * c.params["seq"]) for c in cfgs if c.params["role"] == "query"]
	assert calibrate.scores_from_rows(rows) == []


def test_versions_change_the_key_and_round_trip():
	c = configs.grouped_configs(1)[0]
	v = c.with_versions({"torch": "2.8.0", "corpus": configs.CODE_VERSION})
	assert v.key != c.key and configs.CorpusConfig.from_dict(v.to_dict()).key == v.key


def test_hf_jobs_are_ten_groups_of_five():
	h = configs.hf_configs()
	assert len(h) == 50 and len({c.params["group"] for c in h}) == 10
	assert {c.params["quantization"] for c in h} == {"none", "nf4"}


torch = pytest.importorskip("torch")


@pytest.mark.torch
def test_a_real_cpu_group_runs_through_the_worker_and_calibrator():
	from preflight.corpus import worker

	cfgs = [c for c in configs.grouped_configs(1, seed=3, families=("gpt",))]
	small = [
		configs.CorpusConfig(c.kind, c.family, {**c.params, "n_layers": 2, "d_model": 128, "n_heads": 2, "n_kv_heads": 2, "d_ff": 512, "vocab_size": 1024, "dtype": "fp32", "tf32": False})
		for c in cfgs
	]  # fmt: skip
	rows = []
	for c in small:
		res = worker.run_config(c, device="cpu")
		assert res["status"] == "ok"
		res["memory"] = {
			"peak_reserved": 1e8 + res["work_units"] * 1e3
		}  # CPU has no allocator peak; stand-in
		rows.append({"key": c.key, "gpu": "cpu", "config": c.to_dict(), **res})
	scores = calibrate.scores_from_rows(rows)
	assert any(s.target == "time.step" and math.isfinite(s.value) for s in scores)


def test_oom_pairs_use_other_architectures_and_real_outcomes():
	from preflight.engine import conformal

	rows, scores = [], []
	for c in configs.grouped_configs(15, families=("gpt",)):
		x = c.params["batch"] * c.params["seq"]
		an = calibrate._analytical({"config": c.to_dict()})
		row = _row(c, an * 1.05, x)
		row["env"] = {"total_memory": an * 1.2}  # capacity just above the analytical peak
		if c.params["role"] == "query" and c.params["batch"] > 8:
			row["status"] = "oom"
		rows.append(row)
		if c.params["role"] == "query":
			scores.append(conformal.Score("memory.peak", "L4", "pretrain", c.params["group"], an, an * 1.05))
	pairs = calibrate.oom_pairs(rows, scores)
	assert pairs and all(0 < p < 1 for p, _ in pairs)
	assert any(y for _, y in pairs) and not all(y for _, y in pairs)  # both outcomes present


def test_holdout_gpus_never_calibrate_and_are_reported_once():
	import math

	from preflight.corpus import calibrate as cal
	from preflight.engine import conformal as cf

	def scores(gpu, n, seed):
		import random

		r = random.Random(seed)
		return [
			cf.Score("memory.peak", gpu, "finetune", f"{gpu}{i}", 1e9, 1e9 * math.exp(r.gauss(0, 0.1)))
			for i in range(n)
		]

	s = scores("L4", 40, 0) + scores("A100-SXM-80GB", 20, 1)
	table, report = cal.build(s, holdout_gpus=("A100-SXM-80GB",), built_from="t")
	assert all(k[1] != "A100-SXM-80GB" for k in table.cells)
	assert table.lookup("memory.peak", "", "").n == 40  # the A100 rows never reached calibration
	hits, n = table.holdout["memory.peak"]["A100-SXM-80GB"]
	assert n == 20 and 0 <= hits <= 20
	assert "A100-SXM-80GB" in report and "Wilson" in report


def test_reliability_report_leaves_out_thin_bins():
	from preflight.corpus.calibrate import reliability_lines

	pairs = [(0.01, False)] * 30 + [(0.5, True)] * 3  # the 30-70% bin has only 3 points
	lines = reliability_lines(pairs, min_per_bin=10)
	text = "\n".join(lines)
	assert "[0.00, 0.05) n=30" in text and "Wilson 95%" in text
	assert "[0.30, 0.70): 3 points, too few to draw" in text
