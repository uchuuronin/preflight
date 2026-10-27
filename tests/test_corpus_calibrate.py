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
