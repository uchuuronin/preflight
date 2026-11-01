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


def test_both_a100_boards_are_held_out_by_default():
	from preflight.corpus import calibrate as cal
	from preflight.corpus import sweep

	# a request for the holdout GPU can land on any of these boards; every one must stay out of calibration
	assert set(sweep.landing_gpus("A100-SXM-80GB")) <= set(cal.HOLDOUT_GPUS)
	for g in ("A100-PCIe-80GB", "A100-SXM-40GB", "A100-SXM-80GB-MIG-3g.40gb"):
		assert cal.held_out(g)
	assert not cal.held_out("L4") and not cal.held_out(None) and not cal.held_out("A100-SXM-80GBX")


def test_reliability_report_leaves_out_thin_bins():
	from preflight.corpus.calibrate import reliability_lines

	pairs = [(0.01, False)] * 30 + [(0.5, True)] * 3  # the 30-70% bin has only 3 points
	lines = reliability_lines(pairs, min_per_bin=10)
	text = "\n".join(lines)
	assert "[0.00, 0.05) n=30" in text and "Wilson 95%" in text
	assert "[0.30, 0.70): 3 points, too few to draw" in text


def _two_gpu_rows(order=1):
	cfgs = [c for c in configs.grouped_configs(50) if c.family == "gpt"]  # 10 gpt groups: 20 scores a GPU
	rows = []
	for gpu, f in (("A100-SXM-80GB", 1.3), ("L4", 1.0))[::order]:
		for c in cfgs:
			x = c.params["batch"] * c.params["seq"]
			an = calibrate._analytical({"config": c.to_dict()})
			rows.append(_row(c, an * f + 2e8, x, gpu=gpu))
	return cfgs, rows


@pytest.mark.parametrize("order", [1, -1])
def test_each_gpu_is_fitted_and_labelled_on_its_own(order):
	from collections import Counter

	_, rows = _two_gpu_rows(order)
	s = calibrate.scores_from_rows(rows)
	per_gpu = Counter(x.gpu for x in s)
	assert per_gpu["L4"] == per_gpu["A100-SXM-80GB"] > 0  # same groups on both GPUs, scored separately
	# the group stays the architecture, so leave-one-architecture-out never splits one model across GPUs
	assert {x.group for x in s if x.gpu == "L4"} == {x.group for x in s if x.gpu == "A100-SXM-80GB"}
	table, _ = calibrate.build(s)
	mem = Counter(x.gpu for x in s if x.target == "memory.peak")
	assert table.lookup("memory.peak", "", "").n == mem["L4"]  # only L4 calibrates
	assert sum(n for _, n in table.holdout["memory.peak"].values()) == mem["A100-SXM-80GB"]


def test_oom_pairs_mirror_the_engines_cell_and_skip_the_holdout():
	_, rows = _two_gpu_rows()
	s = calibrate.scores_from_rows(rows)
	pairs = calibrate.oom_pairs(
		[r for r in rows if r["gpu"] != "A100-SXM-80GB"], [x for x in s if x.gpu != "A100-SXM-80GB"]
	)
	assert all(0 < p < 1 for p, _ in pairs)


def test_reliability_needs_real_out_of_memory_runs():
	from preflight.engine import oom

	never = [(0.01, False)] * 60  # one bin, no OOM anywhere: consistent, but says nothing where it matters
	assert not oom.reliable(oom.reliability(never))
	mixed = (
		[(0.01, False)] * 60
		+ [(0.8, True)] * 16
		+ [(0.8, False)] * 4
		+ [(0.5, True)] * 6
		+ [(0.5, False)] * 6
	)
	assert oom.reliable(oom.reliability(mixed))


def test_the_written_table_imports_as_python_and_round_trips(tmp_path):
	import importlib.util

	from preflight.engine import conformal as cf

	table = cf.Table(
		{
			("memory.peak", "L4", "finetune"): cf.Cell(
				0.1, 12, 4, "held_out", ("memory.peak", "L4", "finetune"), (0.01, -0.02)
			)
		},
		"test",
		oom_reliable=False,
		coverage={"memory.peak": (11, 12)},
	)
	out = tmp_path / "_calibration_table.py"
	calibrate.write_table(table, out)
	spec = importlib.util.spec_from_file_location("t", out)
	mod = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(mod)  # JSON's true/false/null would raise NameError here
	back = cf.Table.from_dict(mod.DATA)
	assert back.cells == table.cells and back.oom_reliable is False and back.coverage == table.coverage


def test_a_table_with_non_finite_numbers_is_refused(tmp_path):
	from preflight.engine import conformal as cf

	bad = cf.Table(
		{("memory.peak", "", ""): cf.Cell(float("nan"), 9, 3, "held_out", ("memory.peak", "", ""), ())}
	)
	with pytest.raises(ValueError):
		calibrate.write_table(bad, tmp_path / "t.py")
	assert not (tmp_path / "t.py").exists()


def test_scores_are_grouped_by_architecture_not_by_config_group():
	"""Two config groups that differ only in dtype are one architecture: near-duplicates, never split."""
	base = next(c for c in configs.grouped_configs(50) if c.family == "gpt" and c.params["role"] == "probe")
	a = {**base.params, "dtype": "bf16"}
	b = {**base.params, "dtype": "fp32"}
	assert calibrate.arch_key({"family": "gpt", "params": a}) == calibrate.arch_key(
		{"family": "gpt", "params": b}
	)
	assert calibrate.arch_key(
		{"family": "gpt", "params": {**a, "n_layers": a["n_layers"] + 1}}
	) != calibrate.arch_key({"family": "gpt", "params": a})
	hf = {"family": "hf_lora", "params": {"model_id": "Qwen/Qwen2.5-0.5B", "quantization": "nf4"}}
	assert calibrate.arch_key(hf) == calibrate.arch_key(
		{**hf, "params": {**hf["params"], "quantization": "none"}}
	)


def test_the_generator_writes_the_table_ruff_formatted(tmp_path):
	from pathlib import Path

	# PM, Oct 31: ruff reformatted the generated table during a merge; the generator now writes what ruff would
	from preflight.engine import conformal

	out = tmp_path / "_calibration_table.py"
	calibrate.write_table(conformal.TABLE, out)
	committed = Path(calibrate.__file__).resolve().parents[1] / "engine" / "_calibration_table.py"
	assert out.read_text() == committed.read_text()
	assert calibrate.ruff_formatted(out.read_text(), out) == out.read_text()
