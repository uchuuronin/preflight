import json
import math
from pathlib import Path

import pytest

from preflight.corpus import configs as c
from preflight.corpus import store as s
from preflight.corpus import sweep as sw

# ---------------------------------------------------------------- configs


def test_kernel_configs_are_deterministic():
	a = c.kernel_configs(50, seed=7)
	b = c.kernel_configs(50, seed=7)
	assert [x.key for x in a] == [x.key for x in b]


def test_kernel_configs_cover_every_family():
	fams = {x.family for x in c.kernel_configs(300, seed=0)}
	assert fams == set(c.FAMILIES)


def test_keys_are_unique_and_content_addressed():
	cfgs = c.kernel_configs(300, seed=0)
	assert len({x.key for x in cfgs}) == len(cfgs)
	x = cfgs[0]
	again = c.CorpusConfig.from_dict(json.loads(json.dumps(x.to_dict())))
	assert again.key == x.key


def test_key_changes_with_any_knob():
	x = c.kernel_configs(1, seed=0)[0]
	y = c.CorpusConfig(x.kind, x.family, {**x.params, "batch": x.params["batch"] + 1})
	assert x.key != y.key


def test_amd_slice_is_500_and_a_subset_of_the_nvidia_list():
	nv = c.kernel_configs(1500, seed=0)
	amd = c.amd_slice(nv, 500)
	assert len(amd) == 500
	assert {x.key for x in amd} <= {x.key for x in nv}
	assert {x.family for x in amd} == set(c.FAMILIES)


def test_divergence_configs_span_learning_rates_and_precisions():
	d = c.divergence_configs(800, seed=0)
	assert len(d) == 800
	lrs = [x.params["lr"] for x in d]
	assert min(lrs) < 3e-4 and max(lrs) > 1e-1
	assert {x.params["precision"] for x in d} == {"fp32", "bf16", "fp16"}
	assert {x.params["qk_norm"] for x in d} == {True, False}


# ---------------------------------------------------------------- store


def test_store_writes_once_and_skips_repeats(tmp_path):
	st = s.ResultStore(tmp_path)
	rec = {"status": "ok", "gpu_seconds": 10.0}
	assert st.write_once("kernels", "L4", "abc123", rec) is True
	assert st.write_once("kernels", "L4", "abc123", {"status": "ok", "gpu_seconds": 99}) is False
	assert st.read("kernels", "L4", "abc123")["gpu_seconds"] == 10.0


def test_store_lists_done_keys_per_gpu(tmp_path):
	st = s.ResultStore(tmp_path)
	st.write_once("kernels", "L4", "k1", {"status": "ok", "gpu_seconds": 1})
	st.write_once("kernels", "H100", "k2", {"status": "ok", "gpu_seconds": 1})
	assert st.done_keys("kernels", "L4") == {"k1"}


def test_store_leaves_no_partial_files(tmp_path):
	st = s.ResultStore(tmp_path)
	st.write_once("kernels", "L4", "k1", {"status": "ok", "gpu_seconds": 1})
	assert not list(tmp_path.rglob("*.tmp"))


def test_error_records_can_be_retried(tmp_path):
	st = s.ResultStore(tmp_path)
	st.write_once("kernels", "L4", "k1", {"status": "error", "gpu_seconds": 1})
	assert st.done_keys("kernels", "L4", retry_errors=True) == set()
	assert st.write_once("kernels", "L4", "k1", {"status": "ok", "gpu_seconds": 2}, replace_errors=True)
	assert st.read("kernels", "L4", "k1")["status"] == "ok"


def test_spend_sums_all_records(tmp_path):
	st = s.ResultStore(tmp_path)
	st.write_once("kernels", "L4", "a", {"status": "ok", "gpu_seconds": 3600})
	st.write_once("kernels", "H100", "b", {"status": "oom", "gpu_seconds": 1800})
	prices = {"L4": 0.80, "H100": 3.95}
	assert st.spent_usd(prices) == pytest.approx(0.80 + 3.95 / 2)


# ---------------------------------------------------------------- plan and budget


def test_plan_skips_done_and_stops_at_budget(tmp_path):
	st = s.ResultStore(tmp_path)
	cfgs = c.kernel_configs(20, seed=1)
	st.write_once("kernels", "L4", cfgs[0].key, {"status": "ok", "gpu_seconds": 0})
	# 0.80 $/h, 45 s each, overhead 1.0 -> $0.01 per config; budget for 5
	todo = sw.plan(cfgs, st, gpu="L4", budget_usd=0.05, usd_per_hour=0.80, est_seconds=45, overhead=1.0)
	assert cfgs[0].key not in {x.key for x in todo}
	assert len(todo) == 5


def test_plan_counts_existing_spend_against_budget(tmp_path):
	st = s.ResultStore(tmp_path)
	st.write_once("kernels", "L4", "old", {"status": "ok", "gpu_seconds": 3600})  # $0.80 already spent
	todo = sw.plan(
		c.kernel_configs(20, seed=1),
		st,
		gpu="L4",
		budget_usd=0.85,
		usd_per_hour=0.80,
		est_seconds=45,
		overhead=1.0,
	)
	assert len(todo) == 5


def test_run_local_halts_at_budget_and_resumes(tmp_path):
	st = s.ResultStore(tmp_path)
	cfgs = c.kernel_configs(10, seed=2)
	calls = []

	def fake_run(cfg):
		calls.append(cfg.key)
		return {"status": "ok", "gpu_seconds": 45.0}

	n1 = sw.run_local(
		cfgs, st, gpu="L4", budget_usd=0.03, usd_per_hour=0.80, runner=fake_run, est_seconds=45, overhead=1.0
	)
	assert n1 == 3
	n2 = sw.run_local(
		cfgs, st, gpu="L4", budget_usd=0.06, usd_per_hour=0.80, runner=fake_run, est_seconds=45, overhead=1.0
	)
	# three finished configs make the estimate learned, so one config is held in reserve (S5/B5)
	assert n2 == 2
	assert len(set(calls)) == 5


def test_parse_budget_plan():
	assert sw.parse_budget_plan("L4=25,A100-80GB=45, H100=70") == {
		"L4": 25.0,
		"A100-SXM-80GB": 45.0,
		"H100-SXM-80GB": 70.0,
	}
	with pytest.raises(KeyError):
		sw.parse_budget_plan("Voodoo=5")


def test_modal_gpu_strings_pin_the_exact_gpu():
	assert sw.modal_gpu("H100") == "H100!"  # Modal may silently upgrade H100 to H200
	assert sw.modal_gpu("A100-80GB") == "A100-80GB"
	assert sw.modal_gpu("L4") == "L4"
	with pytest.raises(ValueError):
		sw.modal_gpu("MI300X")


def test_batches_split_evenly():
	cfgs = c.kernel_configs(10, seed=4)
	b = sw.batches(cfgs, 4)
	assert [len(x) for x in b] == [4, 4, 2]


def test_plan_from_ledger_matches_plan(tmp_path):
	cfgs = c.kernel_configs(12, seed=5)
	todo = sw.plan_from(
		cfgs,
		done={cfgs[0].key},
		spent_usd=0.02,
		budget_usd=0.05,
		usd_per_hour=0.8,
		est_seconds=45,
		overhead=1.0,
	)
	assert len(todo) == 3
	assert cfgs[0].key not in {x.key for x in todo}


def test_runner_exception_is_recorded_not_raised(tmp_path):
	st = s.ResultStore(tmp_path)
	cfgs = c.kernel_configs(1, seed=3)

	def boom(cfg):
		raise RuntimeError("driver fell over")

	sw.run_local(cfgs, st, gpu="L4", budget_usd=1, usd_per_hour=0.80, runner=boom, est_seconds=45)
	rec = st.read("kernels", "L4", cfgs[0].key)
	assert rec["status"] == "error"
	assert "driver fell over" in rec["error"]


def test_estimate_is_learned_after_three_configs():
	assert sw.estimate_seconds("kernels", [10, 20]) == (45.0, False)
	est, learned = sw.estimate_seconds("kernels", [10, 20, 30, 40, 100])
	assert learned and est == pytest.approx(sw.quantile([10, 20, 30, 40, 100], 0.9))


def test_plan_holds_a_wave_in_reserve_once_learned():
	cfgs = c.kernel_configs(40, seed=6)
	# 45 s p90 at $0.80/h, overhead 1: $0.01 per config; $0.20 budget, 8-config waves
	free = sw.plan_from(cfgs, set(), 0.0, 0.20, 0.80, overhead=1.0, history=[45.0] * 5, reserve_configs=0)
	held = sw.plan_from(cfgs, set(), 0.0, 0.20, 0.80, overhead=1.0, history=[45.0] * 5, reserve_configs=8)
	assert len(free) == 20 and len(held) == 12


def test_slow_history_shrinks_the_plan():
	cfgs = c.kernel_configs(40, seed=6)
	fast = sw.plan_from(cfgs, set(), 0.0, 0.20, 0.80, overhead=1.0, history=[45.0] * 5)
	slow = sw.plan_from(cfgs, set(), 0.0, 0.20, 0.80, overhead=1.0, history=[90.0] * 5)
	assert len(slow) == len(fast) // 2


def test_store_reports_gpu_seconds(tmp_path):
	st = s.ResultStore(tmp_path)
	st.write_once("kernels", "L4", "a", {"status": "ok", "gpu_seconds": 12.5})
	st.write_once("kernels", "L4", "b", {"status": "oom", "gpu_seconds": 3.0})
	assert sorted(st.gpu_seconds("kernels", "L4")) == [3.0, 12.5]


# ---------------------------------------------------------------- review fixes: spend and robustness


def test_retrying_an_error_keeps_its_spend(tmp_path):
	st = s.ResultStore(tmp_path)
	st.write_once("kernels", "L4", "k", {"status": "timeout", "gpu_seconds": 600.0})
	st.write_once("kernels", "L4", "k", {"status": "ok", "gpu_seconds": 40.0}, replace_errors=True)
	assert st.read("kernels", "L4", "k")["prior_gpu_seconds"] == 600.0
	assert st.spent_usd({"L4": 3600.0}, kind="kernels") == pytest.approx(640.0)


def test_corrupt_rows_are_skipped_not_fatal(tmp_path):
	st = s.ResultStore(tmp_path)
	st.write_once("kernels", "L4", "good", {"status": "ok", "gpu_seconds": 1.0})
	bad = st.path("kernels", "L4", "bad")
	bad.parent.mkdir(parents=True, exist_ok=True)
	bad.write_text("{not json")
	assert st.done_keys("kernels", "L4") == {"good"}


def test_ledger_reads_small_summaries(tmp_path):
	st = s.ResultStore(tmp_path)
	st.write_once("kernels", "L4", "k", {"status": "ok", "gpu_seconds": 3.0, "ops": [{"big": "x" * 1000}]})
	meta = st.meta_path("kernels", "L4", "k")
	assert meta.exists() and "ops" not in meta.read_text()


def test_local_driver_counts_billing_overhead(tmp_path):
	st = s.ResultStore(tmp_path)
	cfgs = c.kernel_configs(10, seed=8)
	n = sw.run_local(
		cfgs,
		st,
		"L4",
		0.05,
		0.80,
		lambda cfg: {"status": "ok", "gpu_seconds": 45.0},
		est_seconds=45,
		log=lambda _m: None,
	)
	# $0.0125 per config with the 1.25 overhead: 4 fit in $0.05, 3 once a config is held in reserve
	assert n == 3


def test_repeated_corpus_bugs_stop_the_sweep(tmp_path):
	st = s.ResultStore(tmp_path)
	cfgs = c.kernel_configs(10, seed=9)
	n = sw.run_local(
		cfgs,
		st,
		"L4",
		10,
		0.80,
		lambda cfg: {"status": "error", "error_code": "user_error", "gpu_seconds": 1.0},
		log=lambda _m: None,
	)
	assert n == sw.MAX_CONSECUTIVE_USER_ERRORS


def test_fp32_configs_sample_tf32_both_ways():
	vals = {x.params.get("tf32") for x in c.kernel_configs(400, seed=0) if x.params.get("dtype") == "fp32"}
	assert vals == {True, False}


def test_uptime_ledger_bills_idle_time_and_stops_at_budget(tmp_path):
	from datetime import UTC, datetime

	from preflight.corpus import configs as C
	from preflight.corpus.ledger import UptimeLedger
	from preflight.corpus.store import ResultStore
	from preflight.corpus.sweep import run_local

	now = [1_000_000.0]
	started = datetime.fromtimestamp(now[0] - 3600, UTC)  # the droplet has been up an hour already
	ledger = UptimeLedger(tmp_path / "ledger.jsonl", started, usd_per_hour=2.0, clock=lambda: now[0])
	assert ledger.spent() == pytest.approx(2.0)

	def runner(cfg):
		now[0] += 1800  # each config takes half an hour of droplet time
		return {"status": "ok", "gpu_seconds": 60.0}

	cfgs = C.kernel_configs(10)
	n = run_local(
		cfgs,
		ResultStore(tmp_path / "out"),
		"MI300X",
		5.0,
		2.0,
		runner,
		est_seconds=60,
		ledger=ledger,
		log=lambda m: None,
	)
	# $2 already spent by uptime; each config adds $1 of droplet time, not $0.03 of GPU seconds
	assert n == 3
	events = [r["event"] for r in ledger.rows()]
	assert events[0] == "start" and events.count("config") == 3 and events[-1] == "stop"
	assert ledger.rows()[-1]["spent_usd"] == pytest.approx(5.0)
	with pytest.raises(ValueError):
		UptimeLedger(tmp_path / "x.jsonl", datetime(2026, 10, 28), 2.0)  # no time zone


def test_amd_grouped_export_matches_the_nvidia_groups(tmp_path):
	from preflight.corpus import configs as C
	from preflight.corpus.sweep import _main

	out = tmp_path / "amd.jsonl"
	_main(["export-amd", "--kind", "grouped", "--n", "300", "--out", str(out)])
	amd = C.read_jsonl(str(out))
	nv = C.configs_for("grouped", 300, 0)
	assert [c.key for c in amd] == [c.key for c in nv]
	# the A100 holdout's 100 configs are the first 20 of the same groups
	assert {c.key for c in C.configs_for("grouped", 100, 0)} <= {c.key for c in nv}
	shipped = C.read_jsonl(str(Path(C.__file__).parent / "data" / "amd_grouped.jsonl"))
	assert [c.key for c in shipped] == [c.key for c in nv]


def test_copy_results_keeps_the_layout(tmp_path):
	from preflight.corpus.sweep import copy_results

	src = tmp_path / "store"
	(src / "kernels" / "MI300X").mkdir(parents=True)
	(src / "kernels" / "MI300X" / "a.json").write_text("{}")
	(src / "ledger-MI300X.jsonl").write_text("{}\n")
	assert copy_results(src, tmp_path / "out") == 2
	assert (tmp_path / "out" / "kernels" / "MI300X" / "a.json").exists()


def test_bf16_configs_skip_gpus_without_bf16():
	pytest.importorskip("modal")
	from preflight.corpus import configs as C
	from preflight.corpus.modal_app import runs_on

	cfgs = C.configs_for("grouped", 300, 0)
	t4 = [c for c in cfgs if runs_on(c, "T4")]
	assert 0 < len(t4) < len(cfgs) and all(c.params.get("dtype") != "bf16" for c in t4)
	assert all(runs_on(c, "L4") for c in cfgs)
	assert not any(runs_on(c, "T4") for c in C.hf_configs())  # HF jobs are bf16 LoRA or bf16-compute QLoRA


def test_calibrate_reads_exported_gzip(tmp_path):
	import gzip
	import json

	from preflight.corpus.calibrate import load_rows

	p = tmp_path / "kernels" / "L4.jsonl.gz"
	p.parent.mkdir(parents=True)
	with gzip.open(p, "wt") as f:
		f.write(
			json.dumps({"status": "ok", "gpu": "L4"})
			+ "\n"
			+ json.dumps({"status": "oom", "gpu": "L4"})
			+ "\n"
		)
	assert [r["status"] for r in load_rows(tmp_path)] == ["ok", "oom"]


def test_deadline_pricing_makes_the_cap_hard_before_any_history():
	from preflight.corpus import configs as C
	from preflight.corpus import sweep as sw

	cfgs = C.configs_for("grouped", 10, 0)
	# $0.30 on an L4 at a 600 s deadline x 1.25 overhead: one config fits, even if it hangs to its deadline
	todo = sw.plan_from(cfgs, set(), 0.0, 0.30, 0.80, worst_seconds=600.0)
	assert len(todo) == 1
	assert len(todo) * sw.cost_usd(600.0, 0.80, sw.DEFAULT_OVERHEAD) <= 0.30
	# after three measured configs the running p90 takes over
	todo = sw.plan_from(cfgs, set(), 0.05, 0.30, 0.80, history=[40.0, 45.0, 50.0], worst_seconds=600.0)
	assert len(todo) > 1


def test_wave_billing_is_an_upper_bound_on_container_time():
	from preflight.corpus import sweep as sw

	# 4 containers for a 10-minute wave, each with its 30 s idle tail: 4 x 630 s
	assert sw.wave_billed_seconds(600.0, 4) == pytest.approx(4 * (600.0 + sw.SCALEDOWN_S))
	assert sw.wave_billed_seconds(600.0, 0) == 0.0
	# phase 1's shape: 32 configs in an 11-minute wave cost far more than their 9 s each
	per = sw.wave_billed_seconds(660.0, 4) / 32
	assert per > 80


def test_container_history_prices_the_next_wave():
	from preflight.corpus import configs as C
	from preflight.corpus import sweep as sw

	cfgs = C.configs_for("grouped", 600, 0)
	# three measured waves at ~85 container-seconds a config; $6 on an L4 with $5 already billed
	todo = sw.plan_from(
		cfgs, set(), 5.0, 6.0, 0.80, overhead=1.0, history=[85.0, 80.0, 90.0], reserve_configs=0
	)
	assert len(todo) == math.floor(1.0 / sw.cost_usd(sw.quantile([85.0, 80.0, 90.0], 0.9), 0.80, 1.0))


def test_billing_file_round_trips(tmp_path):
	from preflight.corpus import sweep as sw

	p = tmp_path / "_billing" / "kernels-L4.jsonl"
	assert sw.billed_seconds(p) == 0.0
	sw.add_billing(p, {"wave": 0, "wall_s": 100.0, "containers": 2, "billed_s": 260.0, "configs": 16})
	sw.add_billing(p, {"wave": 1, "wall_s": 50.0, "containers": 1, "billed_s": 80.0, "configs": 8})
	assert sw.billed_seconds(p) == pytest.approx(340.0)
	assert sw.container_history(p) == pytest.approx([260.0 / 16, 80.0 / 8])


def test_modal_idle_window_matches_what_the_sweep_bills():
	pytest.importorskip("modal")
	import re

	from preflight.corpus import modal_app, sweep

	src = Path(modal_app.__file__).read_text()
	assert float(re.search(r"scaledown_window=(\d+)", src).group(1)) == sweep.SCALEDOWN_S
	assert modal_app.COMMIT_EVERY >= 1
