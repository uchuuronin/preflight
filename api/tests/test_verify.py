"""Verify checks both sides: the target improves and nothing the run means changes."""

import copy
import json
from pathlib import Path

from preflight_api.verify import check, main

GIB = 2**30


def record(**over):
	probe = {
		"id": "p1",
		"status": "ok",
		"knobs": {"batch_size": 4, "eval_batch_size": 32},
		"memory": {"peak_allocated_bytes": 90 * GIB},
		"throughput": {"tokens_per_s": 2600.0},
		"extras": {"samples_per_step": 4, "tokens_per_step": 8192},
		"stability": {"series": {"lr": [1e-5, 2e-5, 3e-5], "loss": [2.31, 2.25, 2.20]}},
	}
	probe.update(over.pop("probe", {}))
	rec = {
		"knobs": [{"id": "batch_size", "current": 4}, {"id": "eval_batch_size", "current": 32}],
		"job": {"script_map": {"training": {"lr": 2e-4, "warmup_steps": 100, "max_steps": 6000}}},
		"probes": [probe],
	}
	for k, v in over.items():
		rec[k] = v
	return rec


def fixed(**probe):
	r = record(probe={"memory": {"peak_allocated_bytes": 66 * GIB}, **probe})
	return r


def test_a_memory_fix_that_changes_nothing_else_passes():
	res = check(record(), fixed())
	assert res["status"] == "passed", res["checks"]
	assert all(c["ok"] for c in res["checks"])
	assert res["effective_batch_unchanged"] is True


def test_each_side_effect_fails_its_own_check():
	cases = {
		"effective_batch": {"extras": {"samples_per_step": 2, "tokens_per_step": 8192}},
		"tokens_seen": {"extras": {"samples_per_step": 4, "tokens_per_step": 4096}},
		"throughput_floor": {"throughput": {"tokens_per_s": 1800.0}},
		"lr_schedule": {"stability": {"series": {"lr": [1e-5, 4e-5, 3e-5], "loss": [2.31, 2.25, 2.20]}}},
		"fixed_seed_loss": {"stability": {"series": {"lr": [1e-5, 2e-5, 3e-5], "loss": [2.31, 2.40, 2.20]}}},
	}
	for name, probe in cases.items():
		res = check(record(), fixed(**probe))
		bad = [c["id"] for c in res["checks"] if c["ok"] is False]
		assert res["status"] == "failed" and bad == [name], (name, res["checks"])


def test_a_changed_learning_rate_in_the_script_fails():
	after = fixed()
	after["job"] = copy.deepcopy(after["job"])
	after["job"]["script_map"]["training"]["lr"] = 1e-4
	assert check(record(), after)["status"] == "failed"


def test_no_improvement_or_missing_measurements_never_pass():
	assert check(record(), record())["status"] == "failed"
	blind = fixed(throughput={})
	res = check(record(), blind)
	assert res["status"] == "error"
	assert [c["id"] for c in res["checks"] if c["ok"] is None] == ["throughput_floor"]


def test_an_out_of_memory_before_counts_as_fixed_when_after_fits():
	before = record(probe={"status": "oom", "memory": {}})
	assert check(before, fixed())["checks"][0]["ok"] is True


def test_the_command_line_is_the_actions_pass_fail(tmp_path: Path, capsys):
	b, a, bad = tmp_path / "b.json", tmp_path / "a.json", tmp_path / "x.json"
	b.write_text(json.dumps(record()))
	a.write_text(json.dumps(fixed()))
	bad.write_text(json.dumps(fixed(extras={"samples_per_step": 2, "tokens_per_step": 8192})))
	assert main([str(b), str(a)]) == 0
	assert main([str(b), str(bad)]) == 1
	out = capsys.readouterr().out
	assert "FAIL" in out and "effective_batch" in out


def test_the_command_line_takes_what_a_fix_may_change(tmp_path: Path):
	b, a = tmp_path / "b.json", tmp_path / "a.json"
	b.write_text(json.dumps(record()))
	a.write_text(json.dumps(fixed()))
	# A fix that touches evaluation must show the same evaluation still ran: unmeasured is an error.
	assert main([str(b), str(a), "--keep-eval"]) == 2
	assert main([str(b), str(a), "--may-change", "throughput_floor, fixed_seed_loss"]) == 0
