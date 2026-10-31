"""recorder/ground_truth.py without a GPU: what it refuses, and where the bare run installs."""

import json
import os
from pathlib import Path

from preflight.recorder import ground_truth as gt


def test_only_the_zoo_scripts_are_measured(tmp_path, capsys):
	(tmp_path / "mine").mkdir()
	assert gt.main([str(tmp_path / "mine"), str(tmp_path / "o.json"), "--bare-steps", "3"]) == 2
	assert not (tmp_path / "o.json").exists()


def test_the_bare_run_installs_into_its_own_prefix_not_the_probes(tmp_path, monkeypatch):
	import preflight.recorder.requirements as req

	monkeypatch.setenv("PREFLIGHT_DEPS_DIR", "/probe/deps")
	monkeypatch.setenv("PREFLIGHT_PIP_CACHE", "/probe/pip")
	seen = {}

	def install(workdir, target, *, log=print):
		seen["env"] = {k: os.environ.get(k) for k in ("PREFLIGHT_DEPS_DIR", "PREFLIGHT_PIP_CACHE")}
		seen["target"] = target
		return None, [Path("/gt/site")]

	def stream(cmd, **kw):
		seen["cmd"], seen["kw"] = cmd, kw
		Path(cmd[4]).write_text(json.dumps({"steps": 3, "walls_s": [None, 1.0, 1.0], "ended": "steps"}))
		return 0, ""

	monkeypatch.setattr(req, "install", install)
	monkeypatch.setattr(gt, "_stream", stream)
	out = gt._bare(tmp_path, 3, install_requirements=True, caches=tmp_path / "c", env={"HF_HOME": "/gt/hf"},
		timeout_s=60)  # fmt: skip
	assert out["ended"] == "steps"
	assert seen["env"] == {"PREFLIGHT_DEPS_DIR": None, "PREFLIGHT_PIP_CACHE": None}
	assert seen["target"] == tmp_path / "c" / "deps"
	assert seen["kw"]["env"]["PYTHONPATH"].split(os.pathsep)[0] == "/gt/site"
	assert seen["kw"]["env"]["HF_HOME"] == "/gt/hf"
	assert seen["cmd"][-2:] == ["--steps", "3"]
	assert os.environ["PREFLIGHT_DEPS_DIR"] == "/probe/deps"  # restored for the probe
	assert os.environ["PREFLIGHT_PIP_CACHE"] == "/probe/pip"


def test_a_bare_run_that_writes_nothing_reports_its_output(tmp_path, monkeypatch):
	monkeypatch.setattr(gt, "_stream", lambda cmd, **kw: (1, "ModuleNotFoundError: No module named 'peft'"))
	out = gt._bare(tmp_path, 3, install_requirements=False, caches=tmp_path, env={}, timeout_s=60)
	assert "peft" in out["failed"]


def test_a_bare_run_that_times_out_still_lets_the_recorder_run(tmp_path, monkeypatch):
	monkeypatch.setattr(gt, "_stream", lambda cmd, **kw: (None, "step 50"))
	out = gt._bare(tmp_path, 3, install_requirements=False, caches=tmp_path, env={}, timeout_s=60)
	assert "longer than 60 s" in out["failed"] and out["tail"] == "step 50"


def test_streamed_output_passes_through_and_a_hang_is_killed_on_time(tmp_path, capsys):
	import sys
	import time

	code, tail = gt._stream([sys.executable, "-c", "print('hello', flush=True)"], env=dict(os.environ),
		cwd=tmp_path, timeout_s=30)  # fmt: skip
	assert code == 0 and "hello" in tail and "hello" in capsys.readouterr().out
	t0 = time.monotonic()
	# a progress bar that never ends its line, then a hang: still killed at the deadline
	hang = "import sys, time; sys.stdout.write('50%\\r'); sys.stdout.flush(); time.sleep(60)"
	code, tail = gt._stream([sys.executable, "-c", hang], env=dict(os.environ), cwd=tmp_path, timeout_s=1.5)
	assert code is None and "50%" in tail and time.monotonic() - t0 < 10


def test_the_counts_b_reads_drop_anything_but_finite_non_negative_numbers():
	from preflight.probes.local import _extras

	got = _extras({"epochs": int("1" + "0" * 400), "planned_steps": float("nan"), "params_total": -5,
		"train_loader_batches": 2500, "tokens_per_step": "512", "samples_per_step": True,
		"micro_steps_per_step": float("inf"), "host_rss_peak_bytes": 3.5e9})  # fmt: skip
	assert got == {"train_loader_batches": 2500.0, "host_rss_peak_bytes": 3.5e9}


def test_a_worker_left_holding_the_pipe_doesnt_hold_the_run(tmp_path):
	"""The command exits but a child it started (a dataloader worker) keeps the output pipe open."""
	import subprocess
	import sys
	import time

	leave = (
		"import subprocess, sys; "
		"subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)']); "
		"print('done', flush=True)"
	)
	t0 = time.monotonic()
	code, tail = gt._stream([sys.executable, "-c", leave], env=dict(os.environ), cwd=tmp_path, timeout_s=30)
	assert code == 0 and "done" in tail
	assert time.monotonic() - t0 < gt.DRAIN_S + 5  # not the 30 s deadline
	assert subprocess.run(["pgrep", "-f", "time.sleep(60)"], capture_output=True).returncode != 0  # killed


def test_files_sent_from_the_checkout_are_measured_where_the_zoo_isnt(tmp_path, monkeypatch):
	"""Oct 31: Modal's probe image has no zoo, so the ops app sends the files with the call."""
	import types

	import preflight.probes.local as local

	seen = {}

	def bare(work, steps, **kw):
		seen["train"] = (work / "train.py").read_text()
		return {"steps": steps, "walls_s": [None], "ended": "steps"}

	def probe(spec, **kw):
		seen["probe_files"] = spec.files
		p = types.SimpleNamespace(status="error", steps=0, step_time=None, memory=None, cost=None,
			step_log=[], error="x", error_code="x")  # fmt: skip
		return types.SimpleNamespace(probe=p, env=None, log_tail="")

	monkeypatch.setattr(gt, "_bare", bare)
	monkeypatch.setattr(local, "run_probe", probe)
	files = {"train.py": "print('hi')\n", "requirements.txt": "peft\n"}
	res = gt.measure(tmp_path / "not-here" / "finetune_lora_hf", bare_steps=3, files=files, budget_s=60)
	assert seen["train"] == "print('hi')\n" and seen["probe_files"] == files
	assert res["script"] == "finetune_lora_hf"
