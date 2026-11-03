"""recorder/ground_truth.py without a GPU: what it refuses, and where the bare run installs."""

import json
import os
from pathlib import Path

from preflight.recorder import ground_truth as gt


def test_a_dir_with_nothing_to_run_is_refused_before_any_gpu_time(tmp_path, capsys):
	# any script dir is measured now (Monday's unseen scripts), but it needs an entrypoint
	(tmp_path / "mine").mkdir()
	(tmp_path / "mine" / "util.py").write_text("x = 1\n")
	assert gt.main([str(tmp_path / "mine"), str(tmp_path / "o.json"), "--bare-steps", "3"]) == 2
	assert gt.main([str(tmp_path / "nope"), str(tmp_path / "o.json"), "--bare-steps", "3"]) == 2
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
			step_log=[], error="x", error_code="x", rl=None, generation=None, extras={},
			model_dump=lambda **kw: {"status": "error"})  # fmt: skip
		return types.SimpleNamespace(probe=p, env=None, log_tail="")

	monkeypatch.setattr(gt, "_bare", bare)
	monkeypatch.setattr(local, "run_probe", probe)
	files = {"train.py": "print('hi')\n", "requirements.txt": "peft\n"}
	res = gt.measure(tmp_path / "not-here" / "finetune_lora_hf", bare_steps=3, files=files, budget_s=60)
	assert seen["train"] == "print('hi')\n" and seen["probe_files"] == files
	assert res["script"] == "finetune_lora_hf"


def test_the_bare_run_keeps_its_own_hf_home_so_the_probe_can_write_its_locks(tmp_path, monkeypatch):
	"""Oct 31, a-gt-lora-l4-4: root's dataset locks in the shared HF_HOME stopped the probe uid."""
	home = tmp_path / "hf"
	weights = tmp_path / "volume" / "models--Qwen--Qwen2.5-1.5B"
	weights.mkdir(parents=True)
	(home / "hub").mkdir(parents=True)
	(home / "hub" / "models--Qwen--Qwen2.5-1.5B").symlink_to(weights, target_is_directory=True)
	monkeypatch.setenv("HF_HOME", str(home))
	seen = {}

	def bare(work, steps, *, env, **kw):
		seen["env"] = env
		seen["link"] = (Path(env["HF_HOME"]) / "hub" / "models--Qwen--Qwen2.5-1.5B").resolve()
		return {"steps": steps, "walls_s": [None], "ended": "steps"}

	import types

	import preflight.probes.local as local

	monkeypatch.setattr(gt, "_bare", bare)
	monkeypatch.setattr(local, "run_probe", lambda spec, **kw: types.SimpleNamespace(
		probe=types.SimpleNamespace(status="error", steps=0, step_time=None, memory=None, cost=None,
			step_log=[], error="x", error_code="x", rl=None, generation=None, extras={},
			model_dump=lambda **kw: {"status": "error"}), env=None, log_tail=""))  # fmt: skip
	gt.measure(tmp_path / "s", bare_steps=1, files={"train.py": ""}, budget_s=60)
	own = Path(seen["env"]["HF_HOME"])
	assert own != home and seen["link"] == weights.resolve()
	assert os.environ["HF_HOME"] == str(home)  # the probe keeps the shared one


def test_the_log_line_summarises_the_rl_lists():
	res = {"script": "grpo_trl", "bare": {"walls_s": [1.0]}, "recorder": {"step_log": [], "log_tail": "",
		"rl": {"group_size": 8, "pass_rate_per_prompt": [0.5] * 200, "kl": [0.1] * 50, "zero_spread_fraction": 0.2}}}  # fmt: skip
	b = gt.brief(res)
	assert b["recorder"]["rl"] == {
		"group_size": 8,
		"prompts": 200,
		"truncation_rate": None,
		"zero_spread_fraction": 0.2,
	}


def test_a_full_run_takes_any_gpu_script_with_its_entrypoint_and_image():
	# Monday's cost table (B, Nov 1): every zoo script, and any script by directory
	from pathlib import Path

	from preflight.recorder.ground_truth import ZOO_SCRIPTS, entrypoint_of, plan
	from preflight.runner.sources import _read_tree

	zoo = Path(__file__).resolve().parents[1] / "zoo"
	want = {"serving_vllm": ("serve.py", "serving", "serving"), "eval_sweep_reasoning": ("run.py", "eval", "probe"),
		"torchtitan_debug": ("train.py", "pretrain", "titan"), "finetune_lora_hf": ("train.py", "finetune", "probe")}  # fmt: skip
	assert len(ZOO_SCRIPTS) == 9 and "agent_eval_spark" not in ZOO_SCRIPTS
	for name in ZOO_SCRIPTS:
		p = plan(_read_tree(zoo / name))
		if name in want:
			assert (p["entrypoint"], p["lens"], p["image"]) == want[name], name
		assert p["generate_sample"] == (32 if p["lens"] == "serving" else 0)
	assert plan(_read_tree(zoo / "agent_eval_spark"))["lens"] == "agent"
	assert entrypoint_of({"main.py": "", "util.py": ""}) == "main.py"
	assert plan({"a.py": "# about torchtitan\nimport torch\n", "main.py": "import a\n"})["image"] == "probe"
	import pytest

	with pytest.raises(ValueError):
		entrypoint_of({"util.py": ""})
	with pytest.raises(ValueError):
		entrypoint_of({"train.py": ""}, "missing.py")


def test_the_bare_run_passes_the_installer_only_its_own_options(tmp_path, monkeypatch):
	import preflight.recorder.requirements as req

	seen = {}

	def install(workdir, target, *, log=print, **kw):
		seen["kw"] = kw
		return None, []

	def stream(cmd, **kw):
		Path(cmd[4]).write_text(json.dumps({"steps": 3, "walls_s": [None, 1.0, 1.0], "ended": "steps"}))
		return 0, ""

	monkeypatch.setattr(req, "install", install)
	monkeypatch.setattr(gt, "_stream", stream)
	gt._bare(tmp_path, 3, install_requirements=True, caches=tmp_path / "c", env={}, timeout_s=60,
		install_opts={"uv": True, "sdists": ("a",), "eval_units": 8})  # fmt: skip
	assert seen["kw"] == {"uv": True, "sdists": ("a",)}  # eval_units is the recorder's, not the installer's
