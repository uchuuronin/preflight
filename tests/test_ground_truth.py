"""recorder/ground_truth.py without a GPU: what it refuses, and where the bare run installs."""

import json
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
		import os

		seen["env"] = {k: os.environ.get(k) for k in ("PREFLIGHT_DEPS_DIR", "PREFLIGHT_PIP_CACHE")}
		seen["target"] = target
		return None, [Path("/gt/site")]

	def run(cmd, **kw):
		seen["cmd"], seen["kw"] = cmd, kw
		Path(cmd[4]).write_text(json.dumps({"steps": 3, "walls_s": [None, 1.0, 1.0], "ended": "steps"}))

		class Done:
			stdout = stderr = ""

		return Done()

	monkeypatch.setattr(req, "install", install)
	monkeypatch.setattr(gt.subprocess, "run", run)
	out = gt._bare(tmp_path, 3, install_requirements=True, caches=tmp_path / "c", env={"HF_HOME": "/gt/hf"})
	assert out["ended"] == "steps"
	assert seen["env"] == {"PREFLIGHT_DEPS_DIR": None, "PREFLIGHT_PIP_CACHE": None}
	assert seen["target"] == tmp_path / "c" / "deps"
	assert seen["kw"]["env"]["PYTHONPATH"].split(":")[0] == "/gt/site"
	assert seen["kw"]["env"]["HF_HOME"] == "/gt/hf"
	assert seen["cmd"][-2:] == ["--steps", "3"]
	import os

	assert os.environ["PREFLIGHT_DEPS_DIR"] == "/probe/deps"  # restored for the probe
	assert os.environ["PREFLIGHT_PIP_CACHE"] == "/probe/pip"


def test_a_bare_run_that_writes_nothing_reports_its_output(tmp_path, monkeypatch):
	class Done:
		stdout, stderr = "", "ModuleNotFoundError: No module named 'peft'"

	monkeypatch.setattr(gt.subprocess, "run", lambda cmd, **kw: Done())
	out = gt._bare(tmp_path, 3, install_requirements=False, caches=tmp_path, env={})
	assert "peft" in out["failed"]
