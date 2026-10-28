"""The probe sandbox: user code is hostile. No secrets, no survivors, no writes outside the workdir."""

import os
import time
from pathlib import Path

import pytest

from preflight.probes.local import (
	LOG_CAP_BYTES,
	InvalidSubmission,
	ProbeSpec,
	parse_result,
	run_probe,
	validate_files,
)

FIX = Path(__file__).resolve().parent / "fixtures"
# tests that launch the recorder need torch in the probe process; path and parsing checks don't
needs_torch = pytest.mark.torch


def spec(script, **kw):
	return ProbeSpec(
		probe_id=kw.pop("probe_id", "p1"), files={script: (FIX / script).read_text()}, entrypoint=script, **kw
	)


@needs_torch
def test_secrets_in_the_worker_never_reach_user_code(monkeypatch, tmp_path):
	monkeypatch.setenv("MODEL_API_KEY", "sk-demo-SECRET")
	home = tmp_path / "home"
	home.mkdir()
	(home / ".modal.toml").write_text('token_secret = "as-SECRET"\n')
	monkeypatch.setenv("HOME", str(home))
	r = run_probe(spec("leak.py", timeout_s=30))
	assert "NOTHING" in r.log_tail, r.log_tail


@needs_torch
def test_processes_started_by_the_probe_do_not_survive_it(tmp_path):
	marker = tmp_path / "marker"
	r = run_probe(spec("orphan.py", args=[str(marker)], timeout_s=30, out_dir=tmp_path / "run"))
	assert r.probe.status == "error"  # finished with 0 steps
	time.sleep(4)
	assert not marker.exists()


@pytest.mark.parametrize(
	"bad", ["../escape.py", "/etc/passwd", "a/../../b.py", "ok/../../x", 'q"uote.py', "new\nline.py"]
)
def test_paths_outside_the_workdir_are_rejected(bad):
	with pytest.raises(InvalidSubmission):
		validate_files({bad: "x = 1"}, entrypoint=bad)


def test_sibling_prefix_directory_is_rejected():
	with pytest.raises(InvalidSubmission):
		validate_files({"../p1-work-evil/x.py": "x"}, entrypoint="../p1-work-evil/x.py")


@pytest.mark.parametrize(
	("text", "says"),
	[("x = '\ud800'\n", "line 1"), ("a = 1\nb = '\udfff'\n", "line 2"), (None, "not text"), (3, "not text")],
)
def test_text_that_is_not_valid_unicode_is_a_visitor_error_not_a_crash(text, says):
	"""D's finding (Oct 28): an unpaired surrogate raised UnicodeEncodeError, a server error."""
	with pytest.raises(InvalidSubmission) as got:
		validate_files({"run.py": text}, entrypoint="run.py")
	assert got.value.code == "submission.invalid"
	assert "run.py" in str(got.value) and says in str(got.value)


def test_a_notebook_with_an_unpaired_surrogate_is_refused_clearly():
	import json

	from preflight.runner.sources import fetch

	nb = {"cells": [{"cell_type": "code", "source": ["x = '\ud800'\n"]}], "metadata": {}, "nbformat": 4}
	with pytest.raises(InvalidSubmission) as got:
		fetch({"kind": "notebook", "files": {"run.ipynb": json.dumps(nb)}})
	assert got.value.code == "submission.invalid" and "unpaired surrogate" in str(got.value)


def test_submission_size_caps():
	with pytest.raises(InvalidSubmission, match="files"):
		validate_files({f"f{i}.py": "" for i in range(5000)}, entrypoint="f0.py")
	with pytest.raises(InvalidSubmission, match="MB"):
		validate_files({"big.py": "x" * (30 * 1024 * 1024)}, entrypoint="big.py")


def test_entrypoint_must_be_submitted():
	with pytest.raises(InvalidSubmission, match="entrypoint"):
		validate_files({"a.py": ""}, entrypoint="b.py")


@needs_torch
def test_stale_results_from_an_earlier_attempt_are_never_reused(tmp_path):
	(tmp_path / "p1.result.json").write_text('{"status": "ok", "error": null, "summary": {"steps": 9}}')
	r = run_probe(spec("hang.py", timeout_s=5, out_dir=tmp_path))
	assert r.probe.status == "timeout"


@needs_torch
def test_output_is_capped(tmp_path):
	run_probe(spec("chatty.py", timeout_s=60, out_dir=tmp_path))
	assert (tmp_path / "p1.log").stat().st_size <= LOG_CAP_BYTES + 4096


@needs_torch
def test_cancellation_reaches_a_running_probe(tmp_path):
	t0 = time.monotonic()
	r = run_probe(spec("hang.py", timeout_s=120), should_stop=lambda: time.monotonic() - t0 > 1.5)
	assert r.probe.status == "cancelled"
	assert time.monotonic() - t0 < 10


@pytest.mark.parametrize(
	"raw",
	[
		"not json",
		'{"status": "ok"}',
		'{"status": "ok", "error": null, "summary": {"steps": 3, "mfu": 2.4, "step_time": {"n_steps": 1, "mean_s": 1}}}',
		'{"status": "weird", "error": null, "summary": {}}',
	],
)
def test_malformed_child_output_becomes_a_recorder_error(raw):
	status, code, _error, _summary, _env = parse_result(raw)
	assert status == "error" and code == "recorder"


def test_valid_child_output_passes_through():
	status, code, _error, summary, _env = parse_result(
		'{"status": "ok", "error": null, "summary": {"steps": 3}, "env": {}}'
	)
	assert (status, code, summary["steps"]) == ("ok", None, 3)


@needs_torch
@pytest.mark.skipif(os.geteuid() != 0, reason="privilege drop only when the worker runs as root")
def test_probe_runs_as_an_unprivileged_user(tmp_path):
	files = {"who.py": "import os\nprint('UID', os.getuid())\n"}
	r = run_probe(ProbeSpec(probe_id="p1", files=files, entrypoint="who.py", timeout_s=30, out_dir=tmp_path))
	assert "UID 0" not in r.log_tail and "UID " in r.log_tail


@needs_torch
def test_visitor_code_runs_under_resource_limits(tmp_path):
	files = {
		"lim.py": "import resource\nfor n in ('RLIMIT_CORE', 'RLIMIT_FSIZE', 'RLIMIT_NOFILE'):\n    print(n, resource.getrlimit(getattr(resource, n))[1])\n"
	}
	r = run_probe(ProbeSpec(probe_id="p1", files=files, entrypoint="lim.py", timeout_s=30, out_dir=tmp_path))
	assert "RLIMIT_CORE 0" in r.log_tail
	assert f"RLIMIT_FSIZE {64 * 2**30}" in r.log_tail
	nofile = int(r.log_tail.split("RLIMIT_NOFILE ")[1].split()[0])
	assert 0 < nofile <= 65536  # lowered to ours, or the host's if that is lower; never raised


def test_a_pinned_probe_uid_is_used_for_every_probe(monkeypatch):
	from preflight.probes.local import _probe_uid

	monkeypatch.setenv("PREFLIGHT_PROBE_UID", "61000")
	if os.geteuid() == 0:
		assert _probe_uid() == _probe_uid() == 61000
	monkeypatch.setenv("PREFLIGHT_PROBE_UID", "0")  # never root, never a system uid
	if os.geteuid() == 0:
		assert _probe_uid() != 0


def test_probes_import_a_copy_of_the_package_when_its_home_is_closed(tmp_path, monkeypatch):
	import preflight.probes.local as L

	locked = tmp_path / "root"
	pkg = locked / "pf" / "python"
	(pkg / "preflight").mkdir(parents=True)
	(pkg / "preflight" / "__init__.py").write_text("x = 1\n")
	locked.chmod(0o700)
	monkeypatch.setattr(L, "_PKG_COPY", None)
	copy = L._package_for_probe(pkg)
	assert copy != pkg and (copy / "preflight" / "__init__.py").read_text() == "x = 1\n"
	assert L._others_can_reach(copy / "preflight")
	assert locked.stat().st_mode & 0o777 == 0o700  # the host's permissions are left alone
	assert L._package_for_probe(pkg) == copy  # one copy per process
	import shutil
	import tempfile

	top = Path(tempfile.mkdtemp())  # pytest's own tmp dirs are 0700, so the open case lives in /tmp
	try:
		top.chmod(0o755)
		open_pkg = top / "python"
		open_pkg.mkdir(mode=0o755)
		assert L._package_for_probe(open_pkg) == open_pkg
	finally:
		shutil.rmtree(top)


@needs_torch
def test_forced_seeds_override_the_scripts_own(tmp_path):
	files = {
		"s.py": "import random, torch\ntorch.manual_seed(42); random.seed(42)\nprint('DRAW', torch.rand(1).item(), random.random())\n"
	}
	draws = []
	for seed in (1, 2, 2):
		r = run_probe(ProbeSpec(probe_id="p1", files=files, entrypoint="s.py", timeout_s=60, seed=seed,
			seed_force=True, out_dir=tmp_path / str(len(draws))))  # fmt: skip
		draws.append(next(ln for ln in r.log_tail.splitlines() if ln.startswith("DRAW")))
	assert draws[0] != draws[1] and draws[1] == draws[2]
