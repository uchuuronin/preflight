"""probes.uv_install and probes.extra_hosts (both off; Nov 2, after freeze): uv's resolver before pip's with the
same constraints, and PyG's wheel index for the packages PyPI has only as source."""

import subprocess

import pytest

from preflight.recorder import requirements as R


@pytest.fixture
def pip(monkeypatch, tmp_path):
	"""Every installer command, answered by `answer(cmd)` (a return code)."""
	seen: list[list[str]] = []
	state = {"answer": lambda cmd: 0}

	def fake_run(cmd, **kw):
		seen.append(list(cmd))
		code = state["answer"](cmd)
		return subprocess.CompletedProcess(cmd, code, "", "" if code == 0 else "No solution found")

	monkeypatch.setattr(R.subprocess, "run", fake_run)
	monkeypatch.setattr(R, "installed", lambda: {"torch": "2.5.1+cu124"})
	monkeypatch.delenv("PREFLIGHT_DEPS_DIR", raising=False)
	(tmp_path / "requirements.txt").write_text("litdata\n")
	return seen, state, tmp_path


def test_uv_goes_first_with_the_same_constraints_and_wheels_only(pip, monkeypatch):
	seen, _, work = pip
	monkeypatch.setattr(R, "_uv", lambda: ["/usr/bin/uv"])
	assert R.install(work, work / "d", log=lambda m: None, uv=True)[0] is None
	assert len(seen) == 1 and seen[0][:3] == ["/usr/bin/uv", "pip", "install"]
	cmd = seen[0]
	assert "--only-binary" in cmd and "-c" in cmd and "--prefix" in cmd and cmd[-1] == "litdata"


def test_pip_runs_when_uv_fails_or_isnt_there(pip, monkeypatch):
	seen, state, work = pip
	monkeypatch.setattr(R, "_uv", lambda: ["/usr/bin/uv"])
	state["answer"] = lambda cmd: 1 if cmd[0] == "/usr/bin/uv" else 0
	notes = []
	assert R.install(work, work / "d1", log=notes.append, uv=True)[0] is None
	assert [c[0] for c in seen][-2:] == ["/usr/bin/uv", R.sys.executable] and any(
		"trying pip" in n for n in notes
	)
	monkeypatch.setattr(R, "_uv", lambda: None)
	seen.clear()
	assert R.install(work, work / "d2", log=lambda m: None, uv=True)[0] is None
	assert len(seen) == 1 and seen[0][0] == R.sys.executable


def test_without_the_flag_its_pip_alone_as_before(pip, monkeypatch):
	seen, _, work = pip
	monkeypatch.setattr(R, "_uv", lambda: pytest.fail("uv looked for"))
	assert R.install(work, work / "d", log=lambda m: None)[0] is None
	assert len(seen) == 1 and seen[0][0] == R.sys.executable and "--find-links" not in seen[0]


def test_pygs_index_is_passed_and_its_packages_count_as_known(pip, monkeypatch):
	seen, _, work = pip
	monkeypatch.setenv(R.PYG_INDEX_ENV, "https://data.pyg.org/whl/torch-2.5.0+cu124.html")
	assert R.install(work, work / "d", log=lambda m: None, pyg=True)[0] is None
	i = seen[0].index("--find-links")
	assert seen[0][i + 1].startswith("https://data.pyg.org/")
	(work / "requirements.txt").unlink()
	(work / "run.py").write_text("import torch_sparse\n")
	import importlib.util

	monkeypatch.setattr(
		importlib.util, "find_spec", lambda name, *a, **k: None if name == "torch_sparse" else object()
	)
	_, imported = R.repo_requirement_sets(work, "run.py", log=lambda m: None, pyg=True)
	assert imported == ["torch-sparse"]
	assert R.repo_requirement_sets(work, "run.py", log=lambda m: None)[1] == []  # off: not installable


def test_the_probe_spec_carries_them_and_the_cache_key_changes_only_for_pyg():
	from preflight.probes.local import ProbeSpec
	from preflight.runner.pipeline import cache_key

	base = ProbeSpec(probe_id="p1", files={"t.py": "x"}, entrypoint="t.py", install_requirements=True)
	uv = ProbeSpec(
		probe_id="p1", files={"t.py": "x"}, entrypoint="t.py", install_requirements=True, install_uv=True
	)
	pyg = ProbeSpec(
		probe_id="p1", files={"t.py": "x"}, entrypoint="t.py", install_requirements=True, install_pyg=True
	)
	# uv installs the same constraints; another index can install another build
	assert cache_key(base, "L4", "i") == cache_key(uv, "L4", "i") != cache_key(pyg, "L4", "i")
