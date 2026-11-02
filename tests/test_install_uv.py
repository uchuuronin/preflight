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
		return subprocess.CompletedProcess(cmd, code, "", "" if code == 0 else "error: something failed")

	monkeypatch.setattr(R.subprocess, "run", fake_run)
	monkeypatch.setattr(R, "installed", lambda: {"torch": "2.5.1+cu124"})
	monkeypatch.delenv("PREFLIGHT_DEPS_DIR", raising=False)
	(tmp_path / "requirements.txt").write_text("litdata\n")
	return seen, state, tmp_path


def test_uv_resolves_and_pip_installs_only_what_the_image_lacks_without_resolving(pip, monkeypatch):
	seen, _, work = pip
	monkeypatch.setattr(R, "_uv", lambda: ["/usr/bin/uv"])
	compiled = "litdata==0.2.30\ntorch==2.5.1\n# a comment\nboto3==1.35.0\n"
	real = R.subprocess.run

	def run(cmd, **kw):
		if cmd[:3] == ["/usr/bin/uv", "pip", "compile"]:
			seen.append(list(cmd))
			assert kw["input"] == "litdata\n" and "PYTHONPATH" not in kw["env"]
			return subprocess.CompletedProcess(cmd, 0, compiled, "")
		return real(cmd, **kw)

	monkeypatch.setattr(R.subprocess, "run", run)
	assert R.install(work, work / "d", log=lambda m: None, uv=True)[0] is None
	compile_cmd, pip_cmd = seen[-2], seen[-1]
	assert "-c" in compile_cmd and "--only-binary" in compile_cmd
	# torch is the image's: never installed again; the rest exactly as uv chose, with no second resolve
	assert pip_cmd[0] == R.sys.executable and "--no-deps" in pip_cmd and "--prefix" in pip_cmd
	assert pip_cmd[-2:] == ["litdata==0.2.30", "boto3==1.35.0"] and "torch==2.5.1" not in pip_cmd


def test_pip_resolves_as_before_when_uvs_route_fails_or_uv_isnt_there(pip, monkeypatch):
	seen, state, work = pip
	monkeypatch.setattr(R, "_uv", lambda: ["/usr/bin/uv"])
	state["answer"] = lambda cmd: 1 if cmd[0] == "/usr/bin/uv" else 0
	notes = []
	assert R.install(work, work / "d1", log=notes.append, uv=True)[0] is None
	assert seen[-2][0] == "/usr/bin/uv" and seen[-1][0] == R.sys.executable and "--no-deps" not in seen[-1]
	assert any("trying pip" in n for n in notes)
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


def test_the_probe_spec_carries_them_and_each_changes_the_cache_key():
	from preflight.probes.local import ProbeSpec
	from preflight.runner.pipeline import cache_key

	kw = {"probe_id": "p1", "files": {"t.py": "x"}, "entrypoint": "t.py", "install_requirements": True}
	keys = {
		cache_key(ProbeSpec(**kw, **x), "L4", "i") for x in ({}, {"install_uv": True}, {"install_pyg": True})
	}
	assert len(keys) == 3  # another resolver can pick other versions, another index another build


def test_with_pygs_index_torch_geometric_brings_its_run_time_extensions(tmp_path, monkeypatch):
	import importlib.util

	(tmp_path / "gnn.py").write_text("import torch_geometric.transforms as T\nT.ToSparseTensor()\n")
	monkeypatch.setattr(
		importlib.util, "find_spec", lambda name, *a, **k: None if name.startswith("torch_") else object()
	)
	_, imported = R.repo_requirement_sets(tmp_path, "gnn.py", log=lambda m: None, pyg=True)
	assert imported == ["torch-geometric", "torch-scatter", "torch-sparse"]
	assert R.repo_requirement_sets(tmp_path, "gnn.py", log=lambda m: None)[1] == ["torch-geometric"]


def test_when_uv_finds_no_solution_pip_isnt_run_and_the_reason_is_said(pip, monkeypatch):
	seen, _, work = pip
	monkeypatch.setattr(R, "_uv", lambda: ["/usr/bin/uv"])
	text = (
		"  × No solution found when resolving dependencies:\n"
		"  ╰─▶ Because litgpt depends on torch>=2.7 and you require torch==2.5.1, we can conclude that your "
		"requirements are unsatisfiable.\n"
	)

	def run(cmd, **kw):
		seen.append(list(cmd))
		assert kw["env"]["UV_STACK_SIZE"] == str(R.UV_STACK_BYTES)
		return subprocess.CompletedProcess(cmd, 1, "", text)

	monkeypatch.setattr(R.subprocess, "run", run)
	err, _ = R.install(work, work / "d", log=lambda m: None, uv=True)
	assert err.startswith(R.NO_SOLUTION) and "torch>=2.7" in err and len(seen) == 1


def test_a_failed_install_names_the_pins_the_code_asks_past(pip, monkeypatch):
	_, state, work = pip
	(work / "requirements.txt").write_text("torch>=2.7\nlitdata\ntqdm>4\n")
	monkeypatch.setattr(R, "installed", lambda: {"torch": "2.5.1+cu124", "tqdm": "4.70.1"})
	state["answer"] = lambda cmd: 1
	err, _ = R.install(work, work / "d", log=lambda m: None)
	assert err.startswith(
		"the code asks for torch>=2.7 (the image has 2.5.1), which the probe image can't change."
	)
	assert "tqdm" not in err.split(".")[0]  # a line the image's copy satisfies isn't named
