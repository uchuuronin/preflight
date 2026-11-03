"""probes.repo_install_exec (off; the PM, Nov 3): a setup.py that builds its dependency list in code
(open-r1's deps_list) is run, inside the probe's sandbox, by pip's metadata-only build, and the list
it gives is installed the usual way. The repo itself isn't installed."""

import json
import subprocess
from pathlib import Path

import pytest

from preflight.recorder import requirements as R

BUILT = """from setuptools import setup
_deps = {"einops": "einops>=0.8.0", "tqdm": "tqdm"}
def deps_list(*names):
    return [_deps[n] for n in names]
setup(name="demo", version="0.1", py_modules=[], install_requires=deps_list("einops", "tqdm"),
      extras_require={"dev": ["pytest"]})
"""


def test_a_list_built_in_code_is_told_apart_from_a_literal_one(tmp_path):
	(tmp_path / "setup.py").write_text(BUILT)
	assert R.declared(tmp_path) == [] and R.built_in_code(tmp_path)
	lit = tmp_path / "lit"
	lit.mkdir()
	(lit / "setup.py").write_text(
		"from setuptools import setup\nsetup(name='x', install_requires=['tqdm'])\n"
	)
	assert R.declared(lit) == ["tqdm"]  # read without running it, as before
	none = tmp_path / "none"
	none.mkdir()
	(none / "setup.py").write_text("from setuptools import setup\nsetup(name='x')\n")
	assert not R.built_in_code(none)
	dyn = tmp_path / "dyn"
	dyn.mkdir()
	(dyn / "pyproject.toml").write_text('[project]\nname = "x"\ndynamic = ["dependencies", "version"]\n')
	assert R.built_in_code(dyn)


def _fake_pip(requires, rc=0):
	def run(cmd, **kw):
		assert cmd[1:5] == ["-I", "-m", "pip", "install"] and "--dry-run" in cmd and "--no-deps" in cmd
		report = Path(cmd[cmd.index("--report") + 1])
		if rc == 0:
			report.write_text(
				json.dumps({"install": [{"metadata": {"name": "demo", "requires_dist": requires}}]})
			)
		return subprocess.CompletedProcess(cmd, rc, "", "error: metadata-generation-failed" if rc else "")

	return run


def test_the_build_s_list_is_read_without_extras(tmp_path, monkeypatch):
	monkeypatch.setattr(R.subprocess, "run", _fake_pip(["einops>=0.8.0", "tqdm", 'pytest; extra == "dev"',
		"numpy<2; python_version < '3.9'"]))  # fmt: skip
	got = R.requires_from_build(tmp_path, log=lambda m: None)
	assert got == ["einops>=0.8.0", "tqdm", "numpy<2; python_version < '3.9'"]


def test_a_failed_build_installs_nothing_from_it_and_says_why(tmp_path, monkeypatch):
	monkeypatch.setattr(R.subprocess, "run", _fake_pip([], rc=1))
	said = []
	assert R.requires_from_build(tmp_path, log=said.append) == []
	assert "metadata-generation-failed" in said[0]


def test_only_with_the_flag_and_only_when_the_list_is_built_in_code(tmp_path, monkeypatch):
	(tmp_path / "setup.py").write_text(BUILT)
	(tmp_path / "train.py").write_text("import os\n")
	calls = []
	monkeypatch.setattr(R, "requires_from_build", lambda w, log: calls.append(w) or ["einops>=0.8.0"])
	off, _ = R.repo_requirement_sets(tmp_path, "train.py", log=lambda m: None)
	assert off == [] and calls == []  # off: setup.py is never run
	on, _ = R.repo_requirement_sets(tmp_path, "train.py", log=lambda m: None, build=True)
	assert on == ["einops>=0.8.0"] and calls == [tmp_path]


def test_pip_s_real_metadata_build_reads_a_list_built_in_code(tmp_path):
	pytest.importorskip("pip")
	pytest.importorskip("setuptools")
	(tmp_path / "setup.py").write_text(BUILT)
	got = R.requires_from_build(tmp_path, log=lambda m: None)
	if not got:
		pytest.skip("pip couldn't build here (no network for setuptools)")
	assert sorted(got) == ["einops>=0.8.0", "tqdm"]


def test_the_spec_carries_it_and_the_cache_key_changes_only_when_set():
	from preflight.probes.local import ProbeSpec

	assert ProbeSpec.__dataclass_fields__["install_exec"].default is False
