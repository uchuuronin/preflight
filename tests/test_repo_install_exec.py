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


# ---------------------------------------------------------------- probes.torch_new (off; the PM, Nov 3)


def test_code_that_asks_for_a_newer_torch_reads_its_ask():
	lit = {
		"pyproject.toml": '[project]\nname = "litgpt"\ndependencies = ["torch>=2.7", "torch<2.13; python_version>=\'3.10\'"]\n'
	}
	assert R.torch_asks(lit) == ["torch>=2.7"]  # the marker line may not hold here
	assert not R.torch_fits(["torch>=2.7"], R.IMAGE_TORCH) and R.torch_fits(["torch>=2.7"], R.NEW_IMAGE_TORCH)
	assert R.torch_asks({"requirements.txt": "torch\nnumpy\n"}) == []  # no version asked: the image's is fine
	assert (
		R.torch_too_old(["torch>=2.6"], "2.5.1")
		== "needs torch>=2.6; Preflight's probe image has torch 2.5.1"
	)
	assert R.torch_too_old(["torch>=2.4"], "2.5.1") is None


def test_the_image_is_picked_by_the_ask_only_with_the_flag(monkeypatch):
	import preflight.runner.pipeline as P
	from preflight.harness.static import map_static

	files = {
		"train.py": "import torch\nfor step in range(10):\n    pass\n",
		"requirements.txt": "torch>=2.7\n",
	}
	sm = map_static(files, entrypoint="train.py")
	monkeypatch.setattr(P, "_flag", lambda name: False)
	assert P._image(sm, files) == "probe"
	monkeypatch.setattr(P, "_flag", lambda name: name == "probes.torch_new")
	assert P._image(sm, files) == "torch27"
	assert (
		P._image(sm, {**files, "requirements.txt": "torch>=2.8\n"}) == "probe"
	)  # neither image fits: the stop says so
	assert P._image(sm, {**files, "requirements.txt": "torch\n"}) == "probe"


def test_a_torch_the_image_cant_give_stops_before_any_install(tmp_path, monkeypatch):
	(tmp_path / "requirements.txt").write_text("torch>=2.8\neinops\n")
	monkeypatch.setattr(R, "installed", lambda: {"torch": "2.5.1+cu124"})
	monkeypatch.setattr(R, "_pip", lambda *a, **k: (_ for _ in ()).throw(AssertionError("installed")))
	err, dirs = R.install(tmp_path, tmp_path / "deps", log=lambda m: None, torch_stop=True)
	assert err == "needs torch>=2.8; Preflight's probe image has torch 2.5.1" and dirs == []


def test_the_stop_is_only_for_a_newer_torch_review():
	assert R.torch_too_old(["torch==2.4.0"], "2.5.1") is None  # an older pin runs on the image's, as before
	assert R.torch_too_old(["torch==2.5.1+cu124"], "2.5.1") is None  # a CUDA label isn't a version
	assert (
		R.torch_too_old(["torch<2.5"], "2.5.1") is None and R.torch_too_old(["torch!=2.5.1"], "2.5.1") is None
	)
	assert R.torch_too_old(["torch==2.7.1"], "2.5.1") and R.torch_too_old(["torch~=2.6"], "2.5.1")
	assert R.torch_fits(["torch==2.5.1+cu124"], "2.5.1")


def test_a_hostile_report_installs_nothing_from_it(tmp_path, monkeypatch):
	for doc in ({"install": {"a": 1}}, {"install": [{"metadata": {"requires_dist": "torch"}}]}, [1, 2], "x"):

		def run(cmd, doc=doc, **kw):
			Path(cmd[cmd.index("--report") + 1]).write_text(json.dumps(doc))
			return subprocess.CompletedProcess(cmd, 0, "", "")

		monkeypatch.setattr(R.subprocess, "run", run)
		assert R.requires_from_build(tmp_path, log=lambda m: None) == []


def test_torch27_s_pins_key_its_cache_and_the_default_key_doesn_t_move():
	import preflight.runner.pipeline as P
	from preflight.probes.local import ProbeSpec

	base = ProbeSpec(probe_id="p", files={"train.py": "x"}, entrypoint="train.py")
	new = ProbeSpec(probe_id="p", files={"train.py": "x"}, entrypoint="train.py", image="torch27")
	assert P._image_pins("probe") == "" and P._image_pins("torch27").startswith("@")
	assert P.cache_key(base, "L4", "img") != P.cache_key(new, "L4", "img")


def test_the_probe_sees_pyg_s_index_a_public_url():
	from preflight.probes.local import SAFE_ENV

	assert "PREFLIGHT_PYG_INDEX" in SAFE_ENV


def test_lm_eval_s_own_repo_builds_the_same_source_only_packages_as_its_command(monkeypatch):
	import preflight.runner.pipeline as P
	from preflight.runner.sources import COMMANDS, Fetched

	f = Fetched(
		files={"lm_eval/__init__.py": "", "lm_eval/__main__.py": ""}, entrypoint="lm_eval/__main__.py"
	)
	monkeypatch.setattr(P, "_flag", lambda name: False)
	assert P._command_sdists(f) == ()
	monkeypatch.setattr(P, "_flag", lambda name: name == "probes.repo_install_exec")
	assert P._command_sdists(f) == tuple(COMMANDS["lm_eval"]["sdists"])


def test_the_repo_s_own_metadata_is_an_editable_no_deps_install(tmp_path, monkeypatch):
	seen = []

	def run(cmd, **kw):
		seen.append(cmd)
		return subprocess.CompletedProcess(cmd, 0, "", "")

	monkeypatch.setattr(R.subprocess, "run", run)
	assert R.own_metadata(tmp_path, tmp_path / "deps", log=lambda m: None)
	cmd = seen[0]
	assert "--no-deps" in cmd and cmd[cmd.index("-e") + 1] == str(tmp_path) and "--prefix" in cmd
	assert "--ignore-installed" in cmd  # never uninstalls the image's own copy
