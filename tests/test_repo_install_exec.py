"""probes.repo_install_exec (off; the PM, Nov 3): a setup.py that builds its dependency list in code
(open-r1's deps_list) is run, inside the probe's sandbox, by pip's metadata-only build, and the list
it gives is installed the usual way. The repo itself isn't installed."""

import json
import subprocess
from pathlib import Path

import pytest

from preflight.recorder import requirements as R


@pytest.fixture(autouse=True)
def _fresh_builds():
	R._BUILT.clear()  # the build is kept by its files' digest: each test's fake pip runs afresh
	yield
	R._BUILT.clear()


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
	(tmp_path / "setup.py").write_text(BUILT)  # built in code: the only case the build runs
	got = R.requires_from_build(tmp_path, log=lambda m: None)
	assert got == ["einops>=0.8.0", "tqdm", "numpy<2; python_version < '3.9'"]


def test_a_failed_build_installs_nothing_from_it_and_says_why(tmp_path, monkeypatch):
	monkeypatch.setattr(R.subprocess, "run", _fake_pip([], rc=1))
	said = []
	(tmp_path / "setup.py").write_text(BUILT)  # built in code: the only case the build runs
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
	(tmp_path / "setup.py").write_text(BUILT)  # built in code: the only case the build runs
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
		== "asks for torch>=2.6; Preflight's probe image has torch 2.5.1"
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
	assert P._image(sm, {**files, "requirements.txt": "torch>=2.8\n"}) == "torch27"  # the nearest torch here
	assert (
		P._image(sm, {**files, "requirements.txt": "torch==2.6.0\n"}) == "torch27"
	)  # pip freeze's pin (review)
	assert P._image(sm, {**files, "requirements.txt": "torch==2.4.0\n"}) == "probe"  # an older pin: as before
	assert P._image(sm, {**files, "requirements.txt": "torch\n"}) == "probe"


def test_a_newer_torch_ask_is_said_plainly_and_the_install_runs_on(tmp_path, monkeypatch):
	(tmp_path / "requirements.txt").write_text("torch==2.6.0\neinops\n")
	monkeypatch.setattr(R, "installed", lambda: {"torch": "2.5.1+cu124"})
	ran, said = [], []
	monkeypatch.setattr(R, "_pip", lambda keep, *a, **k: ran.append(keep))
	err, _ = R.install(tmp_path, tmp_path / "deps", log=said.append, torch_note=True)
	assert err is None and ran == [["einops"]]  # a pip-freeze pin no longer stops the job (review)
	assert any("asks for torch==2.6.0; Preflight's probe image has torch 2.5.1" in m for m in said)


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
		(tmp_path / "setup.py").write_text(BUILT)  # built in code: the only case the build runs
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


def test_the_build_runs_only_when_the_list_is_built_in_code(tmp_path, monkeypatch):
	seen = []
	monkeypatch.setattr(R.subprocess, "run", lambda cmd, **kw: seen.append(cmd))
	(tmp_path / "pyproject.toml").write_text(
		'[project]\nname = "x"\nversion = "1"\ndependencies = ["tqdm"]\n'
	)
	assert R.build_metadata(tmp_path, log=lambda m: None) is None
	assert R.requires_from_build(tmp_path, log=lambda m: None) == []
	(tmp_path / "pyproject.toml").unlink()
	(tmp_path / "setup.py").write_text(
		"from setuptools import setup\nsetup(name='x', install_requires=['tqdm'])\n"
	)
	assert R.build_metadata(tmp_path, log=lambda m: None) is None
	assert seen == []  # a literal list is read, never run


def test_the_build_runs_on_a_copy_with_compilers_that_fail(tmp_path, monkeypatch):
	(tmp_path / "setup.py").write_text(BUILT)
	(tmp_path / "pkg").mkdir()
	(tmp_path / "pkg" / "fast.c").write_text("int f(void){return 1;}")
	seen = []

	def run(cmd, **kw):
		seen.append((cmd, kw))
		where = Path(cmd[-1])
		(where / "demo.egg-info").mkdir()  # what pip's metadata step leaves beside setup.py
		Path(cmd[cmd.index("--report") + 1]).write_text(
			json.dumps(
				{"install": [{"metadata": {"name": "demo", "version": "0.1", "requires_dist": ["tqdm"]}}]}
			)
		)
		return subprocess.CompletedProcess(cmd, 0, "", "")

	monkeypatch.setattr(R.subprocess, "run", run)
	before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
	(tmp_path / "setup.py").write_text(BUILT)  # built in code: the only case the build runs
	assert R.requires_from_build(tmp_path, log=lambda m: None) == ["tqdm"]
	cmd, kw = seen[0]
	assert (
		Path(cmd[-1]) != tmp_path and (Path(cmd[-1]) / "setup.py").exists() is False
	)  # a scratch copy, gone
	assert (
		kw["env"]["CC"] == "false"
		and kw["env"]["CXX"] == "false"
		and "CUDA_HOME" not in kw["env"]
		and "CUDA_PATH" not in kw["env"]
	)  # unset, never a fake path: torch's CUDA lookup then returns None and flash-attn-style builds skip CUDA
	assert (
		sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*")) == before
	)  # the visitor's copy untouched


def test_the_repo_s_own_metadata_is_written_from_its_files_without_running_them(tmp_path, monkeypatch):
	monkeypatch.setattr(R.subprocess, "run", lambda *a, **k: pytest.fail("nothing runs for stated metadata"))
	repo = tmp_path / "repo"
	repo.mkdir()
	(repo / "pyproject.toml").write_text(
		'[project]\nname = "lm_eval"\nversion = "0.4.8"\ndependencies = []\n'
	)
	before = sorted(repo.rglob("*"))
	assert R.own_metadata(repo, tmp_path / "own", log=lambda m: None)
	assert sorted(repo.rglob("*")) == before
	import importlib.metadata as md

	(site,) = R.site_dirs(tmp_path / "own")
	(dist,) = md.distributions(path=[str(site)])
	assert (dist.metadata["Name"], dist.version) == ("lm_eval", "0.4.8")


def test_the_repo_s_own_metadata_builds_only_when_built_in_code(tmp_path, monkeypatch):
	(tmp_path / "setup.py").write_text(BUILT.replace('name="demo", version="0.1"', "name=NAME, version=VER"))
	monkeypatch.setattr(R, "build_metadata", lambda w, **k: {"name": "demo", "version": "0.1"})
	assert R.own_metadata(tmp_path, tmp_path.parent / "own-b", log=lambda m: None)
	nobuild = tmp_path.parent / "nobuild"
	nobuild.mkdir()
	(nobuild / "setup.py").write_text("from setuptools import setup\nsetup(name=NAME, version=VER)\n")
	monkeypatch.setattr(R, "build_metadata", lambda w, **k: pytest.fail("not built in code: never run"))
	assert not R.own_metadata(nobuild, tmp_path.parent / "own-c", log=lambda m: None)


def test_one_build_for_the_same_files_and_a_new_one_when_they_change(tmp_path, monkeypatch):
	(tmp_path / "setup.py").write_text(BUILT)
	runs = []

	def run(cmd, **kw):
		runs.append(cmd)
		return _fake_pip(["tqdm"])(cmd, **kw)

	monkeypatch.setattr(R.subprocess, "run", run)
	assert R.requires_from_build(tmp_path, log=lambda m: None) == ["tqdm"]
	assert R.requires_from_build(tmp_path, log=lambda m: None) == ["tqdm"] and len(runs) == 1
	(tmp_path / "setup.py").write_text(BUILT + "# changed\n")
	R.requires_from_build(tmp_path, log=lambda m: None)
	assert len(runs) == 2


def test_the_own_metadata_goes_in_the_probe_s_prefix_never_the_shared_one(tmp_path, monkeypatch):
	work = tmp_path / "repo"
	work.mkdir()
	(work / "train.py").write_text("print(1)\n")
	(work / "pyproject.toml").write_text(
		'[project]\nname = "lm_eval"\nversion = "0.4.8"\ndependencies = ["zzz-not-here"]\n'
	)
	shared = tmp_path / "shared"
	shared.mkdir()
	monkeypatch.setenv("PREFLIGHT_DEPS_DIR", str(shared))
	monkeypatch.setattr(R, "_pip", lambda keep, target, *a, **k: None)
	before = sorted(work.rglob("*"))
	for _ in range(2):  # the second probe finds the shared install done and still gets its own metadata
		err, dirs = R.install(
			work, tmp_path / "p-deps", log=lambda m: None, repo=True, entrypoint="train.py", build=True
		)
		assert err is None
		assert any("p-deps-own" in str(d) for d in dirs)
	assert not list(shared.rglob("*.dist-info")) and sorted(work.rglob("*")) == before


@pytest.mark.parametrize(
	"setup",
	[
		"setup(name='x', version='1', install_requires=[])",
		"setup(name='x', install_requires=('tqdm',))",
		"setup(name='x', install_requires=None)",
		"REQS = []\nsetup(name='x', install_requires=REQS)",
	],
)
def test_a_literal_list_never_builds_review(tmp_path, setup):
	(tmp_path / "setup.py").write_text("from setuptools import setup\n" + setup + "\n")
	assert not R.built_in_code(tmp_path) and not R.needs_build(tmp_path)


def test_a_static_project_table_wins_over_setup_py_s_code_review(tmp_path):
	(tmp_path / "setup.py").write_text(BUILT)
	(tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\nversion = "1"\ndependencies = []\n')
	assert not R.needs_build(tmp_path)
	(tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\ndynamic = ["dependencies", "version"]\n')
	assert R.needs_build(tmp_path)


def test_the_build_finds_failing_compilers_on_path_and_takes_wheels_only(tmp_path, monkeypatch):
	(tmp_path / "setup.py").write_text(BUILT)
	seen, real = {}, subprocess.run

	def run(cmd, **kw):
		gcc = Path(kw["env"]["PATH"].split(":")[0]) / "gcc"
		seen.update(gcc=real([str(gcc)], capture_output=True).returncode, cmd=cmd)
		return _fake_pip(["tqdm"])(cmd, **kw)

	monkeypatch.setattr(R.subprocess, "run", run)
	assert R.requires_from_build(tmp_path, log=lambda m: None) == ["tqdm"]
	assert seen["gcc"] == 1 and seen["cmd"][seen["cmd"].index("--only-binary") + 1] == ":all:"


def test_a_version_kept_in_the_package_is_read_without_running_it(tmp_path):
	(tmp_path / "pkg").mkdir()
	(tmp_path / "pkg" / "__init__.py").write_text('__version__ = "2.3.4"\nimport os\n')
	(tmp_path / "setup.cfg").write_text("[metadata]\nname = pkg\nversion = attr: pkg.__version__\n")
	assert R.static_name_version(tmp_path) == ("pkg", "2.3.4")
	(tmp_path / "setup.cfg").unlink()
	(tmp_path / "pyproject.toml").write_text(
		'[project]\nname = "pkg"\ndynamic = ["version"]\ndependencies = []\n'
		'[tool.setuptools.dynamic]\nversion = {attr = "pkg.__version__"}\n'
	)
	assert R.static_name_version(tmp_path) == ("pkg", "2.3.4")
	(tmp_path / "pyproject.toml").write_text(
		'[project]\nname = "pkg"\ndynamic = ["version"]\n[tool.hatch.version]\npath = "pkg/__init__.py"\n'
	)
	assert R.static_name_version(tmp_path) == ("pkg", "2.3.4")
	(tmp_path / "pyproject.toml").write_text(
		'[project]\nname = "pkg"\ndynamic = ["version"]\n[tool.hatch.version]\npath = "../outside.py"\n'
	)
	assert R.static_name_version(tmp_path) is None


def test_a_flash_attn_style_build_sees_no_cuda_home_and_no_nvcc_shim(tmp_path, monkeypatch):
	(tmp_path / "setup.py").write_text(BUILT)
	monkeypatch.setenv("CUDA_HOME", "/usr/local/cuda")
	monkeypatch.setenv("CUDA_PATH", "/usr/local/cuda")
	seen = {}

	def run(cmd, **kw):
		seen["env"] = kw["env"]
		seen["shims"] = sorted(p.name for p in Path(kw["env"]["PATH"].split(":")[0]).iterdir())
		return _fake_pip(["tqdm"])(cmd, **kw)

	monkeypatch.setattr(R.subprocess, "run", run)
	assert R.requires_from_build(tmp_path, log=lambda m: None) == ["tqdm"]
	assert "CUDA_HOME" not in seen["env"] and "CUDA_PATH" not in seen["env"]
	assert "nvcc" not in seen["shims"] and "gcc" in seen["shims"]  # the host compiler still fails
