"""probes.repo_install (off): a repo with no requirements.txt gets what it declares and the known
packages its code imports, and a src/ layout's packages on the path (the unseen runs, Nov 2)."""

import importlib.util
import subprocess

import pytest

from preflight.recorder import requirements as R


def _tree(root, files: dict[str, str]):
	for rel, text in files.items():
		(root / rel).parent.mkdir(parents=True, exist_ok=True)
		(root / rel).write_text(text)
	return root


@pytest.fixture
def missing(monkeypatch):
	"""find_spec answers None for these names, whatever this machine has installed."""
	gone: set[str] = set()
	real = importlib.util.find_spec

	def fake(name, *a, **kw):
		return None if name in gone else real(name, *a, **kw)

	monkeypatch.setattr(importlib.util, "find_spec", fake)
	return gone


def test_pyproject_dependencies_are_declared(tmp_path):
	_tree(
		tmp_path,
		{
			"pyproject.toml": '[project]\nname = "x"\ndependencies = ["lightning>=2.4", "jsonargparse[signatures]"]\n'
		},
	)
	assert R.declared(tmp_path) == ["lightning>=2.4", "jsonargparse[signatures]"]


def test_setup_py_install_requires_as_a_list_or_a_name_bound_to_one(tmp_path):
	_tree(
		tmp_path,
		{
			"setup.py": "from setuptools import setup\nsetup(name='x', install_requires=['ogb>=1.3', 'tqdm'])\n"
		},
	)
	assert R.declared(tmp_path) == ["ogb>=1.3", "tqdm"]
	_tree(
		tmp_path, {"setup.py": "from setuptools import setup\nREQ = ['beir']\nsetup(install_requires=REQ)\n"}
	)
	assert R.declared(tmp_path) == ["beir"]
	_tree(tmp_path, {"setup.py": "from setuptools import setup\nsetup(install_requires=deps_list('a'))\n"})
	assert R.declared(tmp_path) == []  # not a literal: nothing guessed


def test_imports_the_image_lacks_are_found_through_the_repos_own_modules(tmp_path, missing):
	missing.update({"torchvision", "torch_geometric", "weird_unknown_mod", "wandb"})
	_tree(tmp_path, {
		"examples/train.py": "import os, json\nfrom mylib.core import thing\nimport torchvision.transforms as T\n",
		"mylib/__init__.py": "",
		"mylib/core.py": "import torch_geometric\nimport weird_unknown_mod\ntry:\n\timport wandb\nexcept ImportError:\n\twandb = None\n",
	})  # fmt: skip
	# the entrypoint runs from the checkout's root, where mylib is importable
	assert R.missing_imports(tmp_path, "examples/train.py") == {
		"torchvision",
		"torch_geometric",
		"weird_unknown_mod",
	}


def test_only_known_modules_become_requirements_and_the_rest_are_noted(tmp_path, missing):
	missing.update({"torchvision", "torch_sparse", "weird_unknown_mod"})
	_tree(tmp_path, {"run.py": "import torchvision\nimport torch_sparse\nimport weird_unknown_mod\n"})
	notes: list[str] = []
	text = R.repo_requirements(tmp_path, "run.py", log=notes.append)
	assert text == "torchvision\n"
	assert any("torch_sparse has no wheels" in n for n in notes)
	assert any("weird_unknown_mod" in n and "add it to requirements.txt" in n for n in notes)


def test_a_declared_package_isnt_added_twice_from_an_import(tmp_path, missing):
	missing.add("lightning")
	_tree(tmp_path, {"pyproject.toml": '[project]\nname = "x"\ndependencies = ["lightning==2.5.0"]\n',
		"run.py": "import lightning\n"})  # fmt: skip
	assert R.repo_requirements(tmp_path, "run.py", log=lambda m: None) == "lightning==2.5.0\n"


def test_a_src_layout_is_on_the_path_and_its_imports_are_scanned(tmp_path, missing):
	missing.add("torchvision")
	_tree(tmp_path, {"src/pkg/__init__.py": "", "src/pkg/grpo.py": "from pkg import util\n",
		"src/pkg/util.py": "import torchvision\n"})  # fmt: skip
	assert R.missing_imports(tmp_path, "src/pkg/grpo.py") == {"torchvision"}
	err, dirs = R.install(tmp_path, tmp_path / "deps", log=lambda m: None, repo=True, entrypoint="nothing.py")
	assert err is None and dirs == [
		tmp_path / "src"
	]  # nothing to install, but the src/ layout is on the path


def test_repo_mode_installs_the_derived_lines_with_the_image_pinned(tmp_path, missing, monkeypatch):
	missing.add("torchvision")
	_tree(tmp_path, {"run.py": "import torchvision\n"})
	seen = {}

	def fake_run(cmd, **kw):
		seen["cmd"] = cmd
		return subprocess.CompletedProcess(cmd, 0, "", "")

	monkeypatch.setattr(R.subprocess, "run", fake_run)
	monkeypatch.delenv("PREFLIGHT_DEPS_DIR", raising=False)
	err, _ = R.install(tmp_path, tmp_path / "deps", log=lambda m: None, repo=True, entrypoint="run.py")
	assert (
		err is None
		and seen["cmd"][-1] == "torchvision"
		and "-c" in seen["cmd"]
		and "--only-binary" in seen["cmd"]
	)


def test_without_repo_mode_nothing_changes(tmp_path, missing, monkeypatch):
	missing.add("torchvision")
	_tree(tmp_path, {"run.py": "import torchvision\n", "src/pkg/__init__.py": ""})
	monkeypatch.setattr(R.subprocess, "run", lambda *a, **k: pytest.fail("pip ran"))
	assert R.install(tmp_path, tmp_path / "deps", log=lambda m: None) == (None, [])


def test_the_flag_is_off_by_default():
	from preflight.runner.pipeline import _repo_install_on

	assert _repo_install_on() is False


def test_a_repo_that_cant_be_parsed_installs_nothing_and_never_raises(tmp_path, missing, monkeypatch):
	_tree(tmp_path, {"pyproject.toml": "a = " + "[" * 5000, "setup.py": "x = " + "-" * 200000 + "1\n",
		"run.py": "import deep\n", "deep.py": "y = " + "not " * 100000 + "1\n"})  # fmt: skip
	monkeypatch.setattr(R.subprocess, "run", lambda *a, **k: pytest.fail("pip ran"))
	assert R.install(tmp_path, tmp_path / "deps", log=lambda m: None, repo=True, entrypoint="run.py") == (
		None,
		[],
	)


def test_a_url_dependency_doesnt_hide_a_known_import(tmp_path, missing):
	missing.add("torchvision")
	_tree(tmp_path, {"pyproject.toml": '[project]\nname = "x"\ndependencies = ["torchvision @ https://example.com/tv.whl"]\n',
		"run.py": "import torchvision\n"})  # fmt: skip
	assert R.repo_requirements(tmp_path, "run.py", log=lambda m: None).splitlines()[-1] == "torchvision"


def test_a_line_plan_wont_install_doesnt_hide_a_known_import(tmp_path, missing):
	missing.add("torchvision")
	_tree(tmp_path, {"pyproject.toml": '[project]\nname = "x"\ndependencies = ["torchvision (>=0.15)", "x" ]\n',
		"run.py": "import torchvision\nimport " + "a" * 300 + "\n"})  # fmt: skip
	assert R.repo_requirements(tmp_path, "run.py", log=lambda m: None).splitlines()[-1] == "torchvision"


def test_a_samples_measured_outputs_are_sent_only_with_the_flag_off():
	"""B, Nov 2: zoo/grpo_trl holds its own run's record; with runner.hide_sample_outputs on it stays out."""
	import json
	import os

	from preflight.runner import flags, sources

	before = os.environ.get("PREFLIGHT_FLAGS")
	try:
		os.environ.pop("PREFLIGHT_FLAGS", None)
		flags.reload()
		assert (
			"record_nov1_a10.json" in sources.fetch({"kind": "sample", "uri": "sample://zoo/grpo_trl"}).files
		)
		os.environ["PREFLIGHT_FLAGS"] = json.dumps({"runner.hide_sample_outputs": True})
		flags.reload()
		got = sources.fetch({"kind": "sample", "uri": "sample://zoo/grpo_trl"}).files
	finally:  # the environment as it was, and the flags read from it again
		os.environ.pop("PREFLIGHT_FLAGS", None)
		if before is not None:
			os.environ["PREFLIGHT_FLAGS"] = before
		flags.reload()
	assert "record_nov1_a10.json" not in got and any(k.endswith(".py") for k in got)


def test_only_the_modules_the_script_imports_are_scanned_not_their_whole_package(tmp_path, missing):
	"""Unseen LitGPT, Nov 2: scanning all of litgpt/ found lm_eval (its eval module), which the
	pretrain script never imports, and that one conflicting package sank the whole install."""
	missing.update({"jsonargparse", "lm_eval"})
	_tree(tmp_path, {"litgpt/__init__.py": "", "litgpt/pretrain.py": "from litgpt.utils import x\n",
		"litgpt/utils.py": "import jsonargparse\n", "litgpt/eval/__init__.py": "import lm_eval\n"})  # fmt: skip
	assert R.missing_imports(tmp_path, "litgpt/pretrain.py") == {"jsonargparse"}


def test_the_entrypoints_own_folder_and_relative_imports_are_the_repos_own(tmp_path, missing):
	missing.update({"torch_geometric", "logger", "loader"})
	_tree(tmp_path, {"examples/arxiv/gnn.py": "import torch_geometric\nfrom logger import Logger\n",
		"examples/arxiv/logger.py": "", "pkg/__init__.py": "from .sub import f\n", "pkg/sub.py": "import loader\n",
		"run.py": "import pkg\n"})  # fmt: skip
	assert R.missing_imports(tmp_path, "examples/arxiv/gnn.py") == {"torch_geometric"}
	assert R.missing_imports(tmp_path, "run.py") == {"loader"}  # found through the relative import


def test_a_declared_package_isnt_noted_as_unknown(tmp_path, missing):
	missing.add("six")
	_tree(tmp_path, {"setup.py": "from setuptools import setup\nsetup(install_requires=['six>=1.12'])\n",
		"run.py": "import six\n"})  # fmt: skip
	notes: list[str] = []
	assert R.repo_requirement_sets(tmp_path, "run.py", log=notes.append) == (["six>=1.12"], [])
	assert not any("add it to requirements.txt" in n for n in notes)


def test_an_imported_package_that_wont_install_doesnt_sink_the_declared_ones(tmp_path, missing, monkeypatch):
	missing.update({"torchvision", "lm_eval"})
	_tree(tmp_path, {"pyproject.toml": '[project]\nname = "x"\ndependencies = ["tinypkg-xyz==1.0"]\n',
		"run.py": "import torchvision\nimport lm_eval\n"})  # fmt: skip
	calls = []

	def fake_run(cmd, **kw):
		pkgs = cmd[cmd.index("-c") + 2 :]
		calls.append(pkgs)
		bad = "lm-eval" in pkgs
		return subprocess.CompletedProcess(cmd, 1 if bad else 0, "", "ResolutionImpossible" if bad else "")

	monkeypatch.setattr(R.subprocess, "run", fake_run)
	monkeypatch.delenv("PREFLIGHT_DEPS_DIR", raising=False)
	notes: list[str] = []
	err, _ = R.install(tmp_path, tmp_path / "deps", log=notes.append, repo=True, entrypoint="run.py")
	assert err is None
	assert calls == [
		["tinypkg-xyz==1.0", "lm-eval", "torchvision"],
		["tinypkg-xyz==1.0"],
		["lm-eval"],
		["torchvision"],
	]
	assert any("lm-eval didn't install" in n for n in notes) and any(
		"torchvision installed" in n for n in notes
	)


def test_a_known_package_can_carry_a_pin():
	assert R.KNOWN_MODULES["torchao"] == "torchao<0.8"  # 0.8 on needs torch 2.6; the image has 2.5.1


def test_small_data_files_are_read_from_a_repo_only_with_the_flag():
	from preflight.runner.sources import _wanted

	assert not _wanted("ogb/nodeproppred/master.csv") and _wanted("ogb/nodeproppred/master.csv", True)
	assert _wanted("train.py") and not _wanted(".hidden/x.csv", True)


def test_every_known_module_maps_to_a_requirement_line():
	assert all(R._LINE.fullmatch(v) for v in R.KNOWN_MODULES.values())
	assert {"torchdata", "tiktoken", "sentencepiece", "torchao"} <= set(R.KNOWN_MODULES)


def test_a_relative_import_never_reads_above_the_checkout(tmp_path, missing):
	missing.add("outside_mod")
	(tmp_path / "outside.py").write_text("import outside_mod\n")
	work = _tree(tmp_path / "work", {"run.py": "from .. import outside\n"})
	assert R.missing_imports(work, "run.py") == set()


def test_setup_cfg_install_requires_is_declared(tmp_path):
	_tree(
		tmp_path,
		{
			"setup.cfg": "[metadata]\nname = x\n[options]\ninstall_requires =\n    beir>=2\n    # a comment\n    tqdm\n"
		},
	)
	assert R.declared(tmp_path) == ["beir>=2", "tqdm"]


def test_an_import_whose_handler_raises_again_is_required(tmp_path, missing):
	missing.update({"openai", "wandb"})
	_tree(tmp_path, {"run.py": "def model():\n\ttry:\n\t\timport openai\n\texcept ModuleNotFoundError:\n"
		"\t\traise ModuleNotFoundError(\"Please install 'openai' extra\")\ntry:\n\timport wandb\nexcept ImportError:\n\twandb = None\n"})  # fmt: skip
	assert R.missing_imports(tmp_path, "run.py") == {"openai"}
