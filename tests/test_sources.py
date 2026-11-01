"""Fetching a job's source: repos read without a checkout, and visitor JSON type-checked."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from preflight.probes.local import MAX_FILE_BYTES, InvalidSubmission
from preflight.runner import sources
from preflight.runner.sources import _clone, _git_files, fetch

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
ENV = {"GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_NOSYSTEM": "1", "PATH": "/usr/bin:/bin"}


def _run(*args, cwd=None, env=None):
	subprocess.run(args, cwd=cwd, check=True, capture_output=True, env={**ENV, **(env or {})})


def _repo(tmp_path: Path) -> Path:
	src = tmp_path / "src"
	src.mkdir()
	(src / "train.py").write_text("batch_size = 8\n")
	(src / "conf").mkdir()
	(src / "conf" / "a.yaml").write_text("lr: 0.1\n")
	(src / ".hidden.py").write_text("x = 1\n")
	(src / "weights.bin").write_bytes(b"\0" * 10)
	(src / "big.txt").write_text("x" * (MAX_FILE_BYTES + 1))
	(src / "latin.py").write_bytes("caf\xe9 = 1\n".encode("latin-1"))
	(src / "link.py").symlink_to("/etc/passwd")
	home = {"HOME": str(tmp_path)}
	_run("git", "init", "-q", "-b", "main", cwd=src, env=home)
	_run("git", "config", "uploadpack.allowFilter", "true", cwd=src, env=home)
	_run("git", "add", "-A", cwd=src, env=home)
	_run("git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init", cwd=src, env=home)
	return src


def test_repo_files_come_from_the_object_store_without_big_blobs_or_symlinks(tmp_path):
	src = _repo(tmp_path)
	dest = tmp_path / "clone"
	home = {"HOME": str(tmp_path)}
	_run("git", "-c", "protocol.file.allow=always", "clone", "-q", "--depth", "1", "--no-checkout",
		f"--filter=blob:limit={MAX_FILE_BYTES}", f"file://{src}", str(dest), env=home)  # fmt: skip
	assert not (dest / "train.py").exists()  # nothing was checked out
	base = ["git", "-c", "core.hooksPath=/dev/null", "-c", "protocol.file.allow=never"]
	files = _git_files(dest, "HEAD", base=base, env={**ENV, **home})
	# big.txt was never downloaded (and protocol.file=never proves no lazy fetch ran);
	# the symlink, the dotfile, the binary and the non-UTF-8 file are skipped
	assert files == {"train.py": "batch_size = 8\n", "conf/a.yaml": "lr: 0.1\n"}


def test_clone_checks_the_url_and_wants_a_full_commit():
	for bad in (
		"https://github.com/a/b/../c",
		"https://github.com/a/b\nhttps://evil",
		"http://github.com/a/b",
		7,
	):
		with pytest.raises(InvalidSubmission, match=r"github\.com"):
			_clone(bad, None)
	for bad in ("abc1234", "A" * 40, "0" * 39, ["x"]):
		with pytest.raises(InvalidSubmission, match="40-character"):
			_clone("https://github.com/a/b", bad)


def test_clone_leaves_no_temp_directory_behind(monkeypatch, tmp_path):
	made = []
	real = sources.tempfile.mkdtemp

	def mkdtemp(**kw):
		made.append(Path(real(dir=tmp_path, **kw)))
		return str(made[-1])

	def boom(*a, **k):
		raise subprocess.CalledProcessError(128, "git")

	monkeypatch.setattr(sources.tempfile, "mkdtemp", mkdtemp)
	monkeypatch.setattr(sources, "_git", boom)
	with pytest.raises(InvalidSubmission, match="could not be cloned"):
		_clone("https://github.com/a/b", None)
	assert made and not made[0].exists()


@pytest.mark.parametrize(
	("source", "match"),
	[
		({"kind": "notebook", "files": {"nb.ipynb": 5}}, "not text"),
		({"kind": "notebook", "files": {"nb.ipynb": "[1, 2]"}}, "list of cells"),
		({"kind": "notebook", "files": {"nb.ipynb": json.dumps({"cells": [{"cell_type": "code", "source": [1]}]})}},
			"not text"),
		({"kind": "notebook", "files": {"nb.py": "{}"}}, ".ipynb"),
		({"kind": "upload", "files": {"t.py": "x"}, "entrypoint": ["t.py"]}, "file path"),
		({"kind": "upload", "files": {"t.py": "x"}, "args": "--lr 1"}, "list of strings"),
		({"kind": "upload", "files": {"t.py": "x"}, "args": [1]}, "list of strings"),
		({"kind": "upload", "files": {"t.py": "x"}, "args": ["a"] * 65}, "up to 64"),
		({"kind": "upload", "files": {"t.py": "x"}, "commit": 5}, "40-character"),
		({"kind": "upload", "files": {"t.py": "x"}, "uri": {"a": 1}}, "uri"),
	],
)  # fmt: skip
def test_malformed_sources_are_visitor_errors_not_crashes(source, match):
	with pytest.raises(InvalidSubmission, match=match):
		fetch(source)


def test_a_notebook_becomes_its_code_cells():
	nb = {"cells": [{"cell_type": "markdown", "source": "# hi"},
		{"cell_type": "code", "source": ["!pip install x\n", "bs = 4\n"]}]}  # fmt: skip
	got = fetch({"kind": "notebook", "files": {"dir/run.ipynb": json.dumps(nb)}})
	assert got.files == {"run.py": "# !pip install x\nbs = 4\n\n", "requirements.txt": "x\n"}
	assert got.entrypoint == "run.py"


def test_notebook_pip_lines_become_plain_requirements():
	from preflight.runner.sources import notebook_requirements

	script = "\n".join([
		"# !pip install -q -U peft==0.13.2 'trl>=0.12' -r reqs.txt",
		"# %pip install --index-url https://evil.example datasets",
		"# !python -m pip install bitsandbytes; rm -rf /",
		"# !pip install peft==0.13.2",
		"x = 1",
	])  # fmt: skip
	assert notebook_requirements(script) == ["peft==0.13.2", "trl>=0.12", "datasets", "bitsandbytes"]


def test_requirements_keep_the_image_stack_and_skip_anything_but_plain_pypi_lines():
	from preflight.recorder.requirements import plan

	text = "\n".join([
		"torch==2.6.0", "Transformers>=4.50", "sentencepiece==0.2.0", "einops", "rich[jupyter] ; python_version>'3'",
		"-e .", "git+https://github.com/a/b", "https://x.example/p.whl", "./local", "--index-url https://x", "# c", "",
	])  # fmt: skip
	keep, notes = plan(text, {"torch": "2.5.1", "transformers": "4.46.3"})
	assert keep == ["sentencepiece==0.2.0", "einops", "rich[jupyter] ; python_version>'3'"]
	assert any("kept the image's torch 2.5.1" in n for n in notes)
	assert sum("only plain PyPI" in n for n in notes) == 5


def test_an_ops_only_zoo_script_is_not_a_sample():
	import pytest

	from preflight.probes.local import InvalidSubmission
	from preflight.runner.sources import _sample

	with pytest.raises(InvalidSubmission):
		_sample("sample://zoo/torchtitan_debug")
	assert _sample("sample://zoo/grpo_trl").name == "grpo_trl"
