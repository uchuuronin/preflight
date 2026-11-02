"""Check in by command (runner.command_checkin, off; Nov 2, after freeze): lm_eval and `vllm bench` are console
tools, not scripts in a repo, so Preflight installs the tool and runs the command instead of reading the repo."""

import pytest

from preflight.probes.local import InvalidSubmission
from preflight.runner import record_jobs, sources
from preflight.runner.sources import COMMAND_ENTRY, fetch

LM_EVAL = "lm_eval --model hf --model_args pretrained=HuggingFaceTB/SmolLM2-1.7B-Instruct,dtype=bfloat16 --tasks arc_easy,piqa"


@pytest.fixture
def on(monkeypatch):
	monkeypatch.setattr("preflight.runner.flags.flag", lambda name: name == "runner.command_checkin")


def test_a_command_becomes_a_wrapper_its_package_and_its_lens(on):
	got = fetch({"kind": "command", "command": LM_EVAL})
	assert got.entrypoint == COMMAND_ENTRY and got.lens == "eval"
	assert got.files["requirements.txt"] == "lm-eval\n" and "import lm_eval" in got.files[COMMAND_ENTRY]
	assert (
		got.args[:3] == ["lm_eval", "--model", "hf"]
		and "pretrained=HuggingFaceTB/SmolLM2-1.7B-Instruct,dtype=bfloat16" in got.args
	)
	assert got.source.kind == "command" and got.source.uri == "command://lm_eval"
	# the whole command as the source's args validates as a Source (C's API checks it so)
	as_args = fetch({"kind": "command", "args": got.args})
	assert as_args.source.content_hash == got.source.content_hash and as_args.args == got.args
	with pytest.raises(
		InvalidSubmission, match="these commands by name: lm_eval"
	):  # the image's vLLM has no bench
		fetch({"kind": "command", "command": "vllm bench throughput --num-prompts 500"})


def test_the_wrapper_runs_the_named_console_entry_point(on, tmp_path):
	import re
	import subprocess
	import sys

	got = fetch({"kind": "command", "command": LM_EVAL})
	script = re.sub(r"^import lm_eval\b.*$", "pass", got.files[COMMAND_ENTRY], flags=re.M)  # no lm-eval here
	(tmp_path / "w.py").write_text(script.replace("'lm_eval'", "'pytest'"))
	r = subprocess.run(
		[sys.executable, "w.py", "pytest", "--version"], cwd=tmp_path, capture_output=True, text=True
	)
	assert r.returncode == 0 and "pytest" in (r.stdout + r.stderr)


def test_a_long_command_is_refused_before_it_is_split(on):
	import time

	t0 = time.monotonic()
	with pytest.raises(InvalidSubmission, match="over 8,192 characters"):
		fetch({"kind": "command", "command": "lm_eval '" + "x" * 5_000_000})
	assert time.monotonic() - t0 < 1


def test_its_source_only_dependencies_build_for_it_alone(on, tmp_path, monkeypatch):
	import subprocess

	from preflight.recorder import requirements as R
	from preflight.runner.pipeline import _command_sdists

	got = fetch({"kind": "command", "command": LM_EVAL})
	assert _command_sdists(got) == ("sqlitedict", "rouge-score", "word2number")
	assert _command_sdists(fetch({"kind": "upload", "files": {"t.py": "x"}})) == ()
	seen = []
	monkeypatch.setattr(
		R.subprocess, "run", lambda cmd, **kw: seen.append(cmd) or subprocess.CompletedProcess(cmd, 0, "", "")
	)
	monkeypatch.setattr(R, "installed", lambda: {})
	monkeypatch.delenv("PREFLIGHT_DEPS_DIR", raising=False)
	(tmp_path / "requirements.txt").write_text("lm-eval\n")
	assert R.install(tmp_path, tmp_path / "d", log=lambda m: None, sdists=_command_sdists(got))[0] is None
	cmd = seen[-1]
	assert "--only-binary" in cmd and [cmd[i + 1] for i, a in enumerate(cmd) if a == "--no-binary"] == [
		"sqlitedict", "rouge-score", "word2number"]  # fmt: skip


def test_only_named_tools_and_only_with_the_flag(on, monkeypatch):
	for bad in ("rm -rf /", "python train.py", "", [], "lm_eval 'unclosed", [3]):
		with pytest.raises(InvalidSubmission) as exc:
			fetch({"kind": "command", "command": bad})
		assert exc.value.code == "submission.invalid"
	monkeypatch.setattr("preflight.runner.flags.flag", lambda name: False)
	with pytest.raises(InvalidSubmission, match="isn't on here"):
		fetch({"kind": "command", "command": LM_EVAL})


def test_a_command_record_runs_the_same_command_again(on):
	got = fetch({"kind": "command", "command": LM_EVAL})
	again = fetch(record_jobs.refetch(got.source, None))
	assert again.source.content_hash == got.source.content_hash and again.args == got.args


def test_the_pipeline_reads_it_as_its_tools_lens(on):
	from preflight.harness.static import map_static
	from preflight.runner.pipeline import _command_lens

	got = fetch({"kind": "command", "command": LM_EVAL})
	assert (
		map_static(got.files, entrypoint=got.entrypoint).lens != "eval"
	)  # the wrapper alone reads otherwise
	assert _command_lens(map_static(got.files, entrypoint=got.entrypoint), got).lens == "eval"
	assert set(sources.COMMANDS) == {"lm_eval"}
