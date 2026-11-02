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
	vllm = fetch({"kind": "command", "command": ["vllm", "bench", "throughput", "--num-prompts", "500"]})
	assert vllm.lens == "serving" and "requirements.txt" not in vllm.files  # the serving image's own vLLM


def test_the_wrapper_runs_the_named_console_entry_point(on, tmp_path):
	import subprocess
	import sys

	got = fetch({"kind": "command", "command": "vllm --version"})
	import re

	script = re.sub(r"^import vllm\b.*$", "pass", got.files[COMMAND_ENTRY], flags=re.M)  # no vLLM here
	script = script.replace("'vllm'", "'pytest'")
	(tmp_path / "w.py").write_text(script)
	r = subprocess.run(
		[sys.executable, "w.py", "pytest", "--version"], cwd=tmp_path, capture_output=True, text=True
	)
	assert r.returncode == 0 and "pytest" in (r.stdout + r.stderr)


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
	assert set(sources.COMMANDS) == {"lm_eval", "vllm"}
