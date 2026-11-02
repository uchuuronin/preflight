"""probes.agent_error_line (off): an agent script that exits with an error leaves its exception line in
the sample, and nothing else of its output (unseen smolagents run, Nov 2: 0 tasks, no reason)."""

import json
import os
import subprocess
import sys
from pathlib import Path

from preflight.probes.agent_remote import ERROR_LINE_CHARS, error_line

ROOT = Path(__file__).resolve().parents[1]


def test_the_exception_line_is_the_one_kept():
	tail = "Traceback (most recent call last):\n  File \"run.py\", line 1\n    import smolagents\nModuleNotFoundError: No module named 'smolagents'\n"
	assert error_line(tail) == "ModuleNotFoundError: No module named 'smolagents'"
	assert error_line("all good\nprinted the prompt: secret text\n") is None
	assert len(error_line("ValueError: " + "x" * 1000)) == ERROR_LINE_CHARS


def _remote(job: dict) -> dict:
	env = {**os.environ, "PYTHONPATH": str(ROOT / "python")}
	r = subprocess.run([sys.executable, "-m", "preflight.probes.agent_remote"], input=json.dumps(job), text=True,
		capture_output=True, env=env, timeout=60)  # fmt: skip
	return json.loads([x for x in r.stdout.splitlines() if x.startswith("{")][-1])


def test_the_sandbox_reports_the_error_only_when_asked():
	job = {"files": {"run.py": "print('the prompt was: hello')\nimport nosuchmod_xyz_123\n"}, "entrypoint": "run.py",
		"timeout_s": 30}  # fmt: skip
	off = _remote(job)
	assert off["code"] != 0 and "error" not in off
	on = _remote({**job, "keep_error": True})
	assert on["error"] == "ModuleNotFoundError: No module named 'nosuchmod_xyz_123'"
	assert "hello" not in json.dumps(on)  # the script's other output stays dropped


def test_the_sample_says_why_the_script_stopped():
	from preflight.recorder.agent_sampler import run_sample

	class Proxy:
		def open_run(self, run_id, **kw):
			return "http://proxy/v1"

		def run_stats(self, run_id):
			return {}

		def run_tasks(self, run_id):
			return []

	class Proc:
		def __init__(self, report):
			self.report = report

		def poll(self):
			return 1

		def wait(self, timeout):
			pass

		def kill(self):
			pass

	failed = {"ended": "exited", "code": 1, "error": "ModuleNotFoundError: No module named 'smolagents'"}
	s = run_sample(None, cwd=None, proxy=Proxy(), population=None, env={}, launch=lambda env: Proc(failed))
	assert "script failed: ModuleNotFoundError: No module named 'smolagents'" in s.selection
	install = {"ended": "install_failed", "error": "pip said no"}
	assert "script failed" not in run_sample(None, cwd=None, proxy=Proxy(), population=None, env={},
		launch=lambda env: Proc(install)).selection  # fmt: skip
