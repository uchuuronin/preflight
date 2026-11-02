"""Inside the agent sandbox: read the script and its run on stdin, install its requirements, run it.

	python -m preflight.probes.agent_remote < {"files": {...}, "entrypoint": "run.py", "args": [],
		"env": {"META_API_BASE": "https://<tunnel>/t/<token>/v1", ...}, "timeout_s": 1800}

The sandbox holds no secrets: its model calls go to the recorder's proxy, which runs outside it
with the key and is reached through a tunnel (runner/agent_jobs.py); its egress allowlist is the
probes' plus that tunnel's host. The script's own output is dropped (it may echo what it was
sent); one JSON line at the end says how it ended. With "keep_error" (probes.agent_error_line,
off), a script that exits with an error adds its last exception line, capped, and nothing else.
"""

import contextlib
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BASE_ENV = ("PATH", "LANG", "LC_ALL", "HF_HOME", "PYTHONPATH", "TZ")
ERROR_LINE_CHARS = 160  # the exception's own line, not the traceback; the rest of the output stays dropped
_EXC = re.compile(r"^(?:[A-Za-z_][\w.]*\.)?[A-Za-z_]\w*(?:Error|Exception|Exit|Interrupt|Warning)\b(?::.*)?$")


def error_line(stderr_tail: str) -> str | None:
	"""The last line of a traceback that names the exception (e.g. "ModuleNotFoundError: No module
	named 'smolagents'"), capped; None when there is none."""
	for line in reversed(stderr_tail.splitlines()):
		line = line.strip()
		if _EXC.match(line):
			return line[:ERROR_LINE_CHARS]
	return None


def main() -> int:
	job = json.loads(sys.stdin.read())
	work = Path(tempfile.mkdtemp(prefix="agent-", dir="/tmp"))
	for rel, text in job["files"].items():
		p = (work / rel).resolve()
		if work.resolve() not in p.parents:
			continue
		p.parent.mkdir(parents=True, exist_ok=True)
		p.write_text(text, encoding="utf-8")
	env = {k: os.environ[k] for k in BASE_ENV if k in os.environ}
	env.update({"HOME": str(work), "PYTHONUNBUFFERED": "1"})
	repo = bool(job.get("install_repo"))  # probes.repo_install: a repo with no requirements.txt too
	if (work / "requirements.txt").exists() or repo:
		from preflight.recorder.requirements import install

		err, dirs = install(work, work.parent / f"{work.name}-deps", log=lambda m: None, repo=repo,
			entrypoint=job["entrypoint"])  # fmt: skip
		if err:
			print(json.dumps({"ended": "install_failed", "error": str(err)[:500]}), flush=True)
			return 0
		env["PYTHONPATH"] = os.pathsep.join([*map(str, dirs), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
	env.update({k: str(v) for k, v in job.get("env", {}).items()})
	t0 = time.monotonic()
	keep = bool(job.get("keep_error"))
	# a file, never a pipe: a chatty script can't block on it; closed once its tail is read
	err_file = tempfile.TemporaryFile() if keep else None  # noqa: SIM115
	proc = subprocess.Popen([sys.executable, job["entrypoint"], *job.get("args", [])], cwd=work, env=env,
		stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=err_file or subprocess.DEVNULL,
		start_new_session=True)  # fmt: skip
	try:
		proc.wait(timeout=float(job.get("timeout_s", 1800)))
		ended = "exited"
	except subprocess.TimeoutExpired:
		ended = "timeout"
	with contextlib.suppress(ProcessLookupError):
		os.killpg(proc.pid, signal.SIGKILL)  # and whatever it left running
	proc.wait()
	report = {"ended": ended, "code": proc.returncode, "wall_s": round(time.monotonic() - t0, 1)}
	if err_file is not None and proc.returncode not in (0, None):
		with contextlib.suppress(OSError):
			size = err_file.seek(0, os.SEEK_END)
			err_file.seek(max(0, size - 8192))  # the tail holds the exception line
			line = error_line(err_file.read().decode("utf-8", "replace"))
			if line:
				report["error"] = line
	if err_file is not None:
		err_file.close()
	print(json.dumps(report), flush=True)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
