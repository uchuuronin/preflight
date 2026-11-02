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
import threading
import time
from pathlib import Path

BASE_ENV = ("PATH", "LANG", "LC_ALL", "HF_HOME", "PYTHONPATH", "TZ")
ERROR_LINE_CHARS = 160  # the exception's own line, not the traceback; the rest of the output stays dropped
_EXC = re.compile(r"^(?:[A-Za-z_][\w.]*\.)?[A-Za-z_]\w*(?:Error|Exception|Exit|Interrupt)\b(?::.*)?$")
TAIL_BYTES = 8192  # of stderr, kept in memory as it streams: never a file that could fill the disk


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
	extra = [x for x in job.get("extra_packages") or [] if isinstance(x, str)]  # 0.6.3, the list's lines
	if (work / "requirements.txt").exists() or repo or extra:
		from preflight.recorder.requirements import install

		err, dirs = install(work, work.parent / f"{work.name}-deps", log=lambda m: None, repo=repo,
			entrypoint=job["entrypoint"], extra=extra)  # fmt: skip
		if err:
			print(json.dumps({"ended": "install_failed", "error": str(err)[:500]}), flush=True)
			return 0
		env["PYTHONPATH"] = os.pathsep.join([*map(str, dirs), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
	env.update({k: str(v) for k, v in job.get("env", {}).items()})
	t0 = time.monotonic()
	keep = bool(job.get("keep_error"))
	proc = subprocess.Popen([sys.executable, job["entrypoint"], *job.get("args", [])], cwd=work, env=env,
		stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE if keep else subprocess.DEVNULL,
		start_new_session=True)  # fmt: skip
	tail = bytearray()
	reader = None
	if keep:  # drained as it comes, so a chatty script never blocks; only the last TAIL_BYTES are kept

		def drain():
			for chunk in iter(lambda: proc.stderr.read(4096), b""):
				tail.extend(chunk)
				del tail[:-TAIL_BYTES]

		reader = threading.Thread(target=drain, daemon=True)
		reader.start()
	try:
		proc.wait(timeout=float(job.get("timeout_s", 1800)))
		ended = "exited"
	except subprocess.TimeoutExpired:
		ended = "timeout"
	with contextlib.suppress(ProcessLookupError):
		os.killpg(proc.pid, signal.SIGKILL)  # and whatever it left running
	proc.wait()
	report = {"ended": ended, "code": proc.returncode, "wall_s": round(time.monotonic() - t0, 1)}
	if reader is not None:
		reader.join(timeout=5)
		# an exception's own exit (a positive code); a killed script's tail may hold an old, handled one
		if ended == "exited" and (proc.returncode or 0) > 0:
			line = error_line(bytes(tail).decode("utf-8", "replace"))
			if line:
				report["error"] = line
	print(json.dumps(report), flush=True)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
