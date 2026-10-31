"""Inside the agent sandbox: read the script and its run on stdin, install its requirements, run it.

	python -m preflight.probes.agent_remote < {"files": {...}, "entrypoint": "run.py", "args": [],
		"env": {"META_API_BASE": "https://<tunnel>/t/<token>/v1", ...}, "timeout_s": 1800}

The sandbox holds no secrets: its model calls go to the recorder's proxy, which runs outside it
with the key and is reached through a tunnel (runner/agent_jobs.py); its egress allowlist is the
probes' plus that tunnel's host. The script's own output is dropped (it may echo what it was
sent); one JSON line at the end says how it ended.
"""

import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

BASE_ENV = ("PATH", "LANG", "LC_ALL", "HF_HOME", "PYTHONPATH", "TZ")


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
	if (work / "requirements.txt").exists():
		from preflight.recorder.requirements import install

		err, dirs = install(work, work.parent / f"{work.name}-deps", log=lambda m: None)
		if err:
			print(json.dumps({"ended": "install_failed", "error": str(err)[:500]}), flush=True)
			return 0
		env["PYTHONPATH"] = os.pathsep.join([*map(str, dirs), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
	env.update({k: str(v) for k, v in job.get("env", {}).items()})
	t0 = time.monotonic()
	proc = subprocess.Popen([sys.executable, job["entrypoint"], *job.get("args", [])], cwd=work, env=env,
		stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)  # fmt: skip
	try:
		proc.wait(timeout=float(job.get("timeout_s", 1800)))
		ended = "exited"
	except subprocess.TimeoutExpired:
		ended = "timeout"
	with contextlib.suppress(ProcessLookupError):
		os.killpg(proc.pid, signal.SIGKILL)  # and whatever it left running
	proc.wait()
	print(json.dumps({"ended": ended, "code": proc.returncode, "wall_s": round(time.monotonic() - t0, 1)}),
		flush=True)  # fmt: skip
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
