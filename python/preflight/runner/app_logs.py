"""Read a Modal app's container logs and its state, for a run whose own log showed nothing.

	python -m preflight.runner.app_logs --env main ap-smlQhBII0aKV0BnMwULRzO

Run through `remote-run.yml` (which only takes `modal run`, `modal deploy` and our own entry
points). `modal run` doesn't pass a remote function's output into the workflow's log, so this is
how a stuck or silent run is diagnosed without starting anything: it reads, it never runs code in
the app. `modal app logs` follows a live app, so it is cut off after --seconds. Writes
out/app_logs.txt (the logs) and out/app_logs.json (the app's state and how the read ended).
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

APP_ID = re.compile(r"ap-[A-Za-z0-9]{6,40}")
MAX_LOG_CHARS = 2_000_000


def _state(env: str, app_id: str) -> str | None:
	try:
		r = subprocess.run(["modal", "app", "list", "--json", "--env", env], capture_output=True, text=True,
			timeout=120)  # fmt: skip
		apps = json.loads(r.stdout) if r.returncode == 0 else []
	except (subprocess.TimeoutExpired, json.JSONDecodeError):
		return None
	for a in apps if isinstance(apps, list) else []:
		if str(a.get("App ID") or a.get("app_id") or "") == app_id:
			return str(a.get("State") or a.get("state") or "?")
	return None


def main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser(prog="python -m preflight.runner.app_logs")
	ap.add_argument("--env", required=True, help="Modal environment, e.g. main")
	ap.add_argument("app_id")
	ap.add_argument("--seconds", type=int, default=90, help="stop following a live app after this long")
	ns = ap.parse_args(argv)
	if not APP_ID.fullmatch(ns.app_id):
		print("an app id looks like ap-...", file=sys.stderr)
		return 2
	cmd = ["modal", "app", "logs", "--env", ns.env, ns.app_id]
	try:
		r = subprocess.run(cmd, capture_output=True, text=True, timeout=ns.seconds)
		text, ended = r.stdout + r.stderr, f"exit {r.returncode}"
	except subprocess.TimeoutExpired as exc:
		out = exc.stdout or b""
		err = exc.stderr or b""
		text = (out.decode("utf-8", "replace") if isinstance(out, bytes) else out) + (
			err.decode("utf-8", "replace") if isinstance(err, bytes) else err
		)
		ended = f"cut off after {ns.seconds} s (the app is still live, or Modal kept the stream open)"
	Path("out").mkdir(exist_ok=True)
	Path("out/app_logs.txt").write_text(text[-MAX_LOG_CHARS:])
	summary = {"app_id": ns.app_id, "env": ns.env, "state": _state(ns.env, ns.app_id), "read": ended,
		"chars": len(text)}  # fmt: skip
	Path("out/app_logs.json").write_text(json.dumps(summary, indent=2))
	print(json.dumps(summary, indent=2))
	print(text[-4000:])
	return 0


if __name__ == "__main__":
	sys.exit(main())
