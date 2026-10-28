"""Stop the deployed Preflight runner, so it stops billing, and say whether it did.

	python -m preflight.runner.stop_app --env main       # stops preflight-runner in Modal's main
	python -m preflight.runner.stop_app --env main --app other-app

Run through `remote-run.yml` (which only takes `modal run`, `modal deploy` and our own entry
points, so `modal app stop` goes through here). A deployed app keeps its always-on job store
container and its once-a-minute tick until it is stopped; stopping ends both, and any sandbox a
worker left behind ends at its idle timeout (ten minutes). Writes out/stop.json with what the app
list says afterwards.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from .modal_app import APP_NAME


def _apps(env: str) -> list[dict]:
	r = subprocess.run(
		["modal", "app", "list", "--json", "--env", env], capture_output=True, text=True, timeout=120
	)
	if r.returncode != 0:
		return []
	try:
		data = json.loads(r.stdout)
	except json.JSONDecodeError:
		return []
	return data if isinstance(data, list) else []


def _state(apps: list[dict], name: str) -> list[str]:
	out = []
	for a in apps:
		label = a.get("Description") or a.get("description") or a.get("Name") or a.get("name")
		if label == name:
			out.append(str(a.get("State") or a.get("state") or "?"))
	return out


def main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser(prog="python -m preflight.runner.stop_app")
	ap.add_argument("--app", default=APP_NAME)
	# required: production is only ever touched by the manual deploy workflow, never by a default
	ap.add_argument("--env", required=True, help="Modal environment, e.g. main")
	ns = ap.parse_args(argv)
	before = _state(_apps(ns.env), ns.app)
	r = subprocess.run(["modal", "app", "stop", "--yes", "--env", ns.env, ns.app], capture_output=True, text=True,
		timeout=180)  # fmt: skip
	after = _state(_apps(ns.env), ns.app)
	running = [s for s in after if s.lower() in ("deployed", "running", "ephemeral", "deploying")]
	result = {"app": ns.app, "env": ns.env, "before": before, "after": after, "stop_exit": r.returncode,
		"stop_output": (r.stdout + r.stderr)[-2000:], "stopped": r.returncode == 0 and not running}  # fmt: skip
	Path("out").mkdir(exist_ok=True)
	Path("out/stop.json").write_text(json.dumps(result, indent=2))
	print(json.dumps(result, indent=2))
	return 0 if result["stopped"] else 1


if __name__ == "__main__":
	sys.exit(main())
