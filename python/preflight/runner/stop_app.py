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
import time
from pathlib import Path

APP_NAME = "preflight-runner"  # runner/modal_app.py's APP_NAME (not imported: that pulls in modal images)
STOP_WAIT_S = 5  # guess: Modal took under 10 s from "stopping..." to "stopped" on Oct 28
STOP_WAIT_POLLS = 24  # two minutes, then report what the list says


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


def _entries(apps: list[dict], name: str) -> list[dict]:
	out = []
	for a in apps:
		label = a.get("Description") or a.get("description") or a.get("Name") or a.get("name")
		if label == name:
			out.append(
				{
					"id": str(a.get("App ID") or a.get("app_id") or ""),
					"state": str(a.get("State") or a.get("state") or "?"),
				}
			)
	return out


def _state(apps: list[dict], name: str) -> list[str]:
	return [e["state"] for e in _entries(apps, name)]


def _live(state: str) -> bool:
	"""Anything but stopped or disabled is still running somewhere (deployed, ephemeral,
	"ephemeral (detached)", "initializing...", "stopping...")."""
	return state.lower() not in ("stopped", "disabled")


def main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser(prog="python -m preflight.runner.stop_app")
	ap.add_argument("--app", default=APP_NAME)
	# required: production is only ever touched by the manual deploy workflow, never by a default
	ap.add_argument("--env", required=True, help="Modal environment, e.g. main")
	ns = ap.parse_args(argv)
	listed = _entries(_apps(ns.env), ns.app)
	before = [e["state"] for e in listed]
	# the deployed app by name, then any temporary copy still live (a detached `modal run` of the
	# runner is exactly the kind that holds the job state), each by its app id
	targets = [ns.app] + [
		e["id"] for e in listed if e["id"] and _live(e["state"]) and e["state"].lower() != "deployed"
	]
	outputs, codes = [], []
	for t in targets:
		r = subprocess.run(["modal", "app", "stop", "--yes", "--env", ns.env, t], capture_output=True, text=True,
			timeout=180)  # fmt: skip
		outputs.append(f"{t}: {(r.stdout + r.stderr).strip()}")
		codes.append(r.returncode)
	after = _state(_apps(ns.env), ns.app)
	for _ in range(STOP_WAIT_POLLS):  # "stopping..." settles within seconds; wait for it to say so
		if not any(x.lower().startswith("stopping") for x in after):
			break
		time.sleep(STOP_WAIT_S)
		after = _state(_apps(ns.env), ns.app)
	running = [s for s in after if _live(s)]
	# `modal app stop` exits 1 on an app that's already stopped: what counts is the list afterwards
	text = "\n".join(outputs)
	result = {"app": ns.app, "env": ns.env, "before": before, "after": after, "stop_exit": max(codes, default=0),
		"stop_output": text[-2000:], "stopped": bool(after) and not running}  # fmt: skip
	Path("out").mkdir(exist_ok=True)
	Path("out/stop.json").write_text(json.dumps(result, indent=2))
	print(json.dumps(result, indent=2))
	return 0 if result["stopped"] else 1


if __name__ == "__main__":
	sys.exit(main())
