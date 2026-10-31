"""Read a Modal app's container logs and its state, for a run whose own log showed nothing.

	python -m preflight.runner.app_logs --env main ap-smlQhBII0aKV0BnMwULRzO

Run through `remote-run.yml` (which only takes `modal run`, `modal deploy` and our own entry
points). `modal run` doesn't pass a remote function's output into the workflow's log, so this is
how a stuck or silent run is diagnosed without starting anything: it reads, it never runs code in
the app. Fetches the last log entries with timestamps; a slow fetch is cut off after --seconds. Writes
out/app_logs.txt (the logs) and out/app_logs.json (the app's state and how the read ended).
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

APP_ID = re.compile(r"ap-[A-Za-z0-9]{6,40}")
# Only our own measurement apps: their logs go to the remote-runs branch, and the job runner's
# (preflight-runner) hold visitors' script output, which must never be published
ALLOWED_APPS = {"preflight-runner-ops"}
# Modal's app list drops apps a while after they stop. Such an app is read only if its own run
# record on the remote-runs branch shows our ops app started it: the run's command was `modal run
# --env <env> python/preflight/runner/ops_app.py::...` and doesn't name the app (an id passed as an
# argument lands in the log of a run that started nothing), and the app is the one Modal's own
# "Initialized. View run at https://modal.com/apps/<workspace>/<env>/<app id>" line names. This
# guards against mistakes; who may dispatch runs at all is the real boundary.
REMOTE_RUNS = "origin/remote-runs"
_OPS_CMD = r"^modal run --env {env} python/preflight/runner/ops_app\.py::[a-z_]+(?: |$)"
_STARTED = (
	r"Initialized\. View run at\s*\n?\s*https://modal\.com/apps/[A-Za-z0-9_-]+/{env}/(ap-[A-Za-z0-9]{{6,40}})"
)
TAIL = 5000  # log entries fetched (Modal's default is 100)
# Modal fetches the last hour unless told otherwise: an app no longer listed is read over its
# run's own window, from the run's start to an hour past its end (an app outlives a cut-off run)
AFTER_RUN_S = 3600
_ISO = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z?")
MAX_LOG_CHARS = 2_000_000


def _app(env: str, app_id: str) -> dict | None:
	try:
		r = subprocess.run(["modal", "app", "list", "--json", "--env", env], capture_output=True, text=True,
			timeout=120)  # fmt: skip
		apps = json.loads(r.stdout) if r.returncode == 0 else []
	except (subprocess.TimeoutExpired, json.JSONDecodeError):
		return None
	for a in apps if isinstance(apps, list) else []:
		if str(a.get("App ID") or a.get("app_id") or "") == app_id:
			return {"name": str(a.get("Description") or a.get("description") or a.get("Name") or ""),
				"state": str(a.get("State") or a.get("state") or "?")}  # fmt: skip
	return None


def _git(*args: str) -> subprocess.CompletedProcess:
	return subprocess.run(["git", *args], capture_output=True, text=True, timeout=120)


def ops_run_for(app_id: str, env: str) -> str | None:
	"""The remote-runs record of the ops-app run that started app_id, or None."""
	_git("fetch", "--quiet", "--depth=1", "origin", f"+refs/heads/remote-runs:refs/remotes/{REMOTE_RUNS}")
	hits = _git("grep", "-l", "-F", app_id, REMOTE_RUNS, "--", "runs/")
	if hits.returncode != 0:
		return None
	found = None
	for line in hits.stdout.splitlines():
		path = line.split(":", 1)[-1]
		if not path.endswith("/log.txt"):
			continue
		run = path.rsplit("/", 1)[0]
		log = _git("show", f"{REMOTE_RUNS}:{path}").stdout
		head = "\n".join(log.splitlines()[:20])  # Modal prints it first; a cut log's tail never counts
		started = re.search(_STARTED.format(env=re.escape(env)), head)
		if started is None or started.group(1) != app_id:
			continue  # this run didn't start the app (it mentions it, or started another)
		try:
			cmd = json.loads(_git("show", f"{REMOTE_RUNS}:{run}/result.json").stdout).get("cmd") or ""
		except ValueError:
			continue
		if "modal_app.py" in cmd:
			return None  # the job runner: its logs hold visitors' output
		if app_id in cmd:
			continue  # the id came in as an argument, not from Modal
		if re.match(_OPS_CMD.format(env=re.escape(env)), cmd):
			found = run
	return found


def run_window(run: str) -> tuple[str, str] | None:
	"""(since, until) for a remote-runs record: its directory's start time to an hour past the
	finished_at in its result.json, as Modal's --since/--until take them."""
	from datetime import UTC, datetime, timedelta

	m = re.search(r"/(\d{8}-\d{6})-", run + "/")
	try:
		start = datetime.strptime(m.group(1), "%Y%m%d-%H%M%S").replace(tzinfo=UTC) if m else None
		res = json.loads(_git("show", f"{REMOTE_RUNS}:{run}/result.json").stdout)
		end = datetime.fromisoformat(str(res.get("finished_at")).replace("Z", "+00:00"))
	except (ValueError, TypeError, AttributeError):
		return None
	if start is None:
		return None
	fmt = "%Y-%m-%dT%H:%M:%S"
	return (start - timedelta(minutes=5)).strftime(fmt), (end + timedelta(seconds=AFTER_RUN_S)).strftime(fmt)


def main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser(prog="python -m preflight.runner.app_logs")
	ap.add_argument("--env", required=True, help="Modal environment, e.g. main")
	ap.add_argument("app_id")
	ap.add_argument("--seconds", type=int, default=90, help="stop a slow fetch after this long")
	ap.add_argument(
		"--since", help="UTC, e.g. 2026-10-28T22:45:00 (default: Modal's last hour, or the run's window)"
	)
	ap.add_argument("--until", help="UTC, same form")
	ns = ap.parse_args(argv)
	bad_time = any(t is not None and not _ISO.fullmatch(t) for t in (ns.since, ns.until))
	if not APP_ID.fullmatch(ns.app_id) or not 5 <= ns.seconds <= 600 or bad_time:
		print("an app id looks like ap-..., --seconds is 5 to 600, times look like 2026-10-28T22:45:00",
			file=sys.stderr)  # fmt: skip
		return 2
	window = (ns.since, ns.until)
	app = _app(ns.env, ns.app_id)
	if app is None:
		run = ops_run_for(ns.app_id, ns.env)
		if run is not None:
			app = {"name": "preflight-runner-ops", "state": f"no longer listed; started by {run}"}
			if ns.since is None and ns.until is None:
				window = run_window(run) or window
	if app is None or app["name"] not in ALLOWED_APPS:
		print(f"only the logs of {sorted(ALLOWED_APPS)} are read here; {ns.app_id} is "
			f"{app['name'] if app else 'not in the app list'}", file=sys.stderr)  # fmt: skip
		return 2
	# fetch (not follow): Modal returns the last TAIL entries and exits; --seconds bounds a slow fetch
	cmd = ["modal", "app", "logs", "--env", ns.env, "--tail", str(TAIL), "--timestamps"]
	for flag, value in zip(("--since", "--until"), window, strict=True):
		if value is not None:
			cmd += [flag, value]
	cmd.append(ns.app_id)
	try:
		r = subprocess.run(cmd, capture_output=True, text=True, timeout=ns.seconds)
		text, ended, code = r.stdout + r.stderr, f"exit {r.returncode}", r.returncode
	except subprocess.TimeoutExpired as exc:
		out = exc.stdout or b""
		err = exc.stderr or b""
		text = (out.decode("utf-8", "replace") if isinstance(out, bytes) else out) + (
			err.decode("utf-8", "replace") if isinstance(err, bytes) else err
		)
		ended = f"cut off after {ns.seconds} s (the app is still live, or Modal kept the stream open)"
		code = 0
	Path("out").mkdir(exist_ok=True)
	Path("out/app_logs.txt").write_text(text[-MAX_LOG_CHARS:])
	summary = {"app_id": ns.app_id, "env": ns.env, "app": app["name"], "state": app["state"], "read": ended,
		"since": window[0], "until": window[1],
		"chars": len(text)}  # fmt: skip
	Path("out/app_logs.json").write_text(json.dumps(summary, indent=2))
	print(json.dumps(summary, indent=2))
	print(text[-4000:])
	return 0 if code == 0 else 1


if __name__ == "__main__":
	sys.exit(main())
