"""stop_app reads the app list after the stop; the runner can't be started as a temporary copy."""

import ast
import json
from pathlib import Path

import pytest

from preflight.runner import stop_app

HERE = Path(stop_app.__file__).parent


class Done:
	def __init__(self, code=0, out="", err=""):
		self.returncode, self.stdout, self.stderr = code, out, err


def _listing(entries):
	return json.dumps([{"App ID": i, "Description": "preflight-runner", "State": st} for i, st in entries])


@pytest.mark.parametrize(
	("lists", "codes", "expect"),
	[
		([[("ap-1", "deployed")], [("ap-1", "stopped")]], {}, True),
		([[("ap-1", "stopped")], [("ap-1", "stopped")]], {"preflight-runner": (1, "App is already stopped.")}, True),
		([[("ap-1", "deployed")], [("ap-1", "deployed")]], {}, False),
		([[], []], {"preflight-runner": (1, "App not found")}, False),
		([[("ap-1", "deployed")], [("ap-1", "stopping...")], [("ap-1", "stopped")]], {}, True),
		# a detached temporary copy is live, and is stopped by its own id
		([[("ap-1", "deployed"), ("ap-2", "ephemeral (detached)")], [("ap-1", "stopped"), ("ap-2", "stopped")]], {}, True),
		([[("ap-1", "deployed"), ("ap-2", "initializing...")], [("ap-1", "stopped"), ("ap-2", "initializing...")]], {}, False),
	],
)  # fmt: skip
def test_stop_reports_what_the_app_list_says(monkeypatch, tmp_path, lists, codes, expect):
	monkeypatch.chdir(tmp_path)
	it = iter(lists)
	stops = []
	last = []

	def run(cmd, **kw):
		nonlocal last
		assert "--env" in cmd and "main" in cmd
		if cmd[:3] == ["modal", "app", "list"]:
			last = next(it, last)
			return Done(0, _listing(last))
		stops.append(cmd[-1])
		code, err = codes.get(cmd[-1], (0, ""))
		return Done(code, "", err)

	monkeypatch.setattr(stop_app.subprocess, "run", run)
	monkeypatch.setattr(stop_app.time, "sleep", lambda s: None)
	assert stop_app.main(["--env", "main"]) == (0 if expect else 1)
	out = json.loads((tmp_path / "out" / "stop.json").read_text())
	assert out["stopped"] is expect and out["env"] == "main"
	assert stops[0] == "preflight-runner"
	copies = [i for i, st in lists[0] if st.lower() not in ("stopped", "disabled", "deployed")]
	assert stops[1:] == copies


def test_the_env_is_required():
	with pytest.raises(SystemExit):
		stop_app.main([])


def test_the_app_name_matches_the_modal_app():
	assert f'APP_NAME = "{stop_app.APP_NAME}"' in (HERE / "modal_app.py").read_text()


def test_the_deployed_app_has_nothing_to_modal_run():
	"""`modal run` of the runner starts a temporary copy; its entry points live in ops_app.py."""
	tree = ast.parse((HERE / "modal_app.py").read_text())
	funcs = (ast.FunctionDef, ast.AsyncFunctionDef)
	decorators = [ast.unparse(d) for n in ast.walk(tree) if isinstance(n, funcs) for d in n.decorator_list]
	assert not [d for d in decorators if "local_entrypoint" in d]
	ops = ast.parse((HERE / "ops_app.py").read_text())
	assert not [n for n in ast.walk(ops) if isinstance(n, ast.ClassDef)]  # no service of its own
	kws = {k.arg for n in ast.walk(ops) if isinstance(n, ast.Call) for k in n.keywords}
	assert not kws & {"schedule", "min_containers", "buffer_containers"}  # nothing that starts on its own


def test_the_job_store_claims_only_in_the_deployed_apps_containers(monkeypatch):
	modal = pytest.importorskip("modal")
	from preflight.runner import modal_app as m
	from preflight.runner.queue import StoreUnavailable

	class App:
		def __init__(self, i):
			self.app_id = i

	monkeypatch.setattr(m, "_container_environment", lambda: "main")
	monkeypatch.setattr(
		modal.App, "lookup", staticmethod(lambda name, environment_name=None: App("ap-deployed"))
	)
	monkeypatch.setattr(modal.App, "_get_container_app", classmethod(lambda cls: App("ap-deployed")))
	assert m.deployed_here()[0] is True
	monkeypatch.setattr(modal.App, "_get_container_app", classmethod(lambda cls: App("ap-temporary")))
	ok, why = m.deployed_here()
	assert ok is False and "ap-temporary" in why and "ap-deployed" in why

	def broken(*a, **k):
		raise RuntimeError("lookup failed")

	monkeypatch.setattr(modal.App, "lookup", staticmethod(broken))
	monkeypatch.setattr("time.sleep", lambda s: None)
	assert m.deployed_here()[0] is None  # can't tell: the store fails to start and Modal retries
	with pytest.raises(StoreUnavailable, match="not the deployed job store"):
		m._Refusing("a temporary copy").submit({}, "v", "k")


def test_ownership_is_claimed_only_on_a_match_and_never_guessed():
	from preflight.runner.ownership import decide_ownership

	class NotFoundError(Exception):
		pass

	def found(i):
		return lambda: i

	def missing():
		raise NotFoundError("App 'preflight-runner' not found in environment 'main'")

	calls = []

	def flaky():
		calls.append(1)
		raise ConnectionError("control plane hiccup")

	assert decide_ownership("ap-1", found("ap-1"), where="x")[0] is True
	assert decide_ownership("ap-2", found("ap-1"), where="x")[0] is False  # a temporary copy
	assert decide_ownership("ap-2", missing, where="x")[0] is False  # nothing deployed there
	assert decide_ownership("ap-2", missing, where="x", env_known=False)[0] is None  # don't guess
	assert decide_ownership("ap-1", flaky, where="x", tries=3)[0] is None and len(calls) == 3  # restart
	assert decide_ownership(None, found("ap-1"), where="x")[0] is None


def test_app_logs_reads_and_never_runs(monkeypatch, tmp_path):
	from preflight.runner import app_logs

	monkeypatch.chdir(tmp_path)
	calls = []

	def run(cmd, **kw):
		calls.append(cmd)
		if cmd[:3] == ["modal", "app", "logs"]:
			raise app_logs.subprocess.TimeoutExpired(
				cmd, kw["timeout"], output=b"[ground truth] bare run", stderr=b""
			)
		return Done(0, json.dumps([{"App ID": "ap-abcdef12", "Description": "preflight-runner-ops", "State": "ephemeral"},
			{"App ID": "ap-runner99", "Description": "preflight-runner", "State": "deployed"}]))  # fmt: skip

	monkeypatch.setattr(app_logs.subprocess, "run", run)
	assert app_logs.main(["--env", "main", "ap-abcdef12", "--seconds", "5"]) == 0
	assert (tmp_path / "out" / "app_logs.txt").read_text() == "[ground truth] bare run"
	got = json.loads((tmp_path / "out" / "app_logs.json").read_text())
	assert got["state"] == "ephemeral" and "cut off" in got["read"]
	assert all(c[:2] == ["modal", "app"] and c[2] in ("logs", "list") for c in calls)  # read-only commands
	assert app_logs.main(["--env", "main", "ap-x; rm -rf /"]) == 2
	n = len(calls)
	assert app_logs.main(["--env", "main", "ap-runner99"]) == 2  # visitors' output is never published
	assert app_logs.main(["--env", "main", "ap-unknown123"]) == 2  # unlisted and not one of ours
	assert not any(c[2] == "logs" for c in calls[n:])


def _remote_runs(tmp_path, runs):
	"""A clone whose origin has a remote-runs branch holding these run records."""
	import subprocess

	def git(*a, cwd):
		subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True)

	src = tmp_path / "src"
	src.mkdir()
	git("init", "-q", "-b", "remote-runs", cwd=src)
	for name, (cmd, log) in runs.items():
		d = src / "runs" / name
		d.mkdir(parents=True)
		(d / "log.txt").write_text(log)
		(d / "result.json").write_text(json.dumps({"cmd": cmd}))
	git("add", "-A", cwd=src)
	git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "runs", cwd=src)
	clone = tmp_path / "clone"
	git("clone", "-q", "--no-checkout", str(src), str(clone), cwd=tmp_path)
	return clone


def test_an_unlisted_app_is_read_only_when_its_run_record_shows_the_ops_app(monkeypatch, tmp_path):
	from preflight.runner import app_logs

	view = "\u2713 Initialized. View run at \nhttps://modal.com/apps/me/main/{}\n"
	clone = _remote_runs(tmp_path, {
		"r1-a-gt-lora-l4-2": ("modal run --env main python/preflight/runner/ops_app.py::ground_truth --script x",
			view.format("ap-opsAAAAAA")),
		"r2-a-stop-ops-main": ("python -m preflight.runner.stop_app --env main --app preflight-runner-ops",
			"stopping ap-runrBBBBBB and ap-opsAAAAAA"),  # mentions apps, started none
		"r3-a-smoke-l4-main": ("modal run --env main python/preflight/runner/modal_app.py::smoke",
			view.format("ap-runrBBBBBB")),
		"r4-a-ops-other-env": ("modal run --env dev python/preflight/runner/ops_app.py::startup_check",
			view.format("ap-devCCCCCC")),
		# a runner id passed as an argument: Modal's arg parser fails and prints it; nothing started
		"r5-a-injected": ("modal run --env main python/preflight/runner/ops_app.py::startup_check ap-runrDDDDDD",
			"Error: Got unexpected extra argument (ap-runrDDDDDD)\n"),
		"r6-a-injected-url": ("modal run --env main python/preflight/runner/ops_app.py::startup_check "
			"https://modal.com/apps/me/main/ap-runrEEEEEE", view.format("ap-runrEEEEEE")),
	})  # fmt: skip
	monkeypatch.chdir(clone)
	real = app_logs.subprocess.run
	logs = []

	def run(cmd, **kw):
		if cmd[0] == "git":
			return real(cmd, **kw)
		if cmd[:3] == ["modal", "app", "list"]:
			return Done(0, "[]")  # nothing listed any more
		logs.append(cmd)
		return Done(0, "[ground truth 22:52:01] bare run of finetune_lora_hf")

	monkeypatch.setattr(app_logs.subprocess, "run", run)
	assert app_logs.ops_run_for("ap-opsAAAAAA", "main") == "runs/r1-a-gt-lora-l4-2"
	assert app_logs.ops_run_for("ap-runrBBBBBB", "main") is None  # the job runner, never
	assert app_logs.ops_run_for("ap-devCCCCCC", "main") is None  # another environment's run
	assert app_logs.ops_run_for("ap-nowhere99", "main") is None
	assert app_logs.ops_run_for("ap-runrDDDDDD", "main") is None  # only mentioned, never started
	assert app_logs.ops_run_for("ap-runrEEEEEE", "main") is None  # its id came from the command
	assert app_logs.main(["--env", "main", "ap-runrBBBBBB"]) == 2 and not logs
	assert app_logs.main(["--env", "main", "ap-opsAAAAAA"]) == 0
	assert logs[-1][-1] == "ap-opsAAAAAA" and "--tail" in logs[-1]
	got = json.loads((clone / "out" / "app_logs.json").read_text())
	assert got["app"] == "preflight-runner-ops" and "r1-a-gt-lora-l4-2" in got["state"]
