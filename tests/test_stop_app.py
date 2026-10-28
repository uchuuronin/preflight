"""stop_app reads the app list after the stop, so an app someone already stopped counts as stopped."""

import json

import pytest

from preflight.runner import stop_app


class Done:
	def __init__(self, code=0, out="", err=""):
		self.returncode, self.stdout, self.stderr = code, out, err


@pytest.mark.parametrize(
	("states", "stop", "expect"),
	[
		([["deployed"], ["stopped"]], Done(0, "stopped"), True),
		([["stopped"], ["stopped"]], Done(1, "", "App is already stopped. (Stopped at ...)"), True),
		([["deployed"], ["deployed"]], Done(0), False),
		([["deployed"], ["stopped"]], Done(1, "", "permission denied"), False),
		([[], []], Done(1, "", "App not found"), False),
	],
)
def test_stop_reports_what_the_app_list_says(monkeypatch, tmp_path, states, stop, expect):
	monkeypatch.chdir(tmp_path)
	lists = iter(states)
	calls = []

	def run(cmd, **kw):
		calls.append(cmd)
		if cmd[:3] == ["modal", "app", "list"]:
			return Done(0, json.dumps([{"Description": "preflight-runner", "State": s} for s in next(lists)]))
		return stop

	monkeypatch.setattr(stop_app.subprocess, "run", run)
	assert stop_app.main(["--env", "main"]) == (0 if expect else 1)
	out = json.loads((tmp_path / "out" / "stop.json").read_text())
	assert out["stopped"] is expect and out["env"] == "main"
	assert all("--env" in c and "main" in c for c in calls)


def test_the_app_name_matches_the_modal_app():
	from pathlib import Path

	src = Path(stop_app.__file__).with_name("modal_app.py").read_text()
	assert f'APP_NAME = "{stop_app.APP_NAME}"' in src
