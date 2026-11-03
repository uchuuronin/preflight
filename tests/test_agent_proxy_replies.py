"""The proxy's own answers on the record, and a script with no task list as one task (Nov 2, unseen script 4:
smolagents' first model call failed in 0.7 s and its SDK showed no message). Flags probes.agent_proxy_replies
and probes.agent_single_task, both off."""

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from preflight.engine.prices import token_rate
from preflight.probes.agent_remote import error_line
from preflight.recorder.agent import AgentProxy
from preflight.recorder.agent_sampler import run_sample
from preflight.runner.agent_jobs import one_task
from preflight.schema.script_map import AgentInfo

sys.path.insert(0, str(Path(__file__).parent))
from test_agent_recorder import Provider


@pytest.fixture
def proxy():
	prov = Provider()
	px = AgentProxy(prov.url, "our-key", rate=token_rate)
	px.start()
	yield prov, px
	px.stop()
	prov.server.shutdown()


def _post(url, body):
	req = urllib.request.Request(f"{url}/chat/completions", json.dumps(body).encode(),
		{"Content-Type": "application/json"})  # fmt: skip
	try:
		with urllib.request.urlopen(req, timeout=10) as r:
			return r.status
	except urllib.error.HTTPError as e:
		return e.code


def _body(text, model="muse-spark-1.3"):
	return {"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": 5}


def test_the_proxys_own_answers_are_counted_by_status_code_and_model(proxy):
	_, px = proxy
	url = px.open_run("r", max_tasks=2)
	assert _post(url, _body("a", model="gpt-4o")) == 400  # no price (it opened task 1)
	assert _post(url, _body("b")) == 200
	assert _post(url, _body("c")) == 429  # a third task: past the sample of 2
	got = px.own_replies()
	assert (400, "unpriced_model", "gpt-4o", 1) in got and (
		429,
		"sample_complete",
		"muse-spark-1.3",
		1,
	) in got


def test_a_single_task_run_keeps_every_call_in_task_1(proxy):
	_, px = proxy
	one = px.open_run("one", max_tasks=1, single=True)
	codes = [_post(one, _body(t)) for t in ("plan the task", "a rewritten memory", "a side call")]
	assert codes == [200, 200, 200] and len(px.run_tasks("one")) == 1
	assert px.task(px.run_tasks("one")[0]).turns == 3
	many = px.open_run("many", max_tasks=1)  # as before: the second opening is past the sample
	assert [_post(many, _body(t)) for t in ("x", "y")] == [200, 429]


def test_only_a_script_with_no_task_list_is_one_task():
	assert one_task(AgentInfo(client="openai"))
	assert not one_task(AgentInfo(client="openai", n_tasks=164))
	assert not one_task(AgentInfo(client="openai", task_source="humaneval"))
	assert not one_task(None)


def test_the_error_line_takes_only_an_api_errors_status_and_code_from_the_next_line():
	tail = (
		'Traceback (most recent call last):\n  File "run.py", line 9\n'
		"smolagents.utils.AgentGenerationError: Error while generating output:\n"
		"Error code: 400 - {'error': {'type': 'preflight_unpriced_model', 'code': 'unpriced_model', 'message': 'x'}}\n"
	)
	head = "smolagents.utils.AgentGenerationError: Error while generating output:"
	assert error_line(tail) == head
	assert error_line(tail, follow=True) == f"{head} Error code: 400 (preflight_unpriced_model)"
	# a provider's message can echo a key, and a line that isn't an API error is the script's own output
	key = "openai.AuthenticationError: Error code: 401 - {'error': {'message': 'Incorrect API key: sk-abc***xyz'}}"
	assert "sk-" not in error_line(f"{head}\n{key.split(': ', 1)[1]}\n", follow=True)
	assert error_line("ValueError: bad:\nTHE NEXT PRINTED ROW: ssn 123\n", follow=True) == "ValueError: bad:"


def test_the_sample_names_the_proxys_answers_only_with_the_flag():
	class Proxy:
		def open_run(self, run_id, **kw):
			self.kw = kw
			return "http://proxy/v1"

		def run_stats(self, run_id):
			return {}

		def run_tasks(self, run_id):
			return []

		def own_replies(self):
			return [(404, "not_found", None, 9), (400, "unpriced_model", "muse-spark-1.3\n<b>x</b>", 2),
				(429, "sample_complete", None, 1)]  # fmt: skip

	class Proc:
		def __init__(self):
			self.report = {"ended": "exited", "code": 1}

		def poll(self):
			return 1

		def wait(self, timeout):
			pass

		def kill(self):
			pass

	px = Proxy()
	s = run_sample(None, cwd=None, proxy=px, population=None, env={}, launch=lambda env: Proc(), note_replies=True,
		single=True)  # fmt: skip
	assert "proxy answered 400 unpriced_model (muse-spark-1.3bx/b) x2, 429 sample_complete" in s.selection
	assert "404" not in s.selection and "<" not in s.selection
	assert px.kw.get("single") is True
	off = run_sample(None, cwd=None, proxy=px, population=None, env={}, launch=lambda env: Proc())
	assert "proxy answered" not in off.selection and "single" not in px.kw


def test_a_long_selection_leaves_room_for_the_top_up_marker():
	from preflight.engine.agent_cost import combine_samples
	from preflight.recorder.agent_sampler import SELECTION_CHARS
	from preflight.schema import AgentSample

	class Proxy:
		def open_run(self, run_id, **kw):
			return "http://proxy/v1"

		def run_stats(self, run_id):
			return {}

		def run_tasks(self, run_id):
			return []

	class Proc:
		def __init__(self):
			self.report = {"ended": "exited", "code": 1, "error": "ValueError: " + "x " * 200}

		def poll(self):
			return 1

		def wait(self, timeout):
			pass

		def kill(self):
			pass

	s = run_sample(None, cwd=None, proxy=Proxy(), population=None, env={}, launch=lambda env: Proc())
	assert len(s.selection) <= SELECTION_CHARS
	both = combine_samples(s, AgentSample(tasks=[], population=None, selection="extra"))
	assert both.selection.endswith("+ top-up of 0")


def test_a_providers_error_is_kept_with_the_requests_shape_never_its_values(proxy):
	from preflight.recorder.agent import request_shape

	prov, px = proxy
	url = px.open_run("r", max_tasks=1)
	prov.status = 400
	body = {"model": "muse-spark-1.3", "max_tokens": 5, "stop": ["<end_code>"],
		"messages": [{"role": "system", "content": "secret plan"}, {"role": "user", "content": [{"type": "text", "text": "x"}]}]}  # fmt: skip
	assert _post(url, body) == 400
	(status, _code, model, shape, n), *_ = px.provider_errors()
	assert status == 400 and model == "muse-spark-1.3" and n == 1
	assert shape == "keys max_tokens,messages,model,stop; roles system,user; content list:text,text"
	assert "secret" not in shape and "end_code" not in shape
	assert request_shape(b"[1]") == "not an object" and request_shape(b"{") == "not JSON"


def test_the_providers_note_fits_beside_a_long_error_line():
	class Proxy:
		def open_run(self, run_id, **kw):
			return "http://proxy/v1"

		def run_stats(self, run_id):
			return {}

		def run_tasks(self, run_id):
			return []

		def own_replies(self):
			return []

		def provider_errors(self):
			return [
				(
					400,
					"invalid_request_error",
					"muse-spark-1.3",
					"keys messages,model,stop; roles system,user",
					1,
				)
			]

	class Proc:
		def __init__(self):
			self.report = {"ended": "exited", "code": 1,
				"error": "smolagents.utils.AgentGenerationError: Error while generating output: Error code: 400"}  # fmt: skip

		def poll(self):
			return 1

		def wait(self, timeout):
			pass

		def kill(self):
			pass

	s = run_sample(
		None, cwd=None, proxy=Proxy(), population=None, env={}, launch=lambda env: Proc(), note_replies=True
	)
	assert (
		"provider answered 400 invalid_request_error (muse-spark-1.3) to keys messages,model,stop"
		in s.selection
	)
	Proxy.own_replies = lambda self: [(429, "sample_complete", None, 1)]
	both = run_sample(
		None, cwd=None, proxy=Proxy(), population=None, env={}, launch=lambda env: Proc(), note_replies=True
	)
	sel = both.selection
	# the provider's refusal first; what fits of the proxy's note and the script's line after it, in that order
	assert len(sel) <= 180 and "provider answered 400" in sel
	at = [sel.find(x) for x in ("provider answered", "proxy answered", "script failed") if x in sel]
	assert at == sorted(at)
