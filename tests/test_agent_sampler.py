"""The agent sampler on a stand-in agent eval and a fake provider, through real sockets."""

import os
import sys
from pathlib import Path

import pytest

from preflight.engine.prices import token_rate
from preflight.recorder.agent import AgentProxy, task_key
from preflight.recorder.agent_sampler import draw_indices, run_sample

sys.path.insert(0, str(Path(__file__).parent))
from test_agent_recorder import Provider

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture
def proxy():
	prov = Provider()
	px = AgentProxy(prov.url, "our-key", rate=token_rate)
	px.start()
	yield prov, px
	px.stop()
	prov.server.shutdown()


def test_thirty_of_a_hundred_tasks_are_run_and_split_by_conversation(proxy, tmp_path):
	prov, px = proxy
	cmd = [sys.executable, str(FIX / "fake_agent.py")]
	sample = run_sample(cmd, cwd=tmp_path, proxy=px, population=100, k=30, env=dict(os.environ))
	assert len(sample.tasks) == 30 and sample.population == 100
	assert all(t.turns == 3 for t in sample.tasks)  # each task's three turns stayed together
	assert len(prov.seen) == 90  # nothing past task 30 reached the provider
	assert {s["auth"] for s in prov.seen} == {"Bearer our-key"}
	assert "first 30 tasks" in sample.selection
	assert all(t.cost_usd and t.cost_usd > 0 for t in sample.tasks)


def _top_up(px, cmd, tmp_path, first, env, k=20):
	return run_sample(cmd, cwd=tmp_path, proxy=px, population=100, k=k, run_id="s2", env=env,
		skip_keys=[task_key(t.task_id) for t in first.tasks], offered_first=draw_indices(100, 30))  # fmt: skip


def _first_users(seen):
	return [s["body"]["messages"][1]["content"] for s in seen]


def test_a_top_up_skips_the_tasks_already_sampled_at_no_cost(proxy, tmp_path):
	prov, px = proxy
	cmd = [sys.executable, str(FIX / "fake_agent.py")]
	first = run_sample(cmd, cwd=tmp_path, proxy=px, population=100, k=30, run_id="s1", env=dict(os.environ))
	before = len(prov.seen)
	more = _top_up(px, cmd, tmp_path, first, dict(os.environ))
	assert len(more.tasks) == 20 and len(prov.seen) - before == 60  # only the 20 new tasks were paid for
	assert "task 30" in _first_users(prov.seen[before:]) and "task 0" not in _first_users(prov.seen[before:])
	assert "30 already sampled skipped" in more.selection
	assert not {t.task_id for t in first.tasks} & {t.task_id for t in more.tasks}


def test_a_top_up_of_a_script_that_reads_the_indices_runs_twenty_new_ones(proxy, tmp_path):
	"""Review, Oct 31: with only the new indices offered, a count-based skip refused all 20."""
	prov, px = proxy
	cmd = [sys.executable, str(FIX / "fake_agent.py")]
	env = {**os.environ, "FAKE_AGENT_READS_INDICES": "1"}
	first = run_sample(cmd, cwd=tmp_path, proxy=px, population=100, k=30, run_id="s1", env=env)
	before = len(prov.seen)
	more = _top_up(px, cmd, tmp_path, first, env)
	assert len(more.tasks) == 20 and len(prov.seen) - before == 60
	new = {int(u.split()[1]) for u in _first_users(prov.seen[before:])}
	assert new == set(draw_indices(100, 20, exclude=draw_indices(100, 30)))


def test_a_concurrent_script_is_skipped_by_content_not_by_order(proxy, tmp_path):
	"""Eight tasks at once start in a different order each run; nothing sampled is paid for twice."""
	prov, px = proxy
	cmd = [sys.executable, str(FIX / "fake_agent.py")]
	env = {**os.environ, "FAKE_AGENT_THREADS": "8"}
	first = run_sample(cmd, cwd=tmp_path, proxy=px, population=100, k=30, run_id="s1", env=env)
	assert len(first.tasks) == 30 and all(t.turns == 3 for t in first.tasks)
	sampled = {u for u in _first_users(prov.seen)}
	before = len(prov.seen)
	more = _top_up(px, cmd, tmp_path, first, env)
	again = set(_first_users(prov.seen[before:]))
	assert not sampled & again and len(more.tasks) == 20


def test_the_script_gets_no_other_provider_keys(proxy, tmp_path):
	_, px = proxy
	out = tmp_path / "env.txt"
	env = {**os.environ, "FAKE_AGENT_ENV_OUT": str(out), "ANTHROPIC_API_KEY": "sk-ant", "LLAMA_API_KEY": "x",
		"MY_COPY": "our-key", "TAVILY_API_KEY": "tool"}  # fmt: skip
	run_sample([sys.executable, str(FIX / "fake_agent.py")], cwd=tmp_path, proxy=px, population=100, k=1,
		env=env)  # fmt: skip
	names = set(out.read_text().split())
	assert "OPENAI_API_KEY" in names and not names & {"ANTHROPIC_API_KEY", "LLAMA_API_KEY", "MY_COPY"}
	assert "TAVILY_API_KEY" in names  # a tool's key stays, or the agent's tools fail


def test_a_cancelled_sample_stops_the_script_and_marks_what_was_running(proxy, tmp_path):
	import time

	_, px = proxy
	slow = tmp_path / "slow.py"
	slow.write_text("import time\ntime.sleep(60)\n")
	t0 = time.monotonic()
	out = run_sample([sys.executable, str(slow)], cwd=tmp_path, proxy=px, population=None, k=3,
		env=dict(os.environ), should_stop=lambda: time.monotonic() - t0 > 1.5)  # fmt: skip
	assert time.monotonic() - t0 < 10 and out.tasks == [] and "cancelled" in out.selection


def test_a_script_that_reads_the_indices_runs_the_random_ones(proxy, tmp_path):
	prov, px = proxy
	cmd = [sys.executable, str(FIX / "fake_agent.py")]
	env = {**os.environ, "FAKE_AGENT_READS_INDICES": "1"}
	run_sample(cmd, cwd=tmp_path, proxy=px, population=100, k=10, env=env)
	seen = sorted({int(s["body"]["messages"][1]["content"].split()[1]) for s in prov.seen})
	assert seen == draw_indices(100, 10)


def test_draws_are_seeded_and_exclude_what_was_run():
	a = draw_indices(500, 30)
	assert a == draw_indices(500, 30) and len(set(a)) == 30
	b = draw_indices(500, 20, exclude=a)
	assert not set(a) & set(b)
	assert draw_indices(None, 30) is None and draw_indices(5, 30) == [0, 1, 2, 3, 4]


def test_the_cli_sampler_runs_the_files_with_the_proxy_holding_the_key(tmp_path):
	from preflight.runner.agent_jobs import local_sampler
	from preflight.runner.sources import Fetched
	from preflight.schema.script_map import AgentInfo, ScriptMap

	prov = Provider()
	try:
		sample = local_sampler(prov.url, "users-key")
		fetched = Fetched(files={"run.py": (FIX / "fake_agent.py").read_text()}, entrypoint="run.py")
		sm = ScriptMap.model_construct(agent=AgentInfo(client="openai", n_tasks=100))
		out = sample(fetched, sm, k=5, env=dict(os.environ))
		assert len(out.tasks) == 5 and out.population == 100
		assert {s["auth"] for s in prov.seen} == {"Bearer users-key"}
	finally:
		prov.server.shutdown()


def _post(url, body):
	import json
	import urllib.error
	import urllib.request

	req = urllib.request.Request(f"{url}/chat/completions", json.dumps(body).encode(),
		{"Content-Type": "application/json"})  # fmt: skip
	try:
		with urllib.request.urlopen(req, timeout=10) as r:
			return r.status, dict(r.headers)
	except urllib.error.HTTPError as e:
		return e.code, dict(e.headers)


def test_the_same_prompt_again_is_a_new_task_and_refusals_say_dont_retry(proxy):
	"""pass@k: five identical calls are five tasks, not one task stopped as a loop (review, Oct 31)."""
	_, px = proxy
	url = px.open_run("r", max_tasks=4)
	body = {"model": "muse-spark-1.3", "messages": [{"role": "user", "content": "same"}], "max_tokens": 5}
	codes = [_post(url, body) for _ in range(5)]
	assert [c for c, _ in codes] == [200, 200, 200, 200, 429]
	assert codes[-1][1].get("x-should-retry") == "false"
	ids = px.run_tasks("r")
	assert len(ids) == 4 and len({task_key(t) for t in ids}) == 1 and ids[1].endswith(".2")
	assert all(px.task(t).stop_reason == "done" for t in ids)
	assert px.run_stats("r")["refused"] == 1


def test_a_retry_after_a_failed_call_stays_in_its_task(proxy):
	prov, px = proxy
	url = px.open_run("r", max_tasks=1)
	body = {"model": "muse-spark-1.3", "messages": [{"role": "user", "content": "x"}], "max_tokens": 5}
	prov.status = 500
	assert _post(url, body)[0] == 500
	prov.status = 200
	assert _post(url, body)[0] == 200  # the retry: not a second task, so not refused
	assert len(px.run_tasks("r")) == 1


def test_a_run_id_is_used_once_per_proxy(proxy):
	_, px = proxy
	px.open_run("s1", max_tasks=1)
	with pytest.raises(ValueError):
		px.open_run("s1", max_tasks=1)


def test_tasks_that_open_the_same_way_cant_be_topped_up(proxy):
	"""Openings like "Begin." can't be told from a framework's retry of a skipped call, so a top-up
	refuses them all rather than pay for a sampled task again (review, Oct 31)."""
	_, px = proxy
	body = {"model": "muse-spark-1.3", "messages": [{"role": "user", "content": "Begin."}], "max_tokens": 5}
	url = px.open_run("s1", max_tasks=2)
	_post(url, body), _post(url, body)
	first = px.run_tasks("s1")
	url2 = px.open_run("s2", max_tasks=5, skip_keys=[task_key(t) for t in first])
	assert [_post(url2, body)[0] for _ in range(4)] == [429] * 4 and px.run_tasks("s2") == []


def test_a_stopped_task_retried_stays_stopped(prov_and_capped):
	"""A task stopped at its call cap: the SDK's retry of the refused call isn't a new task."""
	prov, px = prov_and_capped
	url = px.open_run("r", max_tasks=3)
	b1 = {"model": "muse-spark-1.3", "messages": [{"role": "user", "content": "go"}], "max_tokens": 5}
	assert _post(url, b1)[0] == 200
	b2 = {
		**b1,
		"messages": [
			*b1["messages"],
			{"role": "assistant", "content": "x"},
			{"role": "user", "content": "more"},
		],
	}
	code, headers = _post(url, b2)
	assert code == 429 and headers.get("x-should-retry") == "false"
	assert _post(url, b2)[0] == 429 and len(px.run_tasks("r")) == 1 and len(prov.seen) == 1


@pytest.fixture
def prov_and_capped():
	prov = Provider()
	px = AgentProxy(prov.url, "our-key", rate=token_rate, max_calls_per_task=1)
	px.start()
	yield prov, px
	px.stop()
	prov.server.shutdown()


def test_a_system_prompt_rebuilt_every_call_still_groups_by_task(proxy, tmp_path):
	"""D's agent script puts the time in its system prompt: every turn's prefix differs there."""
	prov, px = proxy
	env = {**os.environ, "FAKE_AGENT_TIMESTAMP_SYSTEM": "1"}
	first = run_sample([sys.executable, str(FIX / "fake_agent.py")], cwd=tmp_path, proxy=px, population=100,
		k=5, env=env)  # fmt: skip
	assert len(first.tasks) == 5 and all(t.turns == 3 for t in first.tasks)
	before = len(prov.seen)
	more = _top_up(px, [sys.executable, str(FIX / "fake_agent.py")], tmp_path, first, env, k=3)
	assert len(more.tasks) == 3 and "task 0" not in _first_users(prov.seen[before:])  # skipped by content


def test_a_run_with_no_model_calls_for_a_while_is_stopped_as_stalled(proxy, tmp_path):
	"""D's planted tool loop: a failing tool retried forever, so the model is never called again."""
	import time

	_, px = proxy
	stuck = tmp_path / "stuck.py"
	stuck.write_text(
		"import json, os, time, urllib.request\n"
		"b = json.dumps({'model': 'muse-spark-1.3', 'messages': [{'role': 'user', 'content': 'x'}]}).encode()\n"
		"urllib.request.urlopen(urllib.request.Request(os.environ['META_API_BASE'] + '/chat/completions', b,"
		" {'Content-Type': 'application/json'})).read()\n"
		"while True: time.sleep(1)\n"
	)
	t0 = time.monotonic()
	out = run_sample([sys.executable, str(stuck)], cwd=tmp_path, proxy=px, population=None, k=3,
		env=dict(os.environ), stall_s=2.0)  # fmt: skip
	assert time.monotonic() - t0 < 15 and len(out.tasks) == 1 and "no model calls" in out.selection


class _FakeExec:
	"""A sandbox exec run as a local process: probes/agent_remote.py, fed on stdin."""

	def __init__(self, cmd):
		import subprocess

		self.proc = subprocess.Popen(list(cmd), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
		self.buf = []
		self.stdin = self
		self.stdout = self

	def write(self, s):
		self.buf.append(s)

	def write_eof(self):
		self.proc.stdin.write("".join(self.buf))
		self.proc.stdin.close()

	def drain(self):
		pass

	def poll(self):
		return self.proc.poll()

	def read(self):
		return self.proc.stdout.read()


class _FakeSandbox:
	def __init__(self, extra_egress):
		self.extra_egress, self.execs, self.terminated = extra_egress, [], False

	def exec(self, *cmd, timeout=None):
		self.execs.append(_FakeExec([sys.executable, *cmd[1:]]))
		return self.execs[-1]

	def terminate(self):
		self.terminated = True
		for e in self.execs:
			if e.proc.poll() is None:
				e.proc.kill()


def test_the_hosted_sampler_keeps_the_key_outside_and_routes_through_the_tunnel(tmp_path, monkeypatch):
	import contextlib
	import types

	from preflight.runner.agent_jobs import sandbox_sampler
	from preflight.runner.sources import Fetched
	from preflight.schema.script_map import AgentInfo, ScriptMap

	prov = Provider()
	made = []

	@contextlib.contextmanager
	def forward(port):
		yield types.SimpleNamespace(url=f"http://127.0.0.1:{port}")

	def create(extra_egress, timeout_s):
		made.append(_FakeSandbox(extra_egress))
		made[-1].timeout_s = timeout_s
		return made[-1]

	monkeypatch.setenv("META_API_KEY", "worker-secret")  # the worker's own env must not reach the script
	try:
		sample = sandbox_sampler(create, prov.url, "our-key", forward=forward)
		fetched = Fetched(files={"run.py": (FIX / "fake_agent.py").read_text()}, entrypoint="run.py")
		sm = ScriptMap.model_construct(agent=AgentInfo(client="openai", n_tasks=100))
		out = sample(fetched, sm, k=4)
	finally:
		prov.server.shutdown()
	assert len(out.tasks) == 4 and all(t.turns == 3 for t in out.tasks)
	assert {s["auth"] for s in prov.seen} == {"Bearer our-key"}
	assert made[0].extra_egress == ["127.0.0.1"] and made[0].terminated
	assert made[0].timeout_s > 1800  # the sandbox outlives the sample's own limit


def test_past_the_dollar_cap_new_tasks_are_refused_not_counted_empty():
	"""Review, Oct 31: tasks opened after the cap became 0-turn, $0 tasks that lowered the projection."""
	prov = Provider()
	px = AgentProxy(prov.url, "our-key", rate=token_rate, max_usd_total=0.02)
	px.start()
	try:
		url = px.open_run("r", max_tasks=30)
		codes = []
		for i in range(10):
			body = {
				"model": "muse-spark-1.3",
				"messages": [{"role": "user", "content": f"t{i}"}],
				"max_tokens": 50,
			}
			codes.append(_post(url, body)[0])
		tasks = [px.task(t) for t in px.run_tasks("r")]
		assert 429 in codes and all(t.turns >= 1 for t in tasks)  # no empty tasks in the sample
		assert px.run_stats("r")["budget_hit"]
	finally:
		px.stop()
		prov.server.shutdown()


def test_tasks_told_apart_only_by_their_system_prompt_can_be_topped_up(proxy):
	_, px = proxy

	def body(i):
		return {"model": "muse-spark-1.3", "max_tokens": 5,
			"messages": [{"role": "system", "content": f"task {i}"}, {"role": "user", "content": "Begin."}]}  # fmt: skip

	url = px.open_run("s1", max_tasks=3)
	for i in range(3):
		_post(url, body(i))
	first = px.run_tasks("s1")
	url2 = px.open_run("s2", max_tasks=3, skip_keys=[task_key(t) for t in first])
	codes = [_post(url2, body(i))[0] for i in range(6)]
	assert codes == [429, 429, 429, 200, 200, 200] and len(px.run_tasks("s2")) == 3


def test_a_task_stuck_in_its_tools_is_a_loop_while_the_others_go_on(proxy, tmp_path):
	"""D's plant 3: one worker's tool retries forever after the model asked for it; the other
	worker keeps calling the model. The stuck task is marked a loop at the 2-minute cutoff (2 s here)."""
	_, px = proxy
	script = tmp_path / "two.py"
	script.write_text(
		"import json, os, threading, time, urllib.request\n"
		"base = os.environ['META_API_BASE']\n"
		"def call(body):\n"
		"    r = urllib.request.Request(base + '/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})\n"
		"    return urllib.request.urlopen(r).read()\n"
		"def stuck():\n"
		"    call({'model': 'muse-spark-1.3', 'stream': True, 'messages': [{'role': 'user', 'content': 'stuck'}]})\n"
		"    while True: time.sleep(1)\n"
		"def busy():\n"
		"    time.sleep(1)  # after the stuck call, so the fake provider's one tool-call answer is its\n"
		"    for i in range(12):\n"
		"        call({'model': 'muse-spark-1.3', 'messages': [{'role': 'user', 'content': f'busy {i}'}]}); time.sleep(0.5)\n"
		"    while True: time.sleep(1)\n"
		"threading.Thread(target=stuck, daemon=True).start(); threading.Thread(target=busy, daemon=True).start()\n"
		"while True: time.sleep(1)\n"
	)
	out = run_sample([sys.executable, str(script)], cwd=tmp_path, proxy=px, population=None, k=20,
		env=dict(os.environ), stall_s=2.0)  # fmt: skip
	by = {t.stop_reason for t in out.tasks if t.turns == 1 and t.tool_calls}
	assert by == {"loop"} and "stalled on their tools" in out.selection
	assert sum(t.stop_reason == "loop" for t in out.tasks) == 1  # the busy worker's tasks finished


def test_a_slow_tool_that_finishes_is_not_a_loop(proxy, tmp_path):
	"""Review, Oct 31: a task that waited past the cutoff on its tool and then finished came out as
	a loop. Only the tasks still open when a stalled run is stopped are."""
	_, px = proxy
	script = tmp_path / "slow.py"
	script.write_text(
		"import json, os, time, urllib.request\n"
		"base = os.environ['META_API_BASE']\n"
		"def call(body):\n"
		"    r = urllib.request.Request(base + '/chat/completions', json.dumps(body).encode(), {'Content-Type': 'application/json'})\n"
		"    return urllib.request.urlopen(r).read()\n"
		"call({'model': 'muse-spark-1.3', 'stream': True, 'messages': [{'role': 'user', 'content': 'slow'}]})\n"
		"time.sleep(3)  # a slow tool, past the 1 s cutoff below, then the task's next turn\n"
		"call({'model': 'muse-spark-1.3', 'messages': [{'role': 'user', 'content': 'slow'}, {'role': 'tool', 'content': 'ok'}]})\n"
	)
	out = run_sample([sys.executable, str(script)], cwd=tmp_path, proxy=px, population=None, k=5,
		env=dict(os.environ), stall_s=10.0)  # fmt: skip
	assert [t.stop_reason for t in out.tasks] == ["done"] and out.tasks[0].turns == 2
