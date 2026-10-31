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


def test_a_key_sampled_twice_is_skipped_twice_and_no_more(proxy):
	"""Tasks that open with the same call ("Begin.") share a key: skip as many as were sampled."""
	_, px = proxy
	body = {"model": "muse-spark-1.3", "messages": [{"role": "user", "content": "Begin."}], "max_tokens": 5}
	url = px.open_run("s1", max_tasks=2)
	_post(url, body), _post(url, body)
	first = px.run_tasks("s1")
	url2 = px.open_run("s2", max_tasks=5, skip_keys=[task_key(t) for t in first])
	codes = [_post(url2, body)[0] for _ in range(4)]
	assert codes == [429, 429, 200, 200] and len(px.run_tasks("s2")) == 2
