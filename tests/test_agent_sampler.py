"""The agent sampler on a stand-in agent eval and a fake provider, through real sockets."""

import os
import sys
from pathlib import Path

import pytest

from preflight.engine.prices import token_rate
from preflight.recorder.agent import AgentProxy
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


def test_a_top_up_skips_the_tasks_already_sampled_at_no_cost(proxy, tmp_path):
	prov, px = proxy
	cmd = [sys.executable, str(FIX / "fake_agent.py")]
	run_sample(cmd, cwd=tmp_path, proxy=px, population=100, k=30, run_id="s1", env=dict(os.environ))
	before = len(prov.seen)
	more = run_sample(
		cmd, cwd=tmp_path, proxy=px, population=100, k=20, skip=30, run_id="s2", env=dict(os.environ)
	)
	assert len(more.tasks) == 20 and len(prov.seen) - before == 60  # only the 20 new tasks were paid for
	first_user = [s["body"]["messages"][1]["content"] for s in prov.seen[before:]]
	assert "task 30" in first_user and "task 0" not in first_user
	assert "skipping the 30" in more.selection


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
