import math

import numpy as np
import pytest

from preflight.engine import agent_cost as ac


def test_cp_upper_with_nothing_seen_is_the_rule_of_three():
	# exact: 1 - 0.05^(1/50); the rule of three says about 3/50
	assert ac.cp_upper(0, 50, 0.95) == pytest.approx(1 - 0.05 ** (1 / 50), rel=1e-6)
	assert ac.cp_upper(0, 50, 0.95) == pytest.approx(3 / 50, rel=0.05)


def test_cp_bounds_bracket_the_rate_and_grow_with_confidence():
	for x in (0, 1, 3, 10):
		lo, hi = ac.cp_lower(x, 50, 0.95), ac.cp_upper(x, 50, 0.95)
		assert lo <= x / 50 <= hi
		assert ac.cp_upper(x, 50, 0.80) < hi
	assert ac.cp_upper(50, 50, 0.9) == 1.0 and ac.cp_lower(0, 50, 0.9) == 0.0


def test_cp_upper_is_stable_for_large_samples():
	u = ac.cp_upper(5, 5000, 0.9)
	assert 0.001 < u < 0.003 and math.isfinite(u)


def _costs(n=50, seed=1):
	return np.random.default_rng(seed).lognormal(np.log(0.05), 0.8, n)


def test_abstains_without_population_or_enough_tasks():
	assert ac.estimate_total(_costs(), None).status == "abstain"
	e = ac.estimate_total(_costs(5), 500)
	assert e.status == "abstain" and "at least 10" in e.reason


def test_every_task_measured_is_exact():
	c = _costs(40)
	e = ac.estimate_total(c, 40)
	assert e.status == "measured" and e.point == e.lo == e.hi == pytest.approx(c.sum())


def test_interval_holds_the_point_and_is_deterministic():
	c = _costs()
	a, b = ac.estimate_total(c, 500), ac.estimate_total(c, 500)
	assert a == b
	assert a.status == "predicted" and a.lo <= a.point <= a.hi
	assert a.point == pytest.approx(c.mean() * 500)


def test_unseen_runaways_still_widen_the_upper_end():
	c = _costs()
	e = ac.estimate_total(c, 500)
	assert e.detail["runaways"] == 0
	expected = 500 * ac.cp_upper(0, 50, 0.80) * 20 * np.median(c)
	assert e.detail["tail_added"] == pytest.approx(expected)
	assert "no sampled task ran away" in e.reason


def test_seen_runaways_are_priced_at_their_own_cost():
	c = _costs()
	c[:2] = 50 * np.median(c)
	e = ac.estimate_total(c, 500)
	assert e.detail["runaways"] == 2
	assert e.detail["runaway_cost"] == pytest.approx(c[:2].mean())
	assert e.detail["tail_added"] > 0 and "2 of 50" in e.reason


def test_unusable_values_are_left_out_and_counted():
	c = [*_costs(30), None, float("nan"), -1.0]
	e = ac.estimate_total(c, 300)
	assert e.detail["dropped"] == 3 and e.detail["n"] == 30 and "left out" in e.reason


def test_free_cached_tasks_do_not_make_everything_a_runaway():
	c = np.r_[np.zeros(30), _costs(20)]
	e = ac.estimate_total(c, 500)
	assert e.detail["runaways"] <= 2


def test_schema_estimate_and_runaway_rate():
	c = _costs()
	est = ac.estimate_total(c, 500).to_estimate()
	assert est.source == "exact" and est.unit == "usd"
	r = ac.runaway_rate(c)
	assert r.point == 0 and r.lo == 0 and 0 < r.hi < 0.1


def test_agent_totals_from_a_sample():
	from preflight.schema.record import AgentSample, AgentTask

	tasks = [
		AgentTask(
			task_id=f"t{i}",
			turns=3,
			input_tokens=1000 + i,
			output_tokens=200,
			latency_s=2.0,
			cost_usd=0.01 + i * 1e-4,
		)
		for i in range(30)
	]
	out = ac.agent_totals(AgentSample(tasks=tasks, population=300))
	assert set(out) == {"cost.total", "tokens.total", "risk.runaway"}
	assert out["cost.total"].status == "predicted" and out["tokens.total"].unit == "tokens"
	no_cost = [t.model_copy(update={"cost_usd": None}) for t in tasks]
	priced = ac.agent_totals(
		AgentSample(tasks=no_cost, population=300), price=lambda t: 1e-5 * t.input_tokens
	)
	assert priced["cost.total"].point == pytest.approx(300 * np.mean([1e-5 * t.input_tokens for t in tasks]))
	unpriced = ac.agent_totals(AgentSample(tasks=no_cost, population=300))
	assert unpriced["cost.total"].status == "abstain"


def test_coverage_on_a_small_slice_of_d3():
	"""A quick version of S6: 400 runs from A's generator, 50 tasks, should cover near 89%."""
	g = np.random.default_rng(3)
	hits = 0
	trials = 400
	for _ in range(trials):
		c = g.lognormal(np.log(0.05), g.uniform(0.5, 1.2), 500)
		pop = np.where(g.random(500) < g.uniform(0, 0.05), c * g.uniform(10, 40), c)
		e = ac.estimate_total(g.choice(pop, 50, replace=False), 500, n_boot=400)
		hits += e.lo <= pop.sum() <= e.hi
	assert hits / trials > 0.83


def test_review_edge_cases():
	assert ac.estimate_total([1e308] * 20, 500).status == "abstain"
	over = ac.estimate_total([1.0] * 20, 10)
	assert over.status == "abstain" and "only 10" in over.reason


def test_large_samples_bootstrap_in_bounded_memory():
	import tracemalloc

	c = np.random.default_rng(0).lognormal(-3, 1, 10_000)
	tracemalloc.start()
	e = ac.estimate_total(c, 100_000)
	peak = tracemalloc.get_traced_memory()[1]
	tracemalloc.stop()
	assert e.status == "predicted" and peak < 60e6


def test_reasoning_tokens_are_not_counted_twice():
	from preflight.schema.record import AgentTask

	t = AgentTask(
		task_id="t", turns=1, input_tokens=100, output_tokens=50, reasoning_tokens=30, latency_s=1.0
	)
	assert ac._tokens(t) == 150


def test_runaway_settings_reach_the_rate_readout():
	from preflight.schema.record import AgentSample, AgentTask

	costs = [0.01] * 25 + [0.05] * 5  # 5x the median: a runaway only at a factor below 5
	tasks = [
		AgentTask(task_id=f"t{i}", turns=1, input_tokens=1, output_tokens=1, latency_s=1.0, cost_usd=c)
		for i, c in enumerate(costs)
	]
	sample = AgentSample(tasks=tasks, population=300)
	assert ac.agent_totals(sample)["risk.runaway"].point == 0
	assert ac.agent_totals(sample, runaway_factor=4.0)["risk.runaway"].point == pytest.approx(5 / 30)
