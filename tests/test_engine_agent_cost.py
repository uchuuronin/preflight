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


# ---------------------------------------------------------------- the top-up (Chiya, Oct 27)


def _est(lo, hi, n=30, pop=500):
	return ac.TotalEstimate((lo + hi) / 2, lo, hi, "usd", "predicted", detail={"n": n, "population": pop})


def test_budget_check_offers_the_top_up_only_when_the_range_straddles():
	assert ac.budget_check(_est(10, 20), 25).verdict == "under"
	assert ac.budget_check(_est(30, 40), 25).verdict == "over"
	b = ac.budget_check(_est(20, 30), 25)
	assert b.verdict == "straddles" and b.offer_top_up and b.top_up_tasks == ac.TOP_UP_TASKS
	assert "20 more tasks" in b.reason


def test_top_up_is_offered_once_and_never_past_the_run():
	assert not ac.budget_check(_est(20, 30, n=50), 25).offer_top_up
	near_end = ac.budget_check(_est(20, 30, n=40, pop=48), 25)
	assert near_end.offer_top_up and near_end.top_up_tasks == 8
	assert ac.budget_check(_est(20, 30), None).verdict == "unknown"
	abst = ac.TotalEstimate(None, None, None, "usd", "abstain", reason="x")
	assert not ac.budget_check(abst, 25).offer_top_up


def test_combine_samples_dedupes_and_keeps_the_population():
	from preflight.schema.record import AgentSample, AgentTask

	def t(i):
		return AgentTask(
			task_id=f"t{i}", turns=1, input_tokens=1, output_tokens=1, latency_s=1.0, cost_usd=0.01
		)

	first = AgentSample(tasks=[t(i) for i in range(30)], population=500, selection="30-70% pass band")
	extra = AgentSample(tasks=[t(i) for i in range(28, 50)])
	both = ac.combine_samples(first, extra)
	assert len(both.tasks) == 50 and both.population == 500
	assert both.selection == "30-70% pass band + top-up of 20"
	assert ac.agent_totals(both)["cost.total"].detail["n"] == 50


def test_top_up_is_not_offered_again_after_dedup():
	e = ac.TotalEstimate(
		25, 20, 30, "usd", "predicted", detail={"n": 45, "population": 500, "topped_up": True}
	)
	assert not ac.budget_check(e, 25).offer_top_up


def _tasks(costs, latencies, stops=None):
	from preflight.schema.record import AgentSample, AgentTask

	stops = stops or [None] * len(costs)
	tasks = [
		AgentTask(
			task_id=f"t{i}", turns=1, input_tokens=1, output_tokens=1, latency_s=w, cost_usd=c, stop_reason=s
		)
		for i, (c, w, s) in enumerate(zip(costs, latencies, stops, strict=True))
	]
	return AgentSample(tasks=tasks, population=500)


def test_a_tool_loop_that_makes_no_model_calls_runs_away_by_time():
	# Chiya, Oct 31: D's planted tool loop costs nothing more, so only its wall time shows it
	costs = [0.014] * 30
	walls = [20.0] * 28 + [600.0, 20.0]
	stops = [None] * 29 + ["timeout"]
	r = ac.agent_totals(_tasks(costs, walls, stops))["risk.runaway"]
	assert r.point == pytest.approx(2 / 30)
	rules = r.detail["by_rule"]
	assert rules["cost"]["runaways"] == 0 and rules["time"]["runaways"] == 2
	assert rules["time"]["hi"] == pytest.approx(ac.cp_upper(2, 30, 0.95))
	assert "by time: 2 of 30" in r.reason and r.lo <= r.point <= r.hi


def test_a_task_that_runs_away_both_ways_counts_once():
	costs = [0.01] * 29 + [1.0]
	walls = [10.0] * 29 + [500.0]
	r = ac.agent_totals(_tasks(costs, walls))["risk.runaway"]
	assert r.point == pytest.approx(1 / 30) and r.detail["by_rule"]["cost"]["runaways"] == 1


def test_no_runaways_either_way_keeps_the_upper_end_from_the_sample_size():
	r = ac.agent_totals(_tasks([0.014] * 30, [20.0] * 30))["risk.runaway"]
	assert r.point == 0 and r.hi == pytest.approx(ac.cp_upper(0, 30, 0.95))
	assert "none of 30" in r.reason and "by time: 0 of 30" in r.reason


def test_the_time_factor_reaches_the_time_rule_alone():
	# review, Oct 31: time_factor once reached estimate_total, which has no such parameter
	walls = [20.0] * 25 + [90.0] * 5
	r = ac.agent_totals(_tasks([0.014] * 30, walls), time_factor=4.0)["risk.runaway"]
	assert r.detail["by_rule"]["time"]["runaways"] == 5


def test_a_capped_task_runs_away_and_a_context_overflow_does_not():
	stops = [None] * 27 + ["max_turns", "context_overflow", "loop"]
	r = ac.agent_totals(_tasks([0.014] * 30, [20.0] * 30, stops))["risk.runaway"]
	assert r.detail["by_rule"]["time"]["runaways"] == 2


def test_with_unpriced_tasks_the_first_rule_goes_by_tokens():
	sample = _tasks([None] * 25 + [0.01] * 5, [20.0] * 30)
	r = ac.agent_totals(sample)["risk.runaway"]
	assert "by tokens: 0 of 30" in r.reason and r.status != "abstain"


def test_short_tasks_dont_run_away_by_time_below_the_floor():
	walls = [0.002] * 27 + [0.05] * 3  # 25x the median, but under a second
	r = ac.agent_totals(_tasks([0.014] * 30, walls))["risk.runaway"]
	assert r.detail["by_rule"]["time"]["runaways"] == 0


def test_a_what_if_turn_cap_isnt_a_runaway():
	stops = [None] * 28 + ["max_turns", "max_turns"]
	sample = _tasks([0.014] * 30, [20.0] * 30, stops)
	assert ac.agent_totals(sample)["risk.runaway"].detail["by_rule"]["time"]["runaways"] == 2
	capped = ac.agent_totals(sample, capped_ids=frozenset({"t28"}))["risk.runaway"]
	assert capped.detail["by_rule"]["time"]["runaways"] == 1


def _stalled(n_open: int, n_done: int, workers: int | None):
	"""A stalled run (contract 0.6.0) with n_open tasks left waiting on a tool; `concurrency` lands in 0.6.1."""
	from types import SimpleNamespace

	s = _tasks([0.014] * (n_open + n_done), [20.0] * (n_open + n_done), [None] * n_open + ["done"] * n_done)
	if workers is None:
		return s.model_copy(update={"ended": "stalled"})
	return SimpleNamespace(
		tasks=s.tasks, population=500, selection=None, ended="stalled", concurrency=workers
	)


def test_a_stalled_run_counts_at_most_one_possibly_stuck_task_per_worker():
	# Chiya, Nov 1: tonight's re-sample has 24 tasks with no stop reason and 4 workers
	r = ac.agent_totals(_stalled(24, 6, 4))["risk.runaway"]
	assert r.point == pytest.approx(4 / 30)
	stalled = r.detail["by_rule"]["stalled"]
	assert stalled["runaways"] == 4 and stalled["open"] == 24 and stalled["workers"] == 4
	assert "stalled with 24 tasks" in r.reason and "one per worker (4)" in r.reason


def test_a_stalled_run_without_a_worker_count_counts_one():
	r = ac.agent_totals(_stalled(24, 6, None))["risk.runaway"]
	assert r.point == pytest.approx(1 / 30) and "isn't recorded" in r.reason


def test_fewer_open_tasks_than_workers_counts_them_all_and_flagged_ones_once():
	assert ac.agent_totals(_stalled(2, 28, 4))["risk.runaway"].point == pytest.approx(2 / 30)
	s = _stalled(3, 27, 2)
	s.tasks[0] = s.tasks[0].model_copy(update={"latency_s": 900.0})  # open, and over 8x the median wall time
	r = ac.agent_totals(s)["risk.runaway"]
	# review, Nov 1: the flagged open task holds one of the 2 workers, so only 1 more counts
	assert r.detail["by_rule"]["time"]["runaways"] == 1 and r.detail["by_rule"]["stalled"]["runaways"] == 1
	assert r.point == pytest.approx(2 / 30) and "1 of them already flagged by the rules above" in r.reason


def test_no_stop_reason_in_a_finished_run_isnt_possibly_stuck():
	s = _tasks([0.014] * 30, [20.0] * 30).model_copy(update={"ended": "finished"})
	r = ac.agent_totals(s)["risk.runaway"]
	assert r.point == 0 and "stalled" not in r.detail["by_rule"]


def test_a_what_if_capped_task_in_a_stalled_run_isnt_open():
	r = ac.agent_totals(_stalled(2, 28, 4), capped_ids=frozenset({"t0"}))["risk.runaway"]
	assert r.detail["by_rule"]["stalled"]["open"] == 1


def test_one_open_task_already_flagged_reads_plainly_without_a_worker_count():
	s = _stalled(1, 29, None)
	s.tasks[0] = s.tasks[0].model_copy(update={"latency_s": 900.0})
	r = ac.agent_totals(s)["risk.runaway"]
	assert r.point == pytest.approx(1 / 30)
	assert "with 1 task waiting on a tool, possibly stuck" in r.reason and "so 0 more counted" in r.reason
	assert "1 of them already flagged" in r.reason and "each" not in r.reason.split("stalled")[1]


def test_a_recorded_worker_count_of_zero_isnt_called_missing():
	from types import SimpleNamespace

	s = _stalled(3, 27, None)
	z = SimpleNamespace(tasks=s.tasks, population=500, selection=None, ended="stalled", concurrency=0)
	r = ac.agent_totals(z)["risk.runaway"]
	assert r.point == pytest.approx(1 / 30) and "count is 0" in r.reason and "isn't recorded" not in r.reason
	assert "1 more counted as a runaway" in r.reason
