"""Campaign planner (campaign/planner.py): its numbers are the engine's, scheduled, segmented and dated."""

import itertools
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from preflight.campaign import planner as P
from preflight.schema import FlightRecord, load_record

EX = Path(__file__).resolve().parents[1] / "schema" / "examples"
START = datetime(2026, 11, 1, tzinfo=UTC)


@pytest.fixture(scope="module")
def ft():
	return load_record((EX / "finetune.json").read_text())


@pytest.fixture(scope="module")
def pt():
	return load_record((EX / "pretrain.json").read_text())


def _with(rec: FlightRecord, **constraints) -> FlightRecord:
	c = rec.job.constraints.model_copy(update=constraints)
	return rec.model_copy(update={"job": rec.job.model_copy(update={"constraints": c})})


def _with_save(rec: FlightRecord, seconds: float, nbytes: int | None = None) -> FlightRecord:
	from preflight.schema.record import CheckpointStats

	probes = [
		p.model_copy(update={"checkpoint": CheckpointStats(saves=2, seconds_mean=seconds, bytes_mean=nbytes)})
		for p in rec.probes
	]
	return rec.model_copy(update={"probes": probes})


# ---------------------------------------------------------------- closed forms


def test_daly_interval_starts_from_young_and_caps_at_the_interruption_interval():
	young = math.sqrt(2 * 60 * 86400)
	r = math.sqrt(60 / (2 * 86400))
	assert P.daly_interval(60, 86400) == pytest.approx(young * (1 + r / 3 + r * r / 9) - 60)
	assert P.daly_interval(60, 86400) == pytest.approx(young, rel=0.02)  # small saves: Young's term dominates
	assert P.daly_interval(5000, 2000) == 2000  # a save longer than twice M: checkpoint every M


def test_youngs_interval_minimises_the_first_order_waste_and_daly_sits_beside_it():
	d, m = 120.0, 6 * 3600.0
	taus = [d * 1.01**k for k in range(1, 700)]
	best = min(taus, key=lambda t: P.waste_fraction(t, d, m, 60))
	assert best == pytest.approx(math.sqrt(2 * d * m), rel=0.01)
	assert P.daly_interval(d, m) == pytest.approx(best, rel=0.05)


def test_interruptions_without_checkpoints_grow_exponentially():
	import numpy as np

	rng = np.random.default_rng(0)
	m, r, t = 3600.0, 60.0, 2 * 3600.0
	lost = P.redone_seconds(rng, np.full(20000, t), None, m, r)
	# restart from zero: expected wall time (M + R)(e^(T/M) - 1), so the loss is that minus T
	assert lost.mean() == pytest.approx((m + r) * math.expm1(t / m) - t, rel=0.05)
	# with checkpoints every 10 min, the same run loses only about T/M x (half an interval + restart)
	small = P.redone_seconds(rng, np.full(20000, t), 600.0, m, r)
	assert small.mean() == pytest.approx(t / m * (300 + r), rel=0.15)
	assert np.isinf(P.redone_seconds(rng, np.full(4, 40 * m), None, m, r)).all()  # it never finishes


def test_segments_bank_only_whole_checkpoint_intervals():
	# 10 h of work, 4 h limit, 60 s restart, a save of 60 s every 30 min: 7 intervals (3.5 h) banked per job
	assert P.segments_needed(36000, 4 * 3600, 1800, 60, 60) == 3
	assert P.segments_needed(3 * 3600, 4 * 3600, 1800, 60, 60) == 1
	assert P.segments_needed(36000, 4 * 3600, 36000, 0, 60) is None  # no checkpoint inside the limit


def test_asha_rungs_and_steps():
	rungs = P.asha_rungs(27, 1, 3, 27)
	assert [(r.steps, r.runs) for r in rungs] == [(1, 27), (3, 9), (9, 3), (27, 1)]
	assert P.asha_steps(rungs) == 27 * 1 + 9 * 2 + 3 * 6 + 1 * 18
	# a lone survivor goes straight to the full length
	assert [(r.steps, r.runs) for r in P.asha_rungs(4, 10, 3, 1000)] == [(10, 4), (1000, 1)]
	with pytest.raises(ValueError):
		P.asha_rungs(4, 10, 1, 1000)


# ---------------------------------------------------------------- the plan


def test_every_cell_is_the_engine_and_the_totals_add_its_intervals(ft):
	from preflight.engine import api

	pl = P.plan(ft, {"batch_size": [1, 2], "seed": [0, 1, 2]}, start=START)
	assert len(pl.cells) == 6
	for c in pl.cells:
		e = api.predict(ft, {"batch_size": c.knobs["batch_size"]}).root["time.total"]
		assert (c.time.point, c.time.lo, c.time.hi) == (e.point, e.lo, e.hi)
	assert pl.usd.point == pytest.approx(sum(c.usd.point for c in pl.cells))
	assert pl.usd.lo == pytest.approx(sum(c.usd.lo for c in pl.cells))
	assert pl.usd.hi == pytest.approx(sum(c.usd.hi for c in pl.cells))
	assert "seed isn't" in pl.notes[0]  # an axis the engine doesn't model still multiplies the runs
	assert "exactly as accurate as the engine" in pl.card.notes and pl.card.kind == "simulation"


def test_one_slot_and_no_interruptions_end_at_the_sum_of_the_runs(ft):
	pl = P.plan(ft, {"batch_size": [1, 2, 4]}, P.Cluster(slots=1), start=START, draws=20000)
	runs = [c.time for c in pl.cells]
	assert pl.finish_s.point == pytest.approx(sum(t.point for t in runs), rel=0.01)
	# the runs err together, so the end date's 5% and 95% are the runs' band edges added up
	assert pl.finish_s.lo == pytest.approx(sum(t.lo for t in runs), rel=0.02)
	assert pl.finish_s.hi == pytest.approx(sum(t.hi for t in runs), rel=0.02)
	assert pl.completion["p10"] < pl.completion["p50"] < pl.completion["p90"]
	assert pl.completion["p50"] == START + timedelta(seconds=pl.finish_s.point)


def test_lanes_split_the_runs_and_more_slots_finish_sooner(ft):
	grid = {"batch_size": [1, 2], "seed": [0, 1]}
	one = P.plan(ft, grid, P.Cluster(slots=1), start=START)
	two = P.plan(ft, grid, P.Cluster(lanes={"ana": 1, "ben": 1}), start=START)
	assert {c.lane for c in two.cells} == {"ana", "ben"} and two.lanes == ("ana", "ben")
	assert two.finish_s.point < 0.6 * one.finish_s.point


def test_interruptions_add_redone_work_and_resubmits_add_waits(ft):
	grid = {"batch_size": [2], "seed": [0, 1]}
	calm = P.plan(ft, grid, start=START)
	stormy = P.plan(ft, grid, P.Cluster(mtti_s=1800, checkpoint_interval_s=600), start=START)
	assert stormy.finish_s.point > calm.finish_s.point
	# about T / M interruptions per run, each redoing half an interval plus the restart
	t = sum(c.time.point for c in calm.cells)
	expected = t / 1800 * (600 / 2 + P.RESTART_S)
	assert stormy.finish_s.point - calm.finish_s.point == pytest.approx(expected, rel=0.25)
	capped = P.plan(
		ft, grid, P.Cluster(wall_time_cap_s=1800, checkpoint_interval_s=300, queue_wait_s=900), start=START
	)
	assert all(c.segments and c.segments > 1 for c in capped.cells)
	assert capped.finish_s.point > calm.finish_s.point + 900


def test_a_run_that_cant_checkpoint_inside_the_limit_says_so(ft):
	# the fixture saves every few hundred steps; a 3 h interval under a 10 min limit never banks any work
	pl = P.plan(
		ft, {"batch_size": [2]}, P.Cluster(wall_time_cap_s=600, checkpoint_interval_s=3 * 3600), start=START
	)
	assert pl.cells[0].segments is None and "10 min job limit without checkpoints" in pl.cells[0].reason
	assert pl.finish_s.status == "abstain" and not pl.completion  # no date for a run that can't end


def test_young_daly_needs_a_measured_save_and_an_interruption_rate(ft):
	assert P.plan(ft, {"batch_size": [2]}, start=START).checkpoint_interval.status == "abstain"
	rec = _with_save(ft, 45.0)
	pl = P.plan(rec, {"batch_size": [2]}, P.Cluster(mtti_s=8 * 3600), start=START)
	assert pl.checkpoint_interval.point == pytest.approx(P.daly_interval(45.0, 8 * 3600))
	assert "Young/Daly" in pl.checkpoint_interval.reason
	assert pl.to_campaign().checkpoint_interval_s == pl.checkpoint_interval.point


def test_the_budget_cuts_the_dearest_runs_and_they_are_not_scheduled(ft):
	full = P.plan(ft, {"batch_size": [1, 2], "seed": [0, 1]}, start=START)
	budget = full.usd.hi * 0.6
	pl = P.plan(ft, {"batch_size": [1, 2], "seed": [0, 1]}, P.Cluster(budget_usd=budget), start=START)
	kept = [c for c in pl.cells if not c.cut]
	assert pl.cut_list and sum(c.usd.hi for c in kept) <= budget
	assert (
		pl.usd.hi == pytest.approx(sum(c.usd.hi for c in kept)) and pl.usd.hi <= budget
	)  # totals are what runs
	assert pl.grid_usd.hi == pytest.approx(full.usd.hi)  # the whole grid is still reported beside it
	assert all(c.lane is None for c in pl.cells if c.cut)
	assert max(c.usd.hi for c in kept) <= min(c.usd.hi for c in pl.cells if c.cut)
	assert pl.finish_s.point < full.finish_s.point


def test_the_quota_check_counts_every_run_going_at_once(ft):
	rec = _with_save(ft, 30.0, nbytes=4 * 2**30)
	ok = P.plan(
		rec,
		{"batch_size": [2], "seed": [0, 1, 2]},
		P.Cluster(slots=3, disk_quota_bytes=30 * 2**30),
		start=START,
	)
	assert ok.quota.needed_bytes.point == 3 * 2 * 4 * 2**30 and ok.quota.fits
	tight = P.plan(
		rec,
		{"batch_size": [2], "seed": [0, 1, 2]},
		P.Cluster(slots=3, disk_quota_bytes=20 * 2**30),
		start=START,
	)
	assert tight.quota.fits is False and "won't fit" in tight.quota.reason
	# no probe saved one: the memory model's weights and optimizer state, with their band
	pred = P.plan(ft, {"batch_size": [2]}, start=START).quota.needed_bytes
	assert pred.status == "predicted" and pred.lo < pred.point < pred.hi


def test_asha_saves_against_the_full_grid(ft):
	pl = P.plan(ft, {"batch_size": [2], "seed": list(range(9))}, asha=P.Asha(min_steps=100), start=START)
	assert pl.asha.rungs[0].runs == 9 and pl.asha.rungs[-1].runs == 1
	assert pl.asha.gpu_hours.point + pl.asha.saved_gpu_hours.point == pytest.approx(pl.gpu_hours.point)
	assert 0 < pl.asha.gpu_hours.point < pl.gpu_hours.point


def test_deadline_probability_and_the_record_block(ft):
	rec = _with(ft, deadline=START + timedelta(days=30))
	pl = P.plan(rec, {"batch_size": [2]}, start=START)
	assert pl.p_by_deadline == 1.0
	soon = P.plan(_with(ft, deadline=START + timedelta(seconds=60)), {"batch_size": [2]}, start=START)
	assert soon.p_by_deadline == 0.0
	block = pl.to_campaign()
	FlightRecord.model_validate(ft.model_copy(update={"campaign": block}).model_dump())
	assert set(block.completion_quantiles) == {"p10", "p50", "p90"}


def test_a_run_the_engine_cant_time_leaves_the_campaign_undated(ft):
	pl = P.plan(ft, {"gpu_type": ["L4", "no-such-gpu-xyz"]}, start=START)
	assert pl.finish_s.status == "abstain" and pl.usd.status == "abstain"
	assert any("no run time" in n for n in pl.notes)


def test_grids_are_bounded(ft):
	with pytest.raises(ValueError):
		P.plan(ft, {"seed": list(range(P.MAX_CELLS + 1))}, start=START)


def test_job_limits_charge_each_killed_jobs_unsaved_tail(ft):
	# 1 h limit, 25 min interval, no measured save: each job banks 2 intervals (50 min) and wastes the last 10
	pl = P.plan(
		ft, {"batch_size": [2]}, P.Cluster(wall_time_cap_s=3600, checkpoint_interval_s=1500), start=START
	)
	c = pl.cells[0]
	assert c.segments == math.ceil(c.time.hi / 3000)
	assert pl.finish_s.point == pytest.approx(c.time.point + (c.segments - 1) * 600, rel=0.02)
	assert "redone after interruptions and job limits" in c.usd.reason  # the GPUs bill those tails too


def test_a_lora_checkpoint_is_the_adapters_not_the_base(ft):
	from preflight.engine import api

	ck = P._checkpoint_bytes(ft, {})
	assert ck.point < 0.5 * api.predict(ft).root["memory.weights"].point  # the frozen base isn't saved
	assert "adapters" in ck.reason


def test_bad_inputs_become_notes_not_crashes(ft):
	pl = P.plan(ft, {"batch_size": [2], "seed": [0, 1]}, asha=P.Asha(min_steps=10**9), start=START)
	assert pl.asha is None and any("first rung" in n for n in pl.notes)
	naive = _with(ft, deadline=datetime(2026, 12, 1))  # no time zone, which the schema accepts
	pl = P.plan(naive, {"batch_size": [2]}, start=START)
	assert pl.p_by_deadline == 1.0 and any("read as UTC" in n for n in pl.notes)
	pl = P.plan(
		ft, {"batch_size": [2], "seed": [0, 1, 2]}, P.Cluster(lanes={"ana": 2, "ben": 0}), start=START
	)
	assert {c.lane for c in pl.cells} == {"ana"} and any("no slots" in n for n in pl.notes)
	with pytest.raises(ValueError):
		P.plan(ft, {"batch_size": [2]}, P.Cluster(lanes={"ana": 0}), start=START)


def test_each_run_has_a_slot_and_a_start_for_the_calendar(ft):
	pl = P.plan(ft, {"batch_size": [2], "seed": [0, 1, 2, 3]}, P.Cluster(slots=2), start=START)
	assert sorted(c.slot for c in pl.cells) == [0, 0, 1, 1]
	firsts = [c for c in pl.cells if c.start_s.point == 0]
	assert len(firsts) == 2  # one run starts at once on each slot
	later = [c for c in pl.cells if c.start_s.point > 0]
	assert all(c.start > START for c in later) and all(
		c.start_s.lo <= c.start_s.point <= c.start_s.hi for c in later
	)


def test_runs_on_a_slot_start_in_order_and_cut_runs_get_no_slot(ft):
	pl = P.plan(ft, {"batch_size": [2], "seed": [0, 1, 2, 3, 4, 5]}, P.Cluster(slots=2), start=START)
	for k in (0, 1):
		on = sorted((c for c in pl.cells if c.slot == k), key=lambda c: c.start_s.point)
		assert on[0].start_s.point == 0
		for a, b in itertools.pairwise(on):  # the next starts once the one before it has run
			assert b.start_s.point > a.start_s.point and b.start_s.lo >= a.start_s.lo
	cut = P.plan(ft, {"batch_size": [2], "seed": [0, 1, 2]}, P.Cluster(slots=1, budget_usd=1e-9), start=START)
	assert all(c.slot is None and c.start_s is None for c in cut.cells if c.cut)
