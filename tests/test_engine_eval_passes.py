"""Eval runs by command (`engine.eval_passes`, off; Chiya, Nov 3): scoring passes x per-pass time from A's eval probe."""

import types

import pytest

from preflight.engine import api
from preflight.schema.record import Probe

EXTRAS = {"eval_forward_batches": 8.0, "eval_forward_s_p50": 0.5, "eval_forward_rows_mean": 64.0}


def _record(**extra):
	p = Probe.model_validate(
		{
			"id": "p1",
			"status": "ok",
			"knobs": {},
			"steps": 0,
			"duration_s": 60.0,
			"extras": {**EXTRAS, **extra},
		}
	)
	return types.SimpleNamespace(probes=[p])


def test_off_by_default_and_in_flags_json():
	assert api.FEATURES["eval_passes"] is False and api.FLAG_OF["eval_passes"] == "engine.eval_passes"


def test_passes_times_the_per_pass_time_plus_the_time_before_the_first_pass():
	e = api._eval_passes_total(_record(eval_requests_total=6400.0))
	# 100 passes x 0.5 s, plus 60 s less the 8 measured and 2 warm-up batches (5 s)
	assert e.status == "predicted" and e.point == pytest.approx(100 * 0.5 + 55.0)
	assert e.lo < e.point < e.hi and "100 scoring passes" in e.reason


def test_abstains_without_the_eval_s_request_count():
	e = api._eval_passes_total(_record())
	assert e.status == "abstain" and "how many requests" in e.reason


def test_abstains_without_an_eval_probe():
	rec = types.SimpleNamespace(
		probes=[Probe.model_validate({"id": "p1", "status": "ok", "knobs": {}, "steps": 0})]
	)
	assert api._eval_passes_total(rec).status == "abstain"
