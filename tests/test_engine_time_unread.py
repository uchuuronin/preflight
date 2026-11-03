"""Knobs the step-time model never reads (api.STEP_TIME_UNREAD; B, Nov 3): moved off every probe, time says it
isn't modelled for them and holds the probed effect, wider; memory still follows them."""

from pathlib import Path

from preflight.engine import api

ZOO = Path(__file__).resolve().parent / "fixtures" / "validation" / "zoo_gt"


def _rec(name):
	from preflight.schema import FlightRecord

	return FlightRecord.model_validate_json((ZOO / name).read_text())


def test_a_model_shape_move_is_said_unmodelled_for_time_and_still_moves_memory():
	rec = _rec("pretrain_nanogpt.record.json")
	n = api._current(rec)["n_layer"]
	base, moved = api.predict(rec).root, api.predict(rec, {"n_layer": n * 2}).root
	t = moved["time.step"]
	assert t.status == "extrapolated" and "n_layer isn't modelled for time yet" in t.reason
	assert "the model carries it there" not in t.reason
	assert (t.hi - t.lo) / t.point > (base["time.step"].hi - base["time.step"].lo) / base["time.step"].point
	assert moved["memory.peak"].point > base["memory.peak"].point


def test_a_knob_the_step_reads_is_unchanged():
	rec = _rec("pretrain_nanogpt.record.json")
	b = api._current(rec)["batch_size"]
	t = api.predict(rec, {"batch_size": b * 2}).root["time.step"]
	assert "isn't modelled for time" not in (t.reason or "")


def test_a_gnn_s_fanout_is_on_the_work_axis_not_unread():
	assert not any(k.startswith("fanout.") for k in api.STEP_TIME_UNREAD)
	from preflight.engine import knobs

	assert knobs.work_units("gnn", {"batch_size": 8, "fanout.hop1": 10}) != knobs.work_units(
		"gnn", {"batch_size": 8, "fanout.hop1": 20}
	)
