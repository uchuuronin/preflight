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


def test_a_fanout_move_is_carried_by_the_model_and_a_packing_move_is_said_once():
	rec = _rec("gnn_pyg_a10.record.json")
	cur = api._current(rec)
	fan = next((k for k in cur if k.startswith("fanout.")), None)
	if fan is not None:
		t = api.predict(rec, {fan: cur[fan] * 2}).root["time.step"]
		assert "isn't modelled for time" not in (t.reason or "")
	vis = _rec("vision_timm_a10.record.json")
	if "channels_last" in api._current(vis):
		cl = api._current(vis)["channels_last"]
		r = api.predict(vis, {"channels_last": not cl}).root["time.step"].reason or ""
		assert "isn't modelled for time yet" in r and "channels_last is off the probed values" not in r


def test_checkpointing_named_none_adds_no_recompute():
	assert (
		not api._checkpointing_on("none")
		and not api._checkpointing_on(False)
		and not api._checkpointing_on(None)
	)
	assert (
		api._checkpointing_on("full") and api._checkpointing_on("selective") and api._checkpointing_on(True)
	)
	assert (
		not api._checkpointing_on("NONE")
		and not api._checkpointing_on(0)
		and api._checkpointing_on("unsloth")
	)
