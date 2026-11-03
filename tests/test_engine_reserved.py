"""The reserved-pool margin (`engine.memory_reserved`, off; Chiya, Nov 3): fitted on the corpus, it raises a memory
prediction only where the probes' own reserved-over-allocated ratio is below the corpus's for that family and size."""

from pathlib import Path

import pytest

from preflight.engine import api
from preflight.engine._reserved_margin import MARGINS
from preflight.schema import FlightRecord

ZOO = Path(__file__).resolve().parent / "fixtures" / "validation" / "zoo_gt"


def _record(name: str) -> FlightRecord:
	return FlightRecord.model_validate_json((ZOO / name).read_text())


def _peak(rec: FlightRecord, on: bool):
	token = api._FEATURES_NOW.set({**api.FEATURES, "memory_reserved": on})
	try:
		return api.predict(rec).root["memory.peak"]
	finally:
		api._FEATURES_NOW.reset(token)


def test_off_by_default_and_in_flags_json():
	assert api.FEATURES["memory_reserved"] is False
	assert api.FLAG_OF["memory_reserved"] == "engine.memory_reserved"
	assert (
		'"engine.memory_reserved": false' in (Path(__file__).resolve().parents[1] / "flags.json").read_text()
	)


def test_the_margin_comes_from_the_corpus_calibration_split():
	assert "calibration split" in MARGINS["$comment"]
	for cell in [*MARGINS["by_size"].values(), *MARGINS["by_family"].values()]:
		assert 1.0 <= cell["median"] <= cell["q90"] and cell["n"] >= 20


def test_a_probe_below_the_margin_is_raised_and_says_why():
	rec = _record("vision_timm_a10.record.json")  # probe ratio 1.08, under the CNN median 1.14
	off, on = _peak(rec, False), _peak(rec, True)
	assert on.point > off.point and on.hi >= off.hi and on.lo == pytest.approx(off.lo)
	assert "reserved pool" in on.reason and "reserved pool" not in off.reason


def test_a_probe_already_above_the_margin_is_unchanged():
	rec = _record("gnn_pyg_a10.record.json")  # probe ratio 1.81, above every family's 90th percentile
	off, on = _peak(rec, False), _peak(rec, True)
	assert (on.point, on.hi, on.reason) == (off.point, off.hi, off.reason)
