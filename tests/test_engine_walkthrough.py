"""Engine items from D's stranger walkthrough of main's production build (from-d.md, 9018a58; Nov 3)."""

from pathlib import Path

from preflight.engine import api
from preflight.schema import FlightRecord


def test_shape_knobs_are_unmodelled_on_a_script_map_with_no_layer_shape():
	# D's walkthrough, Nov 3: the 7B sample's map gives only a parameter count, so lora_r and eval_no_grad had no
	# analytical effect and a move read as "no change"; it abstains on memory now, as for any unmodelled knob
	sample = (
		Path(__file__).resolve().parents[1]
		/ "web"
		/ "src"
		/ "workspace"
		/ "fixtures"
		/ "dev-finetune.record.json"
	)
	rec = FlightRecord.model_validate_json(sample.read_text())
	assert api._generic_shape(rec)
	for move in ({"lora_r": 32}, {"eval_no_grad": True}):
		m = api.predict(rec, move).root["memory.peak"]
		assert m.status == "abstain" and "doesn't model" in m.reason
	assert (
		api.predict(rec, {"precision": "fp16"}).root["memory.peak"].status != "abstain"
	)  # same bytes: no change
	qwen = (
		Path(__file__).resolve().parents[1]
		/ "web"
		/ "src"
		/ "workspace"
		/ "fixtures"
		/ "lora-live-nov1.record.json"
	)
	assert not api._generic_shape(FlightRecord.model_validate_json(qwen.read_text()))


def test_a_seq_len_past_the_scans_cut_is_extrapolated_not_a_silent_no_change():
	# C, Nov 3 (D's item 6): PF102 at seq_len 2,048 read the same time as at 1,024; its rows were scanned at 1,024
	sample = (
		Path(__file__).resolve().parents[1]
		/ "web"
		/ "src"
		/ "workspace"
		/ "fixtures"
		/ "lora-live-nov1.record.json"
	)
	rec = FlightRecord.model_validate_json(sample.read_text())
	base = api.predict(rec).root["time.step"]
	long = api.predict(rec, {"seq_len": 2048}).root["time.step"]
	assert base.status == "predicted" and "scanned at" not in (base.reason or "")
	assert long.status == "extrapolated" and "aren't measured" in long.reason
	assert api.predict(rec, {"seq_len": 512}).root["time.step"].status == "predicted"
