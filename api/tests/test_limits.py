"""The visitor's limits: refused server-side in the live guide, and read from plain words."""

import json

import pytest
from test_guide import SETTING, Model, client, live, reply, turn  # noqa: F401 (live is autouse)

from preflight_api.guide.record_view import RecordView
from preflight_api.routes.limits import checked
from preflight_api.stores import SampleStore

A100 = "A100-SXM-80GB"


def lora() -> RecordView:
	from preflight_api.app import REPO_ROOT

	return RecordView.from_record(SampleStore(REPO_ROOT).get("lora-7b"))


def test_a_move_outside_the_limits_is_refused_with_why_and_the_knob_leaves_set_knob():
	limits = {"allowed": {"gpu_type": [A100]}, "excluded": {"batch_size": [8]}}
	model = Model(
		reply(calls=[("set_knob", {"knob_id": "batch_size", "value": 8, "why": "cheaper"})]),
		reply(text="Batch size 8 is ruled out; nothing else inside your limits helps."),
	)
	with client(model) as c:
		r = turn(c, start=True, setting={**SETTING, "limits": limits}).json()
		sent = model.requests[0]
		set_knob = next(t for t in sent["tools"] if t["name"] == "set_knob")
		# A GPU limited to the one it's on has nowhere to go, so the model can't even name it.
		assert "gpu_type" not in set_knob["parameters"]["properties"]["knob_id"]["enum"]
		assert '"allowed": {"gpu_type": ["A100-SXM-80GB"]}' in sent["input"][-1]["content"]
		assert r["calls"] == []  # the refused move never reaches the browser
		refused = [
			json.loads(i["output"])
			for i in model.requests[1]["input"]
			if i.get("type") == "function_call_output"
		]
		assert refused[0]["refused"] == "outside_limits" and "limits" in refused[0]["hint"]


def test_limits_name_only_knobs_and_values_the_record_has():
	bad = {**SETTING, "limits": {"allowed": {"gpu_type": ["TPU-v9"]}}}
	with client(Model()) as c:
		assert turn(c, start=True, setting=bad).status_code == 422
		assert (
			turn(c, start=True, setting={**SETTING, "limits": {"excluded": {"nope": [1]}}}).status_code == 422
		)


def test_not_this_tells_the_guide_to_offer_another_lever():
	model = Model(
		reply(calls=[("set_knob", {"knob_id": "batch_size", "value": 8, "why": "x"})]),
		reply(text="Batch 8 is cheaper."),
		reply(text="Next lever instead."),
	)
	with client(model) as c:
		r1 = turn(c, start=True).json()
		moved = {
			"tool": "set_knob",
			"call_id": r1["calls"][0]["call_id"],
			"moved": True,
			"knob_id": "batch_size",
			"from_value": 4,
			"to_value": 8,
			"readouts": {},
			"refused": None,
		}
		r2 = turn(c, state=r1["state"], results=[moved]).json()
		assert r2["awaiting"] == "decision"
		r3 = turn(
			c,
			state=r2["state"],
			decision={"knob_id": "batch_size", "action": "declined", "value": 8},
			setting={**SETTING, "limits": {"excluded": {"batch_size": [8]}}},
		)
		assert r3.status_code == 200, r3.text
		said = json.dumps(model.requests[-1]["input"])
		assert "not this" in said and "next-best lever" in said


def test_limits_from_plain_words_keep_only_what_the_record_and_the_text_back():
	rec = lora()
	text = "must stay on A100, under $40, can't touch the dataset"
	out = checked(
		{
			"allowed": [
				{"knob_id": "gpu_type", "values": [A100, "TPU-v9"]},
				{"knob_id": "nope", "values": [1]},
			],
			"keep": ["packing"],
			"budget_usd": 40,
			"deadline_hours": 6,  # not in the text: dropped, never invented
			"gpu_hours": None,
			"not_understood": ["can't touch the dataset"],
		},
		rec,
		text,
	)
	assert out["limits"] == [
		{"kind": "allowed", "knob_id": "gpu_type", "values": [A100]},
		{"kind": "keep", "knob_id": "packing"},
		{"kind": "budget_usd", "value": 40.0},
	]
	assert out["dropped"] == ["deadline_hours"]
	assert out["not_understood"] == ["can't touch the dataset"]
	assert checked({"deadline_hours": 6}, rec, "done in 6 hours")["limits"] == [
		{"kind": "deadline_s", "value": 21600.0}
	]


def test_the_limits_reader_calls_muse_spark_once_with_the_records_knobs(monkeypatch):
	args = {
		"allowed": [{"knob_id": "gpu_type", "values": [A100]}],
		"keep": [],
		"budget_usd": 40,
		"deadline_hours": None,
		"gpu_hours": None,
		"not_understood": ["can't touch the dataset"],
	}
	model = Model(reply(calls=[("set_limits", args)], usage=(900, 80)))
	with client(model) as c:
		r = c.post(
			"/v1/guide/limits",
			json={"record": "lora-7b", "text": "must stay on A100, under $40, can't touch the dataset"},
		)
		assert r.status_code == 200, r.text
		body = r.json()
		assert body["limits"][0] == {"kind": "allowed", "knob_id": "gpu_type", "values": [A100]}
		assert {"kind": "budget_usd", "value": 40.0} in body["limits"]
		assert body["not_understood"] == ["can't touch the dataset"]
		sent = model.requests[0]
		assert [t["name"] for t in sent["tools"]] == ["set_limits"]
		assert "gpu_type: one of" in sent["instructions"] and A100 in sent["instructions"]
		assert body["provenance"]["model"] == "muse-spark-1.3"
		assert c.post("/v1/guide/limits", json={"record": "lora-7b", "text": "x" * 301}).status_code == 422


def test_the_limits_reader_is_off_with_the_live_guide(monkeypatch):
	from preflight_api import flags

	monkeypatch.setenv("PREFLIGHT_FLAGS", json.dumps({"c.guide.live": False}))
	flags.reload()
	with client(Model()) as c:
		r = c.post("/v1/guide/limits", json={"record": "lora-7b", "text": "A100 only"})
		assert r.status_code == 503 and r.json()["error"]["code"] == "limits_reader_off"


@pytest.mark.parametrize("text,expect", [("$40", [40.0]), ("1,200 GPU-hours", [1200.0]), ("2k", [2000.0])])
def test_numbers_in_the_visitors_words(text, expect):
	from preflight_api.routes.limits import _numbers_in

	assert _numbers_in(text) == expect
