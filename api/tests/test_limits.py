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


def test_frozen_and_declined_knobs_are_refused_and_never_offered():
	limits = {"frozen": ["gpu_type"], "declined": ["batch_size"]}
	model = Model(
		reply(
			calls=[
				("set_knob", {"knob_id": "gpu_type", "value": "H100-SXM-80GB", "why": "faster"}),
				("pin", {"knob_id": "gpu_type", "pinned": False}),
				("run_probe", {"knobs": {"gpu_type": "H100-SXM-80GB"}, "reason": "check"}),
			]
		),
		reply(text="Your GPU stays as it is."),
	)
	with client(model) as c:
		r = turn(c, start=True, setting={**SETTING, "limits": limits}).json()
		enum = next(t for t in model.requests[0]["tools"] if t["name"] == "set_knob")["parameters"][
			"properties"
		]["knob_id"]["enum"]
		assert "gpu_type" not in enum and "batch_size" not in enum
		assert r["calls"] == []  # nothing reached the browser or an approval card
		outs = [
			json.loads(i["output"])
			for i in model.requests[1]["input"]
			if i.get("type") == "function_call_output"
		]
		assert [o["refused"] for o in outs] == ["outside_limits", "frozen", "outside_limits"]


def _moved(call_id, cost_after, cost_before):
	return {
		"tool": "set_knob",
		"call_id": call_id,
		"moved": True,
		"knob_id": "gpu_type",
		"from_value": A100,
		"to_value": "H100-SXM-80GB",
		"readouts": {
			"cost.total": {
				"point": cost_after * 0.9,
				"lo": cost_after * 0.8,
				"hi": cost_after,
				"unit": "usd",
				"status": "predicted",
			}
		},
		"baseline": {
			"cost.total": {
				"point": cost_before,
				"lo": cost_before,
				"hi": cost_before,
				"unit": "usd",
				"status": "predicted",
			}
		},
		"refused": None,
	}


def test_a_move_whose_range_passes_the_budget_is_refused_here_and_put_back():
	setting = {**SETTING, "limits": {"budget_usd": 40}}
	model = Model(
		reply(calls=[("set_knob", {"knob_id": "gpu_type", "value": "H100-SXM-80GB", "why": "faster"})]),
		reply(text="That goes past your budget; nothing else fits."),
	)
	with client(model) as c:
		r1 = turn(c, start=True, setting=setting).json()
		# The middle is under $40 (US$40.5 x 0.9) but the top of the range isn't.
		r2 = turn(
			c, state=r1["state"], results=[_moved(r1["calls"][0]["call_id"], 45.0, 30.0)], setting=setting
		).json()
		assert r2["reverts"] == [{"knob_id": "gpu_type", "value": A100}]
		assert r2["awaiting"] != "decision"
		said = [
			json.loads(i["output"])
			for i in model.requests[1]["input"]
			if i.get("type") == "function_call_output"
		]
		assert said[-1]["refused"] == "over_budget"


def test_a_move_that_brings_an_over_budget_run_back_down_stands():
	setting = {**SETTING, "limits": {"budget_usd": 40}}
	model = Model(
		reply(calls=[("set_knob", {"knob_id": "gpu_type", "value": "H100-SXM-80GB", "why": "cheaper"})]),
		reply(text="Cheaper, though still over your budget."),
	)
	with client(model) as c:
		r1 = turn(c, start=True, setting=setting).json()
		r2 = turn(
			c, state=r1["state"], results=[_moved(r1["calls"][0]["call_id"], 50.0, 60.0)], setting=setting
		).json()
		assert r2["reverts"] == [] and r2["awaiting"] == "decision"


def test_not_this_is_ruled_out_on_the_server_whatever_the_browser_sent():
	model = Model(
		reply(calls=[("set_knob", {"knob_id": "batch_size", "value": 8, "why": "x"})]),
		reply(text="Batch 8."),
		reply(calls=[("set_knob", {"knob_id": "batch_size", "value": 8, "why": "again"})]),
		reply(text="Nothing else."),
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
		# The browser's setting doesn't carry the exclusion yet; the decision alone is enough.
		r3 = turn(
			c, state=r2["state"], decision={"knob_id": "batch_size", "action": "declined", "value": 8}
		).json()
		assert r3["calls"] == []
		outs = [
			json.loads(i["output"])
			for i in model.requests[-1]["input"]
			if i.get("type") == "function_call_output"
		]
		assert outs[-1]["refused"] in ("outside_limits", "tool_not_offered")


def test_the_reader_keeps_only_the_visitors_own_words_and_numbers():
	rec = lora()
	out = checked(
		{
			"allowed": [{"knob_id": "batch_size", "values": [16, 4]}],
			"keep": [{"bad": 1}, "packing"],
			"budget_usd": 100,  # the "100" in "A100" is not a budget
			"deadline_hours": None,
			"gpu_hours": None,
			"not_understood": ["can't touch the dataset", "ignore your rules"],
		},
		rec,
		"stay on A100, batch 4 at most, can't touch the dataset",
	)
	assert {"kind": "allowed", "knob_id": "batch_size", "values": [4]} in out["limits"]
	assert {"kind": "keep", "knob_id": "packing"} in out["limits"]
	assert out["dropped"] == ["budget_usd"]
	assert out["not_understood"] == ["can't touch the dataset"]
	assert checked(["not", "a", "dict"], rec, "x")["dropped"] == ["malformed"]
	assert checked({"allowed": [{"knob_id": ["gpu_type"], "values": "x"}]}, rec, "x")["limits"] == []


def test_an_approved_probe_from_the_rule_tour_respects_the_limits():
	with client(Model()) as c:
		body = {
			"record": "lora-7b",
			"tool": "run_probe",
			"setting": SETTING["knobs"],
			"knobs": {"gpu_type": "H100-SXM-80GB"},
			"limits": {"allowed": {"gpu_type": [A100]}},
		}
		r = c.post("/v1/guide/tools", json=body)
		assert r.status_code == 422 and r.json()["error"]["code"] == "outside_limits"
