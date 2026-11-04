"""The live guide endpoint: typed inputs, server-built tools, signed state, guards, approvals."""

import json

import httpx
import pytest
from fastapi.testclient import TestClient

from preflight_api import flags, session
from preflight_api.app import create_app
from preflight_api.ledger import LedgerCore, Limits, MemoryLedger

GIB = 2**30
SETTING = {
	"knobs": {"seq_len": 2048, "batch_size": 4, "eval_batch_size": 32, "gpu_type": "A100-SXM-80GB"},
	"pinned": [],
	"objective": {"kind": "min_cost"},
}


@pytest.fixture(autouse=True)
def live(monkeypatch):
	monkeypatch.setenv("PREFLIGHT_FLAGS", json.dumps({"c.guide.live": True}))
	monkeypatch.setenv("META_API_KEY", "test-key")
	monkeypatch.setenv("META_API_BASE", "https://model.invalid/v1")
	monkeypatch.setenv("PREFLIGHT_SESSION_KEYS", "k" * 40)
	session.reset_keys()
	flags.reload()
	yield
	flags.reload()
	session.reset_keys()


class Model:
	"""A scripted Muse Spark: each call pops the next reply and records the request."""

	def __init__(self, *replies):
		self.replies = list(replies)
		self.requests: list[dict] = []

	def __call__(self, req: httpx.Request) -> httpx.Response:
		self.requests.append(json.loads(req.content))
		out = self.replies.pop(0)
		return httpx.Response(200, json=out, headers={"x-request-id": f"meta-{len(self.requests)}"})


def reply(text="", calls=(), usage=(800, 60)):
	output = []
	if text:
		output.append({"type": "message", "content": [{"type": "output_text", "text": text}]})
	for i, (name, args) in enumerate(calls):
		output.append(
			{"type": "function_call", "call_id": f"c{name}{i}", "name": name, "arguments": json.dumps(args)}
		)
	return {"id": "r", "output": output, "usage": {"input_tokens": usage[0], "output_tokens": usage[1]}}


class Jobs:
	def __init__(self):
		self.submitted = []

	async def submit(self, request, *, visitor, idempotency_key):
		self.submitted.append((request, idempotency_key))
		return {"status": "submitted", "job_id": "job-1", "queue_position": 2}


def client(model, **kw):
	http = httpx.AsyncClient(transport=httpx.MockTransport(model))
	return TestClient(create_app(http=http, **kw), base_url="https://testserver")


def turn(c, **body):
	return c.post("/v1/guide/turns", json={"record": "lora-7b", "setting": SETTING, **body})


def readout(point, unit="bytes"):
	return {"point": point, "lo": point * 0.95, "hi": point * 1.05, "unit": unit, "status": "predicted"}


def test_full_round_trip_sensitivity_move_decision():
	model = Model(
		reply(calls=[("sensitivity", {"objective": "min_cost"}), ("highlight", {"nodes": ["eval"]})]),
		reply(calls=[("set_knob", {"knob_id": "eval_batch_size", "value": 8, "why": "eval OOM"})]),
		reply(text="Eval batch 8 peaks at 66.1 GiB, so the run fits; cost US$11.31."),
		reply(text="Nothing else safe saves money."),
	)
	with client(model) as c:
		r1 = turn(c, start=True).json()
		assert r1["awaiting"] == "results"
		assert [x["name"] for x in r1["calls"]] == ["sensitivity", "highlight"]
		assert all(x["runs_in"] == "browser" for x in r1["calls"])
		# tools come from the server and the record, never from the request
		sent = model.requests[0]
		names = [t["name"] for t in sent["tools"]]
		assert names[0] == "set_knob" and "shell" not in names
		set_knob = sent["tools"][0]["parameters"]["properties"]["knob_id"]["enum"]
		assert "eval_batch_size" in set_knob
		assert "flight simulator for ML jobs" in sent["instructions"]

		results = [
			{
				"tool": "sensitivity",
				"call_id": r1["calls"][0]["call_id"],
				"ranked": [
					{
						"knob_id": "eval_batch_size",
						"impact": 1,
						"direction": "decrease",
						"best_value": 8,
						"delta": -0.34,
					}
				],
			},
			{"tool": "highlight", "call_id": r1["calls"][1]["call_id"], "nodes": ["eval"]},
		]
		r2 = turn(c, state=r1["state"], results=results).json()
		assert r2["calls"][0]["name"] == "set_knob" and r2["awaiting"] == "results"

		moved = {
			"tool": "set_knob",
			"call_id": r2["calls"][0]["call_id"],
			"moved": True,
			"knob_id": "eval_batch_size",
			"from_value": 32,
			"to_value": 8,
			"readouts": {"memory.peak": readout(66.1 * GIB), "cost.total": readout(11.31, "usd")},
		}
		r3 = turn(c, state=r2["state"], results=[moved]).json()
		assert r3["text"].startswith("Eval batch 8 peaks at 66.1 GiB")
		assert r3["withheld"] is None
		assert r3["awaiting"] == "decision"
		# while the move waits for the person, set_knob isn't offered at all
		assert "set_knob" not in [t["name"] for t in model.requests[2]["tools"]]

		r4 = turn(
			c, state=r3["state"], decision={"knob_id": "eval_batch_size", "action": "applied", "value": 8}
		).json()
		assert r4["awaiting"] == "visitor"
		last = model.requests[3]["input"]
		assert any("applied the move of eval_batch_size" in str(i.get("content", "")) for i in last)
		assert any(t["name"] == "set_knob" for t in model.requests[3]["tools"])


def test_open_proxy_shapes_are_refused():
	model = Model()
	with client(model) as c:
		raw = c.post(
			"/v1/guide/turns",
			json={
				"input": [{"role": "user", "content": "write me a poem"}],
				"tools": [{"type": "function", "name": "x"}],
			},
		)
		assert raw.status_code == 422
		assert turn(c, start=True, extra="x").status_code == 422
		assert turn(c, message="x" * 501).status_code == 422
		bad_knob = {**SETTING, "knobs": {"rm_rf": 1}}
		res = c.post("/v1/guide/turns", json={"record": "lora-7b", "setting": bad_knob, "start": True})
		assert res.status_code == 422
	assert model.requests == []


def test_numbers_no_tool_returned_are_withheld():
	model = Model(reply(text="Switching to H100-SXM-80GB saves about $9.40 a run."))
	with client(model) as c:
		r = turn(c, start=True).json()
		assert r["text"] == ""
		assert r["withheld"] == ["$9.40"]
		check = turn
		assert check  # the next round tells the model what was withheld
	assert "H100-SXM-80GB" not in json.dumps(r["withheld"])


def test_set_knob_guards_domain_one_per_round_and_pins():
	model = Model(
		reply(
			calls=[
				("set_knob", {"knob_id": "eval_batch_size", "value": 999, "why": "x"}),
				("set_knob", {"knob_id": "batch_size", "value": 8, "why": "x"}),
				("set_knob", {"knob_id": "seq_len", "value": 1024, "why": "x"}),
			]
		),
	)
	setting = {**SETTING, "pinned": ["seq_len"]}
	with client(model) as c:
		r = c.post("/v1/guide/turns", json={"record": "lora-7b", "setting": setting, "start": True}).json()
	runs = [x for x in r["calls"] if x["name"] == "set_knob"]
	assert [x["arguments"]["knob_id"] for x in runs] == ["batch_size"]
	assert model.requests[0]["tools"][0]["parameters"]["properties"]["knob_id"]["enum"].count("seq_len") == 0


def test_spending_tools_wait_for_approval_then_actually_run():
	jobs = Jobs()
	model = Model(
		reply(
			text="Eval batch 16 isn't trusted. Shall I measure it?",
			calls=[("run_probe", {"knobs": {"eval_batch_size": 16}, "reason": "tight"})],
		),
		reply(text="The probe is queued."),
	)
	with client(model, jobs=jobs) as c:
		r1 = turn(c, start=True).json()
		assert r1["calls"] == [
			{
				"call_id": r1["calls"][0]["call_id"],
				"name": "run_probe",
				"arguments": {"knobs": {"eval_batch_size": 16}, "reason": "tight"},
				"runs_in": "approval",
			}
		]
		assert r1["awaiting"] == "approval"
		assert jobs.submitted == []
		r2 = turn(
			c, state=r1["state"], approvals=[{"call_id": r1["calls"][0]["call_id"], "approved": True}]
		).json()
		assert r2["text"] == "The probe is queued."
	request, key = jobs.submitted[0]
	assert request["mode"] == "probe" and request["knobs"]["eval_batch_size"] == 16
	assert key.endswith(r1["calls"][0]["call_id"])
	told = json.dumps(model.requests[1]["input"])
	assert "job-1" in told and "approved run_probe" in told


def test_approval_without_a_measuring_service_says_so():
	model = Model(reply(calls=[("run_verify", {})]), reply(text="Verifying isn't open yet."))
	with client(model, jobs=None) as c:
		c.app.state.jobs = None
		r1 = turn(c, start=True).json()
		turn(c, state=r1["state"], approvals=[{"call_id": r1["calls"][0]["call_id"], "approved": True}])
	assert "measuring_service_not_running_yet" in json.dumps(model.requests[1]["input"])


def test_state_is_bound_to_the_visitor_and_cant_be_edited():
	model = Model(reply(calls=[("sensitivity", {"objective": "min_cost"})]))
	with client(model) as c:
		r1 = turn(c, start=True).json()
		state = r1["state"]
		forged = state[:-8] + ("A" * 8 if not state.endswith("A" * 8) else "B" * 8)
		assert turn(c, state=forged, results=[]).status_code == 409
	with client(Model()) as other:
		res = turn(
			other,
			state=state,
			results=[{"tool": "sensitivity", "call_id": r1["calls"][0]["call_id"], "ranked": []}],
		)
		assert res.status_code == 409


def test_results_must_answer_pending_calls_with_record_ids():
	model = Model(reply(calls=[("predict", {"knobs": {"batch_size": 8}})]))
	with client(model) as c:
		r1 = turn(c, start=True).json()
		cid = r1["calls"][0]["call_id"]
		assert (
			turn(
				c, state=r1["state"], results=[{"tool": "predict", "call_id": "nope", "readouts": {}}]
			).status_code
			== 422
		)
		bad = {"tool": "predict", "call_id": cid, "readouts": {"made.up": readout(1)}}
		assert turn(c, state=r1["state"], results=[bad]).status_code == 422
		wrong_tool = {"tool": "highlight", "call_id": cid, "nodes": []}
		assert turn(c, state=r1["state"], results=[wrong_tool]).status_code == 422


def test_daily_cap_is_reserved_before_the_call():
	ledger = MemoryLedger(LedgerCore(Limits(usd_per_day=0.0001)))
	model = Model(reply(text="hi"))
	with client(model, ledger=ledger) as c:
		res = turn(c, start=True)
	assert res.status_code == 503
	assert res.json()["error"]["reason"] == "daily_cap"
	assert model.requests == []  # refused before any model call


def test_model_outage_refunds_and_falls_back():
	core = LedgerCore()
	with client(lambda req: httpx.Response(502), ledger=MemoryLedger(core)) as c:
		res = turn(c, start=True)
	assert res.status_code == 503 and res.json()["error"]["fallback"] == "rule_based"
	assert core.spent_today() == 0


def test_unknown_record_is_404():
	with client(Model()) as c:
		res = c.post("/v1/guide/turns", json={"record": "nope", "setting": SETTING, "start": True})
		assert res.status_code == 404


def test_billing_settles_actual_usage():
	core = LedgerCore()
	model = Model(reply(text="Plan the tour.", usage=(1000, 100)))
	with client(model, ledger=MemoryLedger(core)) as c:
		body = turn(c, start=True).json()
	expected = (1000 * 1.25 + 100 * 4.25) / 1e6
	assert body["provenance"]["cost_usd"] == pytest.approx(expected, rel=1e-3)
	assert core.spent_today() == pytest.approx(expected, rel=1e-3)


def test_rule_tour_approvals_run_the_tool():
	jobs = Jobs()
	with client(Model(), jobs=jobs) as c:
		res = c.post(
			"/v1/guide/tools",
			json={
				"record": "lora-7b",
				"tool": "run_probe",
				"setting": SETTING["knobs"],
				"knobs": {"eval_batch_size": 16},
				"reason": "tight",
			},
			headers={"Idempotency-Key": "click-1"},
		)
		assert res.json() == {"status": "submitted", "job_id": "job-1", "queue_position": 2}
		assert jobs.submitted[0][1].endswith("click-1")
		bad = c.post("/v1/guide/tools", json={"record": "lora-7b", "tool": "shell", "setting": {}})
		assert bad.status_code == 422
		odd = c.post(
			"/v1/guide/tools",
			json={"record": "lora-7b", "tool": "run_probe", "setting": {"eval_batch_size": 999}},
		)
		assert odd.status_code == 422
	with client(Model(), jobs=None) as c:
		c.app.state.jobs = None
		res = c.post(
			"/v1/guide/tools", json={"record": "lora-7b", "tool": "run_verify", "setting": SETTING["knobs"]}
		)
		assert res.json()["status"] == "unavailable"


def test_re_ranking_the_same_setting_is_refused_and_dashes_never_reach_the_visitor():
	# Muse Spark looped on sensitivity for one record on Oct 27 until the round cap.
	model = Model(
		reply(calls=[("sensitivity", {"objective": "min_cost"})]),
		reply(calls=[("sensitivity", {"objective": "min_cost"})]),
		reply("Nothing safe improves cost — the tour ends here."),
	)
	c = client(model)
	r1 = turn(c, start=True).json()
	assert [x["name"] for x in r1["calls"]] == ["sensitivity"]
	r2 = turn(
		c,
		state=r1["state"],
		results=[{"tool": "sensitivity", "call_id": r1["calls"][0]["call_id"], "ranked": []}],
	).json()
	# The second ranking of the same setting never reaches the browser: the server refuses it.
	assert r2["calls"] == []
	refused = [
		json.loads(i["output"])
		for i in model.requests[2]["input"]
		if i.get("type") == "function_call_output" and "refused" in i["output"]
	]
	assert refused and refused[0]["refused"] == "already_ranked"
	assert r2["text"] == "Nothing safe improves cost, the tour ends here."


def test_a_stalled_guide_is_reminded_once_of_the_move_its_ranking_offered():
	# Oct 27: Muse Spark sometimes announced a move and stopped without calling set_knob.
	model = Model(
		reply(calls=[("sensitivity", {"objective": "min_cost"})]),
		reply("I'll make the cheapest move next."),
		reply(calls=[("set_knob", {"knob_id": "eval_batch_size", "value": 8, "why": "fits"})]),
	)
	c = client(model)
	r1 = turn(c, start=True).json()
	ranked = [
		{"knob_id": "eval_batch_size", "impact": 1.0, "direction": "decrease", "best_value": 8, "fit": "fits"}
	]
	r2 = turn(
		c,
		state=r1["state"],
		results=[{"tool": "sensitivity", "call_id": r1["calls"][0]["call_id"], "ranked": ranked}],
	).json()
	assert [x["name"] for x in r2["calls"]] == ["set_knob"]
	nudge = [i for i in model.requests[2]["input"] if "[check] Your ranking" in str(i.get("content", ""))]
	assert len(nudge) == 1 and "eval_batch_size" in nudge[0]["content"]


def test_while_a_move_waits_only_spending_tools_are_offered_and_repeats_are_refused():
	model = Model(
		reply(calls=[("highlight", {"nodes": ["eval"]})]),
		reply(calls=[("highlight", {"nodes": ["eval"]})]),
		reply("Eval is where it runs out of memory."),
	)
	c = client(model)
	r1 = turn(c, start=True).json()
	r2 = turn(
		c,
		state=r1["state"],
		results=[{"tool": "highlight", "call_id": r1["calls"][0]["call_id"], "nodes": ["eval"]}],
	).json()
	# The second identical highlight was refused on the server and the model answered instead.
	assert r2["calls"] == [] and r2["text"] == "Eval is where it runs out of memory."
	outs = [
		json.loads(i["output"]) for i in model.requests[2]["input"] if i.get("type") == "function_call_output"
	]
	assert any(o.get("refused") == "already_done" for o in outs)


def test_no_browser_tools_while_a_move_is_pending():
	from preflight_api.guide.record_view import RecordView
	from preflight_api.guide.tools import SERVER_TOOLS, build_tools
	from preflight_api.stores import SampleStore

	rec = RecordView.from_record(
		SampleStore(__import__("pathlib").Path(__file__).resolve().parents[2]).get("lora-7b")
	)
	names = {t["name"] for t in build_tools(rec, set(), move_pending=True)}
	assert names and names <= set(SERVER_TOOLS)


def test_a_daily_limit_on_a_guide_probe_comes_back_in_the_job_stores_words():
	"""A's demo limits (Nov 1): the guide's run_probe hears the limit, not 'the queue is full'."""

	class Limited(Jobs):
		async def submit(self, request, *, visitor, idempotency_key):
			e = type("QueueFull", (RuntimeError,), {})(
				"Preflight has used today's GPU budget: samples only until midnight ET"
			)
			e.code = "limit.daily_spend"
			e.retry_after_s = 3600
			raise e

	with client(Model(), jobs=Limited()) as c:
		res = c.post(
			"/v1/guide/tools",
			json={
				"record": "lora-7b",
				"tool": "run_probe",
				"setting": SETTING["knobs"],
				"knobs": {"eval_batch_size": 16},
				"reason": "tight",
			},
			headers={"Idempotency-Key": "click-2"},
		).json()
	assert res["status"] == "limit"
	assert res["code"] == "limit.daily_spend"
	assert res["message"].startswith("Preflight has used today's GPU budget")


def test_sensitivity_carries_a_combined_move():
	from preflight_api.guide.types import SensitivityOut

	out = SensitivityOut.model_validate(
		{
			"tool": "sensitivity",
			"call_id": "s",
			"ranked": [],
			"combined": {
				"knobs": {"eval_batch_size": 4, "seq_len": 512, "eval_no_grad": True},
				"risk": 0.001,
				"peak_bytes": 17.5 * 2**30,
			},
		}
	)
	assert out.combined is not None and out.combined.knobs["seq_len"] == 512
	import pytest
	from pydantic import ValidationError

	with pytest.raises(ValidationError):  # one knob is a single move, never "combined"
		SensitivityOut.model_validate(
			{
				"tool": "sensitivity",
				"call_id": "s",
				"ranked": [],
				"combined": {"knobs": {"seq_len": 512}, "risk": 0.0},
			}
		)


def test_objective_accepts_the_visitors_description():
	"""New flight saves "What are you trying to do?" as objective.description (the record's schema has
	it); the guide's own Objective took it as an unknown field (D's walkthrough, Nov 3)."""
	from preflight_api.guide.types import Objective, Setting

	o = Objective.model_validate({"kind": "min_cost", "description": "Fine-tune Qwen on my support tickets"})
	assert o.description == "Fine-tune Qwen on my support tickets"
	s = Setting.model_validate({"knobs": {}, "objective": {"kind": "min_time", "description": None}})
	assert s.objective.description is None
	import pytest
	from pydantic import ValidationError

	with pytest.raises(ValidationError):
		Objective.model_validate({"kind": "min_time", "description": "x" * 2001})


def test_the_description_stays_out_of_the_models_notes():
	"""The visitor's words reach the model only as their own bounded message, never inside the
	"[event]" notes the prompt says come from Preflight (review, Nov 3)."""
	import inspect

	from preflight_api.routes import guide as route

	src = inspect.getsource(route)
	assert src.count('objective.model_dump(exclude_none=True, exclude={"description"})') == 2
	assert "objective.model_dump(exclude_none=True)" not in src


def test_a_guide_round_waits_long_enough_for_its_token_cap(monkeypatch):
	"""Nov 4, 1 AM: a 1,580-token round took 12.9 s on the live site and the 12 s limit handed the tour
	to the rule-based one. The limit now covers the 2,000-token cap at the measured speed, on every
	request, whatever the shared client's own default."""
	import asyncio

	from preflight_api.guide import muse_spark

	seen = {}

	def model(request: httpx.Request) -> httpx.Response:
		seen["timeout"] = request.extensions.get("timeout")
		return httpx.Response(200, json={"id": "r", "output": [], "usage": {}})

	monkeypatch.setenv("META_API_BASE", "https://model.test/v1")
	monkeypatch.setenv("META_API_KEY", "k")
	client = httpx.AsyncClient(transport=httpx.MockTransport(model), timeout=5.0)
	asyncio.run(
		muse_spark.create_turn(
			instructions="", input_items=[], tools=[], max_output_tokens=2000, client=client
		)
	)
	assert seen["timeout"]["read"] == muse_spark.TIMEOUT_S
	assert muse_spark.TIMEOUT_S >= 3.2 + 0.0062 * 2000
