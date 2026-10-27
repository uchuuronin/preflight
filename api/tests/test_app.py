import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from preflight_api import flags
from preflight_api.app import create_app
from preflight_api.errors import ApiError
from preflight_api.guide.budget import TURNS_PER_VISITOR_DAY, MemoryLedger
from preflight_api.idempotency import MemoryIdempotencyStore, remember, replay_or_none
from preflight_api.stores import LocalRecordStore, new_share_token

TOOLS = [
	{
		"type": "function",
		"name": "set_knob",
		"description": "Move one unpinned knob.",
		"parameters": {"type": "object", "properties": {"knob_id": {"type": "string", "enum": ["batch_size"]}}},
	}
]


def model_response(calls=True):
	output = [{"type": "message", "content": [{"type": "output_text", "text": "Your eval pass is the risk."}]}]
	if calls:
		output.append(
			{
				"type": "function_call",
				"call_id": "c1",
				"name": "set_knob",
				"arguments": json.dumps({"knob_id": "batch_size", "value": 8}),
			}
		)
	return {
		"id": "r1",
		"output": output,
		"usage": {"input_tokens": 1000, "output_tokens": 100, "input_tokens_details": {"cached_tokens": 400}},
	}


@pytest.fixture
def records(tmp_path: Path):
	token = new_share_token()
	(tmp_path / f"{token}.json").write_text(json.dumps({"schema_version": "0.1.0", "record_id": "01X"}))
	return LocalRecordStore(tmp_path), token


@pytest.fixture
def live_guide(monkeypatch):
	monkeypatch.setenv("PREFLIGHT_FLAGS", json.dumps({"c.guide.live": True}))
	monkeypatch.setenv("META_API_KEY", "test-key")
	monkeypatch.setenv("META_API_BASE", "https://model.invalid/v1")
	flags.reload()
	yield
	flags.reload()


def client_with(model_handler=None, **kw):
	transport = httpx.MockTransport(model_handler or (lambda req: httpx.Response(200, json=model_response())))
	http = httpx.AsyncClient(transport=transport)
	# https, because the session cookie is Secure and a plain-http client never sends it back.
	return TestClient(create_app(http=http, **kw), base_url="https://testserver")


def test_liveness_and_status(records):
	store, _ = records
	with client_with(records=store) as c:
		assert c.get("/livez").json() == {"status": "ok"}
		assert c.get("/readyz").status_code == 200
		status = c.get("/v1/status").json()
		assert status["guide"] == "rule_based"
		assert status["samples_only"] is False


def test_readyz_degraded_when_store_down(tmp_path):
	with client_with(records=LocalRecordStore(tmp_path / "missing")) as c:
		res = c.get("/readyz")
		assert res.status_code == 503
		assert res.json()["status"] == "degraded"


def test_record_by_share_token_and_envelope(records):
	store, token = records
	with client_with(records=store) as c:
		ok = c.get(f"/v1/records/{token}")
		assert ok.status_code == 200
		assert ok.json()["record_id"] == "01X"
		missing = c.get("/v1/records/" + "x" * 24)
		assert missing.status_code == 404
		err = missing.json()["error"]
		assert err["code"] == "record_not_found"
		assert err["retryable"] is False
		assert err["request_id"] == missing.headers["X-Request-Id"]


def test_unknown_route_uses_envelope():
	with client_with() as c:
		res = c.get("/v1/nope")
		assert res.status_code == 404
		assert res.json()["error"]["code"] == "not_found"


def test_session_cookie_is_set_once():
	with client_with() as c:
		first = c.get("/livez")
		assert "pf_session" in first.cookies
		assert "httponly" in first.headers["set-cookie"].lower()
		second = c.get("/livez")
		assert "set-cookie" not in second.headers


def test_guide_off_falls_back_to_rule_tour():
	with client_with() as c:
		res = c.post("/v1/guide/turns", json={"input": [], "tools": TOOLS})
		assert res.status_code == 503
		err = res.json()["error"]
		assert err["fallback"] == "rule_based"
		assert err["reason"] == "disabled"


def test_guide_turn_proxies_and_bills(live_guide):
	seen = {}

	def handler(req: httpx.Request):
		seen["auth"] = req.headers["authorization"]
		seen["body"] = json.loads(req.content)
		return httpx.Response(200, json=model_response(), headers={"x-request-id": "meta-1"})

	ledger = MemoryLedger()
	with client_with(handler, ledger=ledger) as c:
		res = c.post(
			"/v1/guide/turns",
			json={"input": [{"role": "user", "content": "cheapest"}], "tools": TOOLS, "record_ref": "rec"},
		)
		assert res.status_code == 200, res.text
		body = res.json()
	assert seen["auth"] == "Bearer test-key"
	assert seen["body"]["model"] == "muse-spark-1.3"
	# Instructions come from the server, never the browser.
	assert "flight simulator for ML jobs" in seen["body"]["instructions"]
	assert seen["body"]["max_output_tokens"] <= 1200
	assert body["calls"] == [{"call_id": "c1", "name": "set_knob", "arguments": {"knob_id": "batch_size", "value": 8}}]
	assert body["provenance"]["request_id"] == "meta-1"
	expected = (600 * 1.25 + 400 * 0.15 + 100 * 4.25) / 1e6
	assert body["provenance"]["cost_usd"] == pytest.approx(expected, rel=1e-3)
	assert ledger.spent_today() == pytest.approx(expected, rel=1e-3)


def test_guide_rejects_unknown_tools(live_guide):
	with client_with() as c:
		bad = [{**TOOLS[0], "name": "shell_exec"}]
		res = c.post("/v1/guide/turns", json={"input": [], "tools": bad})
		assert res.status_code == 422
		assert "shell_exec" in res.json()["error"]["message"]


def test_guide_turn_budget_per_visitor(live_guide):
	with client_with() as c:
		for _ in range(TURNS_PER_VISITOR_DAY):
			assert c.post("/v1/guide/turns", json={"input": [], "tools": TOOLS}).status_code == 200
		res = c.post("/v1/guide/turns", json={"input": [], "tools": TOOLS})
		assert res.status_code == 503
		assert res.json()["error"]["reason"] == "visitor_turns"


def test_guide_model_outage_falls_back(live_guide):
	with client_with(lambda req: httpx.Response(502)) as c:
		res = c.post("/v1/guide/turns", json={"input": [], "tools": TOOLS})
		assert res.status_code == 503
		assert res.json()["error"]["fallback"] == "rule_based"


def test_addons_are_mounted_and_broken_ones_skipped(tmp_path):
	(tmp_path / "hello.py").write_text(
		"from fastapi import APIRouter\nrouter = APIRouter(prefix='/v1/addons/hello')\n"
		"@router.get('')\ndef hi():\n\treturn {'hi': True}\n"
	)
	(tmp_path / "broken.py").write_text("raise RuntimeError('boom')\n")
	with client_with(addons_dir=tmp_path) as c:
		assert c.get("/v1/addons/hello").json() == {"hi": True}
		assert c.app.state.addons == ["hello"]
		assert c.get("/livez").status_code == 200


def test_idempotency_replays_and_detects_conflicts():
	store = MemoryIdempotencyStore()
	assert replay_or_none(store, "probe", "k1", {"a": 1}) is None
	remember(store, "probe", "k1", {"a": 1}, 202, {"job_id": "j1"})
	stored = replay_or_none(store, "probe", "k1", {"a": 1})
	assert stored is not None and stored.response == {"job_id": "j1"}
	with pytest.raises(ApiError) as e:
		replay_or_none(store, "probe", "k1", {"a": 2})
	assert e.value.code == "idempotency_conflict"
