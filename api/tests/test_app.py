import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from preflight_api.app import create_app
from preflight_api.errors import ApiError
from preflight_api.idempotency import MemoryIdempotencyStore, remember, replay_or_none
from preflight_api.stores import LocalRecordStore, new_share_token


@pytest.fixture
def records(tmp_path: Path):
	token = new_share_token()
	(tmp_path / f"{token}.json").write_text(json.dumps({"schema_version": "0.1.0", "record_id": "01X"}))
	return LocalRecordStore(tmp_path), token


def client_with(model_handler=None, **kw):
	transport = httpx.MockTransport(model_handler or (lambda req: httpx.Response(500)))
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


def test_session_cookie_is_set_once_and_signed():
	with client_with() as c:
		first = c.get("/livez")
		cookie = first.headers["set-cookie"]
		assert cookie.startswith("__Host-pf_session=v1.")
		assert all(p in cookie.lower() for p in ("httponly", "secure", "path=/", "samesite=lax"))
		second = c.get("/livez")
		assert "set-cookie" not in second.headers


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
