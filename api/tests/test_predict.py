"""POST /v1/records/{ref}/predict: the engine's prediction where the record's grids hold no value."""

import json

import pytest
from fastapi.testclient import TestClient

from preflight_api import flags, session
from preflight_api.app import create_app
from preflight_api.routes import predict as predict_route


def _env(monkeypatch, on: bool):
	monkeypatch.setenv("PREFLIGHT_FLAGS", json.dumps({"c.engine_fallback": on}))
	monkeypatch.setenv("PREFLIGHT_SESSION_KEYS", "k" * 40)
	session.reset_keys()
	flags.reload()


@pytest.fixture(autouse=True)
def reset():
	predict_route._buckets.clear()
	yield
	flags.reload()
	session.reset_keys()


def client():
	return TestClient(create_app(), base_url="https://testserver")


def ask(c, cells, ref="lora-qwen"):
	return c.post(f"/v1/records/{ref}/predict", json={"cells": cells})


def test_off_by_default(monkeypatch):
	_env(monkeypatch, False)
	with client() as c:
		assert ask(c, [{"lr": 0.0002}]).status_code == 404


def test_the_engine_predicts_where_the_grids_dont(monkeypatch):
	"""Chiya, Nov 2: on the LoRA record the engine predicts lr 2e-4 (the same step as 1e-4) and the A10
	(extrapolated across GPUs), where the web's grids abstained."""
	_env(monkeypatch, True)
	with client() as c:
		r = ask(c, [{"batch_size": 4, "lr": 0.0002}, {"batch_size": 4, "lr": 0.0001}, {"batch_size": 4, "gpu_type": "A10"}])
	assert r.status_code == 200, r.text
	a, b, gpu = r.json()["cells"]
	assert a["time.step"]["status"] == "predicted"
	assert a["time.step"]["point"] == pytest.approx(b["time.step"]["point"])
	assert a["time.step"]["point"] == pytest.approx(1.709, abs=0.01)
	assert gpu["time.step"]["status"] == "extrapolated"
	assert gpu["time.step"]["point"] == pytest.approx(1.236, abs=0.01)


def test_only_the_records_own_knobs(monkeypatch):
	_env(monkeypatch, True)
	with client() as c:
		r = ask(c, [{"not_a_knob": 1}])
		assert r.status_code == 422
		assert "not_a_knob" in r.json()["error"]["message"]
		assert ask(c, [{"lr": 1}] * (predict_route.MAX_CELLS + 1)).status_code == 422
		assert ask(c, [{"lr": 1}], ref="no-such-record").status_code == 404


def test_a_burst_is_bounded_per_address(monkeypatch):
	_env(monkeypatch, True)
	with client() as c:
		codes = [ask(c, [{"lr": 0.0001}]).status_code for _ in range(predict_route.BURST + 2)]
	assert codes[: predict_route.BURST] == [200] * predict_route.BURST
	assert codes[-1] == 429
