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
	"""Chiya, Nov 2: on the LoRA record the engine predicts lr 2e-4 (the same step as 1e-4) and another
	GPU (extrapolated across GPUs), where the web's grids abstained. The sample is A's Nov 3 A10 record
	(probes on the A10), so the other GPU is the L4."""
	_env(monkeypatch, True)
	with client() as c:
		r = ask(
			c,
			[
				{"batch_size": 4, "lr": 0.0002},
				{"batch_size": 4, "lr": 0.0001},
				{"batch_size": 4, "gpu_type": "L4"},
			],
		)
	assert r.status_code == 200, r.text
	a, b, gpu = r.json()["cells"]
	assert a["time.step"]["status"] == "predicted"
	assert a["time.step"]["point"] == pytest.approx(b["time.step"]["point"])
	assert a["time.step"]["point"] == pytest.approx(1.409, abs=0.01)
	assert gpu["time.step"]["status"] == "extrapolated"
	assert gpu["time.step"]["point"] == pytest.approx(2.190, abs=0.01)


def test_only_the_records_own_knobs(monkeypatch):
	_env(monkeypatch, True)
	with client() as c:
		r = ask(c, [{"not_a_knob": 1}])
		assert r.status_code == 422
		assert "not_a_knob" in r.json()["error"]["message"]
		assert ask(c, [{"lr": 1}] * (predict_route.MAX_CELLS + 1)).status_code == 422
		assert c.post("/v1/records/lora-qwen/predict", content=b"not json").status_code == 422
		assert ask(c, [{"lr": 1}], ref="no-such-record").status_code == 404


def test_a_burst_is_bounded_per_address_one_token_a_setting(monkeypatch):
	_env(monkeypatch, True)
	with client() as c:
		codes = [ask(c, [{"lr": 0.0001}]).status_code for _ in range(predict_route.BURST + 2)]
		assert codes[: predict_route.BURST] == [200] * predict_route.BURST
		assert codes[-1] == 429
		predict_route._buckets.clear()
		assert ask(c, [{"lr": 0.0001}] * predict_route.MAX_CELLS).status_code == 200
		assert ask(c, [{"lr": 0.0001}] * predict_route.MAX_CELLS).status_code == 429  # 30 tokens, 32 asked


def test_off_hides_the_route_before_reading_the_body(monkeypatch):
	_env(monkeypatch, False)
	with client() as c:
		assert c.post("/v1/records/lora-qwen/predict", content=b"not json").status_code == 404


def test_the_engine_answers_with_the_runners_models(monkeypatch):
	"""Review, Nov 2: the engine's own defaults keep engine.rl off, which production turns on, so an RL
	record read here gave another step time than the runner's. The route reads the engine flags."""
	from preflight.engine import api

	seen = []
	real = api.predict

	def spy(rec, knobs=None):
		seen.append(api.feature("rl"))
		return real(rec, knobs)

	monkeypatch.setattr(api, "predict", spy)
	for on in (True, False):
		monkeypatch.setenv("PREFLIGHT_FLAGS", json.dumps({"c.engine_fallback": True, "engine.rl": on}))
		monkeypatch.setenv("PREFLIGHT_SESSION_KEYS", "k" * 40)
		session.reset_keys()
		flags.reload()
		with client() as c:
			assert ask(c, [{"lr": 0.0002}]).status_code == 200
	assert seen == [True, False]
	assert api._FEATURES_NOW.get() is None  # set only in the request's worker thread


@pytest.mark.parametrize(
	"ref,cell",
	[
		("grpo", {"gpu_type": "A10"}),
		("retrieval", {"gpu_type": "A10"}),
		("gnn", {"gpu_type": "A10"}),
		("vision", {"gpu_type": "A10"}),
		("campaign-grid", {"batch_size": 16}),
	],
)
def test_every_sample_answers(monkeypatch, ref, cell):
	"""D's walkthrough (Nov 2): the API knew 5 of the 10 samples, so these five answered 404."""
	_env(monkeypatch, True)
	with client() as c:
		r = ask(c, [cell], ref=ref)
	assert r.status_code == 200, (ref, r.text[:200])
	assert r.json()["cells"][0]


def _local(monkeypatch, on: bool = True):
	monkeypatch.setenv("PREFLIGHT_FLAGS", json.dumps({"c.engine_fallback": True, "c.local_predict": on}))
	monkeypatch.setenv("PREFLIGHT_SESSION_KEYS", "k" * 40)
	session.reset_keys()
	flags.reload()


def _lora_raw() -> dict:
	from pathlib import Path

	p = Path(__file__).resolve().parents[2] / "web/src/workspace/fixtures/lora-live-nov3.record.json"
	return json.loads(p.read_text())


def test_a_local_record_is_predicted_inline_and_answers_as_its_sample_does(monkeypatch):
	"""PM, Nov 3 (Chiya's shot 9b): a record opened from disk sends itself; the engine answers the same."""
	_local(monkeypatch)
	cells = [{"batch_size": 4, "lr": 0.0002}, {"batch_size": 4, "gpu_type": "L4"}]
	with client() as c:
		inline = c.post("/v1/predict", json={"record": _lora_raw(), "cells": cells})
		stored = ask(c, cells)
	assert inline.status_code == 200, inline.text
	assert inline.json() == stored.json()


def test_inline_is_off_by_default_capped_and_charged_for_its_record(monkeypatch):
	_local(monkeypatch, on=False)
	with client() as c:
		assert c.post("/v1/predict", json={"record": {}, "cells": [{"lr": 1}]}).status_code == 404
	_local(monkeypatch)
	with client() as c:
		big = {"record": {"pad": "x" * (predict_route.MAX_INLINE_BYTES + 1)}, "cells": [{"lr": 1}]}
		assert c.post("/v1/predict", json=big).status_code == 413
		assert (
			c.post("/v1/predict", json={"record": {"not": "a record"}, "cells": [{"lr": 1}]}).status_code
			== 422
		)
		# each request costs its record's parse plus its settings (five tokens here): of 30, the unreadable
		# record took five, so five more one-setting requests go through and the sixth waits
		codes = [
			c.post("/v1/predict", json={"record": _lora_raw(), "cells": [{"lr": 0.0002}]}).status_code
			for _ in range(6)
		]
	assert codes[:5] == [200] * 5 and codes[5] == 429
