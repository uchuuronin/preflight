"""/v1/patches over a stand-in for the fix Function: who may race, what is reserved, what is shown."""

import copy
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from preflight_api import flags, session
from preflight_api.app import create_app
from preflight_api.fix import run
from preflight_api.fix.service import MemoryRaces
from preflight_api.jobs import visitor_of
from preflight_api.ledger import MemoryLedger
from preflight_api.owners import MemoryOwners
from preflight_api.routes import patches
from preflight_api.stores import LocalRecordStore

REPO = Path(__file__).resolve().parents[2]
DEV = json.loads((REPO / "web/src/workspace/fixtures/dev-finetune.record.json").read_text())
UPLOAD_LINK = "UpLoAdToKeN0123456789"
REPO_LINK = "RePoToKeN012345678901"
DERIVED_LINK = "DeRiVeDToKeN012345678"

THREE = [
	{"template_id": "eval.wrap_no_grad"},
	{"template_id": "eval.reduce_batch", "value": 8},
	{"template_id": "memory.gradient_checkpointing", "value": True},
]


class Fixer:
	def __init__(self, fail=False):
		self.fail = fail
		self.started: list[dict] = []
		self.result: dict | None = None
		self.cancelled: list[str] = []

	async def start(self, *, record_ref, candidates, measured, stop_usd):
		if self.fail:
			raise RuntimeError("modal down")
		self.started.append(
			dict(record_ref=record_ref, candidates=candidates, measured=measured, stop_usd=stop_usd)
		)
		return f"fc-{len(self.started)}"

	async def poll(self, call_id):
		return self.result

	async def cancel(self, call_id):
		self.cancelled.append(call_id)


def _record(kind: str, *, parent: str | None = None) -> dict:
	rec = copy.deepcopy(DEV)
	rec["job"]["source"] = {
		"kind": kind,
		"uri": "upload://sha256:" + "a" * 64,
		"entrypoint": "train.py",
		"args": [],
	}
	if kind == "repo":
		rec["job"]["source"] = {
			"kind": "repo",
			"uri": "https://github.com/o/r",
			"commit": "b" * 40,
			"entrypoint": "train.py",
			"args": [],
		}
	rec["parent_id"] = parent
	return rec


@pytest.fixture(autouse=True)
def env(monkeypatch):
	monkeypatch.setenv("PREFLIGHT_FLAGS", json.dumps({"c.fix_loop": True}))
	monkeypatch.setenv("PREFLIGHT_SESSION_KEYS", "k" * 40)
	session.reset_keys()
	flags.reload()
	yield
	flags.reload()
	session.reset_keys()


@pytest.fixture
def world(tmp_path):
	for token, rec in (
		(UPLOAD_LINK, _record("upload")),
		(REPO_LINK, _record("repo")),
		(DERIVED_LINK, _record("upload", parent="01PARENT")),
	):
		(tmp_path / f"{token}.json").write_text(json.dumps(rec))
	fixer, ledger, owners = Fixer(), MemoryLedger(), MemoryOwners()
	app = create_app(
		records=LocalRecordStore(tmp_path), ledger=ledger, fixer=fixer, races=MemoryRaces(), owners=owners
	)
	return TestClient(app, base_url="https://testserver"), fixer, ledger, owners


def _sid(c: TestClient) -> str:
	if not c.cookies.get("__Host-pf_session"):
		c.get("/v1/health")
	return c.cookies.get("__Host-pf_session").split(".")[1]


def test_a_race_on_a_sample_reserves_its_cap_and_names_the_problem_in_preflights_words(world):
	c, fixer, ledger, _ = world
	r = c.post("/v1/patches", json={"record": "lora-7b", "finding": "f2", "candidates": THREE})
	assert r.status_code == 202, r.text
	body = r.json()
	assert body["status"] == "running" and body["patch_id"].startswith("fx_")
	assert body["cap_usd"] == run.stop_usd(3) == 0.39
	(s,) = fixer.started
	assert s["record_ref"] == "sample:lora-7b"
	assert s["measured"] == "First eval will run out of memory"
	assert [x["template_id"] for x in s["candidates"]] == [x["template_id"] for x in THREE]
	assert ledger.core.spent_today() == pytest.approx(0.39)
	assert "call_id" not in body and "reservation" not in json.dumps(body)


def test_the_race_settles_at_what_muse_code_reported_and_only_its_visitor_sees_it(world):
	c, fixer, ledger, _ = world
	pid = c.post("/v1/patches", json={"record": "lora-7b", "candidates": THREE[:1]}).json()["patch_id"]
	assert c.get(f"/v1/patches/{pid}").json()["status"] == "running"
	fixer.result = {
		"usd_from_reported_tokens": 0.0912,
		"candidates": [
			{"id": "c1", "template_id": "eval.wrap_no_grad", "status": "patched", "diff": "--- a"}
		],
	}
	done = c.get(f"/v1/patches/{pid}").json()
	assert done["status"] == "done" and done["spent_usd"] == 0.0912
	assert done["candidates"][0]["status"] == "patched"
	assert ledger.core.spent_today() == pytest.approx(0.0912)
	other = TestClient(c.app, base_url="https://testserver")
	assert other.get(f"/v1/patches/{pid}").status_code == 404
	assert c.get("/v1/patches/not-an-id").status_code == 404


def test_only_fixes_for_this_kind_of_run_with_values_the_knob_allows(world):
	c, *_ = world
	bad = [
		[{"template_id": "agent.cap_tool_retries", "value": 3}],  # an agent fix on a fine-tune
		[{"template_id": "eval.reduce_batch", "value": 64}],  # past the knob's range
		[{"template_id": "eval.reduce_batch"}],  # the template needs one
		[{"template_id": "eval.wrap_no_grad"}, {"template_id": "eval.wrap_no_grad"}],
		[*THREE, {"template_id": "data.num_workers", "value": 4}],  # four
	]
	for cands in bad:
		assert c.post("/v1/patches", json={"record": "lora-7b", "candidates": cands}).status_code == 422
	assert (
		c.post("/v1/patches", json={"record": "lora-7b", "finding": "nope", "candidates": THREE}).status_code
		== 422
	)


def test_a_sample_with_no_script_has_nothing_to_fix(world):
	c, fixer, *_ = world
	r = c.post(
		"/v1/patches",
		json={"record": "agent-eval", "candidates": [{"template_id": "agent.cap_tool_retries", "value": 3}]},
	)
	assert r.status_code == 422 and r.json()["error"]["code"] == "no_script"
	assert fixer.started == []


def test_uploaded_code_races_only_for_whoever_checked_it_in(world):
	c, fixer, _, owners = world
	ask = {"record": UPLOAD_LINK, "candidates": THREE[:1]}
	assert c.post("/v1/patches", json=ask).status_code == 403
	import asyncio

	asyncio.run(owners.claim(UPLOAD_LINK, "someone-else"))
	assert c.post("/v1/patches", json=ask).status_code == 403
	owners2 = MemoryOwners()
	asyncio.run(owners2.claim(UPLOAD_LINK, visitor_of(_sid(c))))
	c.app.state.owners = owners2
	assert c.post("/v1/patches", json=ask).status_code == 202
	# A record derived from another (a probe on a shared link) never races uploaded code.
	asyncio.run(owners2.claim(DERIVED_LINK, visitor_of(_sid(c))))
	assert c.post("/v1/patches", json={**ask, "record": DERIVED_LINK}).status_code == 403
	# Public code needs no owner.
	assert c.post("/v1/patches", json={**ask, "record": REPO_LINK}).status_code == 202
	assert [s["record_ref"] for s in fixer.started] == [UPLOAD_LINK, REPO_LINK]


def test_closed_unless_the_flag_is_on_and_a_fixer_exists(world, monkeypatch):
	c, fixer, *_ = world
	monkeypatch.setenv("PREFLIGHT_FLAGS", json.dumps({"c.fix_loop": False}))
	flags.reload()
	r = c.post("/v1/patches", json={"record": "lora-7b", "candidates": THREE})
	assert r.status_code == 503 and r.json()["error"]["code"] == "fixes_not_open"
	assert fixer.started == []


def test_nothing_is_spent_when_the_race_cannot_start(world):
	c, fixer, ledger, _ = world
	fixer.fail = True
	assert c.post("/v1/patches", json={"record": "lora-7b", "candidates": THREE}).status_code == 503
	assert ledger.core.spent_today() == 0


def test_a_race_that_never_answers_is_lost_and_settles_at_its_cap(world, monkeypatch):
	c, _, ledger, _ = world
	pid = c.post("/v1/patches", json={"record": "lora-7b", "candidates": THREE[:2]}).json()["patch_id"]
	t0 = patches.now()
	monkeypatch.setattr(patches, "now", lambda: t0 + patches.RACE_LOST_S + 1)
	out = c.get(f"/v1/patches/{pid}").json()
	assert out["status"] == "failed" and out["error"] == "lost"
	assert ledger.core.spent_today() == pytest.approx(run.stop_usd(2))


def test_a_cancelled_race_stops_the_function(world):
	c, fixer, *_ = world
	pid = c.post("/v1/patches", json={"record": "lora-7b", "candidates": THREE[:1]}).json()["patch_id"]
	out = c.post(f"/v1/patches/{pid}/cancel").json()
	assert out["status"] == "failed" and out["error"] == "cancelled" and fixer.cancelled == ["fc-1"]


def test_run_race_reads_the_code_before_opening_a_box_and_reports_why_it_could_not():
	opened = []
	out = run.run_race(
		"NoSuchToken0123456789",
		[{"template_id": "eval.wrap_no_grad"}],
		None,
		reader=type("R", (), {"get": lambda self, t: None})(),
		blobs=None,
		open_box=lambda: opened.append(1),
		key="k",
	)
	assert out == {"error": "submission.record_unknown", "usd_from_reported_tokens": 0.0, "candidates": []}
	assert opened == []


def test_the_web_apps_copy_of_the_templates_matches():
	from preflight_api.fix.templates import TEMPLATES

	copy_ = json.loads((REPO / "web/src/fix/templates.json").read_text())
	assert copy_["usd_per_candidate"] == run.USD_PER_CANDIDATE
	web = copy_["templates"]
	assert web == [
		{
			"id": t.id,
			"title": t.title,
			"lenses": list(t.lenses),
			"knob": t.knob,
			"metric": t.metric,
			"needs_value": "{value}" in t.task,
			"verifiable": t.verifiable,
		}
		for t in TEMPLATES.values()
	]


def test_one_address_has_one_fix_budget_however_many_sessions_it_makes(world):
	c, *_ = world
	n = int(patches.FIX_USD_PER_ADDRESS_DAY // run.stop_usd(3))
	for _ in range(n):  # a room of visitors behind one address: several races go through
		fresh = TestClient(c.app, base_url="https://testserver")  # a new session, the same address
		assert fresh.post("/v1/patches", json={"record": "lora-7b", "candidates": THREE}).status_code == 202
	last = TestClient(c.app, base_url="https://testserver").post(
		"/v1/patches", json={"record": "lora-7b", "candidates": THREE}
	)
	assert last.status_code == 429 and last.json()["error"]["code"] == "fix_limit"


def test_a_race_keeps_its_reservation_until_it_settles(world, monkeypatch):
	c, fixer, ledger, _ = world
	pid = c.post("/v1/patches", json={"record": "lora-7b", "candidates": THREE}).json()["patch_id"]
	import time as _time

	later = _time.time() + 5 * 60  # well past a guide turn's reservation time
	assert ledger.core.spent_today(now=later) == pytest.approx(run.stop_usd(3))
	fixer.result = {"usd_from_reported_tokens": 0.5, "candidates": []}
	c.get(f"/v1/patches/{pid}")
	assert ledger.core.spent_today() == pytest.approx(0.5)  # what was reported, past the cap


def test_spend_past_the_cap_is_counted_as_reported(world):
	c, fixer, ledger, _ = world
	pid = c.post("/v1/patches", json={"record": "lora-7b", "candidates": THREE[:1]}).json()["patch_id"]
	fixer.result = {"usd_from_reported_tokens": 0.2, "candidates": []}
	c.get(f"/v1/patches/{pid}")
	assert ledger.core.spent_today() == pytest.approx(0.2)


def test_a_fixer_that_errors_leaves_the_race_running_until_it_is_lost(world, monkeypatch):
	c, fixer, *_ = world
	pid = c.post("/v1/patches", json={"record": "lora-7b", "candidates": THREE[:1]}).json()["patch_id"]

	async def broken(call_id):
		raise RuntimeError("modal unreachable")

	fixer.poll = broken
	assert c.get(f"/v1/patches/{pid}").json()["status"] == "running"
	t0 = patches.now()
	monkeypatch.setattr(patches, "now", lambda: t0 + patches.RACE_LOST_S + 1)
	assert c.get(f"/v1/patches/{pid}").json()["error"] == "lost"


def test_a_sandbox_that_will_not_start_or_breaks_is_reported_not_raised():
	def no_box():
		raise RuntimeError("no capacity")

	reader = type("R", (), {"get": lambda self, t: None})()
	out = run.run_race(
		"sample:lora-7b",
		[{"template_id": "eval.wrap_no_grad"}],
		None,
		reader=reader,
		blobs=None,
		open_box=no_box,
		key="k",
	)
	assert out == {"error": "sandbox_failed", "usd_from_reported_tokens": 0.0, "candidates": []}
	closed = []

	class Box:
		spent = 0.0

		def write(self, path, text):
			raise OSError("disk full")

		def run(self, argv, **kw):
			raise OSError("disk full")

	out = run.run_race(
		"sample:lora-7b",
		[{"template_id": "eval.wrap_no_grad"}],
		None,
		reader=reader,
		blobs=None,
		open_box=lambda: (Box(), lambda: closed.append(1)),
		key="k",
	)
	assert out["error"] == "race_failed" and closed == [1]


def test_a_fix_request_with_the_limits_stripped_out_is_still_held_to_the_sessions_limits(world):
	c, fixer, *_ = world
	# The visitor confirms "don't change gradient checkpointing" and rules out eval batch 8.
	saved = c.put(
		"/v1/guide/limits/state",
		json={
			"record": "lora-7b",
			"limits": {"frozen": ["gradient_checkpointing"], "excluded": {"eval_batch_size": [8]}},
		},
	)
	assert saved.status_code == 200, saved.text
	# The fix request itself carries no limits at all (and the route takes none).
	for cands in ([THREE[1]], [THREE[2]]):
		r = c.post("/v1/patches", json={"record": "lora-7b", "candidates": cands})
		assert r.status_code == 422 and r.json()["error"]["code"] == "outside_limits"
	assert (
		c.post("/v1/patches", json={"record": "lora-7b", "candidates": THREE[:1], "limits": {}}).status_code
		== 422
	)
	assert c.post("/v1/patches", json={"record": "lora-7b", "candidates": THREE[:1]}).status_code == 202
	assert len(fixer.started) == 1
	# Another session's limits don't apply here, and this session's come back on reload.
	other = TestClient(c.app, base_url="https://testserver")
	assert other.post("/v1/patches", json={"record": "lora-7b", "candidates": [THREE[2]]}).status_code == 202
	got = c.get("/v1/guide/limits/state", params={"record": "lora-7b"}).json()["limits"]
	assert got["frozen"] == ["gradient_checkpointing"] and got["excluded"] == {"eval_batch_size": [8]}
	assert other.get("/v1/guide/limits/state", params={"record": "lora-7b"}).json()["limits"] is None


def test_stored_limits_name_only_what_the_record_has(world):
	c, *_ = world
	r = c.put(
		"/v1/guide/limits/state", json={"record": "lora-7b", "limits": {"allowed": {"gpu_type": ["TPU-v9"]}}}
	)
	assert r.status_code == 422


def test_a_fix_is_judged_by_what_it_really_sets_not_the_value_a_client_sends(world):
	c, fixer, *_ = world
	c.put(
		"/v1/guide/limits/state",
		json={
			"record": "lora-7b",
			"limits": {"excluded": {"eval_no_grad": [True]}, "allowed": {"precision": ["fp32"]}},
		},
	)
	# "Turn no_grad on", claimed as false: still refused, and the false value itself is refused.
	lie = c.post(
		"/v1/patches",
		json={"record": "lora-7b", "candidates": [{"template_id": "eval.wrap_no_grad", "value": False}]},
	)
	assert lie.status_code == 422
	honest = c.post(
		"/v1/patches", json={"record": "lora-7b", "candidates": [{"template_id": "eval.wrap_no_grad"}]}
	)
	assert honest.status_code == 422 and honest.json()["error"]["code"] == "outside_limits"
	# bf16 autocast sets precision to bf16, which the limits don't allow.
	bf16 = c.post(
		"/v1/patches", json={"record": "lora-7b", "candidates": [{"template_id": "precision.bf16_autocast"}]}
	)
	assert bf16.status_code == 422 and bf16.json()["error"]["code"] == "outside_limits"
	assert fixer.started == []


def test_every_template_without_a_value_says_what_it_sets():
	from preflight_api.fix.templates import APPLIES, TEMPLATES

	assert {t.id for t in TEMPLATES.values() if "{value}" not in t.task} == set(APPLIES)
