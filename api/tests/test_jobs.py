"""/v1/jobs over a stand-in for A's job store service."""

import json

import pytest
from fastapi.testclient import TestClient

from preflight_api import flags, session
from preflight_api.app import create_app
from preflight_api.jobs import visitor_of
from preflight_api.ledger import LedgerCore, Limits, MemoryLedger

BODY = {
	"source": {"kind": "repo", "uri": "https://github.com/karpathy/nanoGPT"},
	"objective": {"kind": "min_cost"},
	"gpu": "L4",
}
KEY = {"Idempotency-Key": "k-0123456789"}


class QueueFull(RuntimeError):
	def __init__(self, msg, retry_after_s=60):
		super().__init__(msg)
		self.retry_after_s = retry_after_s


class UnknownJob(KeyError):
	pass


class Store:
	"""Behaves like A's StoreMethods: idempotent per (visitor, key), events by seq."""

	def __init__(self, full=False):
		self.full = full
		self.jobs: dict[str, dict] = {}
		self.keys: dict[tuple[str, str], str] = {}
		self.events_by: dict[str, list[dict]] = {}

	async def submit(self, request, *, visitor, idempotency_key):
		if self.full:
			raise QueueFull("busy", retry_after_s=90)
		k = (visitor, idempotency_key)
		if k in self.keys:
			return {"job_id": self.keys[k], "created": False, "position": 0}
		jid = f"01JOB{len(self.jobs):021d}"
		self.jobs[jid] = {"request": request, "visitor": visitor}
		self.keys[k] = jid
		self.events_by[jid] = [
			{"job_id": jid, "seq": 0, "at": "2026-10-27T00:00:00Z", "stage": "queued", "queue_position": 0},
			{"job_id": jid, "seq": 1, "at": "2026-10-27T00:00:01Z", "stage": "probing"},
			{"job_id": jid, "seq": 2, "at": "2026-10-27T00:00:02Z", "stage": "done", "record_id": "r1"},
		]
		return {"job_id": jid, "created": True, "position": 0}

	async def status(self, job_id):
		if job_id not in self.jobs:
			raise UnknownJob(job_id)
		return {
			"job_id": job_id,
			"status": "done" if self.jobs[job_id].get("over") else "queued",
			"position": 0,
		}

	async def events(self, job_id, after=-1):
		if job_id not in self.jobs:
			raise UnknownJob(job_id)
		return [e for e in self.events_by[job_id] if e["seq"] > after]

	async def cancel(self, job_id):
		self.jobs[job_id]["cancelled"] = True


@pytest.fixture(autouse=True)
def env(monkeypatch):
	monkeypatch.setenv("PREFLIGHT_FLAGS", json.dumps({"runner.queue": True}))
	monkeypatch.setenv("PREFLIGHT_SESSION_KEYS", "k" * 40)
	session.reset_keys()
	flags.reload()
	yield
	flags.reload()
	session.reset_keys()


def client(store, **kw):
	return TestClient(create_app(jobstore=store, **kw), base_url="https://testserver")


def test_check_in_is_idempotent_and_hashes_the_visitor():
	store = Store()
	c = client(store)
	r1 = c.post("/v1/jobs", json=BODY, headers=KEY)
	r2 = c.post("/v1/jobs", json=BODY, headers=KEY)
	assert r1.status_code == 202 and r1.json()["created"] is True
	assert r2.json() == {**r1.json(), "created": False}
	((visitor, _),) = store.keys
	sid = c.cookies.get("__Host-pf_session").split(".")[1]
	assert visitor == visitor_of(sid) and sid not in visitor


def test_check_in_needs_a_key_a_known_source_and_https():
	c = client(Store())
	assert c.post("/v1/jobs", json=BODY).json()["error"]["code"] == "idempotency_key_required"
	bad = {**BODY, "source": {"kind": "inflight"}}
	assert c.post("/v1/jobs", json=bad, headers=KEY).status_code == 422
	plain = {**BODY, "source": {"kind": "repo", "uri": "http://github.com/x/y"}}
	assert c.post("/v1/jobs", json=plain, headers=KEY).status_code == 422
	extra = {**BODY, "surprise": 1}
	assert c.post("/v1/jobs", json=extra, headers=KEY).status_code == 422


def test_a_full_queue_is_429_with_retry_after():
	r = client(Store(full=True)).post("/v1/jobs", json=BODY, headers=KEY)
	assert r.status_code == 429
	assert r.headers.get("Retry-After") == "90"
	assert r.json()["error"]["code"] == "queue_full"


def test_not_open_without_a_store_or_with_the_flag_off(monkeypatch):
	c = TestClient(create_app(), base_url="https://testserver")
	assert c.post("/v1/jobs", json=BODY, headers=KEY).json()["error"]["code"] == "jobs_not_open"
	monkeypatch.setenv("PREFLIGHT_FLAGS", json.dumps({"runner.queue": False}))
	flags.reload()
	assert client(Store()).post("/v1/jobs", json=BODY, headers=KEY).status_code == 503


def test_events_stream_until_done_and_resume_after_last_event_id():
	store = Store()
	c = client(store)
	jid = c.post("/v1/jobs", json=BODY, headers=KEY).json()["job_id"]
	r = c.get(f"/v1/jobs/{jid}/events")
	assert r.headers["content-type"].startswith("text/event-stream")
	stages = [json.loads(line[6:])["stage"] for line in r.text.splitlines() if line.startswith("data: ")]
	assert stages == ["queued", "probing", "done"]
	again = c.get(f"/v1/jobs/{jid}/events", headers={"Last-Event-ID": "1"})
	assert [json.loads(x[6:])["seq"] for x in again.text.splitlines() if x.startswith("data: ")] == [2]
	# After the last event of a finished job: 204, so the browser stops reconnecting.
	store.jobs[jid]["over"] = True
	assert c.get(f"/v1/jobs/{jid}/events", headers={"Last-Event-ID": "2"}).status_code == 204


def test_unknown_jobs_are_404_and_cancel_reaches_the_store():
	store = Store()
	c = client(store)
	assert c.get("/v1/jobs/01NOPE000000000000000000000").status_code == 404
	assert c.get("/v1/jobs/../etc").status_code == 404
	jid = c.post("/v1/jobs", json=BODY, headers=KEY).json()["job_id"]
	assert c.post(f"/v1/jobs/{jid}/cancel").status_code == 204
	assert store.jobs[jid]["cancelled"] is True


def test_the_ip_bucket_limits_check_ins_from_one_address():
	limits = Limits(ip_burst=2, ip_refill_per_s=0.0)
	c = client(Store(), ledger=MemoryLedger(LedgerCore(limits)))
	codes = [
		c.post("/v1/jobs", json=BODY, headers={"Idempotency-Key": f"k-{i:012d}"}).status_code
		for i in range(3)
	]
	assert codes == [202, 202, 429]
