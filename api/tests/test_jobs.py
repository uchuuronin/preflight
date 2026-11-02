"""/v1/jobs over a stand-in for A's job store service."""

import json
from urllib.parse import unquote

import httpx
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
# A check-in key the way the web app makes them (a request id, not a secret).
KEY = {"Idempotency-Key": "checkin-aaaa-0001"}


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


SHA = "a" * 40


def github(req: httpx.Request) -> httpx.Response:
	"""GitHub's commits API: main and v1 exist on karpathy/nanoGPT; nothing else does."""
	refs = {"HEAD": SHA, "main": SHA, "v1": "b" * 40}
	m = [unquote(x) for x in req.url.raw_path.decode().split("?")[0].split("/")]
	if req.url.host == "api.github.com" and m[2:4] == ["karpathy", "nanoGPT"] and m[5] in refs:
		return httpx.Response(200, text=refs[m[5]])
	if m[5:6] == ["limited"]:
		return httpx.Response(403, text="rate limited")
	return httpx.Response(404, text="Not Found")


def client(store, **kw):
	kw.setdefault("http", httpx.AsyncClient(transport=httpx.MockTransport(github)))
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
	elsewhere = {**BODY, "source": {"kind": "repo", "uri": "https://gitlab.com/x/y"}}
	assert c.post("/v1/jobs", json=elsewhere, headers=KEY).json()["error"]["code"] == "repo_not_usable"
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


def test_branch_links_become_the_bare_repo_and_a_full_commit():
	store = Store()
	c = client(store)
	cases = [
		("github.com/karpathy/nanoGPT", None, SHA, "HEAD"),
		("https://github.com/karpathy/nanoGPT/tree/main/data", None, SHA, "main"),
		("https://github.com/karpathy/nanoGPT.git", "v1", "b" * 40, "v1"),
		("https://github.com/karpathy/nanoGPT", "c" * 40, "c" * 40, "c" * 40),
	]
	for i, (uri, commit, sha, ref) in enumerate(cases):
		src = {"kind": "repo", "uri": uri, **({"commit": commit} if commit else {})}
		r = c.post("/v1/jobs", json={**BODY, "source": src}, headers={"Idempotency-Key": f"k-case-{i:06d}"})
		assert r.status_code == 202, r.text
		assert r.json()["source"] == {"uri": "https://github.com/karpathy/nanoGPT", "commit": sha, "ref": ref}
		sent = store.jobs[r.json()["job_id"]]["request"]["source"]
		assert sent["uri"] == "https://github.com/karpathy/nanoGPT" and sent["commit"] == sha


def test_unknown_branches_and_rate_limits_are_explained():
	c = client(Store())
	gone = {
		**BODY,
		"source": {"kind": "repo", "uri": "https://github.com/karpathy/nanoGPT", "commit": "nope"},
	}
	assert c.post("/v1/jobs", json=gone, headers=KEY).json()["error"]["code"] == "repo_not_usable"
	busy = {**BODY, "source": {"kind": "repo", "uri": "https://github.com/x/y", "commit": "limited"}}
	r = c.post("/v1/jobs", json=busy, headers={"Idempotency-Key": "k-limited-0001"})
	assert r.status_code == 503 and r.json()["error"]["code"] == "repo_unavailable"


def test_run_a_probe_here_takes_one_setting_of_a_real_record_only():
	store = Store()
	c = client(store)
	ref = "test-share-ref-000001"
	probe = {
		"source": {"kind": "record", "record_ref": ref},
		"mode": "probe",
		"knobs": {"seq_len": 2048, "batch_size": 8},
	}
	r = c.post("/v1/jobs", json=probe, headers=KEY)
	assert r.status_code == 202
	sent = next(iter(store.jobs.values()))["request"]
	assert sent["mode"] == "probe" and sent["knobs"] == {"seq_len": 2048, "batch_size": 8}
	assert sent["purpose"] == "cockpit.run_probe_here"  # whatever the browser said
	other = {"Idempotency-Key": "checkin-aaaa-0002"}
	for bad in (
		{**probe, "source": {"kind": "record", "record_ref": "lora-7b"}},  # a sample: nothing to run
		{**probe, "source": {"kind": "record", "record_ref": "sample:lora-7b"}},
		{**probe, "mode": "verify"},
		{**probe, "mode": "sample_more", "knobs": None, "tasks": 20},
		{**BODY, "mode": "probe", "knobs": {"seq_len": 1}},  # a repo check-in can't name a mode
	):
		assert c.post("/v1/jobs", json=bad, headers=other).status_code == 422, bad
	assert len(store.jobs) == 1


def test_a_daily_limit_is_429_in_the_job_stores_own_words():
	"""A's demo limits (Nov 1): QueueFull with a `limit.` code and its message for the visitor."""

	class Limited(Store):
		async def submit(self, request, *, visitor, idempotency_key):
			e = QueueFull(
				"You've run today's 3 probe jobs: samples only until midnight ET", retry_after_s=3600
			)
			e.code = "limit.visitor_daily"
			raise e

	r = client(Limited()).post("/v1/jobs", json=BODY, headers=KEY)
	assert r.status_code == 429
	assert r.headers.get("Retry-After") == "3600"
	err = r.json()["error"]
	assert err["code"] == "limit.visitor_daily"
	# A's way past, less what A's own words already say (midnight, the samples), in plain text.
	assert err["message"] == (
		"You've run today's 3 probe jobs: samples only until midnight ET. You can also probe on your"
		" machine with the preflight run command."
	)
	# A busy queue keeps its own code and wording.
	assert (
		client(Store(full=True)).post("/v1/jobs", json=BODY, headers=KEY).json()["error"]["code"]
		== "queue_full"
	)


def test_packages_lists_what_the_probes_install_by_import_name():
	r = client(Store()).get("/v1/packages")
	assert r.status_code == 200
	body = r.json()
	assert body["known"]["torchvision"] == "torchvision"
	assert body["known"]["torch_geometric"] == "torch-geometric"
	assert "torch_sparse" in body["no_wheels"]
