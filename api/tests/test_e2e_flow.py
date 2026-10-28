"""The end-to-end check's steps (api/scripts/e2e_flow.py) over the real app and a stand-in for A's
job store that writes the record file as A's workers do, so the flow is proven before it spends."""

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

from preflight_api import flags
from preflight_api.app import create_app
from preflight_api.ledger import MemoryLedger
from preflight_api.stores import LocalRecordStore, new_share_token

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from e2e_flow import flow, passed

RECORD = {
	"schema_version": "0.3.0",
	"record_id": "01REC",
	"job": {"lens": "finetune"},
	"env": {"gpu": {"name": "L4"}},
	"graph": {"nodes": [{"id": "model"}, {"id": "loss"}]},
	"knobs": [{"id": "batch_size"}],
	"probes": [{"id": "p1", "status": "ok"}, {"id": "p2", "status": "oom"}],
	"predictions": {"estimates": {"memory.peak": {"point": 1.0}}},
}


class A:
	"""A's job store as the API sees it: each status call moves the job one step on."""

	def __init__(self, root: Path, steps=("queued", "running", "running", "done")):
		self.root, self.steps = root, list(steps)
		self.token = new_share_token()
		self.request = None
		self.cancelled = False
		self.calls = 0

	async def submit(self, request, *, visitor, idempotency_key):
		self.request = request
		return {"job_id": "01JOBE2E000000000000000000", "created": True, "position": 0}

	async def status(self, job_id):
		i = min(self.calls, len(self.steps) - 1)
		self.calls += 1
		st = "cancelled" if self.cancelled else self.steps[i]
		out = {"job_id": job_id, "status": st, "position": 0 if st == "queued" else None}
		if st != "queued":
			(self.root / f"{self.token}.json").write_text(json.dumps({**RECORD, "share_token": self.token}))
			out["share_token"] = self.token
		return out

	async def events(self, job_id, after=-1):
		evs = [
			{"job_id": job_id, "seq": 0, "at": "t", "stage": "queued"},
			{"job_id": job_id, "seq": 1, "at": "t", "stage": "skeleton", "share_token": self.token},
			{"job_id": job_id, "seq": 2, "at": "t", "stage": "cancelled" if self.cancelled else "done"},
		]
		return [e for e in evs if e["seq"] > after]

	async def cancel(self, job_id):
		self.cancelled = True


@pytest.fixture(autouse=True)
def env(monkeypatch):
	monkeypatch.setenv("PREFLIGHT_FLAGS", json.dumps({"runner.queue": True}))
	monkeypatch.setenv("PREFLIGHT_SESSION_KEYS", "k" * 40)
	flags.reload()
	yield
	flags.reload()


class Clock:
	def __init__(self):
		self.t = 0.0

	def __call__(self):
		return self.t

	async def sleep(self, s):
		self.t += s


def run(tmp_path, a, deadline=600.0):
	return asyncio.run(_run(tmp_path, a, deadline))


async def _run(tmp_path, a, deadline):
	api = create_app(records=LocalRecordStore(tmp_path), ledger=MemoryLedger(), jobstore=a)
	clock = Clock()
	async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api), base_url="https://testserver") as c:
		return await flow(
			c,
			{"train.py": "print(1)\n"},
			entrypoint="train.py",
			gpu="L4",
			job_deadline_s=deadline,
			poll_s=3.0,
			label="zoo/test",
			clock=clock,
			sleep=clock.sleep,
		)


def test_upload_to_record_through_the_api(tmp_path):
	a = A(tmp_path)
	out = run(tmp_path, a)
	assert a.request["source"] == {
		"kind": "upload",
		"files": {"train.py": "print(1)\n"},
		"entrypoint": "train.py",
	}
	assert out["submit"]["status"] == 202
	assert [s["status"] for s in out["statuses"]] == ["queued", "running", "done"]
	assert out["share_token"] == a.token
	assert out["record_http"] == 200 and "no-cache" in out["record_cache_control"]
	c = out["checks"]
	assert c == {
		**c,
		"job_done": True,
		"record_served": True,
		"map_nodes": 2,
		"probes": 2,
		"probes_ok": 1,
		"gpu": "L4",
		"terminal_event": "done",
	}
	assert passed(c)
	assert out["marks"]["status.done"] == 9.0  # three polls at 3 s


def test_past_the_deadline_the_job_is_cancelled_and_the_run_fails(tmp_path):
	a = A(tmp_path, steps=("queued", "running"))
	out = run(tmp_path, a, deadline=5.0)
	assert a.cancelled
	assert out["cancelled_at_deadline"] == 204
	assert out["status"]["status"] == "cancelled"
	assert not passed(out["checks"])


def test_a_refused_check_in_stops_before_polling(tmp_path, monkeypatch):
	monkeypatch.setenv("PREFLIGHT_FLAGS", json.dumps({"runner.queue": False}))
	flags.reload()
	a = A(tmp_path)
	out = run(tmp_path, a)
	assert out["submit"]["status"] == 503
	assert a.calls == 0
	assert not passed(out["checks"])
