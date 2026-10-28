"""The end-to-end check's steps against any API client (e2e_upload.py runs them in Modal against
A's deployed job store; tests/test_e2e_flow.py runs them against a stand-in). No Modal here.

Check in a script's files as an upload, follow the job by polling GET /v1/jobs/{id}, read its
events and the record by share token, and time each step. Past the job deadline the job is
cancelled through the API, so no GPU keeps running for a job nobody follows.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Awaitable, Callable

import httpx

TERMINAL = ("done", "failed", "cancelled")
# Source: guess, A's store moves a cancelled job to "cancelled" within a lease renewal (20 s);
# wait twice that before giving up on reading its events.
CANCEL_SETTLE_S = 40.0


def sse_events(text: str) -> list[dict]:
	out = []
	for line in text.splitlines():
		if line.startswith("data: "):
			try:
				out.append(json.loads(line[6:]))
			except ValueError:
				out.append({"unparsed": line[:200]})
	return out


def checks_for(status: dict, record: dict | None, events: list[dict]) -> dict:
	"""What "upload to map on a real GPU" means, checked on what the API served."""
	rec = record or {}
	probes = rec.get("probes") or []
	gpu = ((rec.get("env") or {}).get("gpu") or {}).get("name")
	return {
		"job_done": status.get("status") == "done",
		"record_served": record is not None,
		"map_nodes": len((rec.get("graph") or {}).get("nodes") or []),
		"knobs": len(rec.get("knobs") or []),
		"probes": len(probes),
		"probes_ok": sum(1 for p in probes if p.get("status") in ("ok", "cached")),
		"gpu": gpu,
		"on_gpu": bool(gpu) and str(gpu).upper() != "CPU",
		"estimates": len(((rec.get("predictions") or {}).get("estimates")) or {}),
		"events": len(events),
		"terminal_event": events[-1].get("stage") if events else None,
	}


def passed(checks: dict) -> bool:
	return bool(
		checks.get("job_done")
		and checks.get("record_served")
		and checks.get("on_gpu")
		and checks.get("map_nodes")
	)


async def flow(
	c: httpx.AsyncClient,
	files: dict[str, str],
	*,
	entrypoint: str,
	gpu: str,
	job_deadline_s: float,
	poll_s: float,
	label: str,
	clock: Callable[[], float] = time.monotonic,
	sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> dict:
	t0 = clock()
	marks: dict[str, float] = {}

	def now() -> float:
		return round(clock() - t0, 1)

	def mark(name: str) -> None:
		marks.setdefault(name, now())

	out: dict = {"files": sorted(files), "gpu_asked": gpu, "marks": marks, "statuses": []}
	await c.get("/livez")  # the session cookie, as a browser gets it on first load
	body = {
		"source": {"kind": "upload", "files": files, "entrypoint": entrypoint},
		"objective": {"kind": "min_cost"},
		"gpu": gpu,
		"description": f"Preflight end-to-end check: {label}",
	}
	res = await c.post("/v1/jobs", json=body, headers={"Idempotency-Key": f"e2e-{uuid.uuid4().hex}"})
	out["submit"] = {"status": res.status_code, "body": res.json()}
	out["checks"] = checks_for({}, None, [])
	if res.status_code != 202:
		return out
	job_id = res.json()["job_id"]
	out["job_id"] = job_id
	mark("submitted")

	status: dict = {}
	last = None
	cancel_by: float | None = None
	while True:
		r = await c.get(f"/v1/jobs/{job_id}")
		status = r.json() if r.status_code == 200 else {"http": r.status_code, "body": r.text[:500]}
		state = (status.get("status"), status.get("position"))
		if state != last:
			out["statuses"].append({"at": now(), "status": state[0], "position": state[1]})
			last = state
			if state[0]:
				mark(f"status.{state[0]}")
		if status.get("share_token"):
			mark("share_token")
		if status.get("status") in TERMINAL:
			break
		elapsed = clock() - t0
		if cancel_by is None and elapsed > job_deadline_s:
			cancel = await c.post(f"/v1/jobs/{job_id}/cancel")
			out["cancelled_at_deadline"] = cancel.status_code
			mark("cancel_sent")
			cancel_by = elapsed + CANCEL_SETTLE_S
		elif cancel_by is not None and elapsed > cancel_by:
			out["cancel_unconfirmed"] = True
			break
		await sleep(poll_s)
	out["status"] = status

	events: list[dict] = []
	if status.get("status") in TERMINAL:  # the stream ends at the terminal event, so this returns
		ev = await c.get(f"/v1/jobs/{job_id}/events", params={"after": -1})
		events = sse_events(ev.text) if ev.status_code == 200 else []
	out["events"] = events

	record = None
	token = status.get("share_token") or next(
		(e.get("share_token") for e in events if e.get("share_token")), None
	)
	if token:
		t1 = clock()
		rr = await c.get(f"/v1/records/{token}")
		out["record_read_s"] = round(clock() - t1, 3)
		out["record_http"] = rr.status_code
		out["record_cache_control"] = rr.headers.get("Cache-Control")
		if rr.status_code == 200:
			record = rr.json()
			mark("record_read")
	out["share_token"] = token
	out["record"] = record
	out["checks"] = checks_for(status, record, events)
	return out
