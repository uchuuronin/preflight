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
# Source: guess, every API call here is one round trip to A's store (a Modal method call, seconds
# at most); 30 s means it hung. httpx's ASGITransport ignores client timeouts, so this is the
# only bound on a call.
CALL_TIMEOUT_S = 30.0
# Source: guess, three tries 5 s apart ride out a store restart (its container restores from a
# Dict in seconds); the same Idempotency-Key makes a retry return the job already created.
SUBMIT_TRIES = 3
SUBMIT_RETRY_S = 5.0
# Source: derived, A's cancel is synchronous in the store, but the worker only stops at its next
# lease renewal (A's LEASE_S is 60 s); a status that isn't terminal a minute after the first
# cancel means the cancel didn't land.
CANCEL_SETTLE_S = 60.0


class Hung(Exception):
	"""An API call that didn't answer within CALL_TIMEOUT_S."""


async def _call(coro, limit_s: float | None = None):
	limit_s = CALL_TIMEOUT_S if limit_s is None else limit_s
	try:
		return await asyncio.wait_for(coro, limit_s)
	except TimeoutError as e:
		raise Hung(str(limit_s)) from e


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
	"""Upload to map on a real GPU: the job finished, its record came back through the API with a
	map, and at least one probe measured something on a GPU."""
	return bool(
		checks.get("job_done")
		and checks.get("record_served")
		and checks.get("on_gpu")
		and checks.get("map_nodes")
		and checks.get("probes_ok")
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
	note: Callable[[str, str], Awaitable[None]] | None = None,
	clock: Callable[[], float] = time.monotonic,
	sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> dict:
	t0 = clock()
	marks: dict[str, float] = {}

	def now() -> float:
		return round(clock() - t0, 1)

	def mark(name: str) -> None:
		marks.setdefault(name, now())

	out: dict = {"files": sorted(files), "gpu_asked": gpu, "marks": marks, "statuses": [], "call_errors": []}
	out["checks"] = checks_for({}, None, [])
	body = {
		"source": {"kind": "upload", "files": files, "entrypoint": entrypoint},
		"objective": {"kind": "min_cost"},
		"gpu": gpu,
		"description": f"Preflight end-to-end check: {label}",
	}
	key = f"e2e-{uuid.uuid4().hex}"

	async def tell(name: str, value: str) -> None:
		"""Hand a fact to the caller (the job id, the check-in key) without ever failing the run."""
		if note is None:
			return
		try:
			await _call(note(name, value))
		except Exception as e:
			out["call_errors"].append({"at": now(), "call": f"note {name}", "error": type(e).__name__})

	await _call(c.get("/livez"))  # the session cookie, as a browser gets it on first load
	# Before submitting: if every try hangs after A took the job, the key still finds it by hand.
	await tell("idempotency_key", key)

	job_id = None
	for attempt in range(SUBMIT_TRIES):
		try:
			res = await _call(c.post("/v1/jobs", json=body, headers={"Idempotency-Key": key}))
		except Hung:
			out["call_errors"].append({"at": now(), "call": "submit", "error": "hung"})
		else:
			out["submit"] = {"status": res.status_code, "body": res.json(), "attempt": attempt + 1}
			if res.status_code == 202:
				job_id = res.json()["job_id"]
				break
			if res.status_code < 500:
				return out  # refused: nothing was queued
		await sleep(SUBMIT_RETRY_S)
	if job_id is None:
		return out
	out["job_id"] = job_id
	mark("submitted")

	status: dict = {}
	try:
		await tell("job_id", job_id)
		last = None
		cancel_from: float | None = None
		while True:
			try:
				r = await _call(c.get(f"/v1/jobs/{job_id}"))
				status = r.json() if r.status_code == 200 else {"http": r.status_code, "body": r.text[:500]}
			except Hung:
				out["call_errors"].append({"at": now(), "call": "status", "error": "hung"})
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
			if elapsed > job_deadline_s:
				# Every poll past the deadline asks again, until the store says it's over.
				cancel_from = elapsed if cancel_from is None else cancel_from
				mark("cancel_sent")
				try:
					cancel = await _call(c.post(f"/v1/jobs/{job_id}/cancel"))
					out.setdefault("cancel_replies", []).append(cancel.status_code)
				except Hung:
					out["call_errors"].append({"at": now(), "call": "cancel", "error": "hung"})
				if elapsed - cancel_from > CANCEL_SETTLE_S:
					out["cancel_unconfirmed"] = True
					break
			await sleep(poll_s)
	finally:
		if status.get("status") not in TERMINAL:
			# Leaving early for any reason (an error, the caller's timeout): stop the job first.
			try:
				await _call(c.post(f"/v1/jobs/{job_id}/cancel"), 10.0)
				out["cancelled_on_exit"] = True
			except Exception as e:  # the caller's own cancel through A's store is the backstop
				out["cancelled_on_exit"] = f"{type(e).__name__}"
	out["status"] = status

	events: list[dict] = []
	if status.get("status") in TERMINAL:  # the stream ends at the terminal event, so this returns
		try:
			ev = await _call(c.get(f"/v1/jobs/{job_id}/events", params={"after": -1}))
			events = sse_events(ev.text) if ev.status_code == 200 else []
		except Hung:
			out["call_errors"].append({"at": now(), "call": "events", "error": "hung"})
	out["events"] = events

	record = None
	token = status.get("share_token") or next(
		(e.get("share_token") for e in events if e.get("share_token")), None
	)
	if token:
		t1 = clock()
		try:
			rr = await _call(c.get(f"/v1/records/{token}"))
			out["record_read_s"] = round(clock() - t1, 3)
			out["record_http"] = rr.status_code
			out["record_cache_control"] = rr.headers.get("Cache-Control")
			if rr.status_code == 200:
				record = rr.json()
				mark("record_read")
		except Hung:
			out["call_errors"].append({"at": now(), "call": "record", "error": "hung"})
	out["share_token"] = token
	out["record"] = record
	out["checks"] = checks_for(status, record, events)
	return out
