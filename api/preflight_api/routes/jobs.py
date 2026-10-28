"""/v1/jobs: check a job in, follow it, cancel it. Validated here (C), run by A's job store.

POST   /v1/jobs                 JobRequest + Idempotency-Key -> {job_id, created, position}
GET    /v1/jobs/{job_id}        status, queue position, record link once it exists, why it failed
GET    /v1/jobs/{job_id}/events Server-Sent Events, resumable with Last-Event-ID
POST   /v1/jobs/{job_id}/cancel

Job ids are ULIDs with 80 random bits, so knowing one is the capability to follow it, as with a
record's share token.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import ValidationError

from .. import edge
from ..errors import ApiError
from ..flags import flag
from ..jobs import kind_of, retry_after, visitor_of

router = APIRouter(prefix="/v1")

JOB_ID = re.compile(r"^[0-9A-Za-z_-]{8,64}$")
IDEMPOTENCY_KEY = re.compile(r"^[A-Za-z0-9_.:-]{8,128}$")
TERMINAL = {"done", "failed", "cancelled"}
SOURCE_KINDS = {"repo", "upload", "notebook", "sample", "template"}
MAX_BODY_BYTES = 6 * 2**20  # source: derived, the upload limit shown at check-in (5 MB) plus JSON overhead

# The event stream. Source: guess, one poll a second keeps a stage change under a second late on
# the boarding screen; replaced if A's store gains a blocking wait.
POLL_S = 1.0
# Source: derived, Cloudflare closes a proxied response idle for 100 s; a comment every 20 s
# keeps it open with room to spare.
HEARTBEAT_S = 20.0
# Source: derived, a probe job is about 4 minutes on one GPU (check-in copy) plus a queue wait;
# 15 minutes covers both, after which the browser reconnects with Last-Event-ID.
STREAM_MAX_S = 900.0

NOT_OPEN = "Live probes aren't open yet. Open a sample to explore the cockpit."


def _seconds(value: float | None, default: int = 60) -> int:
	"""A Retry-After header value: whole seconds, at least 1, at most an hour."""
	if value is None or not math.isfinite(value) or value <= 0:
		return default
	return max(1, min(3600, math.ceil(value)))


def _store(request: Request):
	store = request.app.state.jobstore
	if store is None or not flag("runner.queue"):
		raise ApiError(503, "jobs_not_open", NOT_OPEN)
	return store


def _job_id(job_id: str) -> str:
	if not JOB_ID.match(job_id):
		raise ApiError(404, "job_not_found", "No job matches that id.")
	return job_id


def _store_error(e: Exception) -> ApiError:
	k = kind_of(e)
	if k == "QueueFull":
		s = retry_after(e)
		return ApiError(
			429,
			"queue_full",
			f"All probe GPUs are busy. Try again in {s} seconds, or open a sample meanwhile.",
			retryable=True,
			retry_after=s,
		)
	if k == "StoreUnavailable":
		return ApiError(503, "store_unavailable", "The job queue is busy; try again shortly.", retryable=True)
	if k == "UnknownJob":
		return ApiError(404, "job_not_found", "No job matches that id.")
	if k == "IdempotencyConflict":
		return ApiError(
			409, "idempotency_conflict", "That check-in key was already used for a different job."
		)
	return ApiError(502, "store_error", "The job queue didn't answer; try again shortly.", retryable=True)


@router.post("/jobs", status_code=202)
async def create_job(request: Request) -> dict:
	store = _store(request)
	key = request.headers.get("Idempotency-Key", "")
	if not IDEMPOTENCY_KEY.match(key):
		raise ApiError(
			400, "idempotency_key_required", "Send an Idempotency-Key header (8 to 128 characters)."
		)
	raw = await request.body()
	if len(raw) > MAX_BODY_BYTES:
		raise ApiError(413, "too_large", "That job is larger than check-in accepts (5 MB of code).")
	from preflight.schema import JobRequest
	from preflight.schema.record import Source

	try:
		body = JobRequest.model_validate_json(raw)
		source = Source.model_validate({k: v for k, v in body.source.items() if k != "files"})
	except (ValidationError, ValueError) as e:
		raise ApiError(422, "invalid_request", "The check-in isn't in the expected shape.") from e
	if source.kind not in SOURCE_KINDS:
		raise ApiError(422, "invalid_request", f"Jobs can't come from {source.kind!r} here.")
	if source.kind == "repo" and not (source.uri or "").startswith("https://"):
		raise ApiError(422, "invalid_request", "Repo links must start with https://.")

	sid: str = request.state.session
	# The store caps jobs per visitor; the IP bucket stops a visitor minting sessions to get round it.
	ledger = request.app.state.ledger
	gate = await ledger.reserve(
		visitor=sid, run=f"jobs:{sid}", ip=edge.ip_key(edge.client_ip(request)), usd=0.0
	)
	if not gate.ok:
		raise ApiError(
			429,
			"rate_limited",
			"Too many check-ins from here; try again shortly.",
			retryable=True,
			retry_after=_seconds(gate.retry_after_s),
		)
	await ledger.settle(gate.reservation, 0.0)
	try:
		out = await store.submit(
			body.model_dump(mode="json", exclude_none=True), visitor=visitor_of(sid), idempotency_key=key
		)
	except Exception as e:
		raise _store_error(e) from e
	return {"job_id": out["job_id"], "created": bool(out.get("created")), "position": out.get("position")}


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, request: Request) -> dict:
	store = _store(request)
	try:
		return await store.status(_job_id(job_id))
	except ApiError:
		raise
	except Exception as e:
		raise _store_error(e) from e


@router.post("/jobs/{job_id}/cancel", status_code=204)
async def cancel_job(job_id: str, request: Request) -> Response:
	store = _store(request)
	try:
		await store.cancel(_job_id(job_id))
	except ApiError:
		raise
	except Exception as e:
		raise _store_error(e) from e
	return Response(status_code=204)


def _sse(event: dict) -> str:
	# Unnamed events, so EventSource.onmessage receives them.
	return f"id: {event.get('seq', '')}\ndata: {json.dumps(event, separators=(',', ':'))}\n\n"


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: str, request: Request) -> Response:
	store = _store(request)
	_job_id(job_id)
	last = request.headers.get("Last-Event-ID") or request.query_params.get("after") or "-1"
	try:
		after = int(last)
	except ValueError:
		after = -1
	try:
		first = await store.events(job_id, after)  # 404 before the stream starts, not inside it
		if not first and (await store.status(job_id)).get("status") in TERMINAL:
			# Nothing new and the job is over: 204 tells EventSource to stop reconnecting.
			return Response(status_code=204)
	except Exception as e:
		raise _store_error(e) from e

	async def stream() -> AsyncIterator[str]:
		nonlocal after
		started = beat = time.monotonic()
		batch = first
		while True:
			for ev in batch:
				after = max(after, int(ev.get("seq", after)))
				yield _sse(ev)
				if ev.get("stage") in TERMINAL:
					return
			now = time.monotonic()
			if now - started > STREAM_MAX_S or await request.is_disconnected():
				return
			if now - beat > HEARTBEAT_S:
				beat = now
				yield ": still here\n\n"
			await asyncio.sleep(POLL_S)
			try:
				batch = await store.events(job_id, after)
			except Exception as e:
				err = _store_error(e)
				if not err.retryable:
					yield f"event: error\ndata: {json.dumps({'code': err.code, 'message': err.message})}\n\n"
					return
				batch = []

	return StreamingResponse(
		stream(),
		media_type="text/event-stream",
		headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
	)
