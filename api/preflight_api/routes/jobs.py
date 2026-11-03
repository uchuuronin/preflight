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
import contextlib
import json
import math
import re
import time
from collections.abc import AsyncIterator

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import Response, StreamingResponse
from pydantic import ValidationError

from .. import edge, repos
from ..errors import ApiError
from ..flags import flag
from ..jobs import kind_of, limit_of, retry_after, visitor_of

router = APIRouter(prefix="/v1")

JOB_ID = re.compile(r"^[0-9A-Za-z_-]{8,64}$")
IDEMPOTENCY_KEY = re.compile(r"^[A-Za-z0-9_.:-]{8,128}$")
TERMINAL = {"done", "failed", "cancelled"}
SOURCE_KINDS = {"repo", "upload", "notebook", "sample", "template"}
# The cockpit's "Run a probe here": one probe at one setting of a real record, never a sample's
# (a sample has no script to run). The visitor's click is the approval; quotas are A's as for any job.
RECORD_REF = re.compile(r"^[A-Za-z0-9_-]{16,64}$")
PROBE_HERE_PURPOSE = "cockpit.run_probe_here"
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


# A check-in by command (contract 0.6.3, runner.command_checkin, A's ask Nov 2): the whole command line as
# the source's args, e.g. ["lm_eval", "--model", "hf", ...]. The runner reads it (runner/sources.py
# `_command`, then `_args`) and refuses what this lets through, after it has used one of the visitor's
# daily jobs; so this checks the same limits first, read from the runner where it can be.


def _command_rules() -> tuple[tuple[str, ...], int, int, int]:
	"""The command names, and the runner's limits: words, characters a word, characters in all."""
	try:
		from preflight.runner.sources import COMMANDS, MAX_ARG_LEN, MAX_ARGS, MAX_COMMAND_CHARS
	except Exception:
		# source: A's runner/sources.py as of Nov 2, if the runner's module can't be read here
		return ("lm_eval",), 64, 4096, 8192
	return tuple(sorted(COMMANDS)), MAX_ARGS, MAX_ARG_LEN, MAX_COMMAND_CHARS


def _check_command(source) -> None:
	if not flag("runner.command_checkin"):
		raise ApiError(
			422,
			"invalid_request",
			"Checking in by command isn't open here yet; check in the repository or a script.",
		)
	known, max_words, max_word, max_chars = _command_rules()
	words = list(source.args or [])
	if not words or not words[0].strip():
		raise ApiError(
			422, "invalid_request", "Name the command to run, as its words: lm_eval --model hf ..."
		)
	if (
		len(words) > max_words
		or sum(len(w) for w in words) > max_chars
		or any(len(w) > max_word or "\0" in w for w in words)
	):
		raise ApiError(
			422,
			"invalid_request",
			f"A command here is up to {max_words} words of up to {max_word:,} characters each.",
		)
	if words[0] not in known:
		raise ApiError(422, "invalid_request", f"Preflight runs these commands by name: {', '.join(known)}.")


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
	lim = limit_of(e)
	if lim:
		# A's daily limits: its own words ("samples only until midnight ET"), retry at midnight.
		code, message = lim
		return ApiError(
			429,
			code,
			message,
			retryable=True,
			retry_after=retry_after(e),
		)
	if k == "QueueFull":
		s = retry_after(e)
		return ApiError(
			429,
			"queue_full",
			# A's queue_full covers busy GPUs, the visitor's own queued jobs and budget held by jobs in
			# flight (A, Nov 2), so the words don't blame the GPUs alone.
			f"The queue can't take this job right now. Try again in {s} seconds, or open a sample meanwhile.",
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


@router.get("/packages")
async def packages() -> dict:
	"""The packages Preflight's probes install by name, keyed by the import name a script uses (A's
	`preflight.recorder.requirements.KNOWN_MODULES`), and the ones it can't (no wheels on PyPI):
	Boarding fills its "extra packages" box from a missing import's name (C, Nov 2)."""
	try:
		from preflight.recorder.requirements import KNOWN_MODULES, NO_WHEELS
	except Exception:
		return {"known": {}, "no_wheels": []}
	return {"known": dict(sorted(KNOWN_MODULES.items())), "no_wheels": sorted(NO_WHEELS)}


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
	if source.kind == "record":
		ref = source.record_ref or ""
		if body.mode != "probe" or not body.knobs or not RECORD_REF.match(ref):
			raise ApiError(
				422, "invalid_request", "A probe on a record needs mode probe, the setting and a share token."
			)
		body = body.model_copy(update={"purpose": PROBE_HERE_PURPOSE})
	elif source.kind == "command":
		_check_command(source)
		if body.mode != "plan":
			raise ApiError(422, "invalid_request", "Only a record's probe can name a mode here.")
	elif source.kind not in SOURCE_KINDS:
		raise ApiError(422, "invalid_request", f"Jobs can't come from {source.kind!r} here.")
	elif body.mode != "plan":
		raise ApiError(422, "invalid_request", "Only a record's probe can name a mode here.")
	request_body = body.model_dump(mode="json", exclude_none=True)
	resolved = None
	if source.kind == "repo":
		# A's runner fetches GitHub only by a full commit hash: resolve the branch, tag or default
		# branch now, so the job runs the code the visitor saw (preflight_api/repos.py).
		http = request.app.state.http
		own = http is None
		client = http or httpx.AsyncClient()
		try:
			resolved = await repos.resolve(source.uri or "", source.commit, client)
		except repos.RepoError as e:
			if e.retryable:
				raise ApiError(503, "repo_unavailable", str(e), retryable=True, retry_after=30) from e
			raise ApiError(422, "repo_not_usable", str(e)) from e
		finally:
			if own:
				await client.aclose()
		request_body["source"] = {**request_body["source"], "uri": resolved.uri, "commit": resolved.commit}

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
		out = await store.submit(request_body, visitor=visitor_of(sid), idempotency_key=key)
	except Exception as e:
		raise _store_error(e) from e
	reply = {"job_id": out["job_id"], "created": bool(out.get("created")), "position": out.get("position")}
	if resolved:
		reply["source"] = {"uri": resolved.uri, "commit": resolved.commit, "ref": resolved.ref}
	return reply


async def _claim(request: Request, token: object) -> None:
	"""The visitor following their own job sees its record's share token: they own that record
	(preflight_api/owners.py). Best effort: a fix race on it is refused if this never lands."""
	owners = getattr(request.app.state, "owners", None)
	if owners is None or not isinstance(token, str) or not RECORD_REF.match(token):
		return
	with contextlib.suppress(Exception):
		await owners.claim(token, visitor_of(request.state.session))


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, request: Request) -> dict:
	store = _store(request)
	try:
		out = await store.status(_job_id(job_id))
	except ApiError:
		raise
	except Exception as e:
		raise _store_error(e) from e
	await _claim(request, (out or {}).get("share_token"))
	return out


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
				if ev.get("share_token"):
					await _claim(request, ev["share_token"])
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
