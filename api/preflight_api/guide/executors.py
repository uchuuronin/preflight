"""Run a money-spending guide tool once the visitor has clicked to approve it.

The model can only ask for run_probe, run_verify, sample_more or propose_patch; the call
waits in the signed conversation state until the visitor approves or declines it. On
approval the route calls `execute`, and whatever comes back (a submitted job, or the honest
reason it couldn't run) goes back to the model as that call's result, so the guide never
says a probe ran when it didn't.

Where the work runs:
- run_probe, run_verify, sample_more: A's job store service (one Modal container, see
  docs/requests.md, C -> A, Oct 27). Configured with PREFLIGHT_JOBSTORE_APP and
  PREFLIGHT_JOBSTORE_CLS; until A publishes it, these return status "unavailable".
- propose_patch: Muse Code in a sandbox (the fix loop, flag c.fix_loop); until then
  "unavailable".

Each submission is idempotent on (visitor, call id): approving twice never pays twice.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from ..flags import flag
from ..jobs import JobsClient, limit_of, visitor_of
from .record_view import RecordView
from .tools import SAMPLE_MORE_TASKS

log = logging.getLogger("preflight.guide.executors")


@dataclass
class ExecContext:
	record: RecordView
	record_ref: str
	visitor: str
	call_id: str
	knobs: dict[str, Any]


class Executor(Protocol):
	async def submit(self, request: dict, *, visitor: str, idempotency_key: str) -> dict: ...


class JobStoreExecutor:
	"""A's job store service (preflight_api/jobs.py), submitting the guide's approved jobs."""

	def __init__(self, store: JobsClient) -> None:
		self._store = store

	async def submit(self, request: dict, *, visitor: str, idempotency_key: str) -> dict:
		got = await self._store.submit(request, visitor=visitor, idempotency_key=idempotency_key)
		return {"status": "submitted", "job_id": got["job_id"], "queue_position": got.get("position")}


def job_executor(store: JobsClient | None) -> Executor | None:
	return JobStoreExecutor(store) if store is not None else None


def _unavailable(reason: str) -> dict:
	return {"status": "unavailable", "reason": reason}


def job_request(name: str, args: dict, ctx: ExecContext) -> dict:
	"""The JobRequest the guide submits for a server tool (fields requested of A, docs/requests.md, Oct 27)."""
	base = {"source": {"kind": "record", "record_ref": ctx.record_ref}, "purpose": f"guide.{name}"}
	if name == "run_probe":
		knobs = {
			k: v
			for k, v in (args.get("knobs") or {}).items()
			if k in ctx.record.knobs and ctx.record.knobs[k].allows(v)
		}
		return {**base, "mode": "probe", "knobs": {**ctx.knobs, **knobs}}
	if name == "run_verify":
		return {**base, "mode": "verify", "knobs": dict(ctx.knobs)}
	if name == "sample_more":
		return {**base, "mode": "sample_more", "tasks": SAMPLE_MORE_TASKS}
	raise ValueError(name)


async def execute(name: str, args: dict, ctx: ExecContext, jobs: Executor | None) -> dict:
	"""Run an approved server tool and return the typed result the model will see."""
	if name == "propose_patch":
		# The fix loop (Muse Code in a sandbox) isn't built; with or without its flag the honest
		# answer is the same until it is.
		return _unavailable(
			"code_changes_not_open_yet" if not flag("c.fix_loop") else "code_changes_not_built_yet"
		)
	if name in ("run_probe", "run_verify", "sample_more"):
		if name == "sample_more" and not ctx.record.samples_tasks:
			return {"status": "refused", "reason": "not_an_agent_run"}
		if jobs is None:
			return _unavailable("measuring_service_not_running_yet")
		try:
			return await jobs.submit(
				job_request(name, args, ctx),
				visitor=visitor_of(ctx.visitor),
				idempotency_key=f"guide.{ctx.call_id}",
			)
		except Exception as e:
			code = type(e).__name__
			lim = limit_of(e)
			if lim:
				# A daily limit (A, Nov 1): say so in A's words, and point to the samples.
				return {
					"status": "limit",
					"code": lim[0],
					"message": lim[1],
					"retry_after_s": getattr(e, "retry_after_s", None),
				}
			if code == "QueueFull":
				return {"status": "queue_full", "retry_after_s": getattr(e, "retry_after_s", None)}
			log.exception("guide tool %s failed to submit", name)
			return _unavailable("measuring_service_error")
	return {"status": "refused", "reason": "unknown_tool"}
