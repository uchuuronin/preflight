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
import os
from dataclasses import dataclass
from typing import Any, Protocol

from ..flags import flag
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
	"""A's job store service, called from any API container."""

	def __init__(self, app_name: str, cls_name: str) -> None:
		import modal

		self._svc = modal.Cls.from_name(app_name, cls_name)()

	async def submit(self, request: dict, *, visitor: str, idempotency_key: str) -> dict:
		got = await self._svc.submit.remote.aio(request, visitor=visitor, idempotency_key=idempotency_key)
		return {"status": "submitted", "job_id": got["job_id"], "queue_position": got.get("position")}


def job_executor() -> Executor | None:
	app_name, cls_name = os.environ.get("PREFLIGHT_JOBSTORE_APP"), os.environ.get("PREFLIGHT_JOBSTORE_CLS")
	if not app_name or not cls_name:
		return None
	try:
		return JobStoreExecutor(app_name, cls_name)
	except Exception:
		log.exception("job store service not reachable")
		return None


def _unavailable(reason: str) -> dict:
	return {"status": "unavailable", "reason": reason}


def job_request(name: str, args: dict, ctx: ExecContext) -> dict:
	"""The JobRequest the guide submits for a server tool (shape agreed with A, docs/requests.md)."""
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
				visitor=ctx.visitor,
				idempotency_key=f"{ctx.visitor}:{ctx.call_id}",
			)
		except Exception as e:
			code = type(e).__name__
			if code == "QueueFull":
				return {"status": "queue_full", "retry_after_s": getattr(e, "retry_after_s", None)}
			log.exception("guide tool %s failed to submit", name)
			return _unavailable("measuring_service_error")
	return {"status": "refused", "reason": "unknown_tool"}
