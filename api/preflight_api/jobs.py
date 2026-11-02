"""C's side of A's job store service (python/preflight/runner/modal_app.py: JobStoreService in the
Modal app "preflight-runner"): one container that owns every job, called with `.remote()`.

Configured with PREFLIGHT_JOBSTORE_APP and PREFLIGHT_JOBSTORE_CLS; without them (or with the
runner.queue flag off) /v1/jobs answers "not open yet" and the guide's spending tools say the
same, never pretending a probe was queued.

Errors keep their types across Modal (A's remote_store.py), but the API matches them by class
name so it doesn't need A's package importable to tell QueueFull from StoreUnavailable.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from typing import Any, Protocol

log = logging.getLogger("preflight.jobs")


class JobsClient(Protocol):
	async def submit(self, request: dict, *, visitor: str, idempotency_key: str) -> dict: ...
	async def status(self, job_id: str) -> dict: ...
	async def events(self, job_id: str, after: int = -1) -> list[dict]: ...
	async def cancel(self, job_id: str) -> None: ...


class ModalJobs:
	"""A's JobStoreService, reached from any API container."""

	def __init__(self, app_name: str, cls_name: str) -> None:
		import modal

		self._svc = modal.Cls.from_name(app_name, cls_name)()

	async def submit(self, request: dict, *, visitor: str, idempotency_key: str) -> dict:
		return await self._svc.submit.remote.aio(request, visitor, idempotency_key)

	async def status(self, job_id: str) -> dict:
		return await self._svc.status.remote.aio(job_id)

	async def events(self, job_id: str, after: int = -1) -> list[dict]:
		return await self._svc.events.remote.aio(job_id, after)

	async def cancel(self, job_id: str) -> None:
		await self._svc.cancel.remote.aio(job_id)


def jobs_client() -> JobsClient | None:
	app_name, cls_name = os.environ.get("PREFLIGHT_JOBSTORE_APP"), os.environ.get("PREFLIGHT_JOBSTORE_CLS")
	if not app_name or not cls_name:
		return None
	try:
		return ModalJobs(app_name, cls_name)
	except Exception:
		log.exception("job store service not reachable")
		return None


def visitor_of(session_id: str) -> str:
	"""What the job store knows a visitor by: a hash of the signed session id, never the cookie."""
	return hashlib.sha256(f"preflight.visitor.{session_id}".encode()).hexdigest()[:32]


def kind_of(e: BaseException) -> str:
	"""A's error classes by name: QueueFull, StoreUnavailable, UnknownJob, IdempotencyConflict."""
	return type(e).__name__


def limit_of(e: BaseException) -> tuple[str, str] | None:
	"""A demo limit A's job store refused with (Nov 1): `QueueFull` with a `code` starting `limit.`
	(`limit.visitor_daily`, `limit.daily_spend`) and its own message for the visitor. None for a
	plain busy queue (`queue_full`)."""
	code = getattr(e, "code", "")
	if kind_of(e) != "QueueFull" or not isinstance(code, str) or not code.startswith("limit."):
		return None
	message = (str(e.args[0]) if e.args and isinstance(e.args[0], str) else "").strip()[:200]
	if not message:
		message = "Today's limit is reached: samples only until midnight ET."
	elif message[-1] not in ".!?":
		message += "."  # A's messages end without a full stop; ours follow with another sentence
	way = way_past(code, said=message)
	return code, f"{message} {way}" if way else message


def way_past(code: str | None, *, said: str = "") -> str | None:
	"""A's way past a refusal (preflight.runner.failures.describe, Nov 2: "come back after midnight
	ET, open a sample meanwhile, or probe on your machine with `preflight run train.py`"), as a
	sentence, keeping only the options the message before it doesn't already give (A's limit
	messages already say "samples only until midnight ET"). Plain text: no backticks, and "the
	preflight run command" for A's command with its placeholder file name. None when A's package or the code isn't known, or every
	option is already said."""
	try:
		from preflight.runner.failures import FALLBACK, describe
	except Exception:
		return None
	step = describe(code)
	if not code or step == FALLBACK:
		return None
	text = (
		re.sub(r"`preflight run [^`]*`", "the preflight run command", step[1])
		.replace("`", "")
		.strip()
		.rstrip(".")
	)
	options = [o.strip() for o in re.split(r",\s*(?:or\s+)?|\s+or\s+", text) if o.strip()]
	said_l = said.lower()
	keep = [o for o in options if not any(w in said_l for w in ("midnight", "sample") if w in o.lower())]
	if not keep:
		return None
	line = keep[0] if len(keep) == 1 else ", ".join(keep[:-1]) + ", or " + keep[-1]
	if len(keep) < len(options):
		line = "or " + line if not line.lower().startswith("or ") else line
		line = "You can also " + line[3:]
	return f"{line[:1].upper()}{line[1:]}."


def retry_after(e: BaseException, default: int = 60) -> int:
	value: Any = getattr(e, "retry_after_s", None)
	return int(value) if isinstance(value, int | float) and value > 0 else default
