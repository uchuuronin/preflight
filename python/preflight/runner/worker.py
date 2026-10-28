"""Worker loop: lease a job, heartbeat while it runs, report the outcome.

The job function gets the lease and a JobContext. Long-running code calls
ctx.check() at safe points, and run_probe polls ctx.should_stop(); both fire
once the lease is gone, either because the store said so (reaped, cancelled)
or because the worker could not renew it before it ran out (store
unreachable). A worker that can't prove it still holds the lease stops, so a
probe never runs twice and the GPU cap holds.

Failures from the visitor's own input (bad script, bad paths, unsupported
setup) are permanent and shown as they are. Anything else is ours: it is
retried once, logged in full, and shown to the visitor only as an internal
error with a reference.
"""

import contextlib
import logging
import threading
import time
from collections.abc import Callable

from .queue import HEARTBEAT_S, LEASE_S, JobStore, Lease, LeaseLost

log = logging.getLogger("preflight.worker")


class Cancelled(Exception):
	"""The lease was lost; stop work and discard results."""


class PermanentError(Exception):
	"""A failure retrying cannot fix (bad script, unsupported framework). Not retried; message is visitor-safe.

	code: a stable error code (Probe.error_code's families) for the boarding screen's copy.
	"""

	def __init__(self, message: str = "", *, code: str = "unsupported"):
		super().__init__(message)
		self.code = code


def error_code(exc: BaseException) -> str:
	"""The stable code a failed job carries: the exception's own `code`, else one per type."""
	code = getattr(exc, "code", None)
	if isinstance(code, str) and code:
		return code
	from preflight.harness.static import ScriptParseError
	from preflight.probes.local import InvalidSubmission
	from preflight.recorder.overrides import OverrideError

	if isinstance(exc, ScriptParseError):
		return "syntax_error"
	if isinstance(exc, OverrideError):
		return "override.unsupported"
	if isinstance(exc, InvalidSubmission):
		return "submission"
	if isinstance(exc, RecursionError):
		return "submission.too_deep"
	return "internal"


def _permanent_types() -> tuple[type[BaseException], ...]:
	from preflight.harness.static import ScriptParseError
	from preflight.probes.local import InvalidSubmission
	from preflight.recorder.overrides import OverrideError

	return (PermanentError, ScriptParseError, InvalidSubmission, OverrideError, RecursionError)


class JobContext:
	def __init__(self, store: JobStore, lease: Lease, lease_s: float):
		self._store = store
		self.lease = lease
		self.lost = threading.Event()
		self.lease_s = lease_s
		self.renewed_at = time.monotonic()

	def should_stop(self) -> bool:
		if not self.lost.is_set() and time.monotonic() - self.renewed_at > 0.9 * self.lease_s:
			# could not renew in time: the reaper may already have handed the job to someone else
			self.lost.set()
		return self.lost.is_set()

	def check(self) -> None:
		if self.should_stop():
			raise Cancelled(self.lease.job_id)

	def emit(self, stage: str, **fields) -> None:
		self.check()
		self._store.emit(self.lease.job_id, stage=stage, token=self.lease.token, **fields)

	def confirm(self) -> None:
		"""Ask the store, now, whether this attempt still holds the job (check() only sees what the
		heartbeat last saw, up to HEARTBEAT_S ago). Raises LeaseLost after a cancel or a reap."""
		self.check()
		try:
			self._store.heartbeat(self.lease)
		except LeaseLost:
			self.lost.set()
			raise
		self.renewed_at = time.monotonic()


JobFn = Callable[[Lease, JobContext], str]


def _heartbeat(store: JobStore, ctx: JobContext, every: float, stop: threading.Event) -> None:
	while not stop.wait(every):
		try:
			store.heartbeat(ctx.lease)
			ctx.renewed_at = time.monotonic()
		except LeaseLost:
			ctx.lost.set()
			return
		except Exception as exc:
			log.warning("heartbeat for %s failed: %s", ctx.lease.job_id, exc)
			if ctx.should_stop():
				return


def run_one(store: JobStore, worker: str, fn: JobFn, *, heartbeat_s: float = HEARTBEAT_S) -> bool:
	"""Lease and run one job. Returns False when there was nothing to run."""
	lease = store.lease(worker)
	if lease is None:
		return False
	ctx = JobContext(store, lease, getattr(store, "lease_s", LEASE_S))
	stop = threading.Event()
	hb = threading.Thread(target=_heartbeat, args=(store, ctx, heartbeat_s, stop), daemon=True)
	hb.start()
	try:
		record_id = fn(lease, ctx)
		ctx.check()
		store.complete(lease, record_id=record_id)
	except (Cancelled, LeaseLost):
		pass
	except _permanent_types() as exc:
		msg = str(exc)[:1000] or type(exc).__name__
		_safe_fail(store, lease, msg, retryable=False, error_code=error_code(exc))
	except Exception:
		log.exception("job %s failed with an internal error", lease.job_id)
		_safe_fail(
			store,
			lease,
			f"Internal error while running the job (ref {lease.job_id}); retrying once",
			retryable=True,
			error_code="internal",
		)
	finally:
		stop.set()
		hb.join(timeout=1)
	return True


def _safe_fail(
	store: JobStore, lease: Lease, error: str, *, retryable: bool, error_code: str | None = None
) -> None:
	with contextlib.suppress(LeaseLost):
		store.fail(lease, error=error, retryable=retryable, error_code=error_code)
