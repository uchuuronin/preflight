import threading
import time

from preflight.runner.queue import InMemoryJobStore
from preflight.runner.worker import Cancelled, PermanentError, run_one

REQ = {"source": {"kind": "upload"}}


def test_worker_runs_job_and_completes():
	store = InMemoryJobStore()
	job = store.submit(REQ, visitor="v", idempotency_key="k")
	assert run_one(store, "w", lambda lease, ctx: "rec-1", heartbeat_s=0.01)
	assert store.status(job.job_id) == "done"
	assert store.get(job.job_id).record_id == "rec-1"


def test_worker_returns_false_when_queue_is_empty():
	assert run_one(InMemoryJobStore(), "w", lambda lease, ctx: "x") is False


def test_worker_heartbeats_through_a_job_longer_than_its_lease():
	store = InMemoryJobStore(lease_s=0.05)
	job = store.submit(REQ, visitor="v", idempotency_key="k")

	def slow(lease, ctx):
		time.sleep(0.2)
		assert store.reap() == []
		return "rec"

	run_one(store, "w", slow, heartbeat_s=0.01)
	assert store.status(job.job_id) == "done"


def test_cancel_reaches_the_running_job():
	store = InMemoryJobStore()
	job = store.submit(REQ, visitor="v", idempotency_key="k")
	seen = threading.Event()

	def work(lease, ctx):
		store.cancel(job.job_id)
		for _ in range(100):
			time.sleep(0.01)
			ctx.check()  # raises Cancelled once the heartbeat sees the lost lease
		seen.set()
		return "never"

	run_one(store, "w", work, heartbeat_s=0.01)
	assert not seen.is_set()
	assert store.status(job.job_id) == "cancelled"


def test_permanent_error_fails_without_retry():
	store = InMemoryJobStore()
	job = store.submit(REQ, visitor="v", idempotency_key="k")

	def bad(lease, ctx):
		raise PermanentError("could not find a training loop")

	run_one(store, "w", bad)
	assert store.status(job.job_id) == "failed"
	assert "training loop" in store.get(job.job_id).error


def test_unexpected_error_is_retried_once():
	store = InMemoryJobStore()
	job = store.submit(REQ, visitor="v", idempotency_key="k")

	def flaky(lease, ctx):
		raise RuntimeError("container went away")

	run_one(store, "w", flaky)
	assert store.status(job.job_id) == "queued"


def test_ctx_emit_writes_job_events():
	store = InMemoryJobStore()
	job = store.submit(REQ, visitor="v", idempotency_key="k")

	def work(lease, ctx):
		ctx.emit("probing", progress=0.5, message="probe 1 of 2")
		return "r"

	run_one(store, "w", work)
	assert "probing" in [e.stage for e in store.events(job.job_id)]


def test_cancelled_is_exported():
	assert issubclass(Cancelled, Exception)


class FlakyStore(InMemoryJobStore):
	"""Heartbeats fail as if the store were unreachable."""

	def heartbeat(self, lease):
		raise ConnectionError("store unreachable")


def test_worker_stops_itself_when_it_cannot_renew_before_the_lease_runs_out():
	store = FlakyStore(lease_s=0.2)
	store.submit(REQ, visitor="v", idempotency_key="k")
	stopped = threading.Event()

	def work(lease, ctx):
		for _ in range(200):
			time.sleep(0.01)
			if ctx.should_stop():
				stopped.set()
				ctx.check()
		return "never"

	run_one(store, "w", work, heartbeat_s=0.02)
	assert stopped.is_set()


def test_bad_input_errors_are_permanent():
	from preflight.harness.static import ScriptParseError
	from preflight.probes.local import InvalidSubmission
	from preflight.recorder.overrides import OverrideError

	for exc in (
		ScriptParseError("train.py:3: invalid syntax"),
		InvalidSubmission("bad path"),
		OverrideError("no literal"),
	):
		store = InMemoryJobStore()
		job = store.submit(REQ, visitor="v", idempotency_key="k")

		def bad(lease, ctx, exc=exc):
			raise exc

		run_one(store, "w", bad)
		assert store.status(job.job_id) == "failed", type(exc)


def test_internal_errors_are_not_shown_to_visitors():
	store = InMemoryJobStore()
	job = store.submit(REQ, visitor="v", idempotency_key="k")

	def boom(lease, ctx):
		raise RuntimeError("connect to http://10.0.0.7:8080/internal failed: token=abc")

	run_one(store, "w", boom)
	msgs = " ".join(e.message or "" for e in store.events(job.job_id))
	assert "10.0.0.7" not in msgs and "token" not in msgs
	assert "internal error" in msgs.lower()
