"""The job queue: admission control between visitors and GPUs (gap 2).

Pull-based. Workers lease the oldest admissible job for LEASE_S seconds and
renew it with a heartbeat every HEARTBEAT_S. A crashed worker stops
heartbeating, so the reaper frees its slot within a minute; a slow but live
worker keeps renewing, so its probe is never started twice. Each lease carries
a fencing token: once a lease is reaped or cancelled, the old holder's
heartbeat and completion are refused (LeaseLost), so a zombie can't overwrite
the result of the retry.

Submit is idempotent per (visitor, Idempotency-Key): the same key returns the
same job, and the same key with a different body is a conflict. Backpressure
is explicit: a visitor may hold a few queued jobs, the queue has a global cap,
and both answer QueueFull (the API turns it into 429 with Retry-After).
Dispatch is fair: one running job per visitor, and the queue position a
visitor sees counts the jobs that would really be dispatched first.

Persistence: state changes (submit, lease, heartbeat, complete, fail, cancel,
reap) are transactional. The new state is saved before the call returns; if
the save fails the change is rolled back and StoreUnavailable is raised, so a
store outage never orphans a job or burns an attempt. Progress events are
batched (event_flush_s). Finished jobs leave the hot snapshot: they are handed
to on_archive once, and only a small index (status, record id, idempotency
key) stays. Modal Queue has no visibility timeout, so this store is the single
serialised source of truth; in production it runs inside one Modal container
and saves to a Modal Dict.
"""

import copy
import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Literal, Protocol

from preflight.schema import JobEvent
from preflight.schema.ids import new_ulid

log = logging.getLogger("preflight.queue")

LEASE_S = 60  # cited: Contracts tab, gap 2 (60 s leases)
HEARTBEAT_S = 20  # cited: Contracts tab, gap 2 (renewed every 20 s)
MAX_RUNNING = 8  # cited: Contracts tab, public demo limits (8 GPU containers)
MAX_ATTEMPTS = 2  # derived: one retry for our own failures; visitor failures are never retried
MAX_QUEUED = 200  # guess: global queue cap; replaced by the load test before judging (Nov 4)
MAX_QUEUED_PER_VISITOR = 3  # guess: per-visitor backlog; replaced by the load test before judging (Nov 4)
MAX_EVENTS_PER_JOB = 300
RECENT_DONE = 2_000  # finished jobs kept in memory for status and event reads

JobStatus = Literal["queued", "running", "done", "failed", "cancelled"]
TERMINAL = ("done", "failed", "cancelled")


class LeaseLost(RuntimeError):
	"""The caller's lease is no longer current: stop work and discard results."""


class IdempotencyConflict(ValueError):
	"""Same Idempotency-Key, different request body."""


class QueueFull(RuntimeError):
	"""Backpressure: the visitor or the whole queue is at its cap. Retry later."""

	def __init__(self, message: str, retry_after_s: int = 60):
		super().__init__(message)
		self.retry_after_s = retry_after_s


class StoreUnavailable(RuntimeError):
	"""The state could not be saved; the change was rolled back."""


class UnknownJob(KeyError):
	pass


@dataclass
class Submitted:
	job_id: str
	created: bool
	position: int | None


@dataclass
class Lease:
	job_id: str
	token: int
	worker: str
	attempt: int
	request: dict
	expires_at: float
	# the record an earlier attempt started: a retry continues it, so a viewer's link stays good
	record_id: str | None = None
	share_token: str | None = None


@dataclass
class Job:
	job_id: str
	visitor: str
	idempotency_key: str
	body_hash: str
	request: dict
	submitted_at: float
	status: JobStatus = "queued"
	attempt: int = 0
	token: int = 0
	worker: str | None = None
	lease_expires_at: float | None = None
	record_id: str | None = None
	share_token: str | None = None  # set by the first event that carries it (the skeleton)
	error: str | None = None
	retryable: bool | None = None
	events: list[dict] = field(default_factory=list)
	next_seq: int = 0


class JobStore(Protocol):
	lease_s: float

	def submit(self, request: dict, *, visitor: str, idempotency_key: str) -> Submitted: ...
	def lease(self, worker: str) -> Lease | None: ...
	def heartbeat(self, lease: Lease) -> float: ...
	def complete(self, lease: Lease, *, record_id: str) -> None: ...
	def fail(self, lease: Lease, *, error: str, retryable: bool) -> None: ...
	def cancel(self, job_id: str) -> None: ...
	def reap(self) -> list[str]: ...
	def position(self, job_id: str) -> int | None: ...
	def status(self, job_id: str) -> JobStatus: ...
	def emit(self, job_id: str, *, stage: str, **fields) -> JobEvent: ...
	def events(self, job_id: str, after: int = -1) -> list[JobEvent]: ...


def body_hash(request: dict) -> str:
	return hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class InMemoryJobStore:
	"""Thread-safe, clock-injectable. The logic every backend shares."""

	def __init__(
		self,
		*,
		clock: Callable[[], float] = time.time,
		max_running: int = MAX_RUNNING,
		max_running_per_visitor: int = 1,
		max_attempts: int = MAX_ATTEMPTS,
		max_queued: int = MAX_QUEUED,
		max_queued_per_visitor: int = MAX_QUEUED_PER_VISITOR,
		max_events_per_job: int = MAX_EVENTS_PER_JOB,
		lease_s: float = LEASE_S,
		event_flush_s: float = 1.0,
		on_change: Callable[[dict], None] | None = None,
		on_archive: Callable[[dict], None] | None = None,
	):
		self._clock = clock
		self.max_running = max_running
		self.max_running_per_visitor = max_running_per_visitor
		self.max_attempts = max_attempts
		self.max_queued = max_queued
		self.max_queued_per_visitor = max_queued_per_visitor
		self.max_events = max(2, max_events_per_job)
		self.lease_s = lease_s
		self.event_flush_s = event_flush_s
		self._on_change = on_change
		self._on_archive = on_archive
		self._jobs: dict[str, Job] = {}  # live jobs (queued, running)
		self._done: OrderedDict[str, Job] = OrderedDict()  # recently finished, for reads
		self._index: dict[str, dict] = {}  # every finished job: status, record id, key
		self._order: list[str] = []  # submission order of queued jobs
		self._keys: dict[tuple[str, str], str] = {}
		self._lock = threading.RLock()
		self._dirty_events = False
		self._last_flush = 0.0

	# ------------------------------------------------------------- persistence

	def _state(self):
		return (self._jobs, self._done, self._index, self._order, self._keys)

	def _txn(self, mutate: Callable[[], object]):
		"""Run a state change; save it; on a failed save restore the previous state."""
		with self._lock:
			backup = copy.deepcopy(self._state())
			result = mutate()
			if self._on_change is not None:
				try:
					self._on_change(self.snapshot())
				except Exception as exc:
					self._jobs, self._done, self._index, self._order, self._keys = backup
					raise StoreUnavailable(f"job state could not be saved: {type(exc).__name__}") from exc
				self._dirty_events = False
				self._last_flush = self._clock()
			return result

	def _flush_events(self, force: bool = False) -> None:
		if self._on_change is None or not self._dirty_events:
			return
		if not force and self._clock() - self._last_flush < self.event_flush_s:
			return
		try:
			self._on_change(self.snapshot())
			self._dirty_events = False
			self._last_flush = self._clock()
		except Exception as exc:  # events are progress, not state: keep them dirty and try again later
			log.warning("event flush failed: %s", exc)

	def snapshot(self) -> dict:
		with self._lock:
			return {
				"jobs": [asdict(j) for j in self._jobs.values()],
				"order": list(self._order),
				"index": dict(self._index),
			}

	@classmethod
	def restore(cls, snap: dict, **kw) -> "InMemoryJobStore":
		store = cls(**kw)
		for raw in snap["jobs"]:
			job = Job(**raw)
			store._jobs[job.job_id] = job
			store._keys[(job.visitor, job.idempotency_key)] = job.job_id
		for job_id, info in snap.get("index", {}).items():
			store._index[job_id] = info
			store._keys[(info["visitor"], info["idempotency_key"])] = job_id
		store._order = list(snap["order"])
		return store

	# ------------------------------------------------------------- helpers

	def _job(self, job_id: str) -> Job:
		job = self._jobs.get(job_id) or self._done.get(job_id)
		if job is not None:
			return job
		info = self._index.get(job_id)
		if info is None:
			raise UnknownJob(job_id)
		# finished before a restart: rebuild what reads need from the index
		job = Job(
			job_id=job_id,
			visitor=info["visitor"],
			idempotency_key=info["idempotency_key"],
			body_hash=info["body_hash"],
			request={},
			submitted_at=info.get("submitted_at", 0.0),
			status=info["status"],
			record_id=info.get("record_id"),
			share_token=info.get("share_token"),
			error=info.get("error"),
			retryable=info.get("retryable"),
			events=[info["last_event"]] if info.get("last_event") else [],
			next_seq=info.get("next_seq", 0),
		)
		return job

	def _iso(self) -> datetime:
		return datetime.fromtimestamp(self._clock(), UTC)

	def _emit(self, job: Job, stage: str, **fields) -> JobEvent:
		ev = JobEvent(job_id=job.job_id, seq=job.next_seq, at=self._iso(), stage=stage, **fields)
		if ev.share_token:  # the latest attempt's record: a retry writes a fresh one
			job.share_token = ev.share_token
		if ev.record_id:
			job.record_id = ev.record_id
		job.next_seq += 1
		job.events.append(json.loads(ev.model_dump_json(exclude_none=True)))
		if len(job.events) > self.max_events:
			# keep the first event (submission) and the most recent ones
			del job.events[1 : len(job.events) - self.max_events + 1]
		self._dirty_events = True
		return ev

	def _check(self, lease: Lease) -> Job:
		job = self._jobs.get(lease.job_id)
		if job is None or job.status != "running" or job.token != lease.token:
			raise LeaseLost(f"lease {lease.token} on {lease.job_id} is no longer current")
		return job

	def _running(self) -> list[Job]:
		return [j for j in self._jobs.values() if j.status == "running"]

	def _finish(self, job: Job) -> None:
		"""Move a job out of the live set once it is terminal."""
		self._jobs.pop(job.job_id, None)
		self._done[job.job_id] = job
		while len(self._done) > RECENT_DONE:
			self._done.popitem(last=False)
		self._index[job.job_id] = {
			"visitor": job.visitor,
			"idempotency_key": job.idempotency_key,
			"body_hash": job.body_hash,
			"status": job.status,
			"record_id": job.record_id,
			"share_token": job.share_token,
			"error": job.error,
			"retryable": job.retryable,
			"submitted_at": job.submitted_at,
			"last_event": job.events[-1] if job.events else None,
			"next_seq": job.next_seq,
		}
		if self._on_archive is not None:
			try:
				self._on_archive(asdict(job))
			except Exception as exc:
				log.warning("archive of %s failed: %s", job.job_id, exc)

	def _requeue_or_fail(self, job: Job, error: str, retryable: bool) -> None:
		job.worker = None
		job.lease_expires_at = None
		job.token += 1  # fence the old holder
		if retryable and job.attempt < self.max_attempts:
			job.status = "queued"
			job.error = error
			self._order.append(job.job_id)
			self._emit(job, "queued", message=error, level="warn", queue_position=self._position(job.job_id))
		else:
			# retryable here means "resubmitting later may work" (capacity), shown in the UI
			job.status = "failed"
			job.error = error
			job.retryable = retryable
			self._emit(job, "failed", message=error, level="error", retryable=retryable)
			self._finish(job)

	def _rank(self, job_id: str) -> tuple[int, int]:
		"""Fair-dispatch order: a visitor's k-th waiting job goes behind every visitor's (k-1)-th."""
		job = self._jobs[job_id]
		running = sum(1 for j in self._running() if j.visitor == job.visitor)
		ahead_same = 0
		for other in self._order:
			if other == job_id:
				break
			if self._jobs[other].visitor == job.visitor:
				ahead_same += 1
		return (ahead_same + running, self._order.index(job_id))

	def _position(self, job_id: str) -> int | None:
		if job_id not in self._order:
			return None
		mine = self._rank(job_id)
		return sum(1 for other in self._order if other != job_id and self._rank(other) < mine)

	# ------------------------------------------------------------- API

	def submit(self, request: dict, *, visitor: str, idempotency_key: str) -> Submitted:
		h = body_hash(request)
		with self._lock:
			existing = self._keys.get((visitor, idempotency_key))
			if existing is not None:
				job = self._job(existing)
				if job.body_hash != h:
					raise IdempotencyConflict(
						f"Idempotency-Key {idempotency_key!r} was used for a different request"
					)
				return Submitted(job.job_id, False, self._position(job.job_id))
			mine = sum(1 for j in self._jobs.values() if j.visitor == visitor)
			if mine >= self.max_queued_per_visitor:
				raise QueueFull(f"this visitor already has {mine} jobs waiting or running")
			if len(self._order) >= self.max_queued:
				raise QueueFull("the probe queue is full", retry_after_s=120)

			def do():
				job = Job(
					job_id=new_ulid(int(self._clock() * 1000)),
					visitor=visitor,
					idempotency_key=idempotency_key,
					body_hash=h,
					request=request,
					submitted_at=self._clock(),
				)
				self._jobs[job.job_id] = job
				self._keys[(visitor, idempotency_key)] = job.job_id
				self._order.append(job.job_id)
				pos = self._position(job.job_id)
				self._emit(job, "queued", queue_position=pos)
				return Submitted(job.job_id, True, pos)

			return self._txn(do)

	def lease(self, worker: str) -> Lease | None:
		with self._lock:
			running = self._running()
			if len(running) >= self.max_running:
				return None
			per_visitor: dict[str, int] = {}
			for j in running:
				per_visitor[j.visitor] = per_visitor.get(j.visitor, 0) + 1
			pick = next(
				(
					jid
					for jid in self._order
					if per_visitor.get(self._jobs[jid].visitor, 0) < self.max_running_per_visitor
				),
				None,
			)
			if pick is None:
				return None

			def do():
				job = self._jobs[pick]
				self._order.remove(pick)
				job.status = "running"
				job.attempt += 1
				job.token += 1
				job.worker = worker
				job.lease_expires_at = self._clock() + self.lease_s
				self._emit(job, "admitted", message=f"attempt {job.attempt}")
				return Lease(
					job.job_id,
					job.token,
					worker,
					job.attempt,
					job.request,
					job.lease_expires_at,
					record_id=job.record_id,
					share_token=job.share_token,
				)

			return self._txn(do)

	def heartbeat(self, lease: Lease) -> float:
		with self._lock:
			job = self._check(lease)

			def do():
				job.lease_expires_at = self._clock() + self.lease_s
				return job.lease_expires_at

			lease.expires_at = self._txn(do)
			return lease.expires_at

	def complete(self, lease: Lease, *, record_id: str) -> None:
		with self._lock:
			job = self._check(lease)

			def do():
				job.status = "done"
				job.record_id = record_id
				job.worker = None
				job.lease_expires_at = None
				job.token += 1
				self._emit(job, "done", record_id=record_id, progress=1.0)
				self._finish(job)

			self._txn(do)

	def fail(self, lease: Lease, *, error: str, retryable: bool) -> None:
		with self._lock:
			job = self._check(lease)
			self._txn(lambda: self._requeue_or_fail(job, error, retryable))

	def cancel(self, job_id: str) -> None:
		with self._lock:
			job = self._job(job_id)
			if job.status in TERMINAL:
				return

			def do():
				if job.job_id in self._order:
					self._order.remove(job.job_id)
				job.status = "cancelled"
				job.token += 1
				job.worker = None
				job.lease_expires_at = None
				self._emit(job, "cancelled")
				self._finish(job)

			self._txn(do)

	def reap(self) -> list[str]:
		now = self._clock()
		with self._lock:
			expired = [
				j for j in self._running() if j.lease_expires_at is not None and j.lease_expires_at < now
			]
			if not expired:
				return []

			def do():
				for job in expired:
					times = {1: "once", 2: "twice"}.get(job.attempt, f"{job.attempt} times")
					self._requeue_or_fail(job, f"lease expired {times}; worker crashed or hung", True)
				return [j.job_id for j in expired]

			return self._txn(do)

	def emit(self, job_id: str, *, stage: str, token: int | None = None, **fields) -> JobEvent:
		"""Progress from a worker. With its lease token, it is fenced like heartbeat and complete: an
		attempt that was reaped or cancelled can't add events or move the job's record."""
		with self._lock:
			job = self._job(job_id)
			if token is not None and (job.status != "running" or job.token != token):
				raise LeaseLost(job_id)
			ev = self._emit(job, stage, **fields)
			self._flush_events()
			return ev

	def events(self, job_id: str, after: int = -1) -> list[JobEvent]:
		with self._lock:
			return [JobEvent.model_validate(e) for e in self._job(job_id).events if e["seq"] > after]

	def position(self, job_id: str) -> int | None:
		with self._lock:
			self._job(job_id)
			return self._position(job_id)

	def status(self, job_id: str) -> JobStatus:
		with self._lock:
			return self._job(job_id).status

	def info(self, job_id: str) -> dict:
		"""What GET /v1/jobs/{id} shows: status, queue position, where the record is, and why it failed."""
		with self._lock:
			job = self._job(job_id)
			return {
				"job_id": job.job_id,
				"status": job.status,
				"position": self._position(job_id),
				"record_id": job.record_id,
				"share_token": job.share_token,
				"error": job.error,
				"retryable": job.retryable,
				"last_seq": job.next_seq - 1,
			}

	def get(self, job_id: str) -> Job:
		with self._lock:
			return self._job(job_id)

	def queued(self) -> list[str]:
		with self._lock:
			return list(self._order)
