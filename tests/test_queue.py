"""Job queue: idempotent submit, leases with heartbeats, a reaper, admission caps (gap 2)."""

import pytest

from preflight.runner.queue import (
	HEARTBEAT_S,
	LEASE_S,
	IdempotencyConflict,
	InMemoryJobStore,
	LeaseLost,
	QueueFull,
	StoreUnavailable,
)


class Clock:
	def __init__(self, t: float = 1_000.0):
		self.t = t

	def __call__(self) -> float:
		return self.t

	def tick(self, s: float) -> None:
		self.t += s


@pytest.fixture
def clock():
	return Clock()


@pytest.fixture
def store(clock):
	return InMemoryJobStore(clock=clock, max_running=2, max_running_per_visitor=1, max_attempts=2)


REQ = {"source": {"kind": "upload", "uri": "upload://abc"}}


def test_lease_and_heartbeat_constants_match_contract():
	assert LEASE_S == 60
	assert HEARTBEAT_S == 20


def test_submit_returns_job_and_queue_position(store):
	a = store.submit(REQ, visitor="v1", idempotency_key="k1")
	b = store.submit({**REQ, "lens": "pretrain"}, visitor="v2", idempotency_key="k2")
	assert a.created and b.created
	assert store.position(a.job_id) == 0
	assert store.position(b.job_id) == 1


def test_same_idempotency_key_returns_same_job(store):
	a = store.submit(REQ, visitor="v1", idempotency_key="k1")
	again = store.submit(REQ, visitor="v1", idempotency_key="k1")
	assert again.job_id == a.job_id
	assert not again.created
	assert len(store.queued()) == 1


def test_same_key_with_different_body_is_a_conflict(store):
	store.submit(REQ, visitor="v1", idempotency_key="k1")
	with pytest.raises(IdempotencyConflict):
		store.submit({**REQ, "lens": "rl"}, visitor="v1", idempotency_key="k1")


def test_idempotency_keys_are_scoped_per_visitor(store):
	a = store.submit(REQ, visitor="v1", idempotency_key="k1")
	b = store.submit(REQ, visitor="v2", idempotency_key="k1")
	assert a.job_id != b.job_id


def test_lease_takes_oldest_job_and_hides_it(store):
	a = store.submit(REQ, visitor="v1", idempotency_key="k1")
	store.submit(REQ, visitor="v2", idempotency_key="k2")
	lease = store.lease("w1")
	assert lease.job_id == a.job_id
	assert lease.attempt == 1
	assert store.status(a.job_id) == "running"
	assert store.position(a.job_id) is None


def test_global_cap_holds_jobs_in_queue(store):
	for i in range(3):
		store.submit(REQ, visitor=f"v{i}", idempotency_key=f"k{i}")
	assert store.lease("w1") is not None
	assert store.lease("w2") is not None
	assert store.lease("w3") is None  # cap of 2 running


def test_one_visitor_cannot_take_every_slot(store):
	store.submit(REQ, visitor="greedy", idempotency_key="k1")
	store.submit({**REQ, "lens": "x"}, visitor="greedy", idempotency_key="k2")
	other = store.submit(REQ, visitor="polite", idempotency_key="k3")
	store.lease("w1")
	second = store.lease("w2")
	assert second.job_id == other.job_id


def test_heartbeat_extends_the_lease(store, clock):
	store.submit(REQ, visitor="v1", idempotency_key="k1")
	lease = store.lease("w1")
	for _ in range(5):
		clock.tick(HEARTBEAT_S)
		store.heartbeat(lease)
	assert store.reap() == []
	assert store.status(lease.job_id) == "running"


def test_reaper_requeues_a_crashed_worker_within_a_minute(store, clock):
	job = store.submit(REQ, visitor="v1", idempotency_key="k1")
	store.lease("w1")
	clock.tick(LEASE_S + 1)
	assert store.reap() == [job.job_id]
	assert store.status(job.job_id) == "queued"
	again = store.lease("w2")
	assert again.job_id == job.job_id and again.attempt == 2


def test_stale_worker_cannot_complete_after_its_lease_was_reaped(store, clock):
	store.submit(REQ, visitor="v1", idempotency_key="k1")
	old = store.lease("w1")
	clock.tick(LEASE_S + 1)
	store.reap()
	new = store.lease("w2")
	with pytest.raises(LeaseLost):
		store.complete(old, record_id="r-old")
	with pytest.raises(LeaseLost):
		store.heartbeat(old)
	store.complete(new, record_id="r-new")
	assert store.get(new.job_id).record_id == "r-new"


def test_job_dead_letters_after_max_attempts(store, clock):
	job = store.submit(REQ, visitor="v1", idempotency_key="k1")
	for _ in range(2):
		store.lease("w")
		clock.tick(LEASE_S + 1)
		store.reap()
	assert store.status(job.job_id) == "failed"
	assert store.get(job.job_id).error == "lease expired twice; worker crashed or hung"
	assert store.lease("w") is None


def test_complete_frees_the_slot(store):
	for i in range(3):
		store.submit(REQ, visitor=f"v{i}", idempotency_key=f"k{i}")
	a = store.lease("w1")
	store.lease("w2")
	store.complete(a, record_id="r1")
	assert store.status(a.job_id) == "done"
	assert store.lease("w3") is not None


def test_retryable_failure_requeues_once_then_fails(store):
	job = store.submit(REQ, visitor="v1", idempotency_key="k1")
	store.fail(store.lease("w1"), error="modal capacity", retryable=True)
	assert store.status(job.job_id) == "queued"
	store.fail(store.lease("w1"), error="modal capacity", retryable=True)
	assert store.status(job.job_id) == "failed"


def test_permanent_failure_does_not_retry(store):
	job = store.submit(REQ, visitor="v1", idempotency_key="k1")
	store.fail(store.lease("w1"), error="script has a syntax error", retryable=False)
	assert store.status(job.job_id) == "failed"


def test_cancel_queued_job(store):
	job = store.submit(REQ, visitor="v1", idempotency_key="k1")
	store.cancel(job.job_id)
	assert store.status(job.job_id) == "cancelled"
	assert store.lease("w") is None


def test_cancel_running_job_makes_heartbeat_tell_worker_to_stop(store):
	job = store.submit(REQ, visitor="v1", idempotency_key="k1")
	lease = store.lease("w1")
	store.cancel(job.job_id)
	with pytest.raises(LeaseLost):
		store.heartbeat(lease)


def test_events_are_sequenced_and_resumable(store, clock):
	job = store.submit(REQ, visitor="v1", idempotency_key="k1")
	lease = store.lease("w1")
	store.emit(job.job_id, stage="probing", progress=0.5)
	store.complete(lease, record_id="r1")
	events = store.events(job.job_id)
	assert [e.stage for e in events] == ["queued", "admitted", "probing", "done"]
	assert [e.seq for e in events] == [0, 1, 2, 3]
	assert [e.stage for e in store.events(job.job_id, after=1)] == ["probing", "done"]
	assert events[-1].record_id == "r1"
	assert events[0].queue_position == 0


def test_snapshot_restore_survives_a_restart(store, clock):
	job = store.submit(REQ, visitor="v1", idempotency_key="k1")
	lease = store.lease("w1")
	restored = InMemoryJobStore.restore(
		store.snapshot(), clock=clock, max_running=2, max_running_per_visitor=1
	)
	restored.heartbeat(lease)
	again = restored.submit(REQ, visitor="v1", idempotency_key="k1")
	assert again.job_id == job.job_id and not again.created
	assert [e.stage for e in restored.events(job.job_id)] == ["queued", "admitted"]


# ---------------------------------------------------------------- persistence and backpressure (review fixes)


class Boom(Exception):
	pass


def test_a_failed_save_rolls_the_change_back(clock):
	fail = {"on": False}

	def save(snap):
		if fail["on"]:
			raise Boom

	store = InMemoryJobStore(clock=clock, max_running=2, on_change=save)
	job = store.submit(REQ, visitor="v1", idempotency_key="k1")
	fail["on"] = True
	with pytest.raises(StoreUnavailable):
		store.lease("w1")
	fail["on"] = False
	assert store.status(job.job_id) == "queued"
	assert store.get(job.job_id).attempt == 0
	assert store.lease("w1").attempt == 1


def test_snapshot_leaves_out_finished_jobs_but_keeps_their_idempotency(clock):
	archived = []
	store = InMemoryJobStore(clock=clock, on_archive=archived.append)
	job = store.submit(REQ, visitor="v1", idempotency_key="k1")
	store.complete(store.lease("w"), record_id="r1")
	snap = store.snapshot()
	assert all(j["job_id"] != job.job_id for j in snap["jobs"])
	assert archived and archived[0]["job_id"] == job.job_id
	restored = InMemoryJobStore.restore(snap, clock=clock)
	again = restored.submit(REQ, visitor="v1", idempotency_key="k1")
	assert again.job_id == job.job_id and not again.created


def test_events_per_job_are_capped(clock):
	store = InMemoryJobStore(clock=clock, max_events_per_job=50)
	job = store.submit(REQ, visitor="v1", idempotency_key="k1")
	store.lease("w")
	for i in range(500):
		store.emit(job.job_id, stage="probing", progress=i / 500)
	evs = store.events(job.job_id)
	assert len(evs) <= 50
	assert evs[0].stage == "queued"  # the first event is kept
	assert evs[-1].seq == 501  # sequence numbers keep counting


def test_progress_events_do_not_force_a_save_each(clock):
	saves = []
	store = InMemoryJobStore(clock=clock, on_change=saves.append, event_flush_s=1.0)
	job = store.submit(REQ, visitor="v1", idempotency_key="k1")
	store.lease("w")
	n = len(saves)
	for i in range(100):
		store.emit(job.job_id, stage="probing", progress=i / 100)
	assert len(saves) - n <= 1
	clock.tick(2)
	store.emit(job.job_id, stage="probing", progress=1.0)
	assert len(saves) - n >= 1


def test_per_visitor_queue_cap(clock):
	store = InMemoryJobStore(clock=clock, max_queued_per_visitor=2)
	store.submit(REQ, visitor="v1", idempotency_key="a")
	store.submit({**REQ, "lens": "x"}, visitor="v1", idempotency_key="b")
	with pytest.raises(QueueFull, match="visitor"):
		store.submit({**REQ, "lens": "y"}, visitor="v1", idempotency_key="c")


def test_global_queue_cap(clock):
	store = InMemoryJobStore(clock=clock, max_queued=3, max_queued_per_visitor=5)
	for i in range(3):
		store.submit(REQ, visitor=f"v{i}", idempotency_key="k")
	with pytest.raises(QueueFull, match="queue"):
		store.submit(REQ, visitor="v9", idempotency_key="k")


def test_position_reflects_fair_dispatch_not_raw_order(clock):
	store = InMemoryJobStore(clock=clock, max_running=1, max_queued_per_visitor=50)
	for i in range(20):
		store.submit({**REQ, "n": i}, visitor="greedy", idempotency_key=f"g{i}")
	mine = store.submit(REQ, visitor="polite", idempotency_key="p")
	# greedy's first job goes first, then mine: one job ahead, not twenty
	assert store.position(mine.job_id) == 1


def test_info_points_at_the_latest_record_while_the_job_runs():
	store = InMemoryJobStore()
	sub = store.submit({"source": {"kind": "sample"}}, visitor="v", idempotency_key="k")
	lease = store.lease("w")
	store.emit(sub.job_id, stage="skeleton", record_id="R1", share_token="t" * 24, record_rev=1)
	info = store.info(sub.job_id)
	assert info["status"] == "running" and info["share_token"] == "t" * 24 and info["record_id"] == "R1"
	store.complete(lease, record_id="R1")
	assert store.info(sub.job_id)["status"] == "done"
	# archived and restored: still answerable
	again = InMemoryJobStore.restore(store.snapshot())
	assert again.info(sub.job_id)["share_token"] == "t" * 24


def test_a_reaped_attempt_cannot_emit_and_the_retry_keeps_the_record():
	from preflight.runner.queue import LeaseLost

	t = [1000.0]
	store = InMemoryJobStore(clock=lambda: t[0])
	sub = store.submit({"source": {}}, visitor="v", idempotency_key="k")
	first = store.lease("w1")
	store.emit(sub.job_id, stage="skeleton", token=first.token, record_id="R1", share_token="a" * 24)
	t[0] += 1000  # the first worker went silent
	store.reap()
	with pytest.raises(LeaseLost):
		store.emit(sub.job_id, stage="probing", token=first.token, share_token="b" * 24)
	second = store.lease("w2")
	assert (second.record_id, second.share_token) == ("R1", "a" * 24)  # the retry continues R1
	assert store.info(sub.job_id)["share_token"] == "a" * 24
