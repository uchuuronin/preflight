"""runner.nothing_measured and runner.made_by (off; contract 0.6.6; the re-validation, Nov 3): a job whose probes never
reached their first step ends done with nothing_measured and its cause, and a free resend whatever the cause; a record
names the runner commit and the flags that made it."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from preflight.runner.queue import VISITOR_DAILY_JOBS, InMemoryJobStore, QueueFull

REQ = {"source": {"kind": "sample", "uri": "sample://zoo/x"}}


@pytest.fixture
def flags(monkeypatch):
	import preflight.runner.flags as F
	import preflight.runner.pipeline as P

	on: set[str] = set()
	real = F.flag
	monkeypatch.setattr(F, "flag", lambda n: n in on or real(n))
	monkeypatch.setattr(P, "_flag", lambda n: n in on or real(n))
	return on


def test_the_done_event_says_nothing_was_measured_and_why(flags):
	flags.add("runner.nothing_measured")
	store = InMemoryJobStore(clock=lambda: 1_762_000_000.0, free_resend=True)
	store.submit(REQ, visitor="v", idempotency_key="a")
	lease = store.lease("w")
	store.emit(lease.job_id, stage="probing", nothing_measured=True, error_code="script_exit")
	store.complete(lease, record_id="r1", spent_usd=0.01)
	done = store.events(lease.job_id)[-1]
	assert done.stage == "done" and done.nothing_measured is True and done.error_code == "script_exit"
	# a job that measured: done as before
	store.submit({**REQ, "n": 1}, visitor="v2", idempotency_key="b")
	lease = store.lease("w")
	store.complete(lease, record_id="r2", spent_usd=0.01)
	assert store.events(lease.job_id)[-1].nothing_measured is None


def test_a_job_that_measured_nothing_gets_the_free_resend_whatever_the_cause(flags):
	flags.add("runner.nothing_measured")
	store = InMemoryJobStore(clock=lambda: 1_762_000_000.0, free_resend=True)
	ids = []
	for i in range(VISITOR_DAILY_JOBS):
		store.submit({**REQ, "n": i}, visitor="v", idempotency_key=f"k{i}")
		lease = store.lease("w")
		if i == 0:  # a ValueError at import time: not an import stop
			store.emit(lease.job_id, stage="probing", nothing_measured=True, error_code="user_error")
		store.complete(lease, record_id=f"r{i}", spent_usd=0.01)
		ids.append(lease.job_id)
	with pytest.raises(QueueFull):  # a job that measured isn't one
		store.submit({**REQ, "resend_of": ids[1]}, visitor="v", idempotency_key="x")
	s = store.submit({**REQ, "resend_of": ids[0]}, visitor="v", idempotency_key="y")
	assert s.created and store.events(s.job_id)[0].free_resend is True
	flags.discard("runner.nothing_measured")
	store2 = InMemoryJobStore(clock=lambda: 1_762_000_000.0, free_resend=True)
	ids = []
	for i in range(VISITOR_DAILY_JOBS):
		store2.submit({**REQ, "n": i}, visitor="v", idempotency_key=f"k{i}")
		lease = store2.lease("w")
		if i == 0:
			store2.emit(lease.job_id, stage="probing", nothing_measured=True, error_code="user_error")
		store2.complete(lease, record_id=f"r{i}", spent_usd=0.01)
		ids.append(lease.job_id)
	with pytest.raises(QueueFull):  # off: only an import stop is free, as before
		store2.submit({**REQ, "resend_of": ids[0]}, visitor="v", idempotency_key="z")


def _stops(base):
	class Stops(base):  # every probe stops before its first step, not at an import
		def __call__(self, spec, gpu, *, should_stop):
			r = super().__call__(spec, gpu, should_stop=should_stop)
			r.probe = r.probe.model_copy(update={"status": "error", "error_code": "script_exit", "steps": 0,
				"error": "SystemExit: 2"})  # fmt: skip
			return r

	return Stops()


def test_the_pipeline_marks_the_job_and_the_record(tmp_path, flags):
	from preflight.runner.pipeline import _Ctx, run_pipeline
	from test_pipeline import SCRIPT, FakeRunner, deps

	src = {"source": {"kind": "upload", "files": {"train.py": SCRIPT}}}
	off = _Ctx()
	run_pipeline(src, off, deps(tmp_path, _stops(FakeRunner)))
	assert not any(e.get("nothing_measured") for e in off.events)  # off: as before
	flags.add("runner.nothing_measured")
	ctx = _Ctx()
	d = deps(tmp_path, _stops(FakeRunner))
	rec = run_pipeline(src, ctx, d)
	marks = [e for e in ctx.events if e.get("nothing_measured")]
	assert marks and marks[0]["error_code"] == "script_exit"
	f = next(f for f in rec.findings if f.id == "job.nothing_measured")
	assert f.title == "Nothing measured" and "script_exit" in f.detail and f.severity == "error"
	ok = _Ctx()
	run_pipeline(src, ok, deps(tmp_path, FakeRunner()))
	assert not any(e.get("nothing_measured") for e in ok.events)


def test_made_by_names_the_runner_commit_and_the_flags_on(flags, monkeypatch):
	from preflight.runner.pipeline import _made_by

	monkeypatch.setenv("PREFLIGHT_RUNNER_COMMIT", "c6465777" + "0" * 32)
	flags.add("runner.made_by")
	m = _made_by()
	assert m.runner_commit == "c6465777" + "0" * 32 and m.flags_on == sorted(m.flags_on)
	monkeypatch.setenv("PREFLIGHT_RUNNER_COMMIT", "not a sha")
	assert _made_by().runner_commit is None


def test_a_record_carries_made_by_only_with_the_flag(tmp_path, flags, monkeypatch):
	from preflight.runner.pipeline import _Ctx, run_pipeline
	from test_pipeline import SCRIPT, FakeRunner, deps

	monkeypatch.setenv("PREFLIGHT_RUNNER_COMMIT", "a" * 40)
	src = {"source": {"kind": "upload", "files": {"train.py": SCRIPT}}}
	d = deps(tmp_path, FakeRunner())
	rec = run_pipeline(src, _Ctx(), d)
	assert rec.made_by is None
	flags.add("runner.made_by")
	d = deps(tmp_path, FakeRunner())
	rec = run_pipeline(src, _Ctx(), d)
	from preflight.runner.flags import _resolved

	assert rec.made_by.runner_commit == "a" * 40
	assert rec.made_by.flags_on == sorted(
		k for k, v in _resolved().items() if v
	)  # the flags files, as resolved


def test_an_older_record_loads_and_the_new_fields_are_optional():
	from preflight.schema import FlightRecord, JobEvent

	assert FlightRecord.model_fields["made_by"].default is None
	assert JobEvent.model_fields["nothing_measured"].default is None


def test_an_out_of_memory_probe_is_a_measurement(flags):
	from preflight.runner.pipeline import _nothing_measured
	from preflight.schema import Probe

	flags.add("runner.nothing_measured")
	oom = Probe(id="p1", knobs={}, status="oom", steps=0, error="CUDA out of memory", error_code="oom")
	stop = Probe(id="p2", knobs={}, status="error", steps=0, error="SystemExit: 2", error_code="script_exit")
	assert _nothing_measured([oom, stop]) is None
	assert _nothing_measured([stop]).id == "p2"
