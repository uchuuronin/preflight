"""runner.free_resend (off; Chiya, Nov 2): a resend of a job that stopped at an import doesn't count against the
visitor's daily probe jobs, once a day; the daily GPU dollar cap still holds."""

import pytest

from preflight.runner.queue import VISITOR_DAILY_JOBS, InMemoryJobStore, QueueFull

REQ = {"source": {"kind": "sample", "uri": "sample://zoo/x"}}


def _store(**kw):
	t = [1_762_000_000.0]
	s = InMemoryJobStore(clock=lambda: t[0], free_resend=kw.pop("on", True), **kw)
	return s, t


def _finish(store, *, import_stop=False, fail_code=None):
	lease = store.lease("w")
	if import_stop:
		store.emit(lease.job_id, stage="probing", import_stop=True)
	if fail_code:
		store.fail(lease, error="refused", retryable=False, error_code=fail_code, spent_usd=0.0)
	else:
		store.complete(lease, record_id="r1", spent_usd=0.01)
	return lease.job_id


def _use_the_day(store, visitor="v", *, first_import_stop=True):
	ids = []
	for i in range(VISITOR_DAILY_JOBS):
		store.submit({**REQ, "n": i}, visitor=visitor, idempotency_key=f"k{i}")
		ids.append(_finish(store, import_stop=first_import_stop and i == 0))
	return ids


def test_one_resend_after_an_import_stop_is_free_and_says_so():
	store, _ = _store()
	first, *_ = _use_the_day(store)
	with pytest.raises(QueueFull) as exc:  # the day's jobs are used
		store.submit({**REQ, "n": 9}, visitor="v", idempotency_key="k9")
	assert exc.value.code == "limit.visitor_daily"
	s = store.submit({**REQ, "n": 9, "resend_of": first}, visitor="v", idempotency_key="k9")
	ev = store.events(s.job_id)[0]
	assert s.created and ev.stage == "queued" and ev.free_resend is True
	assert store.info(s.job_id)  # a job like any other
	_finish(store)
	with pytest.raises(QueueFull) as exc:  # one a day
		store.submit({**REQ, "n": 10, "resend_of": first}, visitor="v", idempotency_key="k10")
	assert exc.value.code == "limit.visitor_daily"


def test_not_free_without_an_import_stop_for_another_visitor_or_with_the_flag_off():
	store, _ = _store()
	ids = _use_the_day(store, first_import_stop=False)
	with pytest.raises(QueueFull):
		store.submit({**REQ, "resend_of": ids[0]}, visitor="v", idempotency_key="x")
	store2, _ = _store()
	theirs = _use_the_day(store2, visitor="other")[0]
	_use_the_day(store2, visitor="v", first_import_stop=False)
	with pytest.raises(QueueFull):  # someone else's job id
		store2.submit({**REQ, "resend_of": theirs}, visitor="v", idempotency_key="x")
	off, _ = _store(on=False)
	first = _use_the_day(off)[0]
	with pytest.raises(QueueFull):
		off.submit({**REQ, "resend_of": first}, visitor="v", idempotency_key="x")


def test_a_job_refused_for_a_package_counts_as_an_import_stop():
	store, _ = _store()
	store.submit({**REQ, "n": 0}, visitor="v", idempotency_key="k0")
	first = _finish(store, fail_code="missing_package.not_allowed")
	for i in (1, 2):
		store.submit({**REQ, "n": i}, visitor="v", idempotency_key=f"k{i}")
		_finish(store)
	assert store.submit({**REQ, "resend_of": first}, visitor="v", idempotency_key="x").created


def test_the_daily_gpu_dollar_cap_still_holds():
	store, _ = _store(daily_cap_usd=lambda day: 0.65, job_budget_usd=0.60)
	store.submit({**REQ, "n": 0}, visitor="v", idempotency_key="k0")
	first = _finish(store, import_stop=True)  # spent 0.01: 0.61 would pass the cap with one more in flight
	store.submit({**REQ, "n": 1}, visitor="w", idempotency_key="w1")  # another visitor's job holds the budget
	with pytest.raises(QueueFull) as exc:
		store.submit({**REQ, "resend_of": first}, visitor="v", idempotency_key="x")
	assert "budget" in str(exc.value)


def test_the_pipeline_marks_a_job_whose_every_probe_stopped_at_an_import(tmp_path):
	import sys
	from pathlib import Path

	sys.path.insert(0, str(Path(__file__).parent))
	from preflight.runner.pipeline import _Ctx, run_pipeline
	from test_pipeline import SCRIPT, FakeRunner, deps

	class ImportStops(FakeRunner):  # the import fails before the first step, as a real probe's does
		def __call__(self, spec, gpu, *, should_stop):
			r = super().__call__(spec, gpu, should_stop=should_stop)
			r.probe = r.probe.model_copy(
				update={"status": "error", "error_code": "missing_package", "steps": 0}
			)
			return r

	import preflight.runner.pipeline as P

	src = {"source": {"kind": "upload", "files": {"train.py": SCRIPT}}}
	real = P._flag
	P._flag = lambda name: name == "runner.free_resend" or real(name)
	try:
		ctx = _Ctx()
		run_pipeline(src, ctx, deps(tmp_path, ImportStops()))
		assert any(e.get("import_stop") for e in ctx.events)
		ok = _Ctx()
		run_pipeline(src, ok, deps(tmp_path, FakeRunner()))
		assert not any(e.get("import_stop") for e in ok.events)
	finally:
		P._flag = real
	off = _Ctx()  # the flag off: no new event, so a job's events are as before
	run_pipeline(src, off, deps(tmp_path, ImportStops()))
	assert not any(e.get("import_stop") for e in off.events)


def test_a_resend_must_send_the_same_code_and_a_replay_or_a_resend_of_a_resend_isnt_a_second_one():
	store, _ = _store()
	first, *_ = _use_the_day(store)
	other = {"source": {"kind": "sample", "uri": "sample://zoo/y"}, "resend_of": first}
	with pytest.raises(QueueFull):  # another script: not the same code sent again
		store.submit(other, visitor="v", idempotency_key="o")
	# the same code with a way past the import (extra packages, the entry point) is the resend
	again = {**REQ, "extra_packages": ["timm"], "resend_of": first}
	s1 = store.submit(again, visitor="v", idempotency_key="r")
	assert store.submit(again, visitor="v", idempotency_key="r").job_id == s1.job_id  # a replay: the same job
	resend = _finish(store, import_stop=True)
	with pytest.raises(QueueFull):  # the day's free resend is used, even for a resend that stopped again
		store.submit({**REQ, "resend_of": resend}, visitor="v", idempotency_key="r2")


def test_a_retried_attempt_doesnt_keep_the_first_attempts_marker():
	store, _ = _store()
	store.submit(REQ, visitor="v", idempotency_key="k")
	lease = store.lease("w")
	store.emit(lease.job_id, stage="probing", import_stop=True)
	store.fail(lease, error="worker lost", retryable=True, spent_usd=0.0)  # back in the queue
	lease = store.lease("w")
	store.complete(lease, record_id="r", spent_usd=0.1)  # this attempt ran its probes
	assert store._index[lease.job_id]["import_stop"] is False


def test_a_snapshot_still_restores_into_a_job_as_before():
	from dataclasses import fields

	from preflight.runner.queue import Job

	assert {f.name for f in fields(Job)}.isdisjoint(
		{"import_stop", "free_resend"}
	)  # an older build restores it
