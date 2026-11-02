"""The pieces of the Modal execution plane that hold logic: wire format, sandbox runner, remote store.

Modal itself isn't imported here (it isn't installed in CI); the sandbox and the
service are fakes with the same calls.
"""

import contextlib
import json
import os
from pathlib import Path

import pytest

from preflight.probes.local import ProbeResult, ProbeSpec
from preflight.runner.queue import InMemoryJobStore, LeaseLost
from preflight.runner.remote_store import RemoteJobStore, StoreMethods
from preflight.runner.sandbox_runner import SandboxRunner
from preflight.runner.wire import (
	RESULT_PREFIX,
	result_from_wire,
	result_to_wire,
	spec_from_wire,
	spec_to_wire,
)
from preflight.runner.worker import run_one
from preflight.schema import Env, Probe


def spec(pid="p1"):
	return ProbeSpec(
		probe_id=pid, files={"t.py": "print(1)\n"}, entrypoint="t.py", knobs={"batch_size": 4}, profile=True
	)


def ok_result(pid="p1", **probe_kw):
	probe = Probe(id=pid, knobs={"batch_size": 4}, status="ok", steps=7, **probe_kw)
	return ProbeResult(probe=probe, env=Env(backend="cuda", provider="modal", gpu={"name": "L4"}), summary={}, kernels=None,
		arch={"n_layers": 2, "source": "modules"}, trace_file=None, log_tail="hello", workdir=Path("."))  # fmt: skip


# ---------------------------------------------------------------- wire


def test_spec_and_result_round_trip():
	s = spec()
	assert spec_from_wire(spec_to_wire(s)) == s
	r = result_from_wire(s, "noise from the script\n" + result_to_wire(ok_result()) + "\n")
	assert (
		r.probe.status == "ok" and r.env.gpu.name == "L4" and r.arch == {"n_layers": 2, "source": "modules"}
	)


@pytest.mark.parametrize(
	"stdout",
	[
		"",
		"no result here",
		RESULT_PREFIX + "{not json",
		RESULT_PREFIX + json.dumps([1, 2]),
		RESULT_PREFIX + json.dumps({"probe": {"id": "p1", "status": "bogus"}}),
		RESULT_PREFIX + json.dumps({"probe": {"id": "p9", "knobs": {}, "status": "ok", "steps": 1}}),
	],
)
def test_a_sandbox_that_returns_garbage_gives_a_recorder_error_not_a_crash(stdout):
	r = result_from_wire(spec(), stdout)
	assert r.probe.status == "error" and r.probe.error_code == "recorder" and r.probe.id == "p1"


def test_the_sandbox_cannot_change_the_knobs_or_smuggle_a_shape():
	payload = json.loads(result_to_wire(ok_result())[len(RESULT_PREFIX) :])
	payload["probe"]["knobs"] = {"batch_size": 999}
	payload["arch"] = {"norm": "x" * 10_000, "n_layers": -1, "source": "modules"}
	r = result_from_wire(spec(), RESULT_PREFIX + json.dumps(payload))
	assert r.probe.knobs == {"batch_size": 4} and r.arch == {"source": "modules"}


# ---------------------------------------------------------------- sandbox runner


class FakeProc:
	def __init__(self, sb):
		self.sb, self.buf = sb, []
		self.stdin = self
		self.stdout = self

	def write(self, text):
		self.buf.append(text)

	def write_eof(self):
		pass

	def drain(self):
		pass

	def read(self):
		if self.sb.fail:
			raise RuntimeError("sandbox died")
		s = spec_from_wire("".join(self.buf))
		return result_to_wire(ok_result(s.probe_id))

	def wait(self):
		return 0


class FakeSandbox:
	def __init__(self, gpu, fail=False):
		self.gpu, self.fail, self.terminated, self.execs, self.ready = gpu, fail, False, 0, 0

	def exec(self, *args, timeout=None):
		if args == ("true",):  # the runner's wait for the container (and its GPU)
			self.ready += 1
			return FakeProc(self)
		assert args == ("python", "-m", "preflight.probes.remote") and timeout > 0
		self.execs += 1
		return FakeProc(self)

	def terminate(self):
		self.terminated = True


def test_one_sandbox_per_job_and_a_new_one_when_the_gpu_changes():
	made = []

	def create(gpu):
		made.append(FakeSandbox(gpu))
		return made[-1]

	with SandboxRunner(create) as run:
		assert run(spec("p1"), "L4", should_stop=lambda: False).probe.status == "ok"
		run(spec("p2"), "L4", should_stop=lambda: False)
		run(spec("p3"), "L40S", should_stop=lambda: False)
	assert [(s.gpu, s.execs) for s in made] == [("L4", 2), ("L40S", 1)]
	assert all(s.terminated for s in made)  # nothing outlives the job
	assert set(run.gpu_seconds) == {"L4", "L40S"}


def test_a_broken_sandbox_is_dropped_and_the_probe_says_so():
	made = []

	def create(gpu):
		made.append(FakeSandbox(gpu, fail=not made))
		return made[-1]

	with SandboxRunner(create) as run:
		bad = run(spec("p1"), "L4", should_stop=lambda: False)
		good = run(spec("p2"), "L4", should_stop=lambda: False)
	assert bad.probe.status == "error" and bad.probe.error_code == "recorder"
	assert good.probe.status == "ok" and len(made) == 2 and made[0].terminated


# ---------------------------------------------------------------- remote store


def wire(methods):
	"""What Modal does: arguments and results cross as plain data; exceptions keep their type."""

	def call(name, *args, **kwargs):
		args, kwargs = json.loads(json.dumps(args)), json.loads(json.dumps(kwargs))
		return json.loads(json.dumps(getattr(methods, name)(*args, **kwargs), default=str))

	return call


def test_a_worker_runs_a_job_through_the_remote_store():
	methods = StoreMethods(InMemoryJobStore())
	call = wire(methods)
	sub = call("submit", {"source": {"kind": "sample"}}, "v1", "key-1")
	assert sub["created"] and sub["position"] == 0  # jobs ahead of it
	remote = RemoteJobStore(call)

	def job(lease, ctx):
		ctx.emit("skeleton", record_id="R1", share_token="s" * 24, record_rev=1)
		return "R1"

	assert run_one(remote, "w1", job)
	info = call("status", sub["job_id"])
	assert info["status"] == "done" and info["share_token"] == "s" * 24
	stages = [e["stage"] for e in call("events", sub["job_id"], -1)]
	assert stages == ["queued", "admitted", "skeleton", "done"]


def test_a_stale_lease_is_refused_across_the_wire():
	methods = StoreMethods(InMemoryJobStore())
	call = wire(methods)
	sub = call("submit", {"source": {}}, "v1", "k")
	lease = call("lease", "w1")
	call("cancel", sub["job_id"])
	with pytest.raises(LeaseLost):
		methods.heartbeat(lease)


def test_weights_are_linked_into_a_writable_cache(tmp_path, monkeypatch):
	from preflight.probes.remote import link_weights

	src = tmp_path / "weights" / "hub" / "models--org--m"
	src.mkdir(parents=True)
	monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
	assert link_weights(tmp_path / "weights" / "hub") == 1
	link = tmp_path / "hf" / "hub" / "models--org--m"
	assert link.is_symlink() and link.resolve() == src.resolve()
	assert (tmp_path / "hf" / "hub").stat().st_mode & 0o1777 == 0o1777


def test_demand_reports_the_oldest_waiting_job():
	t = [1000.0]
	methods = StoreMethods(InMemoryJobStore(clock=lambda: t[0]))
	methods.submit({"source": {}}, "v1", "k1")
	t[0] += 30
	methods.submit({"source": {}}, "v2", "k2")
	t[0] += 15
	assert methods.demand() == {"queued": 2, "running": 0, "oldest_queued_s": 45.0}


def test_duration_and_cost_are_measured_by_the_worker_not_reported_by_the_sandbox():
	t = [0.0]

	class Slow(FakeSandbox):
		def exec(self, *args, timeout=None):
			t[0] += (
				600.0 if args == ("true",) else 120.0
			)  # ten minutes' wait for a GPU, then a 2-minute probe
			return super().exec(*args, timeout=timeout)

	from dataclasses import replace

	with SandboxRunner(lambda g: Slow(g), clock=lambda: t[0]) as run:
		r = run(replace(spec(), gpu_usd_per_hour=3.6), "L4", should_stop=lambda: False)
		assert r.probe.duration_s == 120.0 and r.probe.cost.usd == pytest.approx(0.12)
		assert run.billed_usd(lambda g: 3.6) == pytest.approx(0.12)


def test_a_sandbox_that_cannot_start_is_a_probe_error_not_a_crash():
	def create(gpu):
		raise ValueError("'nope' is not a GPU Modal offers")

	with SandboxRunner(create) as run:
		r = run(spec(), "nope", should_stop=lambda: False)
	assert r.probe.status == "error" and r.probe.error_code == "recorder"


def test_records_older_than_30_days_are_pruned_by_their_own_dates(tmp_path):
	import os
	from datetime import UTC, datetime

	from preflight.runner.records import RECORD_TTL_S, prune

	now = 2_000_000_000.0

	def rec(name, age_s, mtime_age=0):
		at = datetime.fromtimestamp(now - age_s, UTC).isoformat()
		(tmp_path / name).write_text(json.dumps({"schema_version": "0.3.0", "created_at": at}))
		os.utime(tmp_path / name, (now - mtime_age, now - mtime_age))

	old, fresh = "a" * 22 + ".json", "b" * 22 + ".json"
	rec(old, RECORD_TTL_S + 60)  # file time says new: the record's own date decides
	rec(fresh, 3600, mtime_age=10**9)  # file time says ancient (a reset volume time): kept
	(tmp_path / "d").write_text("x")
	(tmp_path / ("e" * 22 + ".json")).write_text("not json")  # unreadable date: kept
	(tmp_path / ".tmp-x.json").write_text("{}")
	os.utime(tmp_path / ".tmp-x.json", (now - 2 * 86400, now - 2 * 86400))
	(tmp_path / ("c" * 22 + ".json")).symlink_to(tmp_path / fresh)
	assert prune(tmp_path, now=now) == sorted([old, ".tmp-x.json"])
	assert {p.name for p in tmp_path.iterdir()} == {fresh, "d", "e" * 22 + ".json", "c" * 22 + ".json"}


def test_one_prune_pass_removes_at_most_its_share_oldest_first(tmp_path):
	from preflight.runner.records import PRUNE_MIN_GUARD, prune

	for i in range(100):  # all expired: a backlog, or dates gone wrong
		(tmp_path / (f"{i:02d}" + "z" * 20 + ".json")).write_text(
			f'{{"created_at": "2000-01-{1 + i % 28:02d}T00:00:00Z"}}'
		)
	(tmp_path / ("f" * 22 + ".json")).write_text('{"created_at": "2999-01-01T00:00:00Z"}')  # future: kept
	removed = prune(tmp_path, now=2_000_000_000.0)
	assert len(removed) == max(PRUNE_MIN_GUARD, 25)
	left = {p.name for p in tmp_path.iterdir()}
	assert "f" * 22 + ".json" in left and len(left) == 101 - len(removed)


def test_a_serving_probe_gets_the_serving_image_and_its_own_sandbox():
	from dataclasses import replace

	made = []

	def create(gpu, image="probe"):
		made.append((FakeSandbox(gpu), image))
		return made[-1][0]

	with SandboxRunner(create) as run:
		run(spec("p1"), "L4", should_stop=lambda: False)
		run(replace(spec("p2"), image="serving"), "L4", should_stop=lambda: False)
		run(replace(spec("p3"), image="serving"), "L4", should_stop=lambda: False)
	assert [(s.gpu, s.execs, img) for s, img in made] == [("L4", 1, "probe"), ("L4", 2, "serving")]
	assert all(s.terminated for s, _ in made)


def test_a_serving_scripts_probes_ask_for_the_serving_image_and_key_the_cache_apart(tmp_path):
	from dataclasses import replace
	from pathlib import Path

	from preflight.harness.static import map_static
	from preflight.probes.plan import PlannedProbe
	from preflight.runner.pipeline import Deps, _spec, cache_key
	from preflight.runner.records import LocalRecordWriter
	from preflight.runner.sources import Fetched

	root = Path(__file__).resolve().parents[1] / "zoo" / "serving_vllm"
	files = {"serve.py": (root / "serve.py").read_text()}
	sm = map_static(files, entrypoint="serve.py")
	assert sm.lens == "serving"
	deps = Deps(records=LocalRecordWriter(tmp_path))
	s = _spec(
		Fetched(files=files, entrypoint="serve.py"), sm, PlannedProbe(probe_id="p1", role="current"), deps
	)
	assert s.image == "serving"
	assert cache_key(s, "L4", "t") != cache_key(replace(s, image="probe"), "L4", "t")


def test_the_wait_for_a_gpu_is_neither_probe_time_nor_cost_and_a_result_survives_a_bad_wait():
	t = [0.0]

	class Queued(FakeSandbox):
		def exec(self, *args, timeout=None):
			if args == ("true",):
				t[0] += 900.0  # fifteen minutes in Modal's queue for an L4
			return super().exec(*args, timeout=timeout)

	class BadWait(FakeProc):
		def wait(self):
			raise TimeoutError("exec limit while deleting a big dump")

	class SlowCleanup(FakeSandbox):
		def exec(self, *args, timeout=None):
			p = super().exec(*args, timeout=timeout)
			return p if args == ("true",) else BadWait(self)

	with SandboxRunner(lambda g: Queued(g), clock=lambda: t[0]) as run:
		r = run(spec(), "L4", should_stop=lambda: False)
		assert r.probe.duration_s == 0.0 and run.queued_s == 900.0 and run.billed_usd(lambda g: 3.6) == 0.0
	with SandboxRunner(lambda g: SlowCleanup(g)) as run:
		assert run(spec(), "L4", should_stop=lambda: False).probe.status == "ok"


def test_an_l4_that_wont_schedule_falls_back_to_an_a10_priced_as_an_a10(monkeypatch):
	import time as _time

	from preflight.runner import sandbox_runner

	priced = []
	real_timed = sandbox_runner._timed
	monkeypatch.setattr(
		sandbox_runner, "_timed", lambda r, s, sec: priced.append(s.gpu_usd_per_hour) or real_timed(r, s, sec)
	)
	from dataclasses import replace

	class Stuck(FakeSandbox):  # Modal never finds an L4 in time
		def exec(self, *args, timeout=None):
			if args == ("true",) and self.gpu == "L4":
				_time.sleep(1.0)
			return super().exec(*args, timeout=timeout)

	made = []

	def create(gpu):
		made.append(Stuck(gpu))
		return made[-1]

	prices = {"L4": 0.80, "A10": 1.10, "L40S": 1.95}
	with SandboxRunner(create, price=prices.__getitem__, schedule_wait_s=0.1) as run:
		fb = ("A10", "L40S")  # what the pipeline allows here (none when pinned or ruled out)
		r = run(replace(spec("p1"), gpu_usd_per_hour=0.80, gpu_fallback=fb), "L4", should_stop=lambda: False)
		run(replace(spec("p2"), gpu_fallback=fb), "L4", should_stop=lambda: False)  # the A10 keeps serving
	assert r.probe.status == "ok" and [s.gpu for s in made] == ["L4", "A10"]
	assert made[0].terminated and run.unscheduled == ["L4"] and run.ran_on == ["A10", "A10"]
	assert priced == [1.10, 1.10]  # each probe costed at the A10's list price, not the L4's
	assert set(run.gpu_seconds) == {"A10"}


def test_with_no_fallback_allowed_the_runner_waits_for_the_gpu_it_was_given():
	made = []

	def create(gpu):
		made.append(FakeSandbox(gpu))
		return made[-1]

	with SandboxRunner(create, schedule_wait_s=0.0) as run:  # even with no patience, no other GPU
		assert run(spec("p1"), "L4", should_stop=lambda: False).probe.status == "ok"
	assert [s.gpu for s in made] == ["L4"] and run.last_ran == "L4"


def test_fallbacks_respect_a_pinned_gpu_and_the_visitors_allowed_list():
	from preflight.runner.pipeline import _fallbacks

	assert _fallbacks("L4", None, False) == ("A10", "L40S")
	assert _fallbacks("L4", None, True) == ()  # the guide asked for an L4
	assert _fallbacks("L4", ["L4", "L40S"], False) == ("L40S",)
	assert _fallbacks("A100-SXM-80GB", None, False) == ()


def test_the_wait_for_a_gpu_is_logged_and_capped():
	import time as _time

	class Slow(FakeSandbox):
		def exec(self, *args, timeout=None):
			if args == ("true",):
				_time.sleep(0.3)  # Modal finding a GPU
			return super().exec(*args, timeout=timeout)

	with SandboxRunner(lambda g: Slow(g)) as run:
		r = run(spec("p1"), "L4", should_stop=lambda: False)
		again = run(spec("p2"), "L4", should_stop=lambda: False)  # the same sandbox: no wait
	assert r.probe.extras["gpu_wait_s"] >= 0.2 and "gpu_wait_s" not in again.probe.extras

	class Never(FakeSandbox):
		def exec(self, *args, timeout=None):
			if args == ("true",):
				_time.sleep(5)
			return super().exec(*args, timeout=timeout)

	from dataclasses import replace

	with SandboxRunner(lambda g: Never(g), schedule_wait_s=0.1, wait_cap_s=0.3) as run:
		r = run(replace(spec("p1"), gpu_fallback=("A10",)), "L4", should_stop=lambda: False)
	assert r.probe.status == "error" and r.probe.error_code == "gpu.unavailable"
	assert "L4 or A10" in r.probe.error and r.probe.cost.usd == 0 and run.unscheduled == ["L4", "A10"]


def test_a_probe_waiting_for_a_gpu_says_so_as_it_waits_and_when_its_up(monkeypatch):
	# D, Nov 1: an eval run sat 19.5 minutes in "probing" with no events, waiting for an L4
	import time as _time

	from preflight.runner import sandbox_runner

	monkeypatch.setattr(sandbox_runner, "WAIT_NOTE_S", 0.1)

	class Slow(FakeSandbox):
		def exec(self, *args, timeout=None):
			if args == ("true",):
				_time.sleep(0.45)  # Modal finding a GPU
			return super().exec(*args, timeout=timeout)

	told = []
	with SandboxRunner(lambda g: Slow(g)) as run:
		run.notify = lambda kind, gpu, waited: told.append((kind, gpu, round(waited, 1)))
		r = run(spec("p1"), "L4", should_stop=lambda: False)
		run(spec("p2"), "L4", should_stop=lambda: False)  # the same sandbox: nothing to wait for
	kinds = [k for k, _, _ in told]
	assert (
		kinds[0] == "waiting" and kinds.count("waiting") >= 3 and kinds[-1] == "up" and kinds.count("up") == 1
	)
	assert all(g == "L4" for _, g, _ in told) and r.probe.extras["gpu_wait_s"] >= 0.4

	def broken(kind, gpu, waited):
		raise RuntimeError("the status line is down")

	with SandboxRunner(lambda g: Slow(g)) as run:  # a notice that fails never stops the probe
		run.notify = broken
		assert run(spec("p1"), "L4", should_stop=lambda: False).probe.status == "ok"


def test_a_probe_past_its_own_limit_is_stopped_by_the_runner_and_says_timeout(monkeypatch):
	# serving and eval, Nov 1: the exec's timeout didn't end them; the sandbox's 30 min did, no log
	import time as _time
	from dataclasses import replace

	from preflight.runner import sandbox_runner

	monkeypatch.setattr(sandbox_runner, "EXEC_SLACK_S", 0)

	class Hung(FakeProc):
		def read(self):
			while not self.sb.terminated:  # a probe that never returns until its sandbox is ended
				_time.sleep(0.05)
			raise RuntimeError("sandbox terminated")

	class HungBox(FakeSandbox):
		def exec(self, *args, timeout=None):
			p = super().exec(*args, timeout=timeout)
			return p if args == ("true",) else Hung(self)

	with SandboxRunner(lambda g: HungBox(g)) as run:
		t0 = _time.monotonic()
		r = run(replace(spec("p1"), timeout_s=1.5), "L4", should_stop=lambda: False)
	assert _time.monotonic() - t0 < 5
	assert r.probe.status == "timeout" and r.probe.error_code == "timeout" and "s limit" in r.probe.error


def test_modals_own_capacity_wait_in_create_is_told_bounded_and_falls_back(monkeypatch):
	# review, Nov 2: a GPU sandbox's create itself waits up to 21 min for scheduling, then raises
	# "Insufficient capacity": three probes ended "without a result" about 1,280 s in, no wait recorded
	import time as _time
	from dataclasses import replace

	from preflight.runner import sandbox_runner

	monkeypatch.setattr(sandbox_runner, "WAIT_NOTE_S", 0.05)

	class ResourceExhaustedError(Exception):
		pass

	made = []

	def create(gpu):
		if gpu == "L4":
			_time.sleep(0.3)  # Modal waiting for an L4
			raise ResourceExhaustedError("Insufficient capacity to create sandbox.")
		made.append(FakeSandbox(gpu))
		return made[-1]

	told = []
	with SandboxRunner(create, schedule_wait_s=5.0, wait_cap_s=10.0) as run:
		run.notify = lambda kind, gpu, waited: told.append((kind, gpu))
		r = run(replace(spec("p1"), gpu_fallback=("A10",)), "L4", should_stop=lambda: False)
	assert r.probe.status == "ok" and run.ran_on == ["A10"] and run.unscheduled == ["L4"]
	assert ("waiting", "L4") in told  # the wait inside create is told
	# no fallback: NoGPU, naming Modal's reason, with the wait recorded
	with SandboxRunner(create, schedule_wait_s=5.0, wait_cap_s=10.0) as run:
		r = run(spec("p1"), "L4", should_stop=lambda: False)
	assert r.probe.error_code == "gpu.unavailable" and "Insufficient capacity" in r.probe.error
	assert r.probe.extras["gpu_wait_s"] >= 0.25

	def slow_create(gpu):  # a create still waiting past the job's wait cap is abandoned, then ended
		box = FakeSandbox(gpu)
		made.append(box)
		_time.sleep(0.5)
		return box

	with SandboxRunner(slow_create, schedule_wait_s=0.1, wait_cap_s=0.2) as run:
		r = run(spec("p2"), "L4", should_stop=lambda: False)
	assert r.probe.error_code == "gpu.unavailable"
	_time.sleep(0.6)
	assert made[-1].terminated  # ended when its create came back, never left running


def test_a_failed_start_check_is_named_on_the_probe_that_lost_its_result():
	# Nov 1 to 2: three probes ended "without a result" about 1,280 s in with no GPU wait recorded
	class NoReady(FakeSandbox):
		def exec(self, *args, timeout=None):
			if args == ("true",):
				assert timeout < 30 * 60  # under the sandbox's own 30-min life
				raise RuntimeError("exec timeout exceeds the sandbox's")
			return super().exec(*args, timeout=timeout)

	class Silent(FakeProc):
		def read(self):
			return ""

	class Box(NoReady):
		def exec(self, *args, timeout=None):
			super().exec(*args, timeout=timeout)
			return Silent(self)

	with SandboxRunner(lambda g: Box(g)) as run:
		r = run(spec("p1"), "L4", should_stop=lambda: False)
		assert "start check failed: RuntimeError: exec timeout exceeds" in r.probe.error
		assert run.ready_error is None  # told once, on that probe


def test_the_jobs_wall_budget_doesnt_count_the_gpu_wait():
	from preflight.runner.pipeline import Budget

	b = Budget(wall_s=60.0)
	b.started -= 50.0  # 50 s since the job began...
	b.waited = lambda: 45.0  # ...45 of them waiting for a GPU
	assert 54.0 < b.left_s() <= 55.0


def test_the_gpu_wait_cap_covers_the_whole_job_not_each_probe():
	import time as _time

	class Never(FakeSandbox):
		def exec(self, *args, timeout=None):
			if args == ("true",):
				_time.sleep(5)
			return super().exec(*args, timeout=timeout)

	with SandboxRunner(lambda g: Never(g), wait_cap_s=0.3) as run:
		t0 = _time.monotonic()
		first = run(spec("p1"), "L4", should_stop=lambda: False)
		second = run(spec("p2"), "L4", should_stop=lambda: False)  # the cap is spent: no second wait
		took = _time.monotonic() - t0
	assert first.probe.error_code == second.probe.error_code == "gpu.unavailable" and took < 1.0


def test_a_probe_uid_gets_a_user_name_so_getpass_works_without_a_passwd_entry():
	# torchtitan raised KeyError getpwuid() on the hosted runner's uid 61000 (Nov 1): getpass.getuser()
	# reads LOGNAME/USER before /etc/passwd, so the probe sets both whenever it drops to a uid
	import getpass

	from preflight.probes.local import PROBE_USER, _probe_user_env, scrubbed_env

	assert _probe_user_env(None) == {}
	env = scrubbed_env(Path("/tmp/h"), _probe_user_env(61000))
	assert env["USER"] == env["LOGNAME"] == PROBE_USER
	import os
	from unittest import mock

	with mock.patch.dict(os.environ, {"LOGNAME": env["LOGNAME"], "USER": env["USER"]}):
		assert getpass.getuser() == PROBE_USER


def test_the_probe_user_check_reports_the_names_and_imports_it_ran_with(monkeypatch):
	from preflight.probes import local

	monkeypatch.setattr(local, "_probe_uid", lambda: None)  # as this process's own user
	r = local.probe_user_check(("json",))
	assert r["exit"] == 0 and r["imported"] == ["json"] and r["user"] == r["passwd"]


@pytest.mark.skipif(not hasattr(os, "geteuid") or os.geteuid() != 0, reason="drops to a uid: needs root")
def test_the_probe_user_check_reports_a_missing_passwd_entry_and_the_gate_fails_it(monkeypatch):
	# the image bug torchtitan hit on a GPU (Nov 1), caught here on a CPU instead
	import pwd

	from preflight.probes import local

	uid = 64_999
	with contextlib.suppress(KeyError):
		pwd.getpwuid(uid)
		pytest.skip("uid 64999 has a passwd entry on this host")
	monkeypatch.setattr(local, "_probe_uid", lambda: uid)
	r = local.probe_user_check(())
	# the script reports the missing entry (a plain host has none: the AMD droplet's check) and the
	# deploy gate fails it, where the image needs one
	from preflight.runner.probe_user import probe_user_ok

	assert r["exit"] == 0 and r["passwd"] is None and r["group"] is None
	fine = {**r, "uid": 61000, "user": "preflight-probe", "imported": [], "compiled": 1.0}
	assert probe_user_ok({**fine, "passwd": "preflight-probe", "group": "preflight-probe"}, "probe")
	assert not probe_user_ok(fine, "probe")


def test_the_deploy_gate_needs_the_names_the_imports_and_a_compile():
	from preflight.runner.probe_user import probe_user_ok as _probe_user_ok

	good = {"exit": 0, "uid": 61000, "user": "preflight-probe", "passwd": "preflight-probe",
		"group": "preflight-probe", "imported": ["vllm", "datasets"], "compiled": 4.0}  # fmt: skip
	assert _probe_user_ok(good, "serving")
	assert not _probe_user_ok({**good, "compiled": None}, "serving")  # torch.compile didn't run
	assert not _probe_user_ok({**good, "imported": []}, "serving")  # vllm didn't import
	assert not _probe_user_ok({**good, "passwd": None}, "serving")  # the image has no passwd entry
	assert not _probe_user_ok({**good, "group": None}, "serving")  # nor a group entry
	assert not _probe_user_ok({**good, "group": "30002"}, "serving")
	assert not _probe_user_ok({"failed": "TimeoutExpired"}, "probe")
	assert _probe_user_ok({**good, "imported": []}, "probe")


def test_the_cli_as_root_runs_probes_as_itself_when_a_fresh_uid_cant_reach_its_venv(tmp_path, monkeypatch):
	# preflight demo as root with a venv under a 0700 folder: a dropped uid ran a Python without torch
	from preflight.probes import local

	hidden = tmp_path / "private"
	(hidden / "venv" / "bin").mkdir(parents=True)
	hidden.chmod(0o700)
	monkeypatch.setattr(local.os, "geteuid", lambda: 0)
	monkeypatch.delenv("PREFLIGHT_PROBE_UID", raising=False)
	monkeypatch.setattr(local.sys, "executable", str(hidden / "venv" / "bin" / "python"))
	assert local._probe_uid() is None
	monkeypatch.setenv("PREFLIGHT_PROBE_UID", "61000")  # the hosted sandbox pins its uid: unchanged
	assert local._probe_uid() == 61000
