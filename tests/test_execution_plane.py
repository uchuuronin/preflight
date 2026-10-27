"""The pieces of the Modal execution plane that hold logic: wire format, sandbox runner, remote store.

Modal itself isn't imported here (it isn't installed in CI); the sandbox and the
service are fakes with the same calls.
"""

import json
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
		self.gpu, self.fail, self.terminated, self.execs = gpu, fail, False, 0

	def exec(self, *args, timeout=None):
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
			t[0] += 120.0  # two minutes on the worker's clock
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
