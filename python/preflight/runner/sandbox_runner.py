"""Run a job's probe points in one GPU sandbox: one container per job (per GPU, if the job steps up).

Modal-free: the sandbox comes from a factory, so tests drive this with a fake
and runner/modal_app.py passes modal.Sandbox.create. The worker never runs
visitor code itself; it only starts the sandbox, feeds it a spec and reads
one result line back (runner/wire.py).
"""

import contextlib
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import replace

from preflight.probes.local import ProbeResult, ProbeSpec
from preflight.schema import ProbeCost

from .wire import result_from_wire, spec_to_wire

log = logging.getLogger("preflight.sandbox")

ENTRY = ("python", "-m", "preflight.probes.remote")
EXEC_SLACK_S = 90  # over the probe's own cap: model download is inside it, process start is not
READY = ("true",)  # a no-op run first: returns once the container is up, after any wait for a GPU
READY_TIMEOUT_S = 30 * 60  # Modal's queue for a GPU can be long (L4: 14 to 25 min seen, Oct 31 to Nov 1)


def suppress_all():
	return contextlib.suppress(Exception)  # a failed no-op shows itself on the probe's own exec


# When a GPU won't schedule (L4 waits of 14 to 26 min, Oct 31 to Nov 1, against "first map in 5
# minutes"), the spec's gpu_fallback GPUs are tried in turn (the pipeline says which, Chiya, Nov 1:
# L4, then A10, then L40S); the last is waited for as long as it takes, and each probe is priced at
# the list rate of the GPU that ran it
SCHEDULE_WAIT_S = 120
# the hard cap on the whole wait for a GPU, fallbacks included (Chiya, Nov 1): past it the probe is
# an error saying no GPU could be scheduled, not a probe that eats the job's time
WAIT_CAP_S = 20 * 60


class NoGPU(RuntimeError):
	pass


class SandboxRunner:
	"""Callable as a pipeline ProbeRunner. Use as a context manager so the sandbox always ends.

	price: list USD per hour by GPU name, to cost a probe at the GPU that ran it after a fallback."""

	def __init__(self, create: Callable[[str], object], *, clock: Callable[[], float] = time.monotonic,
		price: Callable[[str], float] | None = None, schedule_wait_s: float = SCHEDULE_WAIT_S,
		wait_cap_s: float = WAIT_CAP_S):  # fmt: skip
		self._create = create
		self._clock = clock
		self._price = price
		self._schedule_wait_s = schedule_wait_s
		self._wait_cap_s = wait_cap_s
		self.last_wait_s = 0.0  # the wait before the current sandbox started (0 when it was reused)
		self._sb = None
		self._gpu: str | None = None  # the GPU the sandbox has
		self._asked: str | None = None  # the GPU the job asked for (a fallback keeps serving it)
		self._image: str = "probe"
		self._started = 0.0
		self.gpu_seconds: dict[str, float] = {}  # sandbox lifetime per GPU: what Modal bills
		self.queued_s = 0.0  # time spent waiting for a GPU before any sandbox started
		self.ran_on: list[str] = []  # the GPU each probe ran on, in order
		self.last_ran: str | None = None  # the GPU of the last probe (the pipeline reads it)
		self.unscheduled: list[str] = []  # GPUs given up on after SCHEDULE_WAIT_S

	def __enter__(self):
		return self

	def __exit__(self, *exc):
		self.close()

	def _ready(self, sb, wait_s: float | None) -> bool:
		"""The no-op exec returns once the container is up (after Modal's wait for a GPU)."""
		box: dict = {}

		def go():
			try:
				sb.exec(*READY, timeout=READY_TIMEOUT_S).wait()
				box["ok"] = True
			except Exception:
				box["ok"] = True  # a failed no-op shows itself on the probe's own exec

		t = threading.Thread(target=go, daemon=True)
		t.start()
		t.join(wait_s)
		return bool(box.get("ok"))

	def _sandbox(self, gpu: str, image: str = "probe", fallback: tuple[str, ...] = ()):
		# the job's own GPU, or the one it fell back to (the pipeline then asks for that one by name)
		if self._sb is not None and gpu in (self._asked, self._gpu) and self._image == image:
			return self._sb
		self.close()
		options = (gpu, *(g for g in fallback if g != gpu))
		t = self._clock()
		self.last_wait_s = 0.0
		for i, g in enumerate(options):
			# the probe image by its old one-argument call; another image (serving) by name
			sb = self._create(g) if image == "probe" else self._create(g, image=image)
			last = i == len(options) - 1
			left = max(0.0, self._wait_cap_s - (self._clock() - t))
			# wait out the GPU queue before any clock starts: neither the probe's time limit nor its
			# cost counts the minutes Modal spent finding a GPU (no container runs, nothing is billed)
			if self._ready(sb, left if last else min(self._schedule_wait_s, left)):
				break
			self.unscheduled.append(g)
			with suppress_all():
				sb.terminate()
			if last or left <= 0:
				waited = self._clock() - t
				self.queued_s += waited
				self.last_wait_s = waited
				log.warning("no GPU of %s scheduled in %.0f s; giving up", ", ".join(options), waited)
				raise NoGPU(f"no {' or '.join(options)} could be scheduled in {waited / 60:.0f} min")
			log.info("no %s scheduled in %.0f s; trying %s", g, self._schedule_wait_s, options[i + 1])
		self._sb, self._gpu, self._asked, self._image = sb, g, gpu, image
		waited = self._clock() - t
		self.queued_s += waited
		self.last_wait_s = waited
		if waited > 1:
			log.info("%s scheduled after %.0f s of waiting", g, waited)
		self._started = self._clock()
		return self._sb

	def close(self) -> None:
		if self._sb is None:
			return
		try:
			self._sb.terminate()
		except Exception:
			log.warning("sandbox terminate failed", exc_info=True)
		self.gpu_seconds[self._gpu] = self.gpu_seconds.get(self._gpu, 0.0) + (self._clock() - self._started)
		self._sb, self._gpu = None, None

	def billed_usd(self, price: Callable[[str], float]) -> float:
		"""What the sandboxes have cost so far, the open one included: Modal bills their lifetime."""
		total = sum(price(g) * s / 3600 for g, s in self.gpu_seconds.items())
		if self._sb is not None:
			total += price(self._gpu) * (self._clock() - self._started) / 3600
		return total

	def __call__(self, spec: ProbeSpec, gpu: str, *, should_stop: Callable[[], bool]) -> ProbeResult:
		t0 = self._clock()
		try:
			sb = self._sandbox(gpu, spec.image, tuple(spec.gpu_fallback))
			ran = self._gpu or gpu  # read now: a failure or a cancel below closes the sandbox
			wait, self.last_wait_s = self.last_wait_s, 0.0
			t0 = self._clock()  # after the container is up: the queue for a GPU is neither time nor cost
			proc = sb.exec(*ENTRY, timeout=int(spec.timeout_s + EXEC_SLACK_S))
		except NoGPU as exc:
			self.close()
			self.last_ran = gpu
			self.ran_on.append(gpu)
			r = result_from_wire(spec, "")
			r.probe = r.probe.model_copy(update={"error": str(exc), "error_code": "gpu.unavailable",
				"extras": {**r.probe.extras, "gpu_wait_s": round(self.last_wait_s, 1)}})  # fmt: skip
			return _timed(r, spec, 0.0)
		except Exception:
			log.warning("probe %s: no sandbox on %s", spec.probe_id, gpu, exc_info=True)
			self.close()
			self.last_ran = gpu
			self.ran_on.append(gpu)
			return _timed(result_from_wire(spec, ""), spec, self._clock() - t0)
		done = threading.Event()

		def watch():  # a cancelled job or a lost lease ends the sandbox mid-probe
			while not done.wait(1.0):
				if should_stop():
					log.info("stopping sandbox for %s: job cancelled or lease lost", spec.probe_id)
					self.close()
					return

		watcher = threading.Thread(target=watch, daemon=True)
		watcher.start()
		try:
			proc.stdin.write(spec_to_wire(spec))
			proc.stdin.write_eof()
			proc.stdin.drain()
			out = proc.stdout.read()
		except Exception:
			log.warning("probe %s: sandbox exec failed", spec.probe_id, exc_info=True)
			out = ""
			self.close()  # a broken sandbox is not reused
		try:
			if out:
				proc.wait()  # the result is read: a slow cleanup after it (or a wait error) doesn't lose it
		except Exception:
			log.warning("probe %s: sandbox exec ended badly after its result", spec.probe_id, exc_info=True)
			self.close()
		finally:
			done.set()
			watcher.join(timeout=2)
		self.last_ran = ran
		self.ran_on.append(ran)
		if ran != gpu and self._price is not None:
			spec = replace(spec, gpu_usd_per_hour=self._price(ran))  # list price of the GPU that ran
		r = _timed(result_from_wire(spec, out or ""), spec, self._clock() - t0)
		if wait > 0:  # logged on the probe that waited: how long the GPU took to schedule
			r.probe = r.probe.model_copy(update={"extras": {**r.probe.extras, "gpu_wait_s": round(wait, 1)}})
		return r


def _timed(result: ProbeResult, spec: ProbeSpec, seconds: float) -> ProbeResult:
	"""Duration and cost as the worker measured them: the sandbox runs visitor code, so its own
	report of how long it took (and so what it cost) doesn't count against the budget."""
	cost = ProbeCost(gpu_seconds=round(seconds, 3), usd=round(seconds * spec.gpu_usd_per_hour / 3600, 6))
	result.probe = result.probe.model_copy(
		update={"duration_s": round(seconds, 3), "cost": cost, "trace": None}
	)
	return result
