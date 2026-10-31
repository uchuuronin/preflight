"""Run a job's probe points in one GPU sandbox: one container per job (per GPU, if the job steps up).

Modal-free: the sandbox comes from a factory, so tests drive this with a fake
and runner/modal_app.py passes modal.Sandbox.create. The worker never runs
visitor code itself; it only starts the sandbox, feeds it a spec and reads
one result line back (runner/wire.py).
"""

import logging
import threading
import time
from collections.abc import Callable

from preflight.probes.local import ProbeResult, ProbeSpec
from preflight.schema import ProbeCost

from .wire import result_from_wire, spec_to_wire

log = logging.getLogger("preflight.sandbox")

ENTRY = ("python", "-m", "preflight.probes.remote")
EXEC_SLACK_S = 90  # over the probe's own cap: model download is inside it, process start is not


class SandboxRunner:
	"""Callable as a pipeline ProbeRunner. Use as a context manager so the sandbox always ends."""

	def __init__(self, create: Callable[[str], object], *, clock: Callable[[], float] = time.monotonic):
		self._create = create
		self._clock = clock
		self._sb = None
		self._gpu: str | None = None
		self._image: str = "probe"
		self._started = 0.0
		self.gpu_seconds: dict[str, float] = {}  # sandbox lifetime per GPU: what Modal bills

	def __enter__(self):
		return self

	def __exit__(self, *exc):
		self.close()

	def _sandbox(self, gpu: str, image: str = "probe"):
		if self._sb is not None and self._gpu == gpu and self._image == image:
			return self._sb
		self.close()
		# the probe image by its old one-argument call; another image (serving) by name
		self._sb = self._create(gpu) if image == "probe" else self._create(gpu, image=image)
		self._gpu, self._image, self._started = gpu, image, self._clock()
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
			sb = self._sandbox(gpu, spec.image)
			proc = sb.exec(*ENTRY, timeout=int(spec.timeout_s + EXEC_SLACK_S))
		except Exception:
			log.warning("probe %s: no sandbox on %s", spec.probe_id, gpu, exc_info=True)
			self.close()
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
			proc.wait()
		except Exception:
			log.warning("probe %s: sandbox exec failed", spec.probe_id, exc_info=True)
			out = ""
			self.close()  # a broken sandbox is not reused
		finally:
			done.set()
			watcher.join(timeout=2)
		return _timed(result_from_wire(spec, out or ""), spec, self._clock() - t0)


def _timed(result: ProbeResult, spec: ProbeSpec, seconds: float) -> ProbeResult:
	"""Duration and cost as the worker measured them: the sandbox runs visitor code, so its own
	report of how long it took (and so what it cost) doesn't count against the budget."""
	cost = ProbeCost(gpu_seconds=round(seconds, 3), usd=round(seconds * spec.gpu_usd_per_hour / 3600, 6))
	result.probe = result.probe.model_copy(
		update={"duration_s": round(seconds, 3), "cost": cost, "trace": None}
	)
	return result
