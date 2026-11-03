"""probes.step_clock (off; B's step-time ask, Nov 3): the GPU's SM clock on each timed step, from the
sampler's last NVML read; nothing changes on a CPU or with the flag off."""

import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).parent))

from preflight.probes.local import EXTRAS_KEYS, ProbeSpec, run_probe
from test_recorder import spec


def test_on_a_cpu_the_probe_runs_as_before_and_reports_no_clock():
	r = run_probe(spec("tiny_loop.py", warmup=2, timed=3, timeout_s=120, step_clock=True))
	assert r.probe.status == "ok" and not any(k.startswith("sm_clock") for k in (r.probe.extras or {}))


def test_the_spec_carries_it_and_its_extras_are_named():
	assert ProbeSpec.__dataclass_fields__["step_clock"].default is False
	assert {"sm_clock_mhz_p50", "sm_clock_mhz_min", "sm_clock_mhz_max", "sm_clock_timed_steps"} <= set(
		EXTRAS_KEYS
	)


def test_the_sampler_keeps_the_clock_and_counts_its_readings(monkeypatch):
	import threading
	import time

	import preflight.recorder.runtime as RT

	monkeypatch.setattr(RT, "gpu_clock_reader", lambda: lambda: 1410.0)
	monkeypatch.setattr(RT, "gpu_power_reader", lambda: None)
	monkeypatch.setattr(RT, "SAMPLE_EVERY_S", 0.01)
	rec = RT.Recorder(step_clock=True)
	rec.cuda = True
	t = threading.Thread(target=rec._sample_loop, daemon=True)
	t.start()
	time.sleep(0.1)
	rec._sampler_stop.set()
	t.join(1)
	assert rec._clock_now == 1410.0 and rec._clock_seq >= 2
	off = RT.Recorder()
	assert off.step_clock is False and off._clock_seq == 0
