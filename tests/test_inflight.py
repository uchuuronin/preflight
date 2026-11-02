"""recorder/inflight.py: the collector's windows, bands and alerts, without torch; and one real
CPU run of a tiny loop against a record's predictions when torch is here."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from preflight.recorder.inflight import Bands, Collector, bands_from, live_record
from preflight.schema import load_record

ROOT = Path(__file__).resolve().parents[1]
FIX = Path(__file__).resolve().parent / "fixtures"


class Clock:
	def __init__(self):
		self.t = 0.0

	def __call__(self):
		return self.t


def _run(col, clock, step_s, n):
	for _ in range(n):
		clock.t += step_s
		col.step()


def _go(col, clock, setup_s=30.0):
	"""Start the run, spend setup_s loading, then the first optimizer step (where windows start)."""
	col.start()
	clock.t += setup_s
	col.step()


def test_a_slow_stretch_alerts_once_after_two_windows_and_again_after_recovery():
	clock, said, lines = Clock(), [], []
	col = Collector(Bands(step_s=(1.0, 1.5)), window=10, clock=clock, alert=said.append, write=lines.append)
	_go(col, clock)
	_run(col, clock, 1.2, 20)  # in band
	_run(col, clock, 2.0, 10)  # one slow window: noise
	assert said == []
	_run(col, clock, 2.0, 30)  # still slow: one alert, not three
	assert len(said) == 1 and "above the predicted 1–1.5 s" in said[0]
	_run(col, clock, 1.2, 10)  # back in band
	_run(col, clock, 0.5, 20)  # fast now: a new alert
	assert len(said) == 2 and "below" in said[1]
	assert [w["step"] for w in lines][:3] == [11, 21, 31] and lines[0]["step_s"] == pytest.approx(1.2)
	assert col.setup_s == pytest.approx(30.0)  # loading is not step time


def test_memory_over_the_band_alerts_at_once_and_the_total_is_projected():
	clock, said = Clock(), []
	peak = {"v": 10 * 2**30}
	col = Collector(Bands(peak_bytes=(9 * 2**30, 11 * 2**30), total_s=(90.0, 110.0), total_steps=101), window=10,
		clock=clock, peak=lambda: (peak["v"] - 2**30, peak["v"]), alert=said.append)  # fmt: skip
	_go(col, clock, setup_s=0.0)
	_run(col, clock, 1.0, 10)
	assert said == [] and col.windows[-1]["projected_total_s"] == pytest.approx(100.0)
	peak["v"] = 12 * 2**30
	_run(col, clock, 1.5, 10)
	assert any("peak memory 12.00 GiB" in s for s in said)
	assert any("at this pace the run takes" in s for s in said)


def test_the_last_partial_window_ends_at_the_last_step_not_at_the_stop():
	"""B's A10 run (Nov 2): stopped by SIGTERM a second after its last step, the partial window counted the wait as
	step time (1.79 s against 1.4 to 1.6 s steps), which raised a false total-time alert on a run on plan."""
	clock, said = Clock(), []
	col = Collector(Bands(step_s=(0.9, 1.1), total_s=(95.0, 105.0), total_steps=101), window=10, clock=clock,
		alert=said.append)  # fmt: skip
	_go(col, clock, setup_s=0.0)
	_run(col, clock, 1.0, 29)  # two full windows, then 9 steps
	clock.t += 1.5  # the stop comes 1.5 s after the last step
	col.finish()
	last = col.windows[-1]
	assert last["steps"] == 9 and last["step_s"] == pytest.approx(1.0)
	assert last["elapsed_s"] == pytest.approx(29.0)
	assert said == []


def test_the_summary_thins_a_long_log_to_the_schema_cap():
	clock = Clock()
	col = Collector(Bands(), window=1, clock=clock)
	_go(col, clock)
	_run(col, clock, 0.1, 1000)
	s = col.finish()
	assert s["steps"] == 1001 and len(s["step_log"]) == 256 and s["step_log"][-1]["index"] == 1001
	assert s["step_time"]["n_steps"] == 1000 and s["step_time"]["mean_s"] == pytest.approx(0.1)


def test_the_live_record_is_a_new_record_on_the_predicted_one():
	parent = load_record((ROOT / "schema" / "examples" / "finetune.json").read_text())
	b = bands_from(parent)
	assert b.peak_bytes is not None and b.total_s is not None
	clock = Clock()
	col = Collector(b, window=5, clock=clock)
	_go(col, clock)
	_run(col, clock, 1.0, 12)
	rec = live_record(parent, col.finish(), status="ok")
	assert rec.parent_id == parent.record_id and rec.job.source.kind == "inflight"
	live = rec.probes[-1]
	assert live.id == "live" and live.status == "ok" and live.steps == 13 and len(live.step_log) == 3
	load_record(rec.model_dump_json())  # valid against the schema
	again = live_record(rec, col.summary(), status="ok")  # watching the live record: still one live probe
	assert [p.id for p in again.probes].count("live") == 1


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_a_real_loop_streams_a_live_record(tmp_path):
	import subprocess

	parent = ROOT / "schema" / "examples" / "finetune.json"
	out = tmp_path / "live"
	script = tmp_path / "loop.py"
	script.write_text((FIX / "tiny_loop.py").read_text())
	r = subprocess.run([sys.executable, "-m", "preflight.recorder.inflight", str(parent), str(out), str(script),
		"--window", "2"], capture_output=True, text=True, timeout=300)  # fmt: skip
	assert r.returncode == 0, r.stderr[-2000:]
	recs = list(out.glob("*.json"))
	assert len(recs) == 1
	live = load_record(recs[0].read_text()).probes[-1]
	assert live.id == "live" and live.status == "ok" and live.steps > 0
	lines = [json.loads(x) for x in next(out.glob("*.live.jsonl")).read_text().splitlines()]
	assert lines and all("step_s" in x for x in lines)


def test_a_failing_script_prints_its_traceback_and_keeps_its_exit_code(tmp_path):
	import subprocess

	pytest.importorskip("torch")
	bad = tmp_path / "bad.py"
	bad.write_text("raise RuntimeError('boom in user code')\n")
	r = subprocess.run([sys.executable, "-m", "preflight.recorder.inflight", str(ROOT / "schema" / "examples" /
		"finetune.json"), str(tmp_path / "o"), str(bad)], capture_output=True, text=True, timeout=120)  # fmt: skip
	assert r.returncode == 1 and "Traceback" in r.stderr and "boom in user code" in r.stderr
	live = load_record(next((tmp_path / "o").glob("*.json")).read_text()).probes[-1]
	assert live.status == "error" and "boom" in live.error


def test_a_run_on_another_card_says_different_gpu_instead_of_alerting():
	"""B's A10 check (Nov 2): Modal's A10 gave an A10G, about 11% slower, and the step and total alerts fired."""
	from preflight.recorder.inflight import card_change

	assert card_change("A10", "NVIDIA A10") is None
	assert card_change("L4", "NVIDIA L4") is None
	assert "different GPU" in card_change("A10", "NVIDIA A10G")
	assert "different GPU" in card_change("L4", "NVIDIA A10")
	assert card_change(None, "NVIDIA A10G") is None and card_change("A10", None) is None
	assert card_change("A100-SXM-80GB", "NVIDIA A100-SXM4-80GB") is None  # the same card, named its own way
