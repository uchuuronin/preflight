"""The coverage page's scoring (api/scripts/coverage.py): held-out only, abstentions apart, Wilson
intervals, and the eval-off probes kept out of the memory check."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
pytest.importorskip("preflight.engine.api")
import coverage

REAL = Path(__file__).parent / "fixtures" / "a-smoke-l4-main-4.record.json"


def test_wilson_matches_the_textbook_values():
	assert coverage.wilson(0, 0) is None
	lo, hi = coverage.wilson(0, 3)
	assert lo == 0.0 and abs(hi - 0.5615) < 1e-3
	lo, hi = coverage.wilson(9, 10)
	assert abs(lo - 0.5958) < 1e-3 and abs(hi - 0.9821) < 1e-3
	# Rounded outwards: never narrower than the exact interval.
	from preflight.engine import conformal

	exact = conformal.wilson(9, 10)
	assert lo <= exact[0] and hi >= exact[1]


def test_a_real_run_is_scored_only_on_probes_held_out():
	checks, skipped = coverage.check_record("a-smoke", REAL.read_text())
	steps = [c for c in checks if c["metric"] == "step_time"]
	assert {c["probe"] for c in steps} == {"p1", "p2", "p3"}
	# The held-out probe's own number never reaches the prediction. On this plan p2 (batch 2,
	# sequence 1024) and p3 (batch 4, sequence 512) do the same work per step, so without p2 the
	# engine reads p3's time back: the test is weak here, and the page says so.
	p2 = next(c for c in steps if c["probe"] == "p2")
	assert abs(p2["measured"] - 0.837) < 0.001
	# The exact number moves with the engine (1.666 s before B's tokens-per-step scaling, about 1.07 s
	# after, Oct 31), so check only that it was predicted, not read back from p2 itself.
	assert p2["predicted"]["point"] is not None
	assert p2["predicted"]["point"] != pytest.approx(p2["measured"], rel=0.001)
	# Eval-off re-runs after the eval OOM are scored against the peak less the engine's eval term.
	mem = [c for c in checks if c["metric"] == "peak_memory"]
	assert {c["probe"] for c in mem} == {"p1", "p2", "p3"}
	assert {c["basis"] for c in mem} == {"peak_less_eval_and_context"}
	assert not [s for s in skipped if s.get("metric") == "peak_memory"]
	oom = [c for c in checks if c["metric"] == "oom"]
	assert [c["probe"] for c in oom] == ["p1-eval"]


def test_summary_counts_hits_and_leaves_abstentions_out():
	checks = [
		{"metric": "step_time", "hit": True},
		{"metric": "step_time", "hit": False},
		{"metric": "step_time", "verdict": "abstained"},
		{"metric": "oom", "verdict": "caught"},
		{"metric": "oom", "verdict": "abstained"},
	]
	s = coverage.summarise(checks)
	assert s["step_time"]["n"] == 2 and s["step_time"]["hits"] == 1 and s["step_time"]["abstained"] == 1
	assert s["oom"]["n"] == 1 and s["oom"]["hits"] == 1 and s["oom"]["abstained"] == 1
	assert s["peak_memory"]["n"] == 0 and s["peak_memory"]["wilson95"] is None


def test_a_run_whose_probes_all_errored_is_skipped_with_its_reason():
	rec = json.loads(REAL.read_text())
	for p in rec["probes"]:
		p["status"], p["error_code"] = "error", "recorder"
	checks, skipped = coverage.check_record("broken", json.dumps(rec))
	assert checks == []
	assert skipped == [{"run": "broken", "why": "nothing_measured", "codes": ["recorder"]}]


def test_held_out_records_carry_nothing_derived_from_the_probe():
	from preflight.schema import load_record

	rec = load_record(REAL.read_text())
	p1 = next(p for p in rec.probes if p.id == "p1")
	held = coverage.held_out(rec, p1)
	assert {p.id for p in held.probes} == {"p2", "p3"}  # p1-eval is the same point
	assert held.models == [] and held.findings == [] and not held.predictions.estimates


def test_an_eval_off_probe_is_scored_against_the_engine_training_peak():
	"""Independent of coverage.py's arithmetic: the engine's analytical training-moment peak (eval
	batch removed) plus the same calibration offset the engine applied to its eval-moment peak."""
	from dataclasses import replace

	from preflight.engine import api, knobs, memory
	from preflight.engine.memory import DEFAULT_CONTEXT_BYTES as C
	from preflight.schema import load_record

	rec = load_record(REAL.read_text())
	for pid in ("p1", "p2", "p3"):
		p = next(q for q in rec.probes if q.id == pid)
		held = coverage.held_out(rec, p)
		pred = api.predict(held, dict(p.knobs)).root
		values = api._probe_values(rec, p)
		arch, setup, shape = knobs.setup_from(rec.job.script_map, values)
		cap = api._capacity(held, api._target_gpu(held, values))
		with_eval = memory.breakdown(arch, setup, shape.batch, shape.seq, capacity_bytes=cap)
		training = memory.breakdown(
			arch, replace(setup, eval_batch=None), shape.batch, shape.seq, capacity_bytes=cap
		)
		offset = pred["memory.peak"].point - C - with_eval.allocator_peak
		m, basis = coverage.memory_target(p, pred)
		assert basis == "peak_less_eval_and_context"
		assert m["point"] == pytest.approx(training.allocator_peak + offset, rel=1e-6)


def test_a_probe_that_ran_eval_keeps_the_eval_term():
	from preflight.engine.api import predict
	from preflight.engine.memory import DEFAULT_CONTEXT_BYTES
	from preflight.schema import load_record

	rec = load_record(REAL.read_text())
	p2 = next(p for p in rec.probes if p.id == "p2")
	with_eval = p2.model_copy(update={"eval": next(p for p in rec.probes if p.id == "p1-eval").eval})
	pred = predict(coverage.held_out(rec, p2), dict(p2.knobs)).root
	m, basis = coverage.memory_target(with_eval, pred)
	assert basis == "peak_less_context"
	assert m["point"] == pytest.approx(pred["memory.peak"].point - DEFAULT_CONTEXT_BYTES)


def test_the_eval_off_marker_decides_first_and_missing_stats_only_without_one():
	from preflight.schema import load_record

	rec = load_record(REAL.read_text())
	p1e = next(p for p in rec.probes if p.id == "p1-eval")
	p2 = next(p for p in rec.probes if p.id == "p2")
	mark = lambda p, v: p.model_copy(update={"extras": {**(p.extras or {}), "eval_off": v}})  # noqa: E731
	assert coverage.ran_without_eval(mark(p1e, 1.0))  # marker says off, though eval stats exist
	assert not coverage.ran_without_eval(mark(p2, 0.0))  # marker says eval ran, though stats are missing
	assert coverage.ran_without_eval(p2)  # no marker: missing stats decide
	assert not coverage.ran_without_eval(p1e)
