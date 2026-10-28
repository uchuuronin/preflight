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
	assert p2["predicted"]["point"] == pytest.approx(1.666, rel=0.01)
	# Eval-off re-runs after the eval OOM aren't compared with a job-as-written memory prediction.
	assert {s["probe"] for s in skipped if s.get("metric") == "peak_memory"} == {"p1", "p2", "p3"}
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


def test_a_probe_without_an_eval_pass_is_skipped_only_when_the_engine_peak_includes_eval():
	from preflight.engine.api import predict
	from preflight.schema import load_record

	rec = load_record(REAL.read_text())
	p2 = next(p for p in rec.probes if p.id == "p2")
	as_written = predict(coverage.held_out(rec, p2), dict(p2.knobs)).root
	assert coverage.memory_comparable(p2, as_written) == "no_eval_pass"
	# The same job with an eval that adds nothing to the peak: the training peak is comparable.
	quiet = predict(coverage.held_out(rec, p2), {**p2.knobs, "eval_no_grad": True, "eval_batch_size": 2}).root
	assert (quiet["memory.eval_spike"].point or 0) == 0
	assert coverage.memory_comparable(p2, quiet) is None
