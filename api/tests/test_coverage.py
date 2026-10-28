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


def test_a_real_run_is_scored_only_on_probes_held_out():
	checks, skipped = coverage.check_record("a-smoke", REAL.read_text())
	steps = [c for c in checks if c["metric"] == "step_time"]
	assert {c["probe"] for c in steps} == {"p1", "p2", "p3"}
	# The held-out probe's own number never reaches the prediction: p2 ran at half the batch and
	# took half the time; the engine, without p2, could not have read 0.837 s off the record.
	p2 = next(c for c in steps if c["probe"] == "p2")
	assert abs(p2["measured"] - 0.837) < 0.001
	assert p2["predicted"]["point"] != pytest.approx(p2["measured"], rel=0.01)
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
	assert skipped == [{"run": "broken", "why": "no probe measured anything (recorder)"}]
