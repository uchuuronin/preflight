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


def test_an_eval_off_probe_is_scored_against_the_engine_prediction_as_it_ran():
	"""B's predict_as_run drops the eval pass for a probe that ran none: its peak is below the
	job-as-written peak, and the page compares it after taking off only the context constant."""
	from preflight.engine import api
	from preflight.engine.memory import DEFAULT_CONTEXT_BYTES as C
	from preflight.schema import load_record

	rec = load_record(REAL.read_text())
	for pid in ("p1", "p2", "p3"):
		p = next(q for q in rec.probes if q.id == pid)
		held = coverage.held_out(rec, p)
		as_run = api.predict_as_run(held, p).root
		written = api.predict(held, dict(p.knobs)).root
		assert not api.probe_ran_eval(p)
		assert as_run["memory.peak"].point < written["memory.peak"].point
		m, basis = coverage.memory_target(p, as_run)
		assert basis == "peak_less_eval_and_context"
		assert m["point"] == pytest.approx(as_run["memory.peak"].point - C)


def test_a_probe_whose_rows_were_all_shorter_than_seq_len_is_labelled_not_scored():
	"""Padding is "longest" on short Alpaca rows: p1 and p2 never met a 1024-token row, so the engine
	(sized by seq_len on purpose) is held to nothing there; p3 met its 512 cap and is scored."""
	from preflight.schema import load_record

	rec = load_record(REAL.read_text())
	by_id = {p.id: p for p in rec.probes}
	assert coverage.sized_for_seq_len(by_id["p1"]) is not None
	assert coverage.sized_for_seq_len(by_id["p2"]) is not None
	assert coverage.sized_for_seq_len(by_id["p3"]) is None
	checks, _ = coverage.check_record("a-smoke", REAL.read_text())
	mem = {c["probe"]: c for c in checks if c["metric"] == "peak_memory"}
	assert mem["p1"]["verdict"] == "sized_for_seq_len" and "hit" not in mem["p1"]
	assert mem["p1"]["longest_row"] < mem["p1"]["seq_len"] == 1024
	assert "hit" in mem["p3"]
	s = coverage.summarise(checks)["peak_memory"]
	assert s["n"] == 1 and s["sized_for_seq_len"] == 2


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
