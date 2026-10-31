"""runner/measured_rules.py on the first measured agent record (zoo/agent_eval_spark, Oct 31)."""

from pathlib import Path

from preflight.runner.measured_rules import agent_findings
from preflight.schema import load_record

REC = Path(__file__).resolve().parents[1] / "zoo" / "agent_eval_spark" / "record_oct31.json"


def test_the_agent_record_shows_no_caching_and_an_overflow_at_turn_11():
	rec = load_record(REC.read_text())
	fs = {f.code: f for f in agent_findings(rec)}
	assert "0 of 246,463 input tokens" in fs["agent.no_prompt_caching"].detail
	ov = fs["agent.context_overflow"]
	assert ov.severity == "warn" and "turn 11" in ov.title
	assert {e.metric: e.value for e in ov.evidence}["agent.context_growth_tokens_per_turn"] == 2745


def test_a_cached_sample_raises_no_caching_finding():
	rec = load_record(REC.read_text())
	tasks = [t.model_copy(update={"cached_input_tokens": 100}) for t in rec.agent_sample.tasks]
	rec = rec.model_copy(update={"agent_sample": rec.agent_sample.model_copy(update={"tasks": tasks})})
	assert "agent.no_prompt_caching" not in {f.code for f in agent_findings(rec)}


def test_no_projection_without_a_limit_or_growth():
	rec = load_record(REC.read_text())
	no_limit = rec.model_copy(update={"knobs": [k for k in rec.knobs if k.id != "context_limit"]})
	assert "agent.context_overflow" not in {f.code for f in agent_findings(no_limit)}
	flat = [
		t.model_copy(update={"context_tokens_by_turn": [500] * max(1, t.turns)})
		for t in rec.agent_sample.tasks
	]
	flat_rec = rec.model_copy(update={"agent_sample": rec.agent_sample.model_copy(update={"tasks": flat})})
	assert "agent.context_overflow" not in {f.code for f in agent_findings(flat_rec)}
