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


# ---------------------------------------------------------------- run hygiene

from preflight.runner.measured_rules import hygiene_findings  # noqa: E402

GOLDEN = Path(__file__).resolve().parents[1] / "schema" / "examples" / "finetune.json"


def _gpu_rec(**probe_update):
	rec = load_record(GOLDEN.read_text())
	p = rec.probes[0].model_copy(update=probe_update)
	return rec.model_copy(update={"probes": [p]})


def test_old_checkpoints_shipped_with_the_code_are_flagged():
	files = {"train.py": "print(1)\n", "out/checkpoint-500/trainer_state.json": "{}"}
	codes = [f.code for f in hygiene_findings(_gpu_rec(), files)]
	assert "hygiene.stale_checkpoint" in codes
	assert "hygiene.stale_checkpoint" not in [
		f.code for f in hygiene_findings(_gpu_rec(), {"train.py": "x=1\n"})
	]


def test_a_cache_loaded_from_a_fixed_path_is_flagged_and_a_keyed_one_isnt():
	fixed = "import os, torch\nCACHE = 'tokens.pt'\nif os.path.exists(CACHE):\n    data = torch.load(CACHE)\n"
	keyed = "import os, torch\nmodel = 'gpt2'\nif os.path.exists(f'tokens-{model}.pt'):\n    data = torch.load('x')\n"
	assert "hygiene.cache_not_keyed" in [f.code for f in hygiene_findings(_gpu_rec(), {"train.py": fixed})]
	assert "hygiene.cache_not_keyed" not in [
		f.code for f in hygiene_findings(_gpu_rec(), {"train.py": keyed})
	]


def test_truncated_generations_carry_a_wilson_interval():
	from preflight.schema import GenerationStats

	g = GenerationStats(calls=4, sequences=40, truncation_rate=0.25, max_new_tokens=512, backend="hf")
	f = next(f for f in hygiene_findings(_gpu_rec(generation=g), {}) if f.code == "generation.truncated")
	assert "10 of 40" in f.detail and "95% interval" in f.detail and f.knob_id == "max_new_tokens"


def test_identical_outputs_from_different_weights_are_flagged():
	rec = _gpu_rec(extras={"outputs_compared": 3.0, "outputs_identical": 3.0})
	assert "hygiene.identical_outputs" in [f.code for f in hygiene_findings(rec, {})]
	rec = _gpu_rec(extras={"outputs_compared": 3.0, "outputs_identical": 1.0})
	assert "hygiene.identical_outputs" not in [f.code for f in hygiene_findings(rec, {})]


def test_a_weights_only_checkpoint_is_a_resume_that_wont_match():
	rec = _gpu_rec(extras={"checkpoint_saves_seen": 4.0, "checkpoint_has_optimizer": 0.0,
		"checkpoint_has_scheduler": 0.0, "checkpoint_has_rng": 0.0, "checkpoint_has_step": 0.0})  # fmt: skip
	f = [x for x in hygiene_findings(rec, {}) if x.code == "hygiene.resume_mismatch"]
	assert len(f) == 1 and f[0].severity == "warn" and "optimizer state" in f[0].detail
	full = {
		k: 1.0
		for k in (
			"checkpoint_has_optimizer",
			"checkpoint_has_scheduler",
			"checkpoint_has_rng",
			"checkpoint_has_step",
		)
	}
	rec = _gpu_rec(extras={"checkpoint_saves_seen": 4.0, **full})
	assert not [x for x in hygiene_findings(rec, {}) if x.code == "hygiene.resume_mismatch"]
