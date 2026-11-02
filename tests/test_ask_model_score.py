"""corpus/ask_model_score.py: the model's answers go through the validation scorer's own rows."""

import json
from pathlib import Path

from preflight.corpus import ask_model_score as S

FIX = Path(__file__).resolve().parent / "fixtures" / "validation"


def _answer(mem=(20, 15, 30), step=(1.0, 0.5, 2.0), p=0.9, applies=True):
	iv = lambda t: {"point": t[0], "lo": t[1], "hi": t[2]}  # noqa: E731
	return {"peak_memory_gib": iv(mem), "step_applies": applies, "step_time_s": iv(step), "oom": p >= 0.5,
		"p_oom": p, "reasoning": ""}  # fmt: skip


def test_the_model_is_held_to_the_oom_alarm_rule():
	m = {
		"runs": [
			{"script": "finetune_lora_hf", "set": "zoo", "ground_truth": "ground_truth_finetune_lora_hf.json"}
		]
	}
	rows = {r.metric: r for r in S.build(m, FIX, {"finetune_lora_hf": _answer(p=0.9)})}
	assert rows["OOM prediction"].hits == 1  # the run ran out and the model said 90%
	rows = {r.metric: r for r in S.build(m, FIX, {"finetune_lora_hf": _answer(p=0.2)})}
	assert rows["OOM prediction"].hits == 0 and "missed: finetune_lora_hf" in rows["OOM prediction"].note


def test_a_malformed_answer_scores_nothing():
	bad = _answer(mem=(20, 25, 30))  # point below lo
	assert S.well_formed(bad) and S.well_formed(_answer()) is None
	m = {
		"runs": [
			{"script": "finetune_lora_hf", "set": "zoo", "ground_truth": "ground_truth_finetune_lora_hf.json"}
		]
	}
	rows = {r.metric: r for r in S.build(m, FIX, {"finetune_lora_hf": bad})}
	assert (
		rows["OOM prediction"].hits == 0 and rows["OOM prediction"].n == 1
	)  # held like an abstain: a missed OOM
	assert rows["Peak memory error"].n is None


def test_side_by_side_keeps_the_rows_aligned():
	m = json.loads((FIX / "manifest_nov2.json").read_text())
	model = S.build(m, FIX, {"finetune_lora_hf": _answer()})
	text = S.side_by_side(model, model)
	assert text.count("\n") == 2 + len(S.ROWS)


def test_no_answer_on_a_run_that_ran_out_is_a_missed_oom():
	m = {
		"runs": [
			{"script": "finetune_lora_hf", "set": "zoo", "ground_truth": "ground_truth_finetune_lora_hf.json"}
		]
	}
	rows = {r.metric: r for r in S.build(m, FIX, {})}
	assert rows["OOM prediction"].hits == 0 and rows["OOM prediction"].n == 1
