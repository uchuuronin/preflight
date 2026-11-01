"""The validation scorer (corpus/validate.py, scripts/score_validation.py) on Oct 31's records and on made-up runs."""

import json
from pathlib import Path

import pytest

from preflight.corpus import validate as V

FIX = Path(__file__).parent / "fixtures" / "validation"


@pytest.fixture(scope="module")
def today():
	return V.build(json.loads((FIX / "manifest_oct31.json").read_text()), FIX)


def _row(rows, metric):
	(r,) = [r for r in rows if r.metric == metric]
	return r


def test_every_plan_target_has_a_row(today):
	rows, _ = today
	assert [r.metric for r in rows] == [
		"Peak memory error", "OOM prediction", "Step time, same GPU", "Step time, unmeasured GPU",
		"GNN + GBDT ensemble vs GBDT alone", "Divergence warning", "Rollout time (RL)", "Serving tokens/sec",
		"Agent run total cost", "Interval coverage", "Abstention", "Planted problems caught",
		"Unseen scripts harnessed", "Knob response", "Cost of one Preflight run",
	]  # fmt: skip


def test_the_lora_full_run_on_oct_31(today):
	rows, runs = today
	(lora,) = [r for r in runs if r.script == "finetune_lora_hf"]
	assert lora.oom_actual is True and lora.oom_p == 1.0  # the eval ran out, and the probe said so
	assert lora.memory is None  # a run that ran out never showed its full need, so memory isn't scored on it
	oom = _row(rows, "OOM prediction")
	assert oom.hit and "missed: none" in oom.note
	assert _row(rows, "Planted problems caught").hits == 3
	assert _row(rows, "Step time, same GPU").hits == 1


def test_rows_with_nothing_to_score_say_not_measured(today):
	rows, _ = today
	for m in ("Peak memory error", "Serving tokens/sec", "Knob response", "Unseen scripts harnessed"):
		assert _row(rows, m).cells()[3] == "not measured"


def test_the_gnn_row_follows_the_paired_interval(today):
	r = _row(today[0], "GNN + GBDT ensemble vs GBDT alone")
	assert r.hit is False and "GBDT is used" in r.note


def test_an_unseen_script_marked_for_tuning_is_refused(tmp_path):
	m = {"runs": [{"script": "x", "set": "unseen", "record": "r.json", "used_for_tuning": True}]}
	with pytest.raises(ValueError, match="never tuned on"):
		V.build(m, tmp_path)


def test_a_missed_oom_and_a_false_alarm_are_misses():
	runs = [
		V.Scored("a", "zoo", oom_actual=True, oom_p=0.02),
		V.Scored("b", "zoo", oom_actual=False, oom_p=0.9),
	]
	assert "5%-50%" not in V.reliability(runs) and "0%-5%: 1 of 1 ran out" in V.reliability(runs)


def test_within_share_and_wilson():
	r = V._share_row("m", "t", "o", [(str(i), i < 9) for i in range(10)])
	assert (r.hits, r.n, r.hit) == (9, 10, True) and "misses: 9" in r.note
	r = V._share_row("m", "t", "o", [(str(i), i < 8) for i in range(10)])
	assert r.hit is False
	assert r.cells()[4] == "49% to 94%"


def test_divergence_row_from_the_classifier_report():
	rep = {
		"auroc": {"point": 0.9, "lo": 0.85, "hi": 0.95},
		"lead_steps": {"point": 120, "lo": 80, "hi": 160},
		"n": 300,
	}
	r = V.divergence_row(rep)
	assert r.hit and "120 steps early" in r.note


def test_the_readme_block_is_replaced_between_markers(tmp_path):
	md = tmp_path / "README.md"
	md.write_text(f"intro\n{V.START}\nold\n{V.END}\nrest\n")
	assert V.into_readme(md, "| new |\n")
	assert md.read_text() == f"intro\n{V.START}\n| new |\n{V.END}\nrest\n"
	(tmp_path / "plain.md").write_text("no markers")
	assert not V.into_readme(tmp_path / "plain.md", "x")


def test_the_cli_writes_the_table(tmp_path):
	import runpy
	import sys

	out = tmp_path / "t.md"
	argv = sys.argv
	sys.argv = ["score_validation.py", str(FIX / "manifest_oct31.json"), "--out", str(out)]
	try:
		with pytest.raises(SystemExit) as e:
			runpy.run_path(
				str(Path(__file__).parents[1] / "scripts" / "score_validation.py"), run_name="__main__"
			)
	finally:
		sys.argv = argv
	assert e.value.code == 0 and out.read_text().startswith("| Metric |")


def test_a_classifier_verdict_shows_its_runs_not_a_made_up_proportion():
	# review, Nov 1: it once printed "None of 40" with a Wilson interval
	rep = {
		"auroc": {"point": 0.7, "lo": 0.6, "hi": 0.8},
		"lead_steps": {"point": 50, "lo": 20, "hi": 90},
		"n": 40,
	}
	c = V.divergence_row(rep).cells()
	assert c[3] == "40 runs" and c[4] == "" and c[5] == "miss"
