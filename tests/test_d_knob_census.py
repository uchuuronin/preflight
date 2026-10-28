"""D1 baseline census: which settings popular training examples set, per lens."""

from __future__ import annotations

import json
from pathlib import Path

from preflight.addons.knob_census import (
	Unit,
	assigned_knobs,
	commands_in,
	flags_in,
	knob_for,
	literal_flags,
	rank,
	trl_lens,
)

ROOT = Path(__file__).resolve().parents[1]
CENSUS = ROOT / "web/src/addons/expertise/census.json"


def test_commands_join_continuations_and_skip_non_runs():
	text = """
Some prose.
```bash
python trl/scripts/sft.py \\
    --model_name_or_path Qwen/Qwen2-0.5B \\
    --per_device_train_batch_size 2 \\
    --packing
```
```bash
pip install trl
```
"""
	cmds = commands_in(text)
	assert len(cmds) == 1
	assert "--packing" in cmds[0] and "sft.py" in cmds[0]


def test_commands_found_in_accelerate_and_torchrun_launches():
	text = "```sh\naccelerate launch run_clm.py --block_size 512\ntorchrun --nproc_per_node 8 run_mlm.py --fp16\n```"
	assert len(commands_in(text)) == 2


def test_flags_with_values_equals_and_booleans():
	assert flags_in("python x.py --learning_rate=2e-5 --bf16 --save_steps 500 -v") == [
		"learning_rate",
		"bf16",
		"save_steps",
	]


def test_knob_for_uses_the_harness_vocabulary_plus_cli_precision():
	assert knob_for("per_device_train_batch_size") == "batch_size"
	assert knob_for("gradient_accumulation_steps") == "grad_accum"
	assert knob_for("block_size") == "seq_len"
	assert knob_for("bf16") == "precision"
	assert knob_for("fp16") == "precision"
	assert knob_for("output_dir") is None
	assert knob_for("learning-rate") == "lr"
	assert knob_for("num-generations") == "group_size"


def test_flags_an_example_declares_in_its_code_count():
	code = 'parser.add_argument("--num-generations", type=int)\nargs = ["--learning-rate", "1e-5"]\nx = "--not a flag"\n'
	assert literal_flags(code) == ["num-generations", "learning-rate"]


def test_override_configs_count_only_assigned_names():
	assert assigned_knobs("batch_size = 64\nblock_size = 256 # ctx\ndropout = 0.2\n  lr = 1\n") == {
		"batch_size",
		"seq_len",
	}


def test_trl_files_sorted_into_lenses_and_other_objectives_skipped():
	assert trl_lens("grpo_wordle.py") == "rl"
	assert trl_lens("rloo_math.py") == "rl"
	assert trl_lens("sft_gemma3.py") == "finetune"
	assert trl_lens("dpo_reduce_hallucinations.py") is None


def test_rank_counts_units_not_commands_and_breaks_ties_by_id():
	units = [
		Unit("a", "finetune", frozenset({"lr", "batch_size"})),
		Unit("b", "finetune", frozenset({"lr", "seq_len"})),
		Unit("c", "finetune", frozenset({"seq_len"})),
		Unit("d", "rl", frozenset({"group_size"})),
	]
	out = rank(units)
	assert out["finetune"]["units"] == 3
	assert out["finetune"]["ranking"] == ["lr", "seq_len", "batch_size"]
	assert out["finetune"]["counts"] == {"lr": 2, "seq_len": 2, "batch_size": 1}
	assert out["rl"]["ranking"] == ["group_size"]


def test_census_ranks_only_knobs_the_engine_knows():
	from preflight.engine.knobs import KNOBS

	known = set(KNOBS)
	census = json.loads(CENSUS.read_text())
	for lens, row in census["lenses"].items():
		assert set(row["ranking"]) <= known, (lens, set(row["ranking"]) - known)


def test_committed_census_is_consistent():
	census = json.loads(CENSUS.read_text())
	assert census["sources"], "every census names its sources"
	for src in census["sources"]:
		assert src["commit"] and src["url"].startswith("https://")
	for lens, row in census["lenses"].items():
		counts = row["counts"]
		assert all(0 < c <= row["units"] for c in counts.values()), lens
		expect = sorted(counts, key=lambda k: (-counts[k], k))
		assert row["ranking"] == expect, lens
		assert len(row["units_list"]) == row["units"], lens
