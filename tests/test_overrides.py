"""Knob overrides: rewrite the literal at the knob's verified location, in a copy."""

from pathlib import Path

import pytest

from preflight.harness.static import map_static
from preflight.recorder.overrides import OverrideError, apply_overrides

ROOT = Path(__file__).resolve().parents[1]
NANO = (ROOT / "zoo/pretrain_nanogpt/train.py").read_text()
LORA = (ROOT / "zoo/finetune_lora_hf/train.py").read_text()


def test_rewrites_module_constant():
	sm = map_static({"train.py": NANO}, entrypoint="train.py")
	out = apply_overrides({"train.py": NANO}, sm, {"batch_size": 2, "num_workers": 4})
	assert "batch_size = 2\n" in out["train.py"]
	assert "num_workers = 4  # planted 1" in out["train.py"]
	assert NANO.count("\n") == out["train.py"].count("\n")


def test_original_files_are_untouched():
	files = {"train.py": NANO}
	sm = map_static(files, entrypoint="train.py")
	apply_overrides(files, sm, {"batch_size": 2})
	assert files["train.py"] == NANO


def test_rewrites_keyword_argument():
	sm = map_static({"train.py": LORA}, entrypoint="train.py")
	out = apply_overrides({"train.py": LORA}, sm, {"batch_size": 8, "eval_batch_size": 4})
	assert "per_device_train_batch_size=8," in out["train.py"]
	assert "per_device_eval_batch_size=4,  # planted 1" in out["train.py"]


def test_precision_speaks_the_scripts_vocabulary():
	sm = map_static({"train.py": NANO}, entrypoint="train.py")
	out = apply_overrides({"train.py": NANO}, sm, {"precision": "bf16"})
	assert 'dtype = "bfloat16"' in out["train.py"]


def test_hf_precision_flag_toggles():
	sm = map_static({"train.py": LORA}, entrypoint="train.py")
	out = apply_overrides({"train.py": LORA}, sm, {"precision": "fp32"})
	assert "bf16=False," in out["train.py"]


def test_hf_precision_to_fp16_needs_code():
	sm = map_static({"train.py": LORA}, entrypoint="train.py")
	with pytest.raises(OverrideError, match="precision"):
		apply_overrides({"train.py": LORA}, sm, {"precision": "fp16"})


def test_unknown_knob_is_an_error():
	sm = map_static({"train.py": NANO}, entrypoint="train.py")
	with pytest.raises(OverrideError, match="no_such"):
		apply_overrides({"train.py": NANO}, sm, {"no_such": 1})


def test_override_at_current_value_is_a_no_op():
	sm = map_static({"train.py": NANO}, entrypoint="train.py")
	assert apply_overrides({"train.py": NANO}, sm, {"batch_size": 12})["train.py"] == NANO


def test_non_ascii_before_the_literal_is_spliced_correctly():
	src = 'cfg = dict(note="température", batch_size=4)\nbatch_size = cfg["batch_size"]\n'
	sm = map_static({"t.py": src}, entrypoint="t.py")
	out = apply_overrides({"t.py": src}, sm, {"batch_size": 16})
	assert out["t.py"].startswith('cfg = dict(note="température", batch_size=16)')


ANNOTATED = """from dataclasses import dataclass

learning_rate: float = 3e-4
batch_size: int = 16


@dataclass
class TrainConfig:
	seq_len: int = 512
	num_workers: int = 2
	max_steps: int = 1000
"""

ARGPARSE = """import argparse

p = argparse.ArgumentParser()
p.add_argument("--batch_size", "--bs", type=int, default=16)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--max-steps", dest="max_steps", type=int, default=1000)
args = p.parse_args()
"""


def test_annotated_module_and_config_fields_are_knobs_and_overridable():
	files = {"train.py": ANNOTATED}
	sm = map_static(files, entrypoint="train.py")
	k = {x.id: x.current for x in sm.knobs}
	assert k["batch_size"] == 16 and k["lr"] == 3e-4 and k["seq_len"] == 512 and k["num_workers"] == 2
	out = apply_overrides(files, sm, {"batch_size": 4, "seq_len": 128, "lr": 1e-3})["train.py"]
	assert (
		"batch_size: int = 4\n" in out
		and "seq_len: int = 128\n" in out
		and "learning_rate: float = 0.001" in out
	)


def test_argparse_defaults_are_knobs_and_overridable():
	files = {"train.py": ARGPARSE}
	sm = map_static(files, entrypoint="train.py")
	k = {x.id: x.current for x in sm.knobs}
	assert k["batch_size"] == 16 and k["lr"] == 3e-4 and k["max_steps"] == 1000
	out = apply_overrides(files, sm, {"batch_size": 2, "max_steps": 7})["train.py"]
	assert 'p.add_argument("--batch_size", "--bs", type=int, default=2)' in out
	assert 'p.add_argument("--max-steps", dest="max_steps", type=int, default=7)' in out


def test_flags_on_the_command_line_are_overridden_there_too():
	from preflight.recorder.overrides import override_args

	files = {"train.py": ARGPARSE}
	sm = map_static(files, entrypoint="train.py")
	args = ["--bs", "32", "--lr=0.01", "--max-steps", "500", "--other", "x"]
	got = override_args(files, sm, args, {"batch_size": 4, "lr": 0.002, "max_steps": 9})
	assert got == ["--bs", "4", "--lr=0.002", "--max-steps", "9", "--other", "x"]
	assert override_args(files, sm, args, {"seq_len": 3}) == args  # not a flag: left to the file override


def test_flags_follow_the_scripts_spelling_and_argparse_abbreviations():
	from preflight.recorder.overrides import override_args, values_from_args

	src = (
		"import argparse\np = argparse.ArgumentParser()\n"
		'p.add_argument("--batch_size", type=int, default=8)\n'
		'p.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16", "float16"])\n'
		"a = p.parse_args()\n"
	)
	files = {"train.py": src}
	sm = map_static(files, entrypoint="train.py")
	args = ["--batch", "32", "--dtype", "float32"]
	sm2 = values_from_args(files, sm, args)
	k = {x.id: x.current for x in sm2.knobs}
	assert k["batch_size"] == 32 and k["precision"] == "fp32"  # the prefix counts; canonical precision
	out = override_args(files, sm2, args, {"batch_size": 16, "precision": "bf16"})
	assert out == ["--batch", "16", "--dtype", "bfloat16"]
	strict = src.replace("ArgumentParser()", "ArgumentParser(allow_abbrev=False)")
	assert override_args({"train.py": strict}, sm2, args, {"batch_size": 16})[:2] == ["--batch", "32"]
