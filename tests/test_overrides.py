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
