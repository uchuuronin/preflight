"""harness.lens_by_entry (off; the PM, Nov 3): the unseen runs' lens misreads the import rules leave,
corrected by the entry point's own name, and lm_eval's own entry point read as an eval."""

import pytest

import preflight.harness.static as S

TRAIN = """import torch
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("gpt2")
model = torch.nn.Linear(4, 4)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
for step in range(10):
    loss = model(torch.randn(2, 4)).sum()
    loss.backward()
    opt.step()
"""


@pytest.fixture
def on(monkeypatch):
	import preflight.runner.flags as F

	real = F.flag
	monkeypatch.setattr(F, "flag", lambda name: name == "harness.lens_by_entry" or real(name))


def _lens(files, entry):
	return S.map_static(files, entrypoint=entry).lens


def test_off_the_lens_is_as_before():
	assert _lens(
		{"litgpt/pretrain.py": TRAIN.replace('"gpt2"', '"EleutherAI/pythia-14m"')}, "litgpt/pretrain.py"
	) in (
		"finetune",
		"pretrain",
	)  # whatever the import rules say; the flag changes nothing off


def test_a_pretrain_entry_point_is_pretraining_and_a_lora_one_is_fine_tuning(on):
	assert _lens({"litgpt/pretrain.py": TRAIN}, "litgpt/pretrain.py") == "pretrain"
	plain = TRAIN.replace('tok = AutoTokenizer.from_pretrained("gpt2")\n', "")
	assert (
		_lens({"recipes/lora_finetune_single_device.py": plain}, "recipes/lora_finetune_single_device.py")
		== "finetune"
	)
	assert (
		_lens({"train.py": TRAIN}, "train.py")
		== S.map_static({"train.py": TRAIN}, entrypoint="train.py").lens
	)


def test_lm_eval_s_own_entry_point_is_an_eval(on):
	files = {"lm_eval/__main__.py": "from lm_eval.utils import go\ngo()\n", "lm_eval/__init__.py": "",
		"lm_eval/utils.py": "import json\ndef go():\n    print(json.dumps({}))\n"}  # fmt: skip
	assert _lens(files, "lm_eval/__main__.py") == "eval"
	# a training script that also imports lm_eval keeps its training lens: it has an optimizer
	assert _lens({"train.py": "import lm_eval\n" + TRAIN}, "train.py") != "eval"
