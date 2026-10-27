"""Static script mapping (no model call): framework, lens, knobs with code locations, risks."""

from pathlib import Path

import pytest

from preflight.harness.static import map_static

ROOT = Path(__file__).resolve().parents[1]
LORA = (ROOT / "zoo/finetune_lora_hf/train.py").read_text()
NANO = (ROOT / "zoo/pretrain_nanogpt/train.py").read_text()


def knob(sm, kid):
	for k in sm.knobs:
		if k.id == kid:
			return k
	raise AssertionError(f"no knob {kid}; have {[k.id for k in sm.knobs]}")


def line_of(src, needle):
	for i, line in enumerate(src.splitlines(), 1):
		if needle in line:
			return i
	raise AssertionError(needle)


# ---------------------------------------------------------------- HF Trainer fine-tune


@pytest.fixture(scope="module")
def lora():
	return map_static({"train.py": LORA}, entrypoint="train.py")


def test_lora_framework_and_lens(lora):
	assert lora.framework == "hf_trainer"
	assert lora.lens == "finetune"
	assert lora.produced_by == "adapter"


def test_lora_model_and_adapter(lora):
	assert lora.model.hf_id == "Qwen/Qwen2.5-1.5B"
	assert lora.model.lora.r == 16
	assert "q_proj" in lora.model.lora.target_modules
	assert lora.model.dtype == "bfloat16"


def test_lora_training_args_become_knobs_with_locations(lora):
	bs = knob(lora, "batch_size")
	assert bs.current == 4
	assert bs.location.line == line_of(LORA, "per_device_train_batch_size=4")
	assert knob(lora, "eval_batch_size").current == 32
	assert knob(lora, "grad_accum").current == 4
	assert knob(lora, "lr").current == pytest.approx(1e-4)


def test_lora_seq_len_resolves_module_constant(lora):
	sl = knob(lora, "seq_len")
	assert sl.current == 1024
	assert sl.location.line == line_of(LORA, "MAX_LEN = 1024")


def test_lora_precision_and_data(lora):
	assert lora.training.precision == "bf16"
	assert lora.data.batch_size == 4
	assert lora.data.eval_batch_size == 32
	assert lora.data.padding == "longest"
	assert lora.data.packing is False
	assert lora.data.dataset == "yahma/alpaca-cleaned"


def test_lora_catches_forward_without_no_grad_in_callback(lora):
	assert lora.training.eval_under_no_grad is False
	assert any("no_grad" in r for r in lora.risks)


def test_lora_loop_location(lora):
	assert lora.loop.kind == "hf_trainer"
	assert lora.loop.location.line == line_of(LORA, "trainer.train()")


# ---------------------------------------------------------------- nanoGPT-style plain loop


@pytest.fixture(scope="module")
def nano():
	return map_static({"train.py": NANO}, entrypoint="train.py")


def test_nano_framework_and_lens(nano):
	assert nano.framework == "plain_pytorch"
	assert nano.lens == "pretrain"


def test_nano_constants_become_knobs(nano):
	assert knob(nano, "batch_size").current == 12
	assert knob(nano, "batch_size").location.line == line_of(NANO, "batch_size = 12")
	assert knob(nano, "seq_len").current == 1024
	assert knob(nano, "num_workers").current == 0
	assert knob(nano, "checkpoint_every").current == 20000
	assert knob(nano, "max_steps").current == 60000
	assert knob(nano, "lr").current == pytest.approx(6e-4)
	assert knob(nano, "precision").current == "float32"


def test_nano_model_shape_from_constants(nano):
	m = nano.model
	assert m.family == "decoder"
	assert (m.n_layers, m.d_model, m.n_heads, m.vocab_size) == (12, 768, 12, 50304)
	assert nano.training.precision == "fp32"


def test_nano_loop_location_is_optimizer_step(nano):
	assert nano.loop.kind == "plain"
	assert nano.loop.step_call == "optimizer.step()"
	assert nano.loop.location.line == line_of(NANO, "optimizer.step()")


def test_nano_risks_name_the_planted_problems(nano):
	text = " ".join(nano.risks)
	assert "num_workers" in text
	assert "fp32" in text


# ---------------------------------------------------------------- other lenses


def test_trl_grpo_is_rl():
	src = """
from trl import GRPOConfig, GRPOTrainer
cfg = GRPOConfig(output_dir="o", num_generations=8, max_completion_length=512, per_device_train_batch_size=8)
trainer = GRPOTrainer(model="Qwen/Qwen2.5-0.5B-Instruct", reward_funcs=[r], args=cfg, train_dataset=ds)
trainer.train()
"""
	sm = map_static({"grpo.py": src}, entrypoint="grpo.py")
	assert sm.framework == "trl" and sm.lens == "rl"
	assert sm.rl.algorithm == "grpo"
	assert sm.rl.group_size == 8
	assert sm.rl.max_completion_len == 512
	assert knob(sm, "group_size").current == 8


def test_openai_client_loop_is_agent():
	src = """
from openai import OpenAI
client = OpenAI(base_url="https://api.meta.ai/v1")
MAX_TURNS = 40
def run(task):
	for turn in range(MAX_TURNS):
		r = client.chat.completions.create(model="muse-spark-1.3", messages=msgs, tools=TOOLS)
"""
	sm = map_static({"run.py": src}, entrypoint="run.py")
	assert sm.lens == "agent" and sm.framework == "openai_client"
	assert sm.agent.model == "muse-spark-1.3"
	assert knob(sm, "max_turns").current == 40


@pytest.mark.parametrize(
	("imports", "lens", "framework"),
	[
		("from vllm import LLM\nllm = LLM(model='m')\nllm.generate(p)", "serving", "vllm"),
		(
			"from sentence_transformers import SentenceTransformer\nm = SentenceTransformer('x')",
			"retrieval",
			"sentence_transformers",
		),
		("import torch_geometric\nfrom torch_geometric.loader import NeighborLoader", "gnn", "pyg"),
		("import timm\nm = timm.create_model('resnet50')", "vision", "timm"),
	],
)
def test_lens_from_imports(imports, lens, framework):
	sm = map_static({"s.py": imports}, entrypoint="s.py")
	assert (sm.lens, sm.framework) == (lens, framework)


def test_syntax_error_names_the_file_and_line():
	from preflight.harness.static import ScriptParseError

	with pytest.raises(ScriptParseError, match=r"bad\.py:2"):
		map_static({"bad.py": "x = 1\ndef (:\n"}, entrypoint="bad.py")
