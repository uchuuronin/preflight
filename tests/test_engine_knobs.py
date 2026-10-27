import re
from pathlib import Path

import pytest

from preflight.engine import knobs as K
from preflight.engine import memory as m
from preflight.schema import load_record

ROOT = Path(__file__).resolve().parents[1]


def _harness_knob_ids() -> set[str]:
	"""Knob ids A's harness can declare (INT/FLOAT domains, categories, code knobs)."""
	skel = (ROOT / "python/preflight/harness/skeleton.py").read_text()
	static = (ROOT / "python/preflight/harness/static.py").read_text()
	ids = set(re.findall(r'^\t"([a-z][a-z0-9_]*)": [\(\[]', skel, re.M))  # domains and categories
	ids |= set(re.findall(r'_code_knob\(\s*"([a-z_]+)"', skel))
	ids |= set(re.findall(r'id="([a-z_]+)"', skel))
	ids |= (
		set(re.findall(r'"([a-z][a-z0-9_]*)",', skel.split("GROUPS", 1)[1].split("}", 1)[0]))
		if "GROUPS" in skel
		else set()
	)
	ids |= set(re.findall(r'Candidate\(\s*"([a-z][a-z0-9_]*)"', static))
	labels = static.split("LABELS = {", 1)[1].split("\n}", 1)[0]
	ids |= set(re.findall(r'^\t"([a-z][a-z0-9_]*)":', labels, re.M))
	groups = {
		"memory",
		"speed",
		"data",
		"model",
		"rl",
		"agent",
		"cluster",
		"generation",
		"eval",
		"retrieval",
		"gnn",
		"vision",
	}
	return {i for i in ids if not i.startswith("fanout") and i not in groups}


def test_every_harness_knob_is_in_the_map():
	missing = sorted(k for k in _harness_knob_ids() if K.knob_spec(k).maps_to == "unknown")
	assert not missing, f"knobs A declares that B's map doesn't know: {missing}"


@pytest.mark.parametrize("name", ["finetune", "pretrain", "agent"])
def test_golden_record_knobs_are_known(name):
	rec = load_record((ROOT / f"schema/examples/{name}.json").read_text())
	assert all(K.knob_spec(k.id).maps_to != "unknown" for k in rec.knobs)


def test_fanout_knobs_share_one_spec():
	assert K.knob_spec("fanout.layer2").maps_to == "gnn.fanout"


def test_unknown_knob_is_an_honest_no_op():
	s = K.knob_spec("mystery_flag")
	assert s.maps_to == "unknown" and not s.modelled and s.note


def test_modelled_is_a_subset_of_affects():
	for s in K.KNOBS.values():
		assert s.modelled <= s.affects, s.id


@pytest.mark.parametrize(
	"prec, loaded, expect",
	[
		("bf16", "fp32", ("fp32", "bf16", False, False)),
		("bf16", "bfloat16", ("bf16", "bf16", False, False)),
		("fp16", "fp32", ("fp32", "fp16", False, True)),
		("fp32", None, ("fp32", "fp32", False, False)),
		("mixed", "bfloat16", ("fp32", "bf16", False, False)),
		("fp8", "bf16", ("bf16", "bf16", True, False)),
	],
)
def test_precision_setup(prec, loaded, expect):
	p = K.precision_setup(prec, loaded)
	assert (p.param_dtype, p.compute_dtype, p.master_weights, p.grad_scaler) == expect


@pytest.mark.parametrize(
	"name, gated, norm, tied",
	[
		("meta-llama/Llama-3.1-8B", True, "rmsnorm", False),
		("Qwen/Qwen2.5-1.5B", True, "rmsnorm", None),
		("gpt2", False, "layernorm", True),
		("EleutherAI/pythia-1b", False, "layernorm", False),
		("google/gemma-2-2b", True, "rmsnorm", True),
		("BAAI/bge-small-en-v1.5", False, "layernorm", True),
	],
)
def test_arch_hints(name, gated, norm, tied):
	h = K.arch_hint(name)
	assert (h.gated_mlp, h.norm, h.tie_embeddings) == (gated, norm, tied)


def test_mixtral_hint_is_moe():
	h = K.arch_hint("mistralai/Mixtral-8x7B-v0.1")
	assert (h.n_experts, h.experts_per_token) == (8, 2)


def test_golden_finetune_becomes_engine_inputs():
	rec = load_record((ROOT / "schema/examples/finetune.json").read_text())
	vals = {k.id: k.current for k in rec.knobs}
	arch, setup, shape = K.setup_from(rec.job.script_map, vals)
	assert isinstance(arch, m.TransformerArch)
	assert arch.d_ff == 8960  # solved from the parameter count: Qwen2.5-1.5B's real width
	assert arch.param_count() == pytest.approx(rec.job.script_map.model.param_count, rel=0.001)
	assert setup.method == "lora" and setup.lora_rank == 16
	assert set(setup.lora_targets) == {"q", "k", "v", "o", "gate", "up", "down"}
	assert (shape.batch, shape.seq) == (vals["batch_size"], vals["seq_len"])
	assert setup.eval_batch == vals["eval_batch_size"]


def test_knob_values_override_the_script_map():
	rec = load_record((ROOT / "schema/examples/finetune.json").read_text())
	vals = {k.id: k.current for k in rec.knobs} | {
		"precision": "fp32",
		"gradient_checkpointing": True,
		"gpus": 4,
	}
	_, setup, _ = K.setup_from(rec.job.script_map, vals)
	assert setup.compute_dtype == "fp32"
	assert setup.grad_checkpointing == "full"
	assert setup.parallel.world_size == 4


def test_generic_arch_when_the_shape_is_incomplete():
	class Info:
		family, param_count, n_layers, vocab_size = "cnn", 25_000_000, None, None

	assert isinstance(K.arch_from_model_info(Info()), m.GenericArch)


@pytest.mark.parametrize(
	"lens, values, units",
	[
		("finetune", {"batch_size": 4, "seq_len": 1024}, 4096),
		("rl", {"batch_size": 8, "max_prompt_len": 512, "max_completion_len": 512}, 8192),
		("vision", {"batch_size": 32, "image_size": 224}, 32 * 224 * 224),
		("gnn", {"batch_size": 1024, "fanout.l1": 10, "fanout.l2": 5}, 1024 * 50),
		("agent", {}, 1),
	],
)
def test_work_units_per_lens(lens, values, units):
	assert K.work_units(lens, values) == units


class _Info:
	def __init__(self, **kw):
		self.family, self.d_ff, self.n_kv_heads, self.hf_id = "decoder", None, None, None
		self.__dict__.update(kw)


def test_deepseek_r1_distill_is_dense():
	info = _Info(name="DeepSeek-R1-Distill-Qwen-1.5B", hf_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", param_count=1_777_088_000,
		n_layers=28, d_model=1536, n_heads=12, n_kv_heads=2, vocab_size=151936)  # fmt: skip
	arch = K.arch_from_model_info(info)
	assert arch.n_experts == 1
	assert arch.param_count() == pytest.approx(1.777e9, rel=0.01)


def test_moe_hint_dropped_when_the_size_says_dense():
	info = _Info(
		name="deepseek-v3-tiny-dense-test",
		param_count=1_000_000_000,
		n_layers=24,
		d_model=2048,
		n_heads=16,
		vocab_size=32000,
	)
	assert K.arch_from_model_info(info).n_experts == 1


def test_llama_3_2_small_models_are_tied():
	assert K.arch_hint("meta-llama/Llama-3.2-1B").tie_embeddings is True
	assert K.arch_hint("meta-llama/Llama-3.1-8B").tie_embeddings is False


def test_unknown_precision_uses_the_load_dtype():
	p = K.precision_setup("unknown", "bfloat16")
	assert (p.param_dtype, p.compute_dtype) == ("bf16", "bf16")
	assert K.precision_setup(None, None).param_dtype == "fp32"


def test_frozen_quantised_model_counts_its_quantisation():
	llama8b = m.TransformerArch(
		n_layers=32,
		d_model=4096,
		n_heads=32,
		n_kv_heads=8,
		d_ff=14336,
		vocab_size=128256,
		gated_mlp=True,
		tie_embeddings=False,
	)
	st = m.static_terms(llama8b, m.TrainingSetup(method="frozen", quantization="4bit-nf4"))
	assert 4.5e9 < st["weights"] < 7e9  # not the 16 GB of bf16 weights


@pytest.mark.parametrize(
	"dist, fsdp_mode, ds, expect",
	[
		("deepspeed", None, '{"zero_optimization": {"stage": 3}}', "zero3"),
		("deepspeed", None, "ds_zero1.json", "zero1"),
		("deepspeed", None, None, "zero2"),
		("fsdp", "hybrid_shard", None, "hsdp"),
		("fsdp", "shard_grad_op", None, "fsdp_grad_op"),
		("ddp", None, None, "ddp"),
	],
)
def test_distributed_strategy(dist, fsdp_mode, ds, expect):
	assert K._strategy(dist, fsdp_mode, ds) == expect


def test_hsdp_shards_within_its_group():
	par = m.Parallel("hsdp", world_size=32, shard_group=8)
	assert par.shards() == (8, 8, 8)


# ---------------------------------------------------------------- A's measured shape and new knobs (Oct 27)


def _info(**kw):
	from preflight.schema.script_map import ModelInfo

	base = dict(name="x", family="decoder", n_layers=4, d_model=512, n_heads=8, vocab_size=32000, d_ff=1376)
	return ModelInfo(**{**base, **kw})


def test_group_by_length_is_declared_but_not_claimed_as_modelled():
	spec = K.knob_spec("group_by_length")
	assert spec.affects == {"time", "cost"} and not spec.modelled
	assert not K.knob_spec("packing").modelled


def test_measured_shape_beats_the_name_hint():
	a = K.arch_from_model_info(
		_info(name="meta-llama/Llama-3-8B", gated_mlp=False, tie_embeddings=True, norm="layernorm")
	)
	assert (a.gated_mlp, a.tie_embeddings, a.norm) == (False, True, "layernorm")


def test_measured_dense_beats_a_moe_looking_name():
	a = K.arch_from_model_info(_info(name="deepseek-v3-tiny", n_experts=0))
	assert a.n_experts == 1 and a.experts_per_token == 1


def test_measured_moe_is_used_as_given():
	a = K.arch_from_model_info(_info(name="my-model", n_experts=16, experts_per_token=2))
	assert (a.n_experts, a.experts_per_token) == (16, 2)


def test_head_dim_override_only_when_it_differs():
	assert K.arch_from_model_info(_info(head_dim=64)).head_dim_override is None
	assert K.arch_from_model_info(_info(head_dim=128)).head_dim_override == 128


def test_measured_head_dim_feeds_the_d_ff_solve():
	from preflight.schema.script_map import ModelInfo

	gemma = ModelInfo(
		name="google/gemma-7b", family="decoder", n_layers=28, d_model=3072, n_heads=16, n_kv_heads=16,
		vocab_size=256000, head_dim=256, param_count=8_537_680_896, gated_mlp=True, tie_embeddings=True,
	)  # fmt: skip
	a = K.arch_from_model_info(gemma)
	assert a.d_ff == 24576 and a.param_count() == 8_537_680_896


def test_moe_routing_falls_back_to_the_hint_and_is_capped():
	mix = _info(name="mistralai/Mixtral-8x7B-v0.1", n_experts=8)
	assert K.arch_from_model_info(mix).experts_per_token == 2
	assert K.arch_from_model_info(_info(n_experts=8, experts_per_token=20)).experts_per_token == 8
