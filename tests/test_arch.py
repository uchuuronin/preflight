"""Model shape from the running model (B's ModelInfo fields): HF configs first, module inspection otherwise."""

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from preflight.recorder.arch import model_shape  # noqa: E402

pytestmark = pytest.mark.torch


def hf():
	return pytest.importorskip("transformers")


def test_llama_config_is_read_directly():
	t = hf()
	cfg = t.LlamaConfig(vocab_size=320, hidden_size=64, intermediate_size=176, num_hidden_layers=3,
		num_attention_heads=8, num_key_value_heads=2, max_position_embeddings=128, tie_word_embeddings=False)  # fmt: skip
	s = model_shape([t.LlamaForCausalLM(cfg)])
	assert s["source"] == "hf_config" and s["model_type"] == "llama"
	assert (s["n_layers"], s["d_model"], s["n_heads"], s["n_kv_heads"], s["head_dim"]) == (3, 64, 8, 2, 8)
	assert (s["d_ff"], s["vocab_size"], s["max_positions"]) == (176, 320, 128)
	assert s["gated_mlp"] is True and s["tie_embeddings"] is False and s["norm"] == "rmsnorm"
	assert "attn_impl" in s


def test_gpt2_config_layernorm_tied_and_ungated():
	t = hf()
	cfg = t.GPT2Config(vocab_size=300, n_embd=48, n_layer=2, n_head=4, n_positions=64)
	s = model_shape([t.GPT2LMHeadModel(cfg)])
	assert (s["n_layers"], s["d_model"], s["n_heads"], s["n_kv_heads"], s["head_dim"]) == (2, 48, 4, 4, 12)
	assert s["gated_mlp"] is False and s["tie_embeddings"] is True and s["norm"] == "layernorm"


def test_mixtral_experts_are_counted_and_the_router_is_not_a_gate():
	t = hf()
	if not hasattr(t, "MixtralConfig"):
		pytest.skip("transformers without Mixtral")
	cfg = t.MixtralConfig(vocab_size=300, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
		num_attention_heads=4, num_key_value_heads=2, num_local_experts=4, num_experts_per_tok=2)  # fmt: skip
	s = model_shape([t.MixtralForCausalLM(cfg)])
	assert (s["n_experts"], s["experts_per_token"], s["d_ff"]) == (4, 2, 64)
	assert s["gated_mlp"] is True and s["norm"] == "rmsnorm" and s["n_layers"] == 2


class Block(nn.Module):
	def __init__(self, d, heads, gated):
		super().__init__()
		self.ln_1, self.ln_2 = nn.LayerNorm(d), nn.LayerNorm(d)
		self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
		if gated:
			self.mlp = nn.ModuleDict({"gate_proj": nn.Linear(d, 3 * d), "up_proj": nn.Linear(d, 3 * d),
				"down_proj": nn.Linear(3 * d, d)})  # fmt: skip
		else:
			self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))


class TinyGPT(nn.Module):
	def __init__(self, vocab=512, d=32, layers=4, heads=4, gated=False, tie=True):
		super().__init__()
		self.wte = nn.Embedding(vocab, d)
		self.h = nn.ModuleList(Block(d, heads, gated) for _ in range(layers))
		self.ln_f = nn.LayerNorm(d)
		self.lm_head = nn.Linear(d, vocab, bias=False)
		if tie:
			self.lm_head.weight = self.wte.weight


def test_script_defined_model_is_read_from_its_modules():
	s = model_shape([TinyGPT()])
	assert s["source"] == "modules"
	assert (s["n_layers"], s["d_model"], s["n_heads"], s["head_dim"], s["vocab_size"]) == (4, 32, 4, 8, 512)
	assert (s["d_ff"], s["gated_mlp"], s["tie_embeddings"], s["norm"]) == (128, False, True, "layernorm")
	assert "attn_impl" not in s  # never guessed from module names


def test_gated_untied_and_sdpa_seen():
	s = model_shape([TinyGPT(gated=True, tie=False)], sdpa_seen=True)
	assert s["gated_mlp"] is True and s["tie_embeddings"] is False and s["d_ff"] == 96
	assert s["attn_impl"] == "sdpa"


def test_largest_root_wins_and_nothing_is_invented_for_an_mlp():
	s = model_shape([nn.Linear(4, 4), nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 2))])
	assert s == {"source": "modules"}
	assert model_shape([]) == {}


def test_t5_rms_norm_and_head_dim_and_a_tie_read_from_the_weights():
	t = hf()
	cfg = t.T5Config(vocab_size=300, d_model=64, d_ff=128, num_layers=2, num_heads=4, d_kv=8,
		feed_forward_proj="gated-gelu")  # fmt: skip
	m = t.T5ForConditionalGeneration(cfg)
	s = model_shape([m])
	assert s["norm"] == "rmsnorm" and s["head_dim"] == 8 and s["gated_mlp"] is True
	assert s["tie_embeddings"] is (m.lm_head.weight is m.shared.weight)


def test_meta_device_models_do_not_claim_a_tie():
	with torch.device("meta"):
		m = TinyGPT(tie=False)
	assert "tie_embeddings" not in model_shape([m])


class MoE(nn.Module):
	def __init__(self, d, n_experts):
		super().__init__()
		self.router = nn.Linear(d, n_experts)
		self.experts = nn.ModuleList(
			nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)) for _ in range(n_experts)
		)


class MoEBlock(nn.Module):
	def __init__(self, d, heads, n_experts):
		super().__init__()
		self.norm = nn.LayerNorm(d)
		self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
		self.moe = MoE(d, n_experts)


def test_more_experts_than_layers_still_finds_the_layer_stack():
	m = nn.Module()
	m.emb = nn.Embedding(512, 32)
	m.layers = nn.ModuleList(MoEBlock(32, 4, 8) for _ in range(4))
	s = model_shape([m])
	assert s["n_layers"] == 4 and s["n_heads"] == 4 and s["norm"] == "layernorm"
	assert s.get("gated_mlp") is not True  # eight two-layer experts are not a gate


def test_sampling_top_k_is_not_an_expert_count():
	t = hf()
	cfg = t.LlamaConfig(vocab_size=320, hidden_size=64, intermediate_size=176, num_hidden_layers=2,
		num_attention_heads=8, top_k=50)  # fmt: skip
	s = model_shape([t.LlamaForCausalLM(cfg)])
	assert "experts_per_token" not in s and "n_experts" not in s


def test_zero3_partitioned_params_do_not_claim_a_tie():
	m = TinyGPT(tie=False)
	for p in m.parameters():
		p.data = torch.empty(0)  # what DeepSpeed ZeRO-3 leaves outside a gathered forward
	assert "tie_embeddings" not in model_shape([m])
