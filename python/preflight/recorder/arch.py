"""Model shape read from the running model, for B's memory and FLOP models (docs/requests.md, B -> A).

The static map reads what the code says (a model id, a config call); this reads
what was built, which also covers models defined in the script itself. Two
sources, best first:

  hf_config  a root module carries a transformers PretrainedConfig: its fields
             are authoritative (layers, width, heads, KV heads, head_dim, vocab,
             d_ff, experts, tied embeddings, attention implementation).
  modules    anything else: the repeated block stack, the largest embedding,
             the MLP projections and the norm classes inside the blocks.

Fields nobody can tell are left out rather than guessed; B treats a missing
field as unknown. attn_impl from modules is only "sdpa" when the probe saw
scaled_dot_product_attention run, never inferred from module names.
"""

from collections import Counter

from torch import nn

from .shape_keys import ARCH_KEYS  # torch-free, shared with the parent's checks

# config field names across transformers model families, first match wins
_CFG = {
	"n_layers": ("num_hidden_layers", "n_layer", "num_layers", "n_layers"),
	"d_model": ("hidden_size", "n_embd", "d_model", "dim"),
	"n_heads": ("num_attention_heads", "n_head", "num_heads", "n_heads"),
	"n_kv_heads": ("num_key_value_heads", "num_kv_heads", "n_kv_heads", "multi_query_group_num"),
	"head_dim": ("head_dim", "kv_channels", "d_kv"),
	"d_ff": ("intermediate_size", "n_inner", "ffn_dim", "d_ff", "ffn_hidden_size"),
	"vocab_size": ("vocab_size", "padded_vocab_size"),
	"n_experts": ("num_local_experts", "num_experts", "n_routed_experts", "moe_num_experts"),
	# not "top_k": transformers 4.x puts the sampling default top_k=50 on every config
	"experts_per_token": ("num_experts_per_tok", "moe_topk", "num_selected_experts"),
	"max_positions": ("max_position_embeddings", "n_positions", "max_seq_len", "seq_length"),
}
# activations that come with a gate projection (SwiGLU, GeGLU) in HF configs
_GATED_ACT = ("silu", "swiglu", "swish", "geglu", "gelu_pytorch_tanh_gated")
_GATED_TYPES = ("llama", "mistral", "mixtral", "qwen", "gemma", "phi3", "olmo", "deepseek", "cohere", "granite", "internlm", "yi", "starcoder2_gated")  # fmt: skip
_MLP_NAMES = ("mlp", "ffn", "feed_forward", "feedforward", "fc", "experts")
# a gate projection by name; "gate" alone is a MoE router in Mixtral, so it doesn't count
_GATE_NAMES = ("gate_proj", "gate_up_proj", "wi_0", "w_gate")


def _first(cfg, names):
	for n in names:
		v = getattr(cfg, n, None)
		if isinstance(v, bool):
			continue
		if isinstance(v, int) and v > 0:
			return v
	return None


def _norm_kind(root: nn.Module) -> str | None:
	kinds = Counter()
	for m in root.modules():
		name = type(m).__name__.lower()
		# RMS norms under other names (T5LayerNorm) keep variance_epsilon and have no bias at all;
		# a LayerNorm built without bias still has the attribute, set to None
		if "rmsnorm" in name or (hasattr(m, "variance_epsilon") and not hasattr(m, "bias")):
			kinds["rmsnorm"] += 1
		elif isinstance(m, nn.LayerNorm) or "layernorm" in name:
			kinds["layernorm"] += 1
	return kinds.most_common(1)[0][0] if kinds else None


def _tied(root: nn.Module) -> bool | None:
	emb = [m for m in root.modules() if isinstance(m, nn.Embedding)]
	lin = [m for m in root.modules() if isinstance(m, nn.Linear) and m.weight is not None]
	if not emb or not lin:
		return None
	e = max(emb, key=lambda m: m.weight.numel())
	if e.weight.device.type == "meta" or e.weight.numel() == 0:
		# meta tensors and DeepSpeed ZeRO-3 partitioned params have no storage here: every pointer is 0
		return None
	if any(m.weight is e.weight for m in lin):
		return True
	try:
		ptr = e.weight.data_ptr()
		if ptr == 0:
			return None
		return any(m.weight.data_ptr() == ptr and m.weight.shape == e.weight.shape for m in lin)
	except RuntimeError:  # tensor subclasses without storage (some quantised weights)
		return None


# ModuleLists inside these are a layer's experts or sub-parts, never the layer stack
_INSIDE_A_LAYER = frozenset({*_MLP_NAMES, "moe", "block_sparse_moe", "sparse_moe", "local_experts", "router"})


def _blocks(root: nn.Module, n_layers: int | None = None) -> nn.ModuleList | None:
	"""The layer stack: a ModuleList whose children share one class, not inside an MLP or MoE.

	With the layer count known (from a config) the list of that length wins;
	otherwise the longest. More experts than layers would otherwise pass for the stack.
	"""
	cands = []
	for name, m in root.named_modules():
		if not (isinstance(m, nn.ModuleList) and len(m) >= 2 and len({type(c) for c in m}) == 1):
			continue
		if {p.lower() for p in name.split(".")} & _INSIDE_A_LAYER:
			continue
		cands.append(m)
	if not cands:
		return None
	if n_layers:
		exact = [m for m in cands if len(m) == n_layers]
		if exact:
			return exact[0]
	return max(cands, key=len)


def _gated(block: nn.Module) -> bool | None:
	leaves = {n.rsplit(".", 1)[-1].lower() for n, _ in block.named_modules() if n}
	if leaves & set(_GATE_NAMES) or {"w1", "w3"} <= leaves:  # w1/w3: Llama-reference and Mixtral experts
		return True
	# a dense MLP's own projections; an experts list would count every expert's
	mlp = [
		m
		for n, m in block.named_modules()
		if n.rsplit(".", 1)[-1].lower() in _MLP_NAMES and "experts" not in n.lower()
	]
	if not mlp:
		return None
	lins = [m for m in mlp[0].modules() if isinstance(m, nn.Linear)]
	return len(lins) >= 3 if lins else None


def _d_ff(block: nn.Module, d_model: int | None) -> int | None:
	if not d_model:
		return None
	outs = [
		m.out_features
		for n, m in block.named_modules()
		if isinstance(m, nn.Linear)
		and m.in_features == d_model
		and any(p in n.lower() for p in (*_MLP_NAMES, "up", "w1", "w3", "gate", "intermediate"))
	]
	return max(outs) if outs else None


def _heads(block: nn.Module) -> tuple[int | None, int | None]:
	heads = kv = None
	for m in block.modules():
		for attr in ("num_heads", "n_head", "num_attention_heads", "n_heads"):
			v = getattr(m, attr, None)
			if isinstance(v, int) and v > 0:
				heads = heads or v
		for attr in ("num_key_value_heads", "num_kv_heads", "n_kv_heads"):
			v = getattr(m, attr, None)
			if isinstance(v, int) and v > 0:
				kv = kv or v
		if isinstance(m, nn.MultiheadAttention):
			heads = heads or m.num_heads
	return heads, kv


def _from_config(cfg) -> dict:
	out = {k: _first(cfg, names) for k, names in _CFG.items()}
	if out["n_experts"] is None:
		out["experts_per_token"] = None
	if out["head_dim"] is None and out["d_model"] and out["n_heads"]:
		out["head_dim"] = out["d_model"] // out["n_heads"]
	if out["n_kv_heads"] is None and out["n_heads"]:
		out["n_kv_heads"] = 1 if getattr(cfg, "multi_query", False) else out["n_heads"]
	act = str(getattr(cfg, "hidden_act", None) or getattr(cfg, "activation_function", None) or "").lower()
	mtype = str(getattr(cfg, "model_type", "") or "")
	if act:
		out["gated_mlp"] = act in _GATED_ACT or mtype.startswith(_GATED_TYPES)
	tie = getattr(cfg, "tie_word_embeddings", None)
	if isinstance(tie, bool):
		out["tie_embeddings"] = tie
	impl = getattr(cfg, "_attn_implementation", None)
	if isinstance(impl, str) and impl:
		out["attn_impl"] = impl
	out["model_type"] = mtype or None
	return out


def model_shape(roots: list[nn.Module], *, sdpa_seen: bool = False) -> dict:
	"""Shape of the model the optimizer trains (the largest root); {} when there is none."""
	if not roots:
		return {}
	root = max(roots, key=lambda r: sum(p.numel() for p in r.parameters()))
	cfg = getattr(root, "config", None)
	out: dict = {}
	if cfg is not None and hasattr(cfg, "to_dict"):
		out = _from_config(cfg)
		out["source"] = "hf_config"
	else:
		out["source"] = "modules"
	blocks = _blocks(root, out.get("n_layers"))
	emb = [m for m in root.modules() if isinstance(m, nn.Embedding)]
	big = max(emb, key=lambda m: m.weight.numel()) if emb else None
	if out.get("n_layers") is None and blocks is not None:
		out["n_layers"] = len(blocks)
	if out.get("vocab_size") is None and big is not None and big.num_embeddings >= 256:
		out["vocab_size"] = big.num_embeddings
	if out.get("d_model") is None and big is not None:
		out["d_model"] = big.embedding_dim
	if blocks is not None:
		b0 = blocks[0]
		heads, kv = _heads(b0)
		if out.get("n_heads") is None:
			out["n_heads"] = heads
		if out.get("n_kv_heads") is None:
			out["n_kv_heads"] = kv or heads
		if out.get("d_ff") is None:
			out["d_ff"] = _d_ff(b0, out.get("d_model"))
		if out.get("gated_mlp") is None:
			out["gated_mlp"] = _gated(b0)
	if out.get("head_dim") is None and out.get("d_model") and out.get("n_heads"):
		out["head_dim"] = out["d_model"] // out["n_heads"]
	tied = _tied(root)  # what was built beats what the config says (T5 ties regardless)
	if tied is not None:
		out["tie_embeddings"] = tied
	out["norm"] = _norm_kind(blocks if blocks is not None else root)
	if out.get("attn_impl") is None and sdpa_seen:
		out["attn_impl"] = "sdpa"
	return {k: out[k] for k in ARCH_KEYS if out.get(k) is not None}
