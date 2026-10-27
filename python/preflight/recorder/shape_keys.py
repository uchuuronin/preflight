"""Model-shape keys (recorder/arch.py), importable without torch so the parent can check them."""

INT_KEYS = ("n_layers", "d_model", "n_heads", "n_kv_heads", "head_dim", "d_ff", "vocab_size", "n_experts",
	"experts_per_token", "max_positions")  # fmt: skip
BOOL_KEYS = ("gated_mlp", "tie_embeddings")
TEXT_KEYS = ("norm", "attn_impl", "model_type", "source")
ARCH_KEYS = (*INT_KEYS, *BOOL_KEYS, *TEXT_KEYS)
