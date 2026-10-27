"""RL-shaped loop: generate a rollout with a capped length, then a policy-gradient-like update."""

import torch
from transformers import LlamaConfig, LlamaForCausalLM

MAX_NEW = 8
cfg = LlamaConfig(
	vocab_size=128,
	hidden_size=32,
	intermediate_size=64,
	num_hidden_layers=2,
	num_attention_heads=4,
	num_key_value_heads=2,
	max_position_embeddings=64,
	eos_token_id=1,
	pad_token_id=0,
)
model = LlamaForCausalLM(cfg)
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
for _step in range(1000):
	prompts = torch.randint(2, 128, (4, 6))
	with torch.no_grad():
		out = model.generate(prompts, max_new_tokens=MAX_NEW, do_sample=True, pad_token_id=0)
	logits = model(out).logits
	loss = -logits.log_softmax(-1).mean()
	loss.backward()
	opt.step()
	opt.zero_grad()
