"""Generation whose cap comes in a GenerationConfig, as TRL's GRPOTrainer passes it."""

import torch
from transformers import GenerationConfig, LlamaConfig, LlamaForCausalLM

torch.manual_seed(0)
model = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
	num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64, eos_token_id=63))  # fmt: skip
opt = torch.optim.SGD(model.parameters(), lr=1e-3)
cfg = GenerationConfig(max_new_tokens=4, do_sample=False, pad_token_id=0, eos_token_id=63)
prompts = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
for _ in range(20):
	with torch.no_grad():
		model.generate(prompts, generation_config=cfg)
	model(prompts).logits.mean().backward()
	opt.step()
	opt.zero_grad()
