"""Two 'adapters' that never reach the forward: the eval compares them and gets identical outputs
(the case studies' caching bug, in miniature)."""

import sys

import torch
from transformers import LlamaConfig, LlamaForCausalLM

torch.manual_seed(0)
model = LlamaForCausalLM(LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
	num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64))  # fmt: skip
for p in model.parameters():
	p.requires_grad_(False)
model.adapter = torch.nn.Parameter(torch.zeros(8))  # trained, but the forward never uses it
opt = torch.optim.SGD([model.adapter], lr=1.0)
prompts = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
checkpoints = [torch.zeros(8), torch.ones(8)]  # two "adapters" the eval means to compare
for step in range(50):
	model.adapter.grad = torch.ones(8)
	opt.step()
	if "--train-between" in sys.argv:  # a step between generations: not compared (legitimately equal)
		model.generate(prompts, max_new_tokens=4, do_sample=False, pad_token_id=0)
	elif step % 2 == 0:  # no training between the two: an eval of each checkpoint in turn
		for ck in checkpoints:
			with torch.no_grad():
				model.adapter.copy_(ck)
			model.generate(prompts, max_new_tokens=4, do_sample=False, pad_token_id=0)
