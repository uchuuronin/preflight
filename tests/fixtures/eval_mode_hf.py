"""from_pretrained on a saved tiny Llama returns it in eval mode; the loop never calls .train()."""

import tempfile

import torch
from transformers import LlamaConfig, LlamaForCausalLM

cfg = LlamaConfig(
	vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4
)
path = tempfile.mkdtemp()
LlamaForCausalLM(cfg).save_pretrained(path)
model = LlamaForCausalLM.from_pretrained(path)
assert not model.training
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
loader = torch.utils.data.DataLoader(torch.randint(0, 128, (4000, 16)), batch_size=4)
for step, ids in enumerate(loader):
	loss = model(input_ids=ids, labels=ids).loss
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	if step >= 500:
		break
