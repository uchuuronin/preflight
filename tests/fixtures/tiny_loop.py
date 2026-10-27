"""Tiny plain PyTorch loop for recorder tests (CPU, seconds)."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

batch_size = 4
seq_len = 32
vocab_size = 512
num_workers = 0
learning_rate = 3e-3
max_iters = 1000


class Tokens(Dataset):
	def __len__(self):
		return 100_000

	def __getitem__(self, i):
		g = torch.Generator().manual_seed(i)
		x = torch.randint(0, vocab_size, (seq_len + 1,), generator=g)
		return x[:-1], x[1:]


class Tiny(nn.Module):
	def __init__(self):
		super().__init__()
		self.emb = nn.Embedding(vocab_size, 64)
		self.body = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64))
		self.lm_head = nn.Linear(64, vocab_size, bias=False)

	def forward(self, x):
		return self.lm_head(self.body(self.emb(x)))


model = Tiny()
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
loader = DataLoader(Tokens(), batch_size=batch_size, num_workers=num_workers)
it = 0
for x, y in loader:
	logits = model(x)
	loss = nn.functional.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
	optimizer.zero_grad(set_to_none=True)
	loss.backward()
	optimizer.step()
	it += 1
	if it >= max_iters:
		break
print("finished", it)
