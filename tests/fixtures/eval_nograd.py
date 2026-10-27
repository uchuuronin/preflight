"""Eval every EVAL_EVERY steps with gradients kept (the zoo's planted bug), over a long eval loader."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

EVAL_EVERY = 1000
eval_batch_size = 16
model = nn.Sequential(nn.Embedding(256, 32), nn.Linear(32, 256))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
train = DataLoader(TensorDataset(torch.randint(0, 256, (4000, 16))), batch_size=4)
evals = DataLoader(TensorDataset(torch.randint(0, 256, (100_000, 16))), batch_size=eval_batch_size)


def evaluate():
	model.eval()
	losses = []
	for (x,) in evals:
		out = model(x)  # no torch.no_grad()
		losses.append(nn.functional.cross_entropy(out.view(-1, 256), x.view(-1)))
	model.train()
	return torch.stack(losses).mean().item()


for step, (x,) in enumerate(train, 1):
	loss = nn.functional.cross_entropy(model(x).view(-1, 256), x.view(-1))
	loss.backward()
	opt.step()
	opt.zero_grad()
	if step % EVAL_EVERY == 0:
		evaluate()
