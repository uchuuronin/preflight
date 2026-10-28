"""A validation pass before training (Lightning's sanity check, HF eval_on_start), then a plain loop."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

model = nn.Sequential(nn.Embedding(128, 32), nn.Linear(32, 128))
val = DataLoader(TensorDataset(torch.randint(0, 128, (100_000, 16))), batch_size=8)
model.eval()
seen = 0
with torch.no_grad():
	for (x,) in val:
		model(x)
		seen += 1
print("VAL BATCHES", seen, flush=True)
model.train()
opt = torch.optim.SGD(model.parameters(), lr=0.1)
for (x,) in DataLoader(TensorDataset(torch.randint(0, 128, (4000, 16))), batch_size=4):
	loss = model(x).logsumexp(-1).mean()
	opt.zero_grad()
	loss.backward()
	opt.step()
