"""A plain loop over a model left in eval mode, as HF's from_pretrained leaves it: the script never calls .train()."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

batch_size = 4
seq_len = 16
vocab_size = 128


class Tokens(Dataset):
	def __len__(self):
		return 10_000

	def __getitem__(self, i):
		g = torch.Generator().manual_seed(i)
		x = torch.randint(0, vocab_size, (seq_len + 1,), generator=g)
		return x[:-1], x[1:]


model = nn.Sequential(nn.Embedding(vocab_size, 32), nn.Dropout(0.1), nn.Linear(32, vocab_size))
model.eval()  # what from_pretrained does before returning the model
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
for step, (x, y) in enumerate(DataLoader(Tokens(), batch_size=batch_size)):
	loss = nn.functional.cross_entropy(model(x).view(-1, vocab_size), y.view(-1))
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()
	if step >= 500:
		break
