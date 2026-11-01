"""optimizer.step(closure): the forward and backward run inside the step (Lightning's automatic
optimization, LBFGS). From the Oct 31 review of the recorder's forward windows."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)
model = nn.Sequential(nn.Embedding(512, 256), nn.Linear(256, 2048), nn.ReLU(), nn.Linear(2048, 512))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
for (x,) in DataLoader(TensorDataset(torch.randint(0, 512, (100000, 64))), batch_size=256):  # forward well above the step under load

	def closure():
		opt.zero_grad()
		loss = model(x).logsumexp(-1).mean()  # noqa: B023 (called at once, inside the step)
		loss.backward()
		return loss

	opt.step(closure)
