"""A tiny image classifier on random 3x32x32 images, channels-last, batch 16."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

torch.manual_seed(0)
model = nn.Sequential(
	nn.Conv2d(3, 8, 3),
	nn.Conv2d(8, 8, 3, groups=8),
	nn.ReLU(),
	nn.AdaptiveAvgPool2d(1),
	nn.Flatten(),
	nn.Linear(8, 10),
)  # a depthwise conv too (MobileNet-like)
model = model.to(memory_format=torch.channels_last)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
data = TensorDataset(torch.randn(4096, 3, 32, 32), torch.randint(0, 10, (4096,)))
for x, y in DataLoader(data, batch_size=16):
	x = x.contiguous(memory_format=torch.channels_last)
	loss = nn.functional.cross_entropy(model(x), y)
	loss.backward()
	opt.step()
	opt.zero_grad()
