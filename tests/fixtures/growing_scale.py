"""GradScaler whose scale doubles every step (growth_interval=1): timed steps must see each step's own scale."""

import torch
import torch.nn as nn

model = nn.Linear(16, 16)
opt = torch.optim.SGD(model.parameters(), lr=0.01)
scaler = torch.amp.GradScaler("cpu", init_scale=1024.0, growth_interval=1)
loader = torch.utils.data.DataLoader(torch.randn(4000, 16), batch_size=8)
for x in loader:
	loss = model(x).pow(2).mean()
	opt.zero_grad()
	scaler.scale(loss).backward()
	scaler.step(opt)
	scaler.update()
