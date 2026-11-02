# the notebook tiny_train.ipynb as the runner converts it (runner/sources.py), for a ground-truth run
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
batch_size = 64
lr = 3e-3
max_steps = 200

model = nn.Sequential(nn.Linear(32, 256), nn.ReLU(), nn.Linear(256, 1)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=lr)
w = torch.randn(32, 1, device=device)

for step in range(max_steps):
	x = torch.randn(batch_size, 32, device=device)
	y = x @ w + 0.1 * torch.randn(batch_size, 1, device=device)
	loss = nn.functional.mse_loss(model(x), y)
	opt.zero_grad()
	loss.backward()
	opt.step()
	if step % 50 == 0:
		print(step, loss.item())
