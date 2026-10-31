"""DDP on CPU (gloo): what torchrun runs, one process per rank, each with its own shard."""

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

dist.init_process_group("gloo")
torch.manual_seed(0)
model = DDP(nn.Sequential(nn.Linear(64, 256), nn.ReLU(), nn.Linear(256, 8)))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
data = TensorDataset(torch.randn(8192, 64), torch.randint(0, 8, (8192,)))
for x, y in DataLoader(data, batch_size=32, sampler=DistributedSampler(data)):
	loss = nn.functional.cross_entropy(model(x), y)
	loss.backward()
	opt.step()
	opt.zero_grad()
dist.destroy_process_group()
