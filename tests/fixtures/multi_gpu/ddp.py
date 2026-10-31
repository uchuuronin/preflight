"""DDP on one node over NCCL: a 134M-parameter MLP stack, batch 32 per rank, AdamW."""

import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("nccl")
rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(rank)
torch.manual_seed(0)
layers = [m for _ in range(8) for m in (nn.Linear(4096, 4096), nn.GELU())]
model = DDP(nn.Sequential(*layers).cuda(), device_ids=[rank])
opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
for _ in range(10_000):
	x = torch.randn(32, 4096, device="cuda")
	loss = model(x).pow(2).mean()
	loss.backward()
	opt.step()
	opt.zero_grad(set_to_none=True)
dist.destroy_process_group()
