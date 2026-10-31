"""A GNN-shaped loop without PyG: each batch is a sampled subgraph shaped as a NeighborLoader's
(num_nodes, edge_index, batch_size seeds, num_sampled_nodes per hop)."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

torch.manual_seed(0)


class Sub:
	def __init__(self, seeds, fan):
		hops = [seeds, seeds * fan[0], seeds * fan[0] * fan[1]]
		self.num_sampled_nodes = hops
		self.num_nodes = sum(hops)
		self.batch_size = seeds
		self.x = torch.randn(self.num_nodes, 16)
		self.edge_index = torch.randint(0, self.num_nodes, (2, hops[1] + hops[2]))


class Seeds(Dataset):
	def __len__(self):
		return 10_000

	def __getitem__(self, i):
		return i


loader = DataLoader(Seeds(), batch_size=32, collate_fn=lambda ids: Sub(len(ids), (10, 5)))
model = nn.Linear(16, 4)
opt = torch.optim.Adam(model.parameters())
for b in loader:
	h = model(b.x)
	src, dst = b.edge_index
	agg = torch.zeros_like(h).index_add_(0, dst, h[src])
	agg[: b.batch_size].sum().backward()
	opt.step()
	opt.zero_grad()
