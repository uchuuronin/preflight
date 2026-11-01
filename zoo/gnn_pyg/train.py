"""GNN: GraphSAGE node classification with PyG on a 200,000-node graph, mini-batches by neighbour sampling.

BUG ZOO: this script carries a planted problem on purpose. Do not fix it;
Preflight must catch it from a few minutes of probing.

  1. Fan-out too high on the outer hops: num_neighbors=[15, 25, 25] draws up to
     15 x 25 x 25 = 9,375 neighbours per seed by the third hop. On this graph (200,494 nodes,
     average degree 50 once made undirected) one batch of 1,024 seeds reaches every node, and
     the outer hop alone carries about 3.6 million messages [simulated: this script's sampler
     on the CPU at seed 0 gives 15,360, 323,000 and 3,591,700 edges per hop; the GPU's random
     numbers differ, so the counts move a little]. Memory and time grow with each hop and the
     outer layer dominates. A smaller outer fan-out such as [15, 10, 5] shrinks the outer hops;
     by how much is the probe's to measure, and FakeDataset's random edges carry no signal, so
     accuracy can't show the trade-off here.

Read from the code: none (the per-hop fan-out is read as knobs; the growth needs measuring).
Also read, not planted: data.repetition (3 epochs; normal for node classification).
Found by the probe: peak memory per hop at the planted fan-out (memory growth per hop on the
fit map), and whether a batch fits the GPU at all.

Ground truth: not measured yet. A's probe and full run fill it in.

Data: a synthetic graph made in the script by PyG's FakeDataset, seeded so every run builds
the same graph (no download, nothing to license), so the run needs only PyPI wheels. Neighbour sampling is plain PyTorch because PyG's
fast sampler (pyg-lib) has no PyPI wheels.
"""

import json
import random
import time

import torch
import torch.nn.functional as F
from torch_geometric.datasets import FakeDataset
from torch_geometric.nn import SAGEConv
from torch_geometric.utils import to_undirected

SEED = 0
NUM_NODES = 200_000
AVG_DEGREE = 25
NUM_FEATURES = 128
NUM_CLASSES = 47
BATCH_SIZE = 1024
EPOCHS = 3
LR = 3e-3
# Seed nodes the sampler draws batches from, each epoch: the run's length is
# ceil(N_TRAIN_NODES / BATCH_SIZE) = 157 steps an epoch, 471 in all.
N_TRAIN_NODES = 160_000


def csr(edge_index: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
	src, dst = edge_index
	order = torch.argsort(dst)
	col = src[order]
	counts = torch.bincount(dst, minlength=n)
	rowptr = torch.zeros(n + 1, dtype=torch.long, device=edge_index.device)
	rowptr[1:] = torch.cumsum(counts, 0)
	return rowptr, col


def sample_subgraph(rowptr, col, seeds, num_neighbors):
	"""Sample with replacement, hop by hop; return the local node ids and one edge_index per hop."""
	frontier = seeds
	hop_edges = []
	for k in num_neighbors:
		start = rowptr[frontier]
		deg = rowptr[frontier + 1] - start
		keep = deg > 0
		frontier, start, deg = frontier[keep], start[keep], deg[keep]
		offs = (torch.rand(len(frontier), k, device=seeds.device) * deg.unsqueeze(1)).long()
		nbr = col[start.unsqueeze(1) + offs].reshape(-1)
		hop_edges.append((nbr, frontier.repeat_interleave(k)))
		frontier = nbr.unique()
	nodes = torch.cat([seeds] + [s for s, _ in hop_edges]).unique()
	local = torch.full((int(nodes.max()) + 1,), -1, dtype=torch.long, device=seeds.device)
	local[nodes] = torch.arange(len(nodes), device=seeds.device)
	edge_indexes = [torch.stack([local[s], local[d]]) for s, d in hop_edges]
	return nodes, local[seeds], edge_indexes


class SAGE(torch.nn.Module):
	def __init__(self):
		super().__init__()
		self.convs = torch.nn.ModuleList(
			[
				SAGEConv(NUM_FEATURES, 256),
				SAGEConv(256, 256),
				SAGEConv(256, NUM_CLASSES),
			]
		)

	def forward(self, x, edge_indexes):
		# The outermost hop's edges feed the first layer; the seeds' own edges feed the last.
		for i, (conv, ei) in enumerate(zip(self.convs, reversed(edge_indexes), strict=True)):
			x = conv(x, ei)
			if i < len(self.convs) - 1:
				x = F.relu(x)
		return x


def main() -> None:
	random.seed(SEED)  # FakeDataset draws the node count from Python's random
	torch.manual_seed(SEED)
	data = FakeDataset(
		num_graphs=1,
		avg_num_nodes=NUM_NODES,
		avg_degree=AVG_DEGREE,
		num_channels=NUM_FEATURES,
		num_classes=NUM_CLASSES,
		task="node",
	)[0]
	n = data.num_nodes
	edge_index = to_undirected(data.edge_index, num_nodes=n).cuda()
	rowptr, col = csr(edge_index, n)
	x, y = data.x.cuda(), data.y.cuda()
	model = SAGE().cuda()
	opt = torch.optim.Adam(model.parameters(), lr=LR)
	assert n >= N_TRAIN_NODES, f"the graph has {n} nodes, fewer than N_TRAIN_NODES"
	train_nodes = torch.randperm(n, device="cuda")[:N_TRAIN_NODES]
	started = time.monotonic()
	for epoch in range(EPOCHS):
		perm = train_nodes[torch.randperm(len(train_nodes), device="cuda")]
		total = 0.0
		for i in range(0, len(perm), BATCH_SIZE):
			seeds = perm[i : i + BATCH_SIZE]
			nodes, seed_pos, edge_indexes = sample_subgraph(rowptr, col, seeds, num_neighbors=[15, 25, 25])
			out = model(x[nodes], edge_indexes)[seed_pos]
			loss = F.cross_entropy(out, y[seeds])
			opt.zero_grad()
			loss.backward()
			opt.step()
			total += loss.item()
		print(
			json.dumps(
				{"epoch": epoch, "loss": round(total, 3), "peak_bytes": torch.cuda.max_memory_allocated()}
			),
			flush=True,
		)
	print(json.dumps({"wall_s": round(time.monotonic() - started, 1)}), flush=True)


if __name__ == "__main__":
	main()
