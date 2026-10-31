"""Learned step-time cost models on the corpus: a GraphSAGE network over the module graph against a GBDT on the same
config and GPU features, both scored on architectures they never saw.

This is the plan's "GNN cost model ... beats the GBDT alone on held-out architectures, or is weighted down". Each
corpus row is (module graph, batch and sequence, dtype, GPU) and its measured step. Both models predict log(step)
with no probe at all, which is what the GPU dial needs before anything has run. Folds are by architecture
(calibrate.arch_key): every row of an architecture, on every GPU, is in the same fold, so no architecture is
trained and tested on. The A100 boards never enter.

	uv run --extra probe --with scikit-learn python scripts/fit_cost_gnn.py <store> [<store> ...]

Hyperparameters are fixed here, before any fold is scored.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from preflight.corpus import calibrate as cal
from preflight.engine import conformal, gpus

TOL = 0.20
FOLDS = 5
SEED = 0


def node_vocab(rows: list[dict], top: int = 24) -> list[str]:
	c = Counter(n["target"] if n["op"] == "call_module" else n["op"] for r in rows for n in r["graph"]["nodes"])
	return [k for k, _ in c.most_common(top)]


def node_features(r: dict, vocab: list[str]) -> np.ndarray:
	out = []
	for n in r["graph"]["nodes"]:
		kind = n["target"] if n["op"] == "call_module" else n["op"]
		onehot = [1.0 if kind == v else 0.0 for v in vocab] + [0.0 if kind in vocab else 1.0]
		numel = math.prod(n["shape"]) if n.get("shape") and all(isinstance(x, int) for x in n["shape"]) else 1
		out.append([*onehot, math.log1p(n.get("params") or 0), math.log1p(numel)])
	return np.asarray(out, dtype=np.float32)


def global_features(r: dict) -> list[float]:
	p = r["config"]["params"]
	dtype = p.get("dtype") or p.get("precision") or "fp32"
	spec = gpus.get_gpu(r["gpu"])
	peak = spec.peak_flops(dtype, bool(r.get("tf32"))) or spec.peak_flops("fp32")
	d = gpus.normalise_dtype(dtype)
	return [
		math.log(max(r.get("work_units") or 1, 1)),
		math.log(float(p.get("batch") or 1)),
		math.log(float(p.get("seq") or p.get("image") or 1)),
		math.log1p(r.get("n_params") or 0),
		math.log1p(r.get("n_trainable") or 0),
		math.log(peak),
		math.log(spec.bandwidth_bytes()),
		math.log(spec.compute_units),
		1.0 if d == "fp32" else 0.0,
		1.0 if d in ("bf16", "fp16") else 0.0,
		1.0 if (p.get("grad_checkpointing") or "none") != "none" else 0.0,
	]


def graph_summary(r: dict, vocab: list[str]) -> list[float]:
	"""What the GBDT sees of the graph: counts of each node kind and the log parameter mass, no structure."""
	x = node_features(r, vocab)
	return [*x[:, : len(vocab) + 1].sum(axis=0).tolist(), float(np.log1p(np.expm1(x[:, -2]).sum()))]


def dataset(rows: list[dict]):
	keep = [
		r
		for r in rows
		if r.get("status") == "ok"
		and not cal.held_out(r.get("gpu"))
		and (r.get("timing") or {}).get("step_s")
		and (r.get("graph") or {}).get("nodes")
	]
	return keep


def folds_by_arch(rows: list[dict]) -> list[int]:
	groups = sorted({cal.arch_key(r["config"]) for r in rows})
	rng = np.random.default_rng(SEED)
	rng.shuffle(groups)
	fold = {g: i % FOLDS for i, g in enumerate(groups)}
	return [fold[cal.arch_key(r["config"])] for r in rows]


def fit_gbdt(x, y):
	from sklearn.ensemble import GradientBoostingRegressor

	return GradientBoostingRegressor(
		n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, min_samples_leaf=10, random_state=SEED
	).fit(x, y)


def fit_gnn(graphs, globals_, y, epochs: int = 150):
	import torch
	from torch import nn

	torch.manual_seed(SEED)
	d_node, d_glob = graphs[0][0].shape[1], globals_.shape[1]
	mu_g, sd_g = globals_.mean(0), globals_.std(0) + 1e-6

	class Sage(nn.Module):
		def __init__(self, h=64):
			super().__init__()
			self.l1 = nn.Linear(2 * d_node, h)
			self.l2 = nn.Linear(2 * h, h)
			self.head = nn.Sequential(nn.Linear(2 * h + d_glob, h), nn.ReLU(), nn.Linear(h, 1))

		def forward(self, x, adj, g):
			deg = adj.sum(1, keepdim=True).clamp(min=1)
			h = torch.relu(self.l1(torch.cat([x, adj @ x / deg], 1)))
			h = torch.relu(self.l2(torch.cat([h, adj @ h / deg], 1)))
			pooled = torch.cat([h.mean(0), h.max(0).values])
			return self.head(torch.cat([pooled, g])).squeeze()

	model = Sage()
	opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
	data = [
		(torch.tensor(x), torch.tensor(a), torch.tensor((g - mu_g) / sd_g, dtype=torch.float32), torch.tensor(t, dtype=torch.float32))
		for (x, a), g, t in zip(graphs, globals_, y, strict=True)
	]
	rng = np.random.default_rng(SEED)
	for _ in range(epochs):
		for i in rng.permutation(len(data))[:256]:
			x, a, g, t = data[i]
			loss = (model(x, a, g) - t) ** 2
			opt.zero_grad()
			loss.backward()
			opt.step()

	def predict(gs, globs):
		with torch.no_grad():
			return np.asarray(
				[float(model(torch.tensor(x), torch.tensor(a), torch.tensor((g - mu_g) / sd_g, dtype=torch.float32))) for (x, a), g in zip(gs, globs, strict=True)]
			)

	return predict


def adjacency(r: dict) -> np.ndarray:
	n = len(r["graph"]["nodes"])
	a = np.zeros((n, n), dtype=np.float32)
	for s, t in r["graph"]["edges"]:
		if s < n and t < n:
			a[s, t] = a[t, s] = 1.0
	return a


def report(name: str, pred: np.ndarray, y: np.ndarray) -> tuple[int, int]:
	err = np.abs(np.exp(pred - y) - 1)
	h, n = int((err <= TOL).sum()), len(y)
	lo, hi = conformal.wilson(h, n)
	print(f"  {name}: within 20% {h}/{n} ({lo:.0%} to {hi:.0%}); median |log| {np.median(np.abs(pred - y)):.3f}")
	return h, n


def main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument("stores", nargs="+", type=Path)
	args = ap.parse_args(argv)
	rows = dataset([r for p in args.stores for r in cal.load_rows(p)])
	vocab = node_vocab(rows)
	y = np.log(np.asarray([r["timing"]["step_s"] for r in rows]))
	glob = np.asarray([global_features(r) for r in rows])
	summ = np.asarray([graph_summary(r, vocab) for r in rows])
	xg = np.hstack([glob, summ])
	graphs = [(node_features(r, vocab), adjacency(r)) for r in rows]
	fold = np.asarray(folds_by_arch(rows))
	print(f"{len(rows)} rows, {len({cal.arch_key(r['config']) for r in rows})} architectures, {FOLDS} folds by architecture")
	p_gb = np.zeros(len(rows))
	p_gnn = np.zeros(len(rows))
	for k in range(FOLDS):
		tr, te = fold != k, fold == k
		p_gb[te] = fit_gbdt(xg[tr], y[tr]).predict(xg[te])
		pred = fit_gnn([graphs[i] for i in np.where(tr)[0]], glob[tr], y[tr])
		p_gnn[te] = pred([graphs[i] for i in np.where(te)[0]], glob[te])
		print(f"fold {k}: {int(te.sum())} rows held out")
	print("held-out architectures:")
	gb = report("GBDT (config, GPU and graph counts)", p_gb, y)
	gn = report("GNN (GraphSAGE over the module graph)", p_gnn, y)
	ens_w = 0.5
	report("mean of the two", ens_w * p_gb + (1 - ens_w) * p_gnn, y)
	print("verdict:", "the GNN beats the GBDT" if gn[0] > gb[0] else "the GNN does not beat the GBDT: not used")
	return 0


if __name__ == "__main__":
	sys.exit(main())
