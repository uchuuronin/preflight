"""Synthetic model families for the corpus. FX-traceable when checkpointing is off.

Kept dependency-free (plain PyTorch, no torchvision or PyG) so the same code runs
on CUDA and ROCm builds.
"""

from __future__ import annotations

import itertools
import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
	def __init__(self, d: int, eps: float = 1e-6):
		super().__init__()
		self.weight = nn.Parameter(torch.ones(d))
		self.eps = eps

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		var = x.float().pow(2).mean(-1, keepdim=True)
		return (x.float() * torch.rsqrt(var + self.eps)).type_as(x) * self.weight


class LoRALinear(nn.Module):
	def __init__(self, base: nn.Linear, rank: int, alpha: float = 16.0):
		super().__init__()
		self.base = base
		self.base.weight.requires_grad_(False)
		if self.base.bias is not None:
			self.base.bias.requires_grad_(False)
		self.lora_a = nn.Parameter(torch.empty(rank, base.in_features))
		self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank))
		nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
		self.scale = alpha / rank

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.base(x) + (x @ self.lora_a.t() @ self.lora_b.t()) * self.scale


class Attention(nn.Module):
	def __init__(self, d: int, n_heads: int, n_kv: int, causal: bool, impl: str, qk_norm: bool = False):
		super().__init__()
		self.h, self.kv, self.hd = n_heads, n_kv, d // n_heads
		self.q = nn.Linear(d, d, bias=False)
		self.k = nn.Linear(d, n_kv * self.hd, bias=False)
		self.v = nn.Linear(d, n_kv * self.hd, bias=False)
		self.o = nn.Linear(d, d, bias=False)
		self.causal, self.impl = causal, impl
		self.q_norm = RMSNorm(self.hd) if qk_norm else None
		self.k_norm = RMSNorm(self.hd) if qk_norm else None
		self.track_logits = False
		self.last_max_logit = 0.0

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		b, s, d = x.shape
		q = self.q(x).view(b, s, self.h, self.hd).transpose(1, 2)
		k = self.k(x).view(b, s, self.kv, self.hd).transpose(1, 2)
		v = self.v(x).view(b, s, self.kv, self.hd).transpose(1, 2)
		if self.q_norm is not None:
			q, k = self.q_norm(q), self.k_norm(k)
		if self.kv != self.h:
			rep = self.h // self.kv
			k = k.repeat_interleave(rep, dim=1)
			v = v.repeat_interleave(rep, dim=1)
		if self.track_logits:
			with torch.no_grad():
				logits = (q[:1].float() @ k[:1].float().transpose(-1, -2)) / math.sqrt(self.hd)
				self.last_max_logit = float(logits.abs().max())
		if self.impl == "math":
			att = (q @ k.transpose(-1, -2)) / math.sqrt(self.hd)
			if self.causal:
				mask = torch.ones(s, s, device=x.device, dtype=torch.bool).triu(1)
				att = att.masked_fill(mask, float("-inf"))
			y = att.softmax(-1) @ v
		else:
			y = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
		return self.o(y.transpose(1, 2).reshape(b, s, d))


class MLP(nn.Module):
	def __init__(self, d: int, ff: int, gated: bool):
		super().__init__()
		self.gated = gated
		if gated:
			self.gate = nn.Linear(d, ff, bias=False)
		self.up = nn.Linear(d, ff, bias=False)
		self.down = nn.Linear(ff, d, bias=False)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if self.gated:
			return self.down(F.silu(self.gate(x)) * self.up(x))
		return self.down(F.gelu(self.up(x)))


class Block(nn.Module):
	def __init__(self, p: dict[str, Any], qk_norm: bool = False):
		super().__init__()
		d = p["d_model"]
		norm = RMSNorm if p.get("causal", True) else nn.LayerNorm
		self.n1, self.n2 = norm(d), norm(d)
		self.attn = Attention(
			d,
			p["n_heads"],
			p.get("n_kv_heads", p["n_heads"]),
			p.get("causal", True),
			p.get("attention", "sdpa"),
			qk_norm,
		)
		self.mlp = MLP(d, p["d_ff"], p.get("gated_mlp", False))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = x + self.attn(self.n1(x))
		return x + self.mlp(self.n2(x))


class Transformer(nn.Module):
	def __init__(self, p: dict[str, Any], qk_norm: bool = False, checkpointing: bool = False):
		super().__init__()
		d, v = p["d_model"], p["vocab_size"]
		self.emb = nn.Embedding(v, d)
		self.blocks = nn.ModuleList(Block(p, qk_norm) for _ in range(p["n_layers"]))
		self.norm = RMSNorm(d) if p.get("causal", True) else nn.LayerNorm(d)
		self.head = nn.Linear(d, v, bias=False)
		self.head.weight = self.emb.weight  # tied
		self.checkpointing = checkpointing

	def forward(self, idx: torch.Tensor) -> torch.Tensor:
		x = self.emb(idx)
		for blk in self.blocks:
			x = checkpoint(blk, x, use_reentrant=False) if self.checkpointing else blk(x)
		return self.head(self.norm(x))


def apply_lora(model: Transformer, rank: int, targets: list[str]) -> None:
	for p in model.parameters():
		p.requires_grad_(False)
	for blk in model.blocks:
		for t in targets:
			parent = blk.attn if t in ("q", "k", "v", "o") else blk.mlp
			if hasattr(parent, t):
				setattr(parent, t, LoRALinear(getattr(parent, t), rank))


def init_transformer(model: Transformer, std: float) -> None:
	for name, p in model.named_parameters():
		if p.dim() >= 2 and "lora" not in name:
			nn.init.normal_(p, std=std)


# ---------------------------------------------------------------- CNN


class BasicBlock(nn.Module):
	def __init__(self, cin: int, cout: int, stride: int):
		super().__init__()
		self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
		self.b1 = nn.BatchNorm2d(cout)
		self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
		self.b2 = nn.BatchNorm2d(cout)
		self.skip = None
		if stride != 1 or cin != cout:
			self.skip = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		y = F.relu(self.b1(self.c1(x)))
		y = self.b2(self.c2(y))
		return F.relu(y + (x if self.skip is None else self.skip(x)))


_STAGES = {10: [1, 1, 1, 1], 18: [2, 2, 2, 2], 34: [3, 4, 6, 3]}


class ResNet(nn.Module):
	def __init__(self, depth: int, width: int, n_classes: int):
		super().__init__()
		self.stem = nn.Sequential(nn.Conv2d(3, width, 3, 1, 1, bias=False), nn.BatchNorm2d(width), nn.ReLU())
		layers, cin = [], width
		for i, n in enumerate(_STAGES[depth]):
			cout = width * 2**i
			for j in range(n):
				layers.append(BasicBlock(cin, cout, 2 if (j == 0 and i > 0) else 1))
				cin = cout
		self.body = nn.Sequential(*layers)
		self.fc = nn.Linear(cin, n_classes)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.fc(self.body(self.stem(x)).mean((2, 3)))


# ---------------------------------------------------------------- MLP


def make_mlp(p: dict[str, Any]) -> nn.Module:
	layers: list[nn.Module] = []
	d = p["in_features"]
	for _ in range(p["depth"]):
		layers += [nn.Linear(d, p["width"]), nn.ReLU()]
		d = p["width"]
	layers.append(nn.Linear(d, 10))
	return nn.Sequential(*layers)


# ---------------------------------------------------------------- GNN


class GraphConv(nn.Module):
	def __init__(self, din: int, dout: int, kind: str):
		super().__init__()
		self.kind = kind
		self.lin = nn.Linear(2 * din if kind == "sage" else din, dout)

	def forward(
		self, x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, inv_deg: torch.Tensor
	) -> torch.Tensor:
		msg = x.index_select(0, src)
		agg = x.new_zeros(x.shape).index_add(0, dst, msg) * inv_deg
		if self.kind == "sage":
			return self.lin(torch.cat([x, agg], dim=-1))
		return self.lin(agg + x * inv_deg)


class GNN(nn.Module):
	def __init__(self, p: dict[str, Any]):
		super().__init__()
		dims = [p["in_features"]] + [p["hidden"]] * (p["layers"] - 1) + [p["n_classes"]]
		self.convs = nn.ModuleList(GraphConv(a, b, p["conv"]) for a, b in itertools.pairwise(dims))

	def forward(
		self, x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor, inv_deg: torch.Tensor
	) -> torch.Tensor:
		for i, conv in enumerate(self.convs):
			x = conv(x, src, dst, inv_deg)
			if i < len(self.convs) - 1:
				x = F.relu(x)
		return x


# ---------------------------------------------------------------- build


def build(family: str, p: dict[str, Any], device: torch.device | str, for_trace: bool = False) -> nn.Module:
	if family in ("gpt", "encoder", "lora"):
		ckpt = p.get("grad_checkpointing", "none") == "full" and not for_trace
		model = Transformer(p, checkpointing=ckpt)
		if family == "lora":
			apply_lora(model, p["lora_rank"], p["lora_targets"])
	elif family == "cnn":
		model = ResNet(p["depth"], p["width"], p["n_classes"])
	elif family == "mlp":
		model = make_mlp(p)
	elif family == "gnn":
		model = GNN(p)
	else:
		raise ValueError(family)
	return model.to(device)


def make_inputs(
	family: str, p: dict[str, Any], device: torch.device | str, gen: torch.Generator | None = None
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
	"""(model inputs, targets), created on the device."""
	kw = {"device": device}
	if family in ("gpt", "encoder", "lora"):
		idx = torch.randint(0, p["vocab_size"], (p["batch"], p["seq"] + 1), generator=gen).to(device)
		return (idx[:, :-1],), idx[:, 1:]
	if family == "cnn":
		x = torch.randn(p["batch"], 3, p["image"], p["image"], generator=gen).to(**kw)
		if p.get("channels_last"):
			x = x.to(memory_format=torch.channels_last)
		return (x,), torch.randint(0, p["n_classes"], (p["batch"],), generator=gen).to(device)
	if family == "mlp":
		x = torch.randn(p["batch"], p["in_features"], generator=gen).to(**kw)
		return (x,), torch.randint(0, 10, (p["batch"],), generator=gen).to(device)
	if family == "gnn":
		n, e = p["nodes"], p["nodes"] * p["avg_degree"]
		src = torch.randint(0, n, (e,), generator=gen).to(device)
		dst = torch.randint(0, n, (e,), generator=gen).to(device)
		deg = torch.zeros(n, device=device).index_add(0, dst, torch.ones(e, device=device))
		inv_deg = (1.0 / deg.clamp(min=1)).unsqueeze(-1)
		x = torch.randn(n, p["in_features"], generator=gen).to(**kw)
		y = torch.randint(0, p["n_classes"], (n,), generator=gen).to(device)
		return (x, src, dst, inv_deg), y
	raise ValueError(family)


def work_units(family: str, p: dict[str, Any]) -> int:
	"""The calibration axis: tokens for LMs, samples for vision/MLP, edges for GNNs."""
	if family in ("gpt", "encoder", "lora"):
		return p["batch"] * p["seq"]
	if family in ("cnn", "mlp"):
		return p["batch"]
	return p["nodes"] * p["avg_degree"]
