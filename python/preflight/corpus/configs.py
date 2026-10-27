"""Deterministic config lists for the corpus sweep.

Two kinds:
- kernels: one training step profiled per (config, GPU); feeds the GBDT residual,
  the GNN cost model and memory calibration studies
- divergence: short small-GPT runs swept over learning rate, warmup, init,
  precision and qk-norm, labelled diverged or not (Wortsman et al., ICLR 2024)

Every config is content-addressed: its key is a hash of kind, family and params,
so the sweep can write each result once and resume after preemption (gap 13).
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import Any

FAMILIES = ("gpt", "encoder", "lora", "cnn", "mlp", "gnn")
_FAMILY_WEIGHTS = {"gpt": 0.35, "encoder": 0.10, "lora": 0.15, "cnn": 0.15, "mlp": 0.10, "gnn": 0.15}
SCHEMA = "preflight-corpus/0.1"


@dataclass(frozen=True)
class CorpusConfig:
	kind: str  # "kernels" | "divergence"
	family: str
	params: dict[str, Any]

	@property
	def key(self) -> str:
		blob = json.dumps({"kind": self.kind, "family": self.family, "params": self.params}, sort_keys=True)
		return hashlib.sha256(blob.encode()).hexdigest()[:16]

	def to_dict(self) -> dict[str, Any]:
		return {"kind": self.kind, "family": self.family, "params": self.params, "key": self.key}

	@classmethod
	def from_dict(cls, d: dict[str, Any]) -> CorpusConfig:
		return cls(d["kind"], d["family"], dict(d["params"]))


def _dtype(r: random.Random) -> str:
	return r.choices(["bf16", "fp16", "fp32"], weights=[0.7, 0.1, 0.2])[0]


def _with_tf32(p: dict[str, Any], r: random.Random) -> dict[str, Any]:
	"""fp32 configs run with TF32 on or off (PyTorch's default is off), so the corpus sees both."""
	if p.get("dtype") == "fp32":
		p["tf32"] = r.random() < 0.5
	return p


def _transformer(r: random.Random, causal: bool) -> dict[str, Any]:
	d_model = r.choice([256, 384, 512, 768, 1024, 1536, 2048])
	head_dim = r.choice([64, 128]) if d_model >= 512 else 64
	n_heads = d_model // head_dim
	gated = causal and r.random() < 0.5
	d_ff = int(math.ceil(8 * d_model / 3 / 64) * 64) if gated else 4 * d_model
	kv = n_heads if (not causal or r.random() < 0.6) else max(1, n_heads // r.choice([2, 4]))
	return {
		"n_layers": r.choice([2, 4, 6, 8, 12]),
		"d_model": d_model,
		"n_heads": n_heads,
		"n_kv_heads": kv,
		"d_ff": d_ff,
		"gated_mlp": gated,
		"vocab_size": r.choice([8192, 32000, 50257]) if causal else r.choice([8192, 30522]),
		"seq": r.choice([128, 256, 512, 1024, 2048]),
		"batch": r.choice([1, 2, 4, 8, 16, 32]),
		"dtype": _dtype(r),
		"attention": r.choices(["sdpa", "math"], weights=[0.75, 0.25])[0],
		"grad_checkpointing": r.choices(["none", "full"], weights=[0.8, 0.2])[0],
		"causal": causal,
	}


def _sample(family: str, r: random.Random) -> dict[str, Any]:
	return _with_tf32(_sample_raw(family, r), r)


def _sample_raw(family: str, r: random.Random) -> dict[str, Any]:
	if family == "gpt":
		return _transformer(r, causal=True)
	if family == "encoder":
		return _transformer(r, causal=False)
	if family == "lora":
		p = _transformer(r, causal=True)
		p["lora_rank"] = r.choice([4, 8, 16, 64])
		p["lora_targets"] = r.choice([["q", "v"], ["q", "k", "v", "o"], ["q", "k", "v", "o", "up", "down"]])
		if p["dtype"] == "fp32":
			p["dtype"] = "bf16"
		return p
	if family == "cnn":
		return {
			"depth": r.choice([10, 18, 34]),
			"width": r.choice([32, 64, 96]),
			"image": r.choice([32, 64, 128, 224]),
			"batch": r.choice([8, 16, 32, 64, 128, 256]),
			"channels_last": r.random() < 0.5,
			"dtype": _dtype(r),
			"n_classes": 1000,
		}
	if family == "mlp":
		return {
			"depth": r.choice([2, 3, 4, 6, 8]),
			"width": r.choice([512, 1024, 2048, 4096, 8192]),
			"batch": r.choice([64, 256, 1024, 4096, 8192]),
			"in_features": 784,
			"dtype": _dtype(r),
		}
	if family == "gnn":
		return {
			"conv": r.choice(["gcn", "sage"]),
			"nodes": r.choice([2_000, 10_000, 50_000, 200_000]),
			"avg_degree": r.choice([5, 10, 25, 50]),
			"hidden": r.choice([64, 128, 256, 512]),
			"layers": r.choice([2, 3, 4]),
			"in_features": 128,
			"n_classes": 40,
			"dtype": r.choices(["fp32", "bf16"], weights=[0.7, 0.3])[0],
		}
	raise ValueError(family)


def kernel_configs(n: int, seed: int = 0) -> list[CorpusConfig]:
	r = random.Random(seed)
	out: list[CorpusConfig] = []
	seen: set[str] = set()
	fams = list(_FAMILY_WEIGHTS)
	weights = [_FAMILY_WEIGHTS[f] for f in fams]
	# guarantee every family appears early, then sample by weight
	order = fams + r.choices(fams, weights=weights, k=max(0, 4 * n))
	for fam in order:
		if len(out) >= n:
			break
		cfg = CorpusConfig("kernels", fam, _sample(fam, r))
		if cfg.key not in seen:
			seen.add(cfg.key)
			out.append(cfg)
	return out


def amd_slice(configs: list[CorpusConfig], k: int) -> list[CorpusConfig]:
	"""A family-stratified subset in list order, so it overlaps the NVIDIA slice."""
	by_fam: dict[str, list[CorpusConfig]] = {f: [] for f in FAMILIES}
	for cfg in configs:
		by_fam[cfg.family].append(cfg)
	quota = {f: max(1, round(k * len(v) / len(configs))) for f, v in by_fam.items() if v}
	while sum(quota.values()) > k:
		quota[max(quota, key=quota.get)] -= 1
	while sum(quota.values()) < k:
		f = max(quota, key=lambda f: len(by_fam[f]) - quota[f])
		quota[f] += 1
	keep = {c.key for f, n in quota.items() for c in by_fam[f][:n]}
	return [c for c in configs if c.key in keep]


def divergence_configs(n: int, seed: int = 0) -> list[CorpusConfig]:
	r = random.Random(seed)
	out: list[CorpusConfig] = []
	seen: set[str] = set()
	while len(out) < n:
		p = {
			"n_layers": 4,
			"d_model": 256,
			"n_heads": 4,
			"seq": 256,
			"batch": 16,
			"steps": 600,
			"lr": round(10 ** r.uniform(math.log10(1e-4), math.log10(0.5)), 6),
			"warmup": r.choice([0, 20, 100]),
			"init_std": r.choice([0.02, 0.05, 0.1, 0.2]),
			"precision": r.choices(["fp32", "bf16", "fp16"], weights=[0.3, 0.4, 0.3])[0],
			"qk_norm": r.random() < 0.4,
			"z_loss": r.choice([0.0, 1e-4]),
			"beta2": r.choice([0.95, 0.999]),
			"weight_decay": 0.1,
			"grad_clip": r.choice([None, 1.0]),
			"seed": r.randrange(1_000_000),
		}
		cfg = CorpusConfig("divergence", "gpt-small", p)
		if cfg.key not in seen:
			seen.add(cfg.key)
			out.append(cfg)
	return out


def write_jsonl(configs: list[CorpusConfig], path: str) -> None:
	# the content hash is recomputed on read, and leaving it out keeps secret scanners quiet
	with open(path, "w") as f:
		for cfg in configs:
			d = cfg.to_dict()
			d.pop("key")
			f.write(json.dumps(d, sort_keys=True) + "\n")


def read_jsonl(path: str) -> list[CorpusConfig]:
	with open(path) as f:
		return [CorpusConfig.from_dict(json.loads(line)) for line in f if line.strip()]
