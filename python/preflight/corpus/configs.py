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
_FAMILY_WEIGHTS = {
	"gpt": 0.35,
	"encoder": 0.10,
	"lora": 0.15,
	"cnn": 0.15,
	"mlp": 0.10,
	"gnn": 0.15,
}  # guess: design weights; revisited Oct 31 against the GNN cost model's 3,000-sample need
SCHEMA = "preflight-corpus/0.1"
# derived: bumped whenever the worker measures differently, so old and new rows never share a key
CODE_VERSION = "corpus@0.3.0"


@dataclass(frozen=True)
class CorpusConfig:
	kind: str  # "kernels" | "divergence" | "hf"
	family: str
	params: dict[str, Any]
	versions: tuple[tuple[str, str], ...] = ()  # code and library versions the row was measured with

	@property
	def key(self) -> str:
		"""Content hash of what was run and with what (Review, Oct 27: code and library versions in the key)."""
		body = {"kind": self.kind, "family": self.family, "params": self.params}
		if self.versions:
			body["versions"] = dict(self.versions)
		blob = json.dumps(body, sort_keys=True)
		return hashlib.sha256(blob.encode()).hexdigest()[:16]

	def to_dict(self) -> dict[str, Any]:
		d = {"kind": self.kind, "family": self.family, "params": self.params, "key": self.key}
		if self.versions:
			d["versions"] = dict(self.versions)
		return d

	@classmethod
	def from_dict(cls, d: dict[str, Any]) -> CorpusConfig:
		return cls(
			d["kind"], d["family"], dict(d["params"]), tuple(sorted((d.get("versions") or {}).items()))
		)

	def with_versions(self, versions: dict[str, str]) -> CorpusConfig:
		return CorpusConfig(
			self.kind, self.family, self.params, tuple(sorted({**dict(self.versions), **versions}.items()))
		)


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


# ---------------------------------------------------------------- grouped, probe-like configs (Review, Oct 27)

ROLE_PROBE, ROLE_QUERY = "probe", "query"


def _group_id(kind: str, family: str, params: dict[str, Any]) -> str:
	shape_free = {k: v for k, v in params.items() if k not in ("batch", "seq", "group", "role")}
	blob = json.dumps({"kind": kind, "family": family, "params": shape_free}, sort_keys=True)
	return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _points(fam: str, b: int, s: int | None) -> list[tuple[str, int, int | None]]:
	"""A's probe plan (current, half batch, half sequence) plus two held-out queries: one inside the
	queried range (2x batch) and one a halving below the smallest probe."""
	if s is None:  # vision and MLP: batch is the only work axis
		return [(ROLE_PROBE, b, None), (ROLE_PROBE, max(1, b // 2), None), (ROLE_PROBE, max(1, b // 4), None),
				(ROLE_QUERY, b * 2, None), (ROLE_QUERY, max(1, b // 8), None)]  # fmt: skip
	return [(ROLE_PROBE, b, s), (ROLE_PROBE, max(1, b // 2), s), (ROLE_PROBE, b, max(16, s // 2)),
			(ROLE_QUERY, b * 2, s), (ROLE_QUERY, max(1, b // 2), max(16, s // 2))]  # fmt: skip


def grouped_configs(
	n_groups: int, seed: int = 0, families: tuple[str, ...] = ("gpt", "lora", "encoder", "cnn", "mlp")
) -> list[CorpusConfig]:
	"""One architecture at five batch x sequence points each, so calibration data looks like a user's probes."""
	r = random.Random(seed)
	out: list[CorpusConfig] = []
	for i in range(n_groups):
		fam = families[i % len(families)]
		base = _sample(fam, r)
		seq = None
		if fam in ("gpt", "lora", "encoder"):
			base["batch"], base["seq"] = r.choice([4, 8, 16]), r.choice([256, 512, 1024])
			seq = base["seq"]
		else:
			base["batch"] = r.choice([16, 32, 64, 128])
		gid = _group_id("kernels", fam, base)
		for role, b, s in _points(fam, base["batch"], seq):
			p = {**base, "batch": b, "group": gid, "role": role}
			if s is not None:
				p["seq"] = s
			out.append(CorpusConfig("kernels", fam, p))
	return out


# ---------------------------------------------------------------- real Hugging Face jobs (Review, Oct 27)

# Ungated models that fit an L4 (24 GB) as LoRA in bf16 or QLoRA in 4-bit; weights download at run time.
HF_MODELS = (
	"Qwen/Qwen2.5-0.5B",
	"HuggingFaceTB/SmolLM2-360M",
	"TinyLlama/TinyLlama-1.1B-Chat-v1.0",
	"Qwen/Qwen2.5-1.5B",
	"HuggingFaceTB/SmolLM2-1.7B",
)


def hf_configs(models: tuple[str, ...] = HF_MODELS, rank: int = 16) -> list[CorpusConfig]:
	"""About ten real fine-tuning jobs (each model as bf16 LoRA and 4-bit QLoRA), each at the five probe-like points."""
	out: list[CorpusConfig] = []
	for model in models:
		for quant in ("none", "nf4"):
			base = {
				"model_id": model,
				"quantization": quant,
				"lora_rank": rank,
				"lora_alpha": 2 * rank,
				"lora_targets": "all-linear",
				"dtype": "bf16",
				"grad_checkpointing": quant
				== "nf4",  # what QLoRA recipes (prepare_model_for_kbit_training) turn on
				"batch": 4,
				"seq": 512,
			}
			gid = _group_id("hf", "hf_lora", base)
			for role, b, s in _points("hf_lora", 4, 512):
				out.append(
					CorpusConfig("hf", "hf_lora", {**base, "batch": b, "seq": s, "group": gid, "role": role})
				)
	return out


def divergence_real_configs(n: int, seed: int = 0) -> list[CorpusConfig]:
	"""Chiya's review (Nov 2): the Nov 1 sweep's 44 divergences all came at learning rates of 0.07 or more, none at
	the rates people train at, so the classifier was judged on the wrong runs. This sweep keeps the learning rate where
	people train (1e-4 to 3e-2, log-uniform) and turns on the usual triggers instead: no clipping, no warm-up, and fp16
	without a loss scaler for half the runs; the same small GPT, data and 600 steps. Its rows are kind
	"divergence_real", so they never mix into the Nov 1 sweep's fit."""
	r = random.Random(seed)
	out: list[CorpusConfig] = []
	seen: set[str] = set()
	while len(out) < n:
		precision = r.choices(["fp16", "bf16", "fp32"], weights=[0.5, 0.3, 0.2])[0]
		p = {
			"n_layers": 4,
			"d_model": 256,
			"n_heads": 4,
			"seq": 256,
			"batch": 16,
			"steps": 600,
			"lr": round(10 ** r.uniform(math.log10(1e-4), math.log10(3e-2)), 6),
			"warmup": 0,
			"init_std": r.choice([0.02, 0.05, 0.1, 0.2]),
			"precision": precision,
			"scaler": False,  # fp16 with no GradScaler; bf16 and fp32 never use one
			"qk_norm": r.random() < 0.4,
			"z_loss": r.choice([0.0, 1e-4]),
			"beta2": r.choice([0.95, 0.999]),
			"weight_decay": 0.1,
			"grad_clip": None,
			"seed": r.randrange(1_000_000),
		}
		cfg = CorpusConfig("divergence_real", "gpt-small", p)
		if cfg.key not in seen:
			seen.add(cfg.key)
			out.append(cfg)
	return out


DIVERGENCE_KINDS = ("divergence", "divergence_real")
KINDS = ("kernels", "grouped", "hf", "divergence", "divergence_real")


def configs_for(kind: str, n: int, seed: int = 0) -> list[CorpusConfig]:
	"""The config list for a sweep kind; grouped configs store under kernels, HF jobs under hf."""
	if kind == "kernels":
		return kernel_configs(n, seed)
	if kind == "grouped":
		return grouped_configs(max(1, n // 5), seed)
	if kind == "hf":
		return hf_configs()
	if kind == "divergence":
		return divergence_configs(n, seed)
	if kind == "divergence_real":
		return divergence_real_configs(n, seed)
	raise ValueError(f"unknown kind {kind!r}; one of {KINDS}")


def local_versions(kind: str) -> dict[str, str]:
	"""Versions of what will measure, from this environment, for the key (Review, Oct 27)."""
	import importlib.metadata as md

	out = {"corpus": CODE_VERSION}
	pkgs = ("torch", "transformers", "peft", "bitsandbytes", "accelerate") if kind == "hf" else ("torch",)
	for pkg in pkgs:
		try:
			out[pkg] = md.version(pkg)
		except md.PackageNotFoundError:
			continue
	return out
