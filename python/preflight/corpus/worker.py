"""Runs one corpus config on this machine's GPU and returns a JSON-able record.

kernels: >= 2 warm-up optimizer steps (optimizer state appears at the first step),
then unprofiled timed steps for phase times, then a short profiled window
(wait 1, warmup 1, active 3) used only for per-op proportions, FLOPs and shapes,
because profiler overhead distorts small kernels. Peak memory is read after all
steps. The module graph comes from an FX trace on the meta device.

divergence: a short small-GPT run logging the stability signals from Wortsman et
al. (loss, grad-norm, per-layer update/weight ratio, max attention logit, output
logit norm, loss-scale events, non-finite counts), labelled after the fact.

Works on CUDA and ROCm builds of PyTorch ("cuda" is the device name on both).
"""

from __future__ import annotations

import gc
import math
import platform
import statistics
import time
from contextlib import nullcontext, suppress
from typing import Any

import torch
import torch.nn.functional as F

from preflight.corpus import models
from preflight.corpus.configs import CorpusConfig

# the D9 label lives in corpus/labels.py, which needs no torch (Chiya, Nov 1: so its tests leave the torch job)
from preflight.corpus.labels import (  # noqa: F401
	COLLAPSE_NO_LEARNING,
	COLLAPSE_SPIKE,
	LABEL_FLOOR,
	LABEL_RATIO,
	LABEL_TAIL,
	LABEL_WINDOW,
	label_divergence,
)
from preflight.engine.steptime import classify_kernel
from preflight.recorder.errors import classify

WARMUP_STEPS = 2  # cited: A's probe-plan D1 (warmup 2, timed 5)
TIMED_STEPS = 5  # cited: A's probe-plan D1 (warmup 2, timed 5)
MAX_OP_ROWS = 200  # derived: keeps a corpus row under about 100 KB
_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def classify_exception(exc: BaseException) -> str:
	"""Probe.status for a failure: oom or error. Codes come from A's classifier, shared with probes."""
	return classify(exc, steps=0, in_eval=False).status


def _sync(device: torch.device) -> None:
	if device.type == "cuda":
		torch.cuda.synchronize(device)


class _PhaseTimer:
	"""CUDA events on GPU (also ROCm), perf_counter on CPU."""

	def __init__(self, device: torch.device):
		self.device = device
		self.marks: list[tuple[str, Any]] = []

	def mark(self, name: str) -> None:
		if self.device.type == "cuda":
			ev = torch.cuda.Event(enable_timing=True)
			ev.record()
			self.marks.append((name, ev))
		else:
			self.marks.append((name, time.perf_counter()))

	def phases(self) -> dict[str, float]:
		_sync(self.device)
		out: dict[str, float] = {}
		for (_, a), (name, b) in zip(self.marks, self.marks[1:], strict=False):
			out[name] = a.elapsed_time(b) / 1e3 if self.device.type == "cuda" else b - a
		return out


def _env(device: torch.device) -> dict[str, Any]:
	env: dict[str, Any] = {
		"torch": torch.__version__,
		"python": platform.python_version(),
		"cuda": torch.version.cuda,
		"hip": getattr(torch.version, "hip", None),
		"device": str(device),
	}
	if device.type == "cuda":
		props = torch.cuda.get_device_properties(device)
		env |= {
			"device_name": props.name,
			"total_memory": props.total_memory,
			"multi_processor_count": props.multi_processor_count,
		}
	return env


def _round(x: float, sig: int = 5) -> float:
	if x is None or not math.isfinite(x) or x == 0:
		return x
	return round(x, sig - 1 - math.floor(math.log10(abs(x))))


# ---------------------------------------------------------------- kernels


def _loss(family: str, out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
	if family in ("gpt", "encoder", "lora"):
		return F.cross_entropy(out.float().flatten(0, 1), y.flatten())  # HF-style upcast
	return F.cross_entropy(out.float(), y)


def _op_rows(
	prof: torch.profiler.profile, active_steps: int, elem_bytes: int, device: torch.device
) -> list[dict[str, Any]]:
	rows = []
	for ev in prof.key_averages(group_by_input_shape=True):
		if device.type == "cuda":
			t = getattr(ev, "self_device_time_total", None)
			if t is None:
				t = getattr(ev, "self_cuda_time_total", 0)
		else:
			t = ev.self_cpu_time_total
		if not t:
			continue
		shapes = [list(s) for s in (ev.input_shapes or []) if isinstance(s, (list, tuple))]
		numel = sum(math.prod(s) for s in shapes if s)
		rows.append(
			{
				"name": ev.key,
				"class": classify_kernel(ev.key),
				"count": ev.count / active_steps,
				"device_s": t / 1e6 / active_steps,
				"flops": (ev.flops or 0) / active_steps or None,
				"bytes": numel * elem_bytes or None,  # inputs only: an estimate
				"input_shapes": shapes[:4],
			}
		)
	rows.sort(key=lambda r: -r["device_s"])
	return rows[:MAX_OP_ROWS]


def _graph(family: str, p: dict[str, Any]) -> dict[str, Any]:
	"""Module graph from an FX trace with shapes propagated on the meta device."""
	from torch.fx import symbolic_trace
	from torch.fx.passes.shape_prop import ShapeProp

	with torch.device("meta"):
		model = models.build(family, p, "meta", for_trace=True)
		inputs, _ = models.make_inputs(family, p, "meta")
	gm = symbolic_trace(model)
	ShapeProp(gm).propagate(*inputs)
	modules = dict(gm.named_modules())
	index = {n: i for i, n in enumerate(gm.graph.nodes)}
	nodes, edges = [], []
	for n in gm.graph.nodes:
		meta = n.meta.get("tensor_meta")
		shape = list(meta.shape) if meta is not None and hasattr(meta, "shape") else None
		if n.op == "call_module":
			mod = modules[n.target]
			target = type(mod).__name__
			n_params = sum(t.numel() for t in mod.parameters(recurse=False))
		else:
			target = n.target if isinstance(n.target, str) else getattr(n.target, "__name__", str(n.target))
			n_params = 0
		nodes.append({"op": n.op, "target": target, "shape": shape, "params": n_params})
		for inp in n.all_input_nodes:
			edges.append([index[inp], index[n]])
	return {"nodes": nodes, "edges": edges}


def _run_kernels(cfg: CorpusConfig, device: torch.device) -> dict[str, Any]:
	p, fam = cfg.params, cfg.family
	dtype = _DTYPES[p.get("dtype", "fp32")]
	gen = torch.Generator().manual_seed(0)
	torch.manual_seed(0)
	if device.type == "cuda":
		torch.cuda.reset_peak_memory_stats(device)
		tf32 = bool(p.get("tf32", False))  # PyTorch's default is off; fp32 configs sample both
		torch.backends.cuda.matmul.allow_tf32 = tf32
		torch.backends.cudnn.allow_tf32 = tf32
	model = models.build(fam, p, device)
	if fam == "cnn" and p.get("channels_last"):
		model = model.to(memory_format=torch.channels_last)
	inputs, y = models.make_inputs(fam, p, device, gen)
	params = [t for t in model.parameters() if t.requires_grad]
	opt = torch.optim.AdamW(params, lr=1e-4)
	use_amp = dtype != torch.float32
	scaler = torch.amp.GradScaler(device.type, enabled=dtype == torch.float16 and device.type == "cuda")
	amp = torch.autocast(device.type, dtype=dtype) if use_amp else nullcontext()

	def step(timer: _PhaseTimer | None = None) -> None:
		opt.zero_grad(set_to_none=True)
		if timer:
			timer.mark("start")
		with amp:
			out = model(*inputs)
			loss = _loss(fam, out, y)
		if timer:
			timer.mark("forward")
		scaler.scale(loss).backward()
		if timer:
			timer.mark("backward")
		scaler.step(opt)
		scaler.update()
		if timer:
			timer.mark("optimizer")

	for _ in range(WARMUP_STEPS):
		step()
	_sync(device)
	steady_allocated = torch.cuda.memory_allocated(device) if device.type == "cuda" else None

	phase_runs: list[dict[str, float]] = []
	step_times: list[float] = []
	power: list[float] = []
	for _ in range(TIMED_STEPS):
		timer = _PhaseTimer(device)
		t0 = time.perf_counter()
		step(timer)
		_sync(device)
		step_times.append(time.perf_counter() - t0)
		phase_runs.append(timer.phases())
		if device.type == "cuda":
			with suppress(Exception):  # not every driver exposes power (ROCm, MIG)
				power.append(torch.cuda.power_draw(device) / 1e3)  # mW to W

	memory = None
	if device.type == "cuda":
		memory = {
			"peak_allocated": torch.cuda.max_memory_allocated(device),
			"peak_reserved": torch.cuda.max_memory_reserved(device),
			"steady_allocated": steady_allocated,
		}

	activities = [torch.profiler.ProfilerActivity.CPU]
	if device.type == "cuda":
		activities.append(torch.profiler.ProfilerActivity.CUDA)
	sched = torch.profiler.schedule(wait=1, warmup=1, active=3)
	with torch.profiler.profile(
		activities=activities, schedule=sched, record_shapes=True, with_flops=True
	) as prof:
		for _ in range(5):
			step()
			prof.step()
	_sync(device)
	elem_bytes = torch.finfo(dtype).bits // 8

	try:
		graph = _graph(fam, p)
	except Exception as exc:  # recorded, not fatal: the timing data is still useful
		graph = {"nodes": [], "edges": [], "error": f"{type(exc).__name__}: {exc}"}

	return {
		"status": "ok",
		"n_params": sum(t.numel() for t in model.parameters()),
		"n_trainable": sum(t.numel() for t in params),
		"work_units": models.work_units(fam, p),
		"optimizer_steps_before_measure": WARMUP_STEPS,
		"tf32": bool(p.get("tf32", False)),
		"timing": {
			"step_s": statistics.median(step_times),
			"step_s_all": step_times,
			"phases": {
				k: statistics.median(r[k] for r in phase_runs) for k in ("forward", "backward", "optimizer")
			},
			"n_steps": TIMED_STEPS,
			"source": "unprofiled steps",
		},
		"memory": memory,
		"power_w": statistics.mean(power) if power else None,
		"ops": _op_rows(prof, 3, elem_bytes, device),
		"graph": graph,
	}


# ---------------------------------------------------------------- divergence


def _text_tokens(data_path: str | None, gen: torch.Generator, n: int = 2_000_000) -> torch.Tensor:
	if data_path:
		with open(data_path, "rb") as f:
			return torch.tensor(list(f.read()), dtype=torch.long)
	# synthetic: a sparse random Markov chain over bytes, learnable but not trivial
	trans = torch.rand(256, 256, generator=gen).pow(8)
	trans /= trans.sum(-1, keepdim=True)
	out = torch.empty(min(n, 200_000), dtype=torch.long)
	out[0] = 0
	draws = torch.rand(len(out), generator=gen)
	cdf = trans.cumsum(-1)
	for i in range(1, len(out)):
		out[i] = int(torch.searchsorted(cdf[out[i - 1]], draws[i]).clamp(max=255))
	return out


def _run_divergence(cfg: CorpusConfig, device: torch.device, data_path: str | None) -> dict[str, Any]:
	p = cfg.params
	gen = torch.Generator().manual_seed(p["seed"])
	torch.manual_seed(p["seed"])
	arch = {
		"n_layers": p["n_layers"],
		"d_model": p["d_model"],
		"n_heads": p["n_heads"],
		"n_kv_heads": p["n_heads"],
		"d_ff": 4 * p["d_model"],
		"gated_mlp": False,
		"vocab_size": 256,
		"causal": True,
		"attention": "sdpa",
	}
	model = models.Transformer(arch, qk_norm=p["qk_norm"]).to(device)
	models.init_transformer(model, p["init_std"])
	data = _text_tokens(data_path, gen)
	precision = _DTYPES[p["precision"]]
	amp = torch.autocast(device.type, dtype=precision) if precision != torch.float32 else nullcontext()
	scaler = torch.amp.GradScaler(device.type, enabled=precision == torch.float16 and device.type == "cuda")
	opt = torch.optim.AdamW(
		model.parameters(), lr=p["lr"], betas=(0.9, p["beta2"]), weight_decay=p["weight_decay"]
	)
	blocks = list(model.blocks)
	attn0 = blocks[0].attn

	series: dict[str, list[Any]] = {
		k: []
		for k in (
			"loss",
			"grad_norm",
			"update_ratio",
			"max_attn_logit",
			"output_logit_norm",
			"loss_scale",
			"nonfinite_grads",
		)
	}
	b, s = p["batch"], p["seq"]
	stopped_early = False
	for step in range(p["steps"]):
		lr = p["lr"] * min(1.0, (step + 1) / p["warmup"]) if p["warmup"] else p["lr"]
		for g in opt.param_groups:
			g["lr"] = lr
		starts = torch.randint(0, len(data) - s - 1, (b,), generator=gen)
		chunk = torch.stack([data[i : i + s + 1] for i in starts.tolist()]).to(device)
		x, y = chunk[:, :-1], chunk[:, 1:]
		attn0.track_logits = True
		opt.zero_grad(set_to_none=True)
		with amp:
			logits = model(x)
			loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten())
			if p["z_loss"]:
				loss = loss + p["z_loss"] * torch.logsumexp(logits.float(), -1).pow(2).mean()
		attn0.track_logits = False
		scaler.scale(loss).backward()
		scaler.unscale_(opt)
		grads = [t.grad for t in model.parameters() if t.grad is not None]
		gnorm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.float()) for g in grads]))
		nonfinite = sum(int(~torch.isfinite(g).all()) for g in grads)
		if p["grad_clip"]:
			torch.nn.utils.clip_grad_norm_(model.parameters(), p["grad_clip"])
		before = [[t.detach().clone() for t in blk.parameters()] for blk in blocks]
		scaler.step(opt)
		scaler.update()
		ratios = []
		for blk, old in zip(blocks, before, strict=True):
			num = sum(
				float((t.detach() - o).float().norm() ** 2)
				for t, o in zip(blk.parameters(), old, strict=True)
			)
			den = sum(float(o.float().norm() ** 2) for o in old)
			ratios.append(_round(math.sqrt(num) / math.sqrt(den)) if den > 0 else None)
		lv = float(loss.detach())
		series["loss"].append(_round(lv) if math.isfinite(lv) else None)
		series["grad_norm"].append(_round(float(gnorm)) if math.isfinite(float(gnorm)) else None)
		series["update_ratio"].append(ratios)
		series["max_attn_logit"].append(_round(attn0.last_max_logit))
		series["output_logit_norm"].append(_round(float(logits.detach().float().norm(dim=-1).mean())))
		series["loss_scale"].append(scaler.get_scale() if scaler.is_enabled() else None)
		series["nonfinite_grads"].append(nonfinite)
		if not math.isfinite(lv):
			stopped_early = True
			break
	losses = [x if x is not None else math.nan for x in series["loss"]]
	return {
		"status": "ok",
		"n_params": sum(t.numel() for t in model.parameters()),
		"steps_run": len(losses),
		"stopped_early": stopped_early,
		"series": series,
		"label": label_divergence(losses, vocab_size=256),
		"data": "file" if data_path else "synthetic-markov",
	}


# ---------------------------------------------------------------- entry


def run_config(cfg: CorpusConfig, device: str = "cuda", data_path: str | None = None) -> dict[str, Any]:
	dev = torch.device(device)
	t0 = time.monotonic()
	try:
		if cfg.kind == "kernels":
			result = _run_kernels(cfg, dev)
		elif cfg.kind == "divergence":
			result = _run_divergence(cfg, dev, data_path)
		elif cfg.kind == "hf":
			from preflight.corpus.hf_worker import run_hf

			result = run_hf(cfg, dev)
		else:
			raise ValueError(f"unknown kind {cfg.kind!r}")
	except Exception as exc:
		c = classify(exc, steps=0, in_eval=False)
		result = {
			"status": c.status,
			"error_code": c.code,
			"error": f"{type(exc).__name__}: {str(exc)[:500]}",
		}
		if c.status == "error" and c.code == "user_error":
			raise  # a bug in the corpus code itself, not a property of the config: fail loudly
		if dev.type == "cuda":
			result["memory"] = {
				"peak_allocated": torch.cuda.max_memory_allocated(dev),
				"peak_reserved": torch.cuda.max_memory_reserved(dev),
				"total": torch.cuda.get_device_properties(dev).total_memory,
			}
	finally:
		gc.collect()  # drop tensors an OOM left behind before the allocator releases its cache
		if dev.type == "cuda":
			torch.cuda.empty_cache()
	result["gpu_seconds"] = time.monotonic() - t0
	result["env"] = _env(dev)
	return result
