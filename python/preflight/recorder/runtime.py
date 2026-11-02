"""The recorder: process-wide hooks that measure whatever training loop a script runs.

Installed before the user's script starts. What it hooks, and what for:

  optimizer step (torch's global pre/post hooks)  step count, grad norm, update-to-weight ratio,
                                                  optimizer time, peak memory per step, loss scale
  Tensor.backward                                 backward time, the loss, micro-steps per step, NaN/Inf
  DataLoader iterator                             time waiting on data, tokens, samples, padding;
                                                  caps eval loaders at a few batches
  nn.Module.train / eval (outermost call)         eval phases: time, peak memory, grads kept or not
  output head (found at the first step)           logits size and norm
  F.scaled_dot_product_attention                  max attention logit on two calls per step
  torch.save, safetensors save                    checkpoint time and size, kept out of step time
  GradScaler.scale                                which scaler to read the loss scale from
  HF generate, vLLM LLM.generate                  rollout and generation time, lengths, truncation
  HF TrainingArguments, Trainer.log               log every step, no external reporters; TRL/GRPO
                                                  metrics (truncation, zero-spread groups, KL, entropy)
  a sampler thread                                host RSS (with workers) and GPU power

Timings are taken with a device synchronise at phase boundaries, on steps
without the profiler attached; checkpoint, eval and instrumentation time are
subtracted from step time and reported on their own. Profiling (wait 1,
warmup 1, active 3) runs on extra steps after the timed ones. When the target
is reached the recorder stops at the next batch fetch, backward or step,
so the last step's logging, save and eval still run and get measured.
StopProbe is a BaseException, so a script's `except Exception` can't swallow it.
"""

import copy
import functools
import gc
import importlib.abc
import importlib.machinery
import itertools
import json
import math
import os
import re
import statistics
import sys
import threading
import time
import weakref
from contextlib import suppress
from pathlib import Path

import torch
import torch.nn as nn

from .arch import model_shape
from .gpuinfo import canonical_gpu_name, peak
from .kernels import KERNEL_ROWS  # noqa: F401  (re-exported for callers and tests)
from .rl_groups import reward_groups

PROFILE_ACTIVE = 3  # guess: enough steps for a stable kernel table; check on zoo traces (Oct 28-29)
# guess: batches a loop may fetch in eval mode before any backward; a sanity check or eval_on_start
# pass is cut there. A training loop does its first backward on its first batch.
PRETRAIN_EVAL_CAP = 8
# synchronised steps after the timed ones, for the phase split (data/forward/backward/optimizer):
# timed steps carry no recorder sync points, so their host-side phase timings mean nothing on a GPU
# a serving probe's new tokens per prompt (Chiya, Nov 1): 32 prompts at a script's 4,096 tokens
# decoded for 272 s on an A10 and ran past the probe's 600 s on an L4 (a-serving-debug-1). The
# script's own cap stays the record's max_new_tokens; extras.serving_probe_max_tokens says the probe
# stopped sooner, so B's simulation extrapolates to the script's length
SERVING_PROBE_MAX_TOKENS = 1024
OUTPUT_WATCH = 64  # derived: generate calls watched for identical outputs; each costs one small read
LENS_STATS_BATCHES = 64  # derived: enough batches to average a sampler's varying graph sizes
PHASE_STEPS = (
	2  # guess: two synchronised steps for the phase split; compare with the profiled window on zoo runs
)
PROFILE_STEPS = 2 + PROFILE_ACTIVE  # wait 1 + warmup 1 + active 3
RANGE_STALL_MICRO = 16  # guess: backwards with no optimizer step before a range test counts as stalled
RANGE_STALL_FACTOR = 4  # guess: that many times the most backwards any recorded step took
RANGE_STALL_FIRST = 64  # guess: before the first step (a GradScaler's start-up skips are fewer than 20)
RANGE_SAMPLE = 4  # guess: weights whose update the range test reads; few, as each is copied every step
RANGE_SAMPLE_MAX = 4_000_000  # guess: elements per sampled weight, so the copies stay small
ATTN_CALLS_PER_STEP = 2  # guess: two attention layers sampled per step bound the check's cost
ATTN_QUERY_SAMPLE = 128  # guess: query rows sampled for the max-logit check
ATTN_KEY_CHUNK = 4096  # keys per matmul in the attention-logit check: bounds its extra memory
SAMPLE_EVERY_S = 0.25  # guess: host RSS and GPU power sampling period
ENTROPY_MC_MAX = 4096  # policy forwards kept for the entropy estimate (one device scalar each)
ENTROPY_CHUNK_BYTES = 64 * 2**20  # guess: the exact entropy's fp32 softmax scratch per chunk of positions

# TRL / HF log keys -> what they mean for Preflight's RL lens
TRL_KEYS = {
	"truncation": ("completions/clipped_ratio", "completion_clipped_ratio"),
	"zero_spread": ("frac_reward_zero_std",),
	"kl": ("kl", "objective/kl"),
	"entropy": ("entropy", "objective/entropy"),
	"clip": ("clip_ratio/region_mean", "clip_ratio", "policy/clipfrac"),
	"reward": ("reward", "rewards/mean", "objective/scores"),
	"len_mean": ("completions/mean_length", "completion_length"),
	"len_max": ("completions/max_length",),
}


def _step_has(field_name: str) -> bool:
	try:
		from preflight.schema.record import StepSample

		return field_name in StepSample.model_fields
	except Exception:
		return False


_STEP_HAS_RESERVED = _step_has("reserved_bytes")


class StopProbe(BaseException):
	pass


def _bytes(t: torch.Tensor) -> int:
	return t.numel() * t.element_size()


def _unique_bytes(tensors) -> int:
	seen, total = set(), 0
	for t in tensors:
		if t is None or not isinstance(t, torch.Tensor) or t.device.type == "meta":
			continue
		key = (t.device, t.data_ptr()) if t.data_ptr() else id(t)
		if key in seen:
			continue
		seen.add(key)
		total += _bytes(t)
	return total


def _batch_stats(batch) -> tuple[int | None, int | None, int | None]:
	"""(tokens, real tokens, samples) of a batch, when it looks like token ids."""
	ids = mask = None
	if hasattr(batch, "keys") and hasattr(batch, "get"):
		ids, mask = batch.get("input_ids"), batch.get("attention_mask")
		if ids is None:
			first = next((v for v in batch.values() if isinstance(v, torch.Tensor)), None)
			return None, None, int(first.shape[0]) if first is not None and first.dim() else None
	elif isinstance(batch, list | tuple) and batch and isinstance(batch[0], torch.Tensor):
		ids = batch[0]
	elif isinstance(batch, torch.Tensor):
		ids = batch
	if not isinstance(ids, torch.Tensor) or ids.dim() == 0:
		return None, None, None
	samples = int(ids.shape[0])
	if ids.dtype not in (torch.int64, torch.int32) or ids.dim() != 2:
		return None, None, samples
	tokens = int(ids.numel())
	real = int(mask.sum().item()) if isinstance(mask, torch.Tensor) else tokens
	return tokens, real, samples


def _sample_prompts(args: tuple, kwargs: dict, n: int) -> tuple[tuple, dict]:
	"""vLLM generate's prompts cut to the first n, each prompt kept whole and its reply at the
	length the script asked for; per-prompt lists (sampling params, LoRA requests, priorities) are
	cut to match. One prompt given as a flat list of token ids is one prompt, never cut."""

	def many(v) -> bool:  # a batch of prompts, not one prompt's token ids
		return isinstance(v, list | tuple) and len(v) > n and not all(isinstance(x, int) for x in v)

	def cut(v):
		return v[:n] if isinstance(v, list | tuple) and len(v) > n else v

	args = list(args)
	batch = (args and many(args[0])) or any(
		many(kwargs.get(k)) for k in ("prompts", "prompt_token_ids", "inputs")
	)
	if not batch:
		return tuple(args), kwargs
	if args and many(args[0]):
		args[0] = args[0][:n]
	if len(args) > 1:
		args[1] = cut(args[1])
	for k in ("prompts", "prompt_token_ids", "inputs"):
		if many(kwargs.get(k)):
			kwargs[k] = kwargs[k][:n]
	for k in ("sampling_params", "lora_request", "priority", "prompt_adapter_request"):
		if k in kwargs:
			kwargs[k] = cut(kwargs[k])
	return tuple(args), kwargs


def _cap_new_tokens(args: tuple, kwargs: dict, cap: int) -> tuple[tuple, dict, int | None, int | None]:
	"""vLLM generate's sampling params with max_tokens at most cap, copied (the script's own objects
	are left alone): (args, kwargs, the script's largest max_tokens, the cap applied or None when no
	prompt asked for more). No sampling params means vLLM's default of 16 tokens: nothing to cap."""
	args = list(args)
	where = ("kw", None) if "sampling_params" in kwargs else ("arg", 1) if len(args) > 1 else (None, None)
	sp = kwargs["sampling_params"] if where[0] == "kw" else args[1] if where[0] == "arg" else None
	if sp is None:
		return tuple(args), kwargs, None, None
	many = isinstance(sp, list | tuple)
	items = list(sp) if many else [sp]
	asked = [getattr(x, "max_tokens", None) for x in items]

	def over(a) -> bool:  # None: vLLM decodes to the model's context length, the longest of all
		return a is None or (isinstance(a, int) and a > cap)

	if not any(over(a) for a in asked):
		return tuple(args), kwargs, None, None
	out = []
	for x, a in zip(items, asked, strict=True):
		if over(a):
			x = x.clone() if hasattr(x, "clone") else copy.copy(x)
			x.max_tokens = cap
		out.append(x)
	new = out if many else out[0]
	if where[0] == "kw":
		kwargs = {**kwargs, "sampling_params": new}
	else:
		args[1] = new
	# the script's own cap: None when any prompt had none (to the context length)
	script = None if any(a is None for a in asked) else max(a for a in asked if isinstance(a, int))
	return tuple(args), kwargs, script, cap


def _lens_stats(batch) -> dict[str, float]:
	"""What the vision and graph lenses read from a training batch, by shape alone (no sync):
	images (N, C, H, W) as the first tensor of the batch, and a PyG-style graph batch (num_nodes,
	edge_index, batch_size seeds, num_sampled_nodes per hop)."""
	out: dict[str, float] = {}
	with suppress(Exception):
		if hasattr(batch, "edge_index") and hasattr(batch, "num_nodes"):
			out["graph_nodes_per_batch"] = float(batch.num_nodes)
			ei = batch.edge_index
			if isinstance(ei, torch.Tensor) and ei.dim() == 2:
				out["graph_edges_per_batch"] = float(ei.shape[1])
			seeds = getattr(batch, "batch_size", None)
			if isinstance(seeds, int) and seeds > 0:
				out["graph_seed_nodes"] = float(seeds)
			hops = getattr(batch, "num_sampled_nodes", None)
			if isinstance(hops, list | tuple):
				for i, n in enumerate(hops[:4]):
					out[f"graph_nodes_hop{i}"] = float(n)
			return out
		first = batch
		if hasattr(batch, "keys") and hasattr(batch, "get"):
			first = batch.get("pixel_values", batch.get("image", batch.get("images")))
		elif isinstance(batch, list | tuple) and batch:
			first = batch[0]
		if (
			isinstance(first, torch.Tensor)
			and first.dim() == 4
			and (first.is_floating_point() or first.dtype == torch.uint8)
		):
			# uint8 too: DALI, FFCV and torchvision.io hand over raw images and normalise on the GPU
			n, c, h, w = (int(x) for x in first.shape)
			if c in (1, 3, 4):
				out["images_per_batch"] = float(n)
				out["image_height"], out["image_width"] = float(h), float(w)
	return out


def _quantile(xs, q):
	xs = sorted(xs)
	return xs[min(len(xs) - 1, round(q * (len(xs) - 1)))]


class _PostImport(importlib.abc.MetaPathFinder):
	"""Run a callback right after a module finishes importing (or now, if it already has)."""

	def __init__(self):
		self.callbacks: dict[str, list] = {}
		self._busy = threading.local()

	def on(self, name, fn):
		if name in sys.modules:
			with suppress(Exception):
				fn(sys.modules[name])
			return
		self.callbacks.setdefault(name, []).append(fn)

	def find_spec(self, fullname, path, target=None):
		if fullname not in self.callbacks or getattr(self._busy, "name", None) == fullname:
			return None
		self._busy.name = fullname
		try:
			spec = importlib.machinery.PathFinder.find_spec(fullname, path)
			if spec is None:
				for finder in sys.meta_path:
					if finder is self or not hasattr(finder, "find_spec"):
						continue
					spec = finder.find_spec(fullname, path, target)
					if spec is not None:
						break
		finally:
			self._busy.name = None
		if spec is None or spec.loader is None or not hasattr(spec.loader, "exec_module"):
			return spec
		loader, cbs = spec.loader, self.callbacks.pop(fullname)

		class Wrapped(importlib.abc.Loader):
			def create_module(self, s):
				return loader.create_module(s)

			def exec_module(self, module):
				loader.exec_module(module)
				for fn in cbs:
					with suppress(Exception):
						fn(module)

		spec.loader = Wrapped()
		return spec


def _device_index(*names: str) -> int:
	for n in names:
		v = (os.environ.get(n) or "").split(",")[0].strip()
		if v.isdigit():
			return int(v)
	return 0


def gpu_power_reader():
	"""A function returning the probe GPU's board power in watts, or None when nothing can read it.

	NVIDIA through NVML (nvidia-ml-py); AMD through amdsmi, which ROCm ships for its Python
	(on a ROCm box: pip install /opt/rocm/share/amd_smi). The visible-devices variables only pick the
	first index; with a remapped mask the reading may be another GPU's, so it stays a side signal.
	"""
	try:
		import torch

		hip = getattr(torch.version, "hip", None)
	except Exception:
		hip = None
	if hip:
		try:
			import amdsmi

			amdsmi.amdsmi_init()
			handles = amdsmi.amdsmi_get_processor_handles()
			h = handles[
				min(
					_device_index("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"),
					len(handles) - 1,
				)
			]

			def amd() -> float | None:
				info = amdsmi.amdsmi_get_power_info(h)
				# MI300 reports current_socket_power; older parts only the average (both in W)
				for key in ("current_socket_power", "average_socket_power", "socket_power"):
					v = info.get(key) if isinstance(info, dict) else None
					if isinstance(v, int | float) and not isinstance(v, bool):
						return float(v)
				return None

			return amd
		except Exception:
			return None
	try:
		import pynvml

		pynvml.nvmlInit()
		h = pynvml.nvmlDeviceGetHandleByIndex(_device_index("CUDA_VISIBLE_DEVICES"))
		return lambda: pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
	except Exception:
		return None


class Recorder:
	def __init__(
		self,
		*,
		warmup: int = 2,
		timed: int = 5,
		profile: bool = False,
		trace_path: str | None = None,
		progress_path: str | None = None,
		eval_batches: int = 2,
		hf_overrides: bool = True,
		alternate_sync: bool = False,
		generate_sample: int = 0,
		generate_calls: int = 0,
		lr_range: dict | None = None,
	):
		if warmup + timed < 2:
			raise ValueError("a probe runs at least 2 optimizer steps; optimizer state appears at the first")
		self.warmup, self.timed, self.profile = warmup, timed, profile
		# ground truth only (recorder/overhead.py): after warmup, clean and synced steps alternate,
		# and every step is timed from its boundary events, so the two kinds compare like with like
		self.alternate_sync = alternate_sync
		self.generate_sample = generate_sample  # a serving probe: prompts kept per generate call
		self.generate_calls = generate_calls  # an eval probe: generate calls measured before it ends
		self._gen_calls_done = 0
		self.target = warmup + timed + (0 if alternate_sync else PHASE_STEPS)
		# the learning-rate range test (D, Nov 2; probes/lr_range.py, behind probes.lr_range): before
		# each optimizer step the rate is set on an exponential ramp; the run ends at the ramp's end,
		# or where the loss diverges (no phase steps: their rates would be past the ramp). With
		# start == end it holds one rate (the GPU check's confirmation runs)
		self.lr_range = dict(lr_range) if lr_range else None
		self._lr_scales: list[float] | None = None
		self._lr_div = None  # probes/lr_range.py Divergence, made at the first step
		self.lr_range_error: str | None = None
		self._range_params: list | None = None  # the hooked optimizer's weights whose update is sampled
		self._range_others: set[int] = set()  # other optimizers that stepped other weights (their ids)
		self._range_overlap_steps = 0  # steps by another optimizer over the ramp's weights since its own
		self.lr_range_stalled: int | None = None  # steps done when a GradScaler stopped the stepping
		self.lr_range_replaced = False  # the script stepped another optimizer over the same weights instead
		if self.lr_range:
			self.target = int(self.lr_range["steps"])
		self._boundary_events: list = []  # CUDA event at each step boundary, read once at the end
		self.trace_path = trace_path
		self.progress_path = progress_path
		self.eval_cap = max(1, eval_batches)
		self.hf_overrides = hf_overrides
		self.cuda = torch.cuda.is_available()
		self.steps: list[dict] = []
		self.opt = None
		self._reset_acc()
		self._last_post = None
		self._data_end = None
		self._data_end_bwd = False  # the forward window was started by a backward, not a fetch
		self._generating = 0  # HF generate calls in progress (a rollout is not a reference forward)
		self._t_start = time.perf_counter()
		self._first_fetch = None
		self._prof = None
		self._prof_steps = 0
		self._stop_pending = False
		self._kernels: dict | None = None
		self._resolved = False
		self._sdpa_seen = False  # attn_impl for script-defined models: only what actually ran
		self._roots: list[nn.Module] = []
		self._head_out = {"bytes": None, "norm": None, "dtype": None}
		self._weights = self._params_total = self._params_trainable = None
		self._sample: list = []
		self._sample_params: list = []
		self._handles = []
		self._head_handles = []
		self._orig: dict = {}
		self._importer = _PostImport()
		self.num_workers = None
		self.stage = "running script"
		# eval
		self.eval_active = False
		# .eval() before any backward is loading, not evaluating: HF's from_pretrained calls it, and a
		# loop that never calls .train() would otherwise have its data loader cut to the eval cap
		self._trained = False
		self._pretrain_eval = None  # a module put in eval mode before training started
		self._pretrain_batches: dict[int, int] = {}
		self._eval_module = None
		self._eval_t0 = 0.0
		self._eval_batches_now = 0
		self.evals: list[dict] = []
		self._eval_peak_stash = 0
		self._train_peak_before_eval: int | None = None
		# io, generation, logs, scalers, sampler
		self._io_depth = 0
		self.gens: list[dict] = []
		self._outputs: dict[int, dict] = {}  # input row hash -> {weights mark: output hash}
		self.checkpoints: dict = {}  # what the script's training checkpoints hold (_inspect_save)
		self._any_opt_steps = 0
		self._all_opts: weakref.WeakSet = weakref.WeakSet()
		self._output_watch_calls = 0
		self.trainer_logs: list[dict] = []
		self._entropy_mc: list = []  # (optimizer steps so far, the policy's mean -log p as a device scalar)
		self._entropy_exact: list = []  # (optimizer steps so far, the policy's mean entropy, TRL 0.14's logits)
		self.rl_entropy_estimated = False
		self.rl_entropy_exact = False
		# GRPO (TRL): each reward call's values, so pass rate per prompt and zero-spread groups can
		# be read without trusting the trainer's own logs; the group size from the trainer
		self.reward_calls: list[dict] = []
		self.group_size: int | None = None
		# how long the real run is, for B's time.total: the training loader's batches per epoch,
		# and from an HF Trainer its epochs and the optimizer steps it plans (state.max_steps)
		self.train_loader_batches: int | None = None
		self._train_iter: int | None = None
		self.retrieval: dict[str, dict] = {}  # encode, index_add, index_train, search: calls, items, seconds
		self._retrieval_depth = threading.local()
		self._retrieval_lock = threading.Lock()
		self.lens_stats: list[dict] = []  # vision and graph batch shapes, the first training batches
		self._train_dataset = None
		self._train_sampler = None
		self.run_length: dict = {}
		self._scalers: weakref.WeakSet = weakref.WeakSet()
		self._attn_calls = 0
		self._rss_peak = 0
		self._power: list[float] = []
		self._sampler_stop = threading.Event()
		self._depth = threading.local()
		self._progress()

	# ------------------------------------------------------------- plumbing

	def _sync(self):
		if self._clean:
			return  # timed steps: the only synchronisation is the event at the step boundary
		if self.cuda and torch.cuda.is_initialized():
			torch.cuda.synchronize()

	def _hard_sync(self):
		"""Synchronise even in a timed step: around eval and checkpoint saves, whose time is taken out
		of the step, so it must not include training work still queued on the device."""
		if self.cuda and torch.cuda.is_initialized():
			torch.cuda.synchronize()

	@property
	def _clean(self) -> bool:
		"""True during the timed steps: no recorder sync points, no scalar reads off the GPU. Never in a
		range test: every step's loss and update are read (its timings aren't kept)."""
		if self.lr_range:
			return False
		n = len(self.steps)
		if self._prof is not None or self.eval_active or not self.warmup <= n < self.warmup + self.timed:
			return False
		return not self.alternate_sync or (n - self.warmup) % 2 == 0

	def _kind(self) -> str:
		n = len(self.steps)
		if self._prof is not None:
			return "profiled"
		if n < self.warmup:
			return "warmup"
		if self.alternate_sync:
			return "timed" if (n - self.warmup) % 2 == 0 else "phase"
		return "timed" if n < self.warmup + self.timed else "phase"

	@staticmethod
	def _scalar(v):
		"""A deferred scalar (a 0-d tensor kept on device during timed steps) read as a float."""
		if v is None:
			return None
		if torch.is_tensor(v):
			try:
				return float(v.float())
			except Exception:
				return None
		return v

	def _now(self):
		return time.perf_counter()

	def _progress(self, **extra):
		if not self.progress_path:
			return
		data = {
			"stage": self.stage,
			"steps": len(self.steps),
			"t": round(self._now() - self._t_start, 3),
			"in_eval": self.eval_active,
			**extra,
		}
		tmp = self.progress_path + ".tmp"
		with open(tmp, "w") as f:
			json.dump(data, f)
		os.replace(tmp, self.progress_path)

	def _reset_acc(self):
		self.acc = {
			"data": 0.0,
			"forward": 0.0,
			"backward": 0.0,
			"overhead": 0.0,
			"loss": 0.0,
			"loss_n": 0,
			"micro": 0,
			"tokens": 0,
			"real": 0,
			"samples": 0,
			"batches": 0,
			"nonfinite": 0,
			"io": 0.0,
			"io_bytes": 0,
			"eval": 0.0,
			"generate": 0.0,
			"reference": 0.0,  # RL: the reference model's log-probs (a forward with no gradient)
			"reward": 0.0,  # RL: reward functions and reward models
			"weight_sync": 0.0,  # RL: the policy's weights copied into the vLLM engine
			"attn_max": None,
			"loss_t": [],  # timed steps: losses kept on device, read after the probe
			"scales_t": [],
			"padded_len": 0,
		}
		self._attn_calls = 0

	def _maybe_stop(self):
		if self._stop_pending:
			self._stop_pending = False
			raise StopProbe

	# ------------------------------------------------------------- install

	def install(self):
		from torch.optim.optimizer import register_optimizer_step_post_hook, register_optimizer_step_pre_hook
		from torch.utils.data.dataloader import _BaseDataLoaderIter

		self._handles.append(register_optimizer_step_pre_hook(self._pre))
		self._handles.append(register_optimizer_step_post_hook(self._post))
		rec = self
		F = torch.nn.functional
		self._orig = {
			"backward": torch.Tensor.backward,
			"next": _BaseDataLoaderIter.__next__,
			"iter_cls": _BaseDataLoaderIter,
			"train": nn.Module.train,
			"save": torch.save,
			"sdpa": F.scaled_dot_product_attention,
			"scaler_scale": torch.amp.GradScaler.scale,
		}
		o = self._orig

		def backward(t, *args, **kwargs):
			rec._maybe_stop()
			rec._trained = True
			if rec.eval_active:
				# gradients flowing means training resumed without a .train() call: the eval pass is over
				rec._end_eval()
			rec._sync()
			t0 = rec._now()
			if rec._data_end is not None:
				rec.acc["forward"] += t0 - rec._data_end
				rec._data_end = None
			if t.numel() == 1:
				if rec._clean:
					# kept on device and read after the probe: .item() here would sync every micro-step
					rec.acc["loss_t"].append(t.detach().float())
					# the scale as it is now: GradScaler.update() changes the same tensor in place later
					rec.acc["scales_t"].append(
						[
							sc.detach().clone()
							for sc in (getattr(s_, "_scale", None) for s_ in rec._scalers)
							if sc is not None
						]
					)
				else:
					v = float(t.detach().float().item())
					for s_ in list(rec._scalers):
						with suppress(Exception):
							v /= float(s_.get_scale())  # scaler.scale(loss).backward(): report the real loss
					if v != v or v in (float("inf"), float("-inf")):
						rec.acc["nonfinite"] += 1
					else:
						rec.acc["loss"] += v
						rec.acc["loss_n"] += 1
			if not rec.steps and rec.stage in ("running script", "first batch"):
				rec.stage = "first backward"
				rec._progress()
			out = o["backward"](t, *args, **kwargs)
			rec._sync()
			rec.acc["backward"] += rec._now() - t0
			rec.acc["micro"] += 1
			# the next micro-step's forward starts here when its batch was fetched ahead (HF Trainer
			# fetches every accumulation batch up front); a fetch in between moves the start again
			rec._data_end = rec._now()
			rec._data_end_bwd = True
			if rec.lr_range and rec.lr_range_stalled is None and not rec.lr_range_replaced:
				rec._range_stall_check()
			return out

		def next_(it):
			if rec.eval_active:
				ev = rec.evals[-1] if rec.evals else None
				if ev is not None:
					if ev.get("loader_batches") is None:
						with suppress(Exception):
							ev["loader_batches"] = len(it)  # the whole pass, for extrapolating it
					mem = ev.get("memory_by_batch", [])
					if (
						rec._eval_batches_now
						and rec.cuda
						and torch.cuda.is_initialized()
						and len(mem) <= rec.eval_cap
					):
						# what the previous eval batch left allocated: outputs kept for metrics grow it
						ev.setdefault("memory_by_batch", []).append(torch.cuda.memory_allocated())
				if rec._eval_batches_now >= rec.eval_cap:
					raise StopIteration  # an eval loop gets a few batches, enough to measure it
				batch = o["next"](it)
				rec._eval_batches_now += 1
				if rec.evals and rec.evals[-1].get("batch_size") is None:
					rec.evals[-1]["batch_size"] = _batch_stats(batch)[2]
				return batch
			pending = rec._pretrain_eval
			if pending is not None and not rec._trained and not pending.training:
				n = rec._pretrain_batches.get(id(it), 0) + 1
				rec._pretrain_batches[id(it)] = n
				if n > PRETRAIN_EVAL_CAP:
					raise StopIteration  # a validation pass before training, cut like any eval
			elif not rec._trained:
				# the loader whose batch feeds the first training step: kept until the first backward
				# (a batch drawn from another loader earlier, for a fixed sample or stats, is replaced),
				# then frozen. An IterableDataset has no length: the run's length stays unknown
				n = None
				with suppress(Exception):
					n = len(it)
				rec.train_loader_batches = n if isinstance(n, int) and n > 0 else None
				rec._train_dataset = getattr(it, "_dataset", None)  # its row lengths, read at the end
				# what the loader draws per epoch: a SubsetRandomSampler over a train split, or a
				# WeightedRandomSampler's num_samples, is fewer rows than the dataset (A review, Nov 1)
				# a DistributedSampler's length is one rank's share: the dataset is the run's. Inside
				# suppress: a custom sampler's property must never raise in the script's next()
				rec._train_sampler = None
				with suppress(Exception):
					smp = getattr(getattr(it, "_index_sampler", None), "sampler", None)
					rec._train_sampler = None if hasattr(smp, "num_replicas") else smp
				rec._train_iter = id(it)
			rec._maybe_stop()
			rec._sync()
			t0 = rec._now()
			if rec._first_fetch is None:
				rec._first_fetch = t0
				rec.num_workers = getattr(it, "_num_workers", None)
				if rec.stage == "running script":
					rec.stage = "first batch"
					rec._progress()
			batch = o["next"](it)
			t1 = rec._now()
			rec.acc["data"] += t1 - t0
			tokens, real, samples = _batch_stats(batch)
			if tokens and samples:
				rec.acc["padded_len"] = max(rec.acc["padded_len"], tokens // samples)
			rec.acc["overhead"] += rec._now() - t1
			rec.acc["batches"] += 1
			if tokens is not None:
				rec.acc["tokens"] += tokens
				rec.acc["real"] += real
			if samples is not None:
				rec.acc["samples"] += samples
			if not rec.eval_active and id(it) == rec._train_iter and len(rec.lens_stats) < LENS_STATS_BATCHES:
				st = _lens_stats(batch)
				if st:
					rec.lens_stats.append(st)
			rec._data_end = rec._now()
			rec._data_end_bwd = False
			return batch

		def train(m, mode=True):
			depth = getattr(rec._depth, "n", 0)
			rec._depth.n = depth + 1
			try:
				if depth == 0:
					rec._mode_switch(m, mode)
				return o["train"](m, mode)
			finally:
				rec._depth.n = depth

		def save(obj, f, *args, **kwargs):
			with suppress(Exception):
				rec._inspect_save(obj, f)
			return rec._timed_io(o["save"], f, obj, f, *args, **kwargs)

		def sdpa(q, k, v, *args, **kwargs):
			out = o["sdpa"](q, k, v, *args, **kwargs)
			rec._sdpa_seen = True
			if (
				torch.is_grad_enabled()
				and not rec.eval_active
				and not rec._clean
				and rec._attn_calls < ATTN_CALLS_PER_STEP
			):
				rec._attn_calls += 1
				t0 = rec._now()
				with suppress(Exception):
					rec._attn_logit(q, k, kwargs.get("scale"))
				rec.acc["overhead"] += rec._now() - t0
			return out

		def scaler_scale(s, outputs):
			rec._scalers.add(s)
			return o["scaler_scale"](s, outputs)

		torch.Tensor.backward = backward
		_BaseDataLoaderIter.__next__ = next_
		nn.Module.train = train
		torch.save = save
		F.scaled_dot_product_attention = sdpa
		torch.amp.GradScaler.scale = scaler_scale

		sys.meta_path.insert(0, self._importer)
		self._importer.on("safetensors.torch", self._hook_safetensors)
		self._importer.on("transformers.generation.utils", self._hook_hf_generate)
		self._importer.on("transformers.trainer", self._hook_trainer)
		self._importer.on("trl.trainer.grpo_trainer", self._hook_grpo)
		for name in ("lightning.pytorch.trainer.trainer", "pytorch_lightning.trainer.trainer"):
			self._importer.on(name, self._hook_lightning)
		if self.hf_overrides:
			self._importer.on("transformers.training_args", self._hook_training_args)
		self._importer.on("vllm.entrypoints.llm", self._hook_vllm)
		self._importer.on("sentence_transformers.SentenceTransformer", self._hook_st_encode)
		self._importer.on("faiss", self._hook_faiss)
		threading.Thread(target=self._sample_loop, daemon=True).start()
		return self

	def uninstall(self):
		for h in self._handles + self._head_handles:
			h.remove()
		self._handles.clear()
		self._head_handles.clear()
		if self._orig:
			o = self._orig
			torch.Tensor.backward = o["backward"]
			o["iter_cls"].__next__ = o["next"]
			nn.Module.train = o["train"]
			torch.save = o["save"]
			torch.nn.functional.scaled_dot_product_attention = o["sdpa"]
			torch.amp.GradScaler.scale = o["scaler_scale"]
		for undo in self._orig.get("undo", []):
			with suppress(Exception):
				undo()
		with suppress(ValueError):
			sys.meta_path.remove(self._importer)
		self._sampler_stop.set()

	def _patch(self, owner, name, wrapper_factory):
		orig = getattr(owner, name)
		setattr(owner, name, wrapper_factory(orig))
		self._orig.setdefault("undo", []).append(lambda: setattr(owner, name, orig))

	# ------------------------------------------------------------- instruments

	def _timed_io(self, fn, target, *args, **kwargs):
		if self._io_depth:
			return fn(*args, **kwargs)
		self._io_depth += 1
		try:
			self._hard_sync()
			t0 = self._now()
			out = fn(*args, **kwargs)
			self._hard_sync()
			dt = self._now() - t0
		finally:
			self._io_depth -= 1
		size = 0
		if isinstance(target, str | os.PathLike):
			with suppress(OSError):
				size = os.path.getsize(target)
		elif hasattr(target, "tell"):
			with suppress(Exception):
				size = target.tell()
		self.acc["io"] += dt
		self.acc["io_bytes"] += size
		return out

	def _hook_safetensors(self, mod):
		rec = self
		for name in ("save_file", "save_model"):
			if hasattr(mod, name):

				def factory(orig, name=name):
					def wrapped(*args, **kwargs):
						target = kwargs.get("filename") or (args[1] if len(args) > 1 else None)
						return rec._timed_io(orig, target, *args, **kwargs)

					return wrapped

				self._patch(mod, name, factory)

	def _mode_switch(self, m: nn.Module, mode: bool):
		if self._roots:
			if not any(m is r for r in self._roots):
				return
		elif not any(p.requires_grad for p in m.parameters()):
			return  # a frozen reference or encoder switching modes is not an eval phase
		if not mode and not self._trained:
			# .eval() before any backward: from_pretrained, or a pre-training validation pass (HF
			# eval_on_start, Lightning's sanity check). Not an eval phase yet; the data loader caps a
			# loop that runs on in eval mode without a backward (see next_)
			if m.training:
				self._pretrain_eval = m
			return
		if mode and m is self._pretrain_eval:
			self._pretrain_eval = None
		if not mode and not self.eval_active and m.training:
			self._hard_sync()
			self.eval_active = True
			self._eval_module = m
			self._eval_t0 = self._now()
			self._eval_batches_now = 0
			if self.cuda and torch.cuda.is_initialized():
				self._eval_peak_stash = max(self._eval_peak_stash, torch.cuda.max_memory_allocated())
				if self._train_peak_before_eval is None:
					# A's verify design (Nov 1): training's own peak up to the first eval, so a verify
					# window that ends with an eval still has training's peak apart from the eval's
					steps_peak = max((s["peak"] or 0 for s in self.steps), default=0)
					self._train_peak_before_eval = max(steps_peak, self._eval_peak_stash)
				torch.cuda.reset_peak_memory_stats()
			self.evals.append({"batch_size": None, "grad_enabled": None, "peak": None})
			self._progress()
		elif mode and self.eval_active and m is self._eval_module:
			self._end_eval()

	def _end_eval(self):
		self._hard_sync()
		dt = self._now() - self._eval_t0
		ev = self.evals[-1]
		ev["seconds"] = dt
		ev["batches"] = self._eval_batches_now
		if self.cuda and torch.cuda.is_initialized() and len(ev.get("memory_by_batch", [])) < ev["batches"]:
			ev.setdefault("memory_by_batch", []).append(torch.cuda.memory_allocated())
		if self.cuda and torch.cuda.is_initialized():
			ev["peak"] = torch.cuda.max_memory_allocated()
			torch.cuda.reset_peak_memory_stats()
		self.acc["eval"] += dt
		self.eval_active = False
		self._eval_module = None
		self._data_end = None
		self._progress()

	def _attn_logit(self, q, k, scale):
		"""Largest |q·k|·scale over a sample of queries, in the native dtype and in key chunks, so a
		long context doesn't allocate a float32 [heads, queries, keys] block on top of the step."""
		if q.dim() < 3 or k.dim() < 3:
			return
		q0, k0 = q[0].detach(), k[0].detach()
		if q0.dim() == 2:
			q0, k0 = q0[None], k0[None]
		if k0.shape[0] != q0.shape[0] and q0.shape[0] % k0.shape[0] == 0:
			k0 = k0.repeat_interleave(q0.shape[0] // k0.shape[0], dim=0)
		n = q0.shape[-2]
		idx = torch.linspace(0, n - 1, min(n, ATTN_QUERY_SAMPLE), device=q0.device).long()
		qs = q0[..., idx, :]
		s = scale if scale is not None else q0.shape[-1] ** -0.5
		best = None
		with torch.no_grad():
			for start in range(0, k0.shape[-2], ATTN_KEY_CHUNK):
				part = (qs @ k0[..., start : start + ATTN_KEY_CHUNK, :].transpose(-1, -2)).abs().amax()
				best = part if best is None else torch.maximum(best, part)
		if best is None:
			return
		m = float(best.float()) * s
		self.acc["attn_max"] = m if self.acc["attn_max"] is None else max(self.acc["attn_max"], m)

	def _hook_hf_generate(self, mod):
		rec = self
		gm = getattr(mod, "GenerationMixin", None)
		if gm is None:
			return

		def factory(orig):
			def generate(model, *args, **kwargs):
				rec._maybe_stop()
				rec._gen_call_starts()
				rec._sync()
				t0 = rec._now()
				rec._generating += 1
				try:
					out = orig(model, *args, **kwargs)
				finally:
					rec._generating -= 1
				rec._sync()
				dt = rec._now() - t0
				with suppress(Exception):
					rec._record_hf_generation(model, args, kwargs, out, dt)
				rec.acc["generate"] += dt
				if not rec._generating:  # the outermost call only: assisted decoding nests generate
					rec._gen_calls_done += 1
				return out

			return generate

		self._patch(gm, "generate", factory)

	def _record_hf_generation(self, model, args, kwargs, out, dt):
		seqs = getattr(out, "sequences", out)
		inputs = kwargs.get("input_ids", kwargs.get("inputs", args[0] if args else None))
		# TRL passes its settings as generation_config=; the model's own config is the fallback
		given = kwargs.get("generation_config")
		cfgs = [c for c in (given, getattr(model, "generation_config", None)) if c is not None]

		def setting(name):
			if kwargs.get(name) is not None:
				return kwargs[name]
			return next((getattr(c, name) for c in cfgs if getattr(c, name, None) is not None), None)

		cap = setting("max_new_tokens")
		eos = setting("eos_token_id")
		eos = set(eos if isinstance(eos, list | tuple) else [eos] if eos is not None else [])
		pad = setting("pad_token_id")
		in_len = (
			inputs.shape[-1] if isinstance(inputs, torch.Tensor) and seqs.shape[-1] >= inputs.shape[-1] else 0
		)
		gen = seqs[:, in_len:].tolist()
		lengths, truncated = [], []
		for row in gen:
			end = next((i + 1 for i, t in enumerate(row) if t in eos), None)
			n = end if end is not None else sum(1 for t in row if t != pad or pad in eos)
			lengths.append(n)
			truncated.append((end is None and n >= cap) if cap is not None else None)  # no cap: unknown
		self.gens.append(
			{"seconds": dt, "lengths": lengths, "truncated": truncated, "cap": cap, "backend": "hf"}
		)
		with suppress(Exception):
			self._watch_outputs(model, inputs, gen)

	def _inspect_save(self, obj, f) -> None:
		"""Hygiene: what a training checkpoint holds, so a resume that can't match is found before
		it happens. A torch.save of a dict holding the trained parameters themselves (tensors sharing
		storage with the optimizer's), read for optimizer state (a dict with 'state' and
		'param_groups', or a list of them as Lightning keeps), a scheduler, RNG state and a step
		count. A save named as a final or best export is skipped: weights alone are its purpose. HF
		Trainer saves optimizer.pt, scheduler.pt and rng_state.pth beside the weights, read by name."""
		name = str(getattr(f, "name", f)).rsplit("/", 1)[-1].lower()
		ck = self.checkpoints
		if name in ("optimizer.pt", "scheduler.pt") or name.startswith("rng_state"):
			ck[{"optimizer.pt": "optimizer", "scheduler.pt": "scheduler"}.get(name, "rng")] = True
			return
		if not isinstance(obj, dict) or not self._all_opts or re.search(r"final|best|export|inference", name):
			return
		# every optimizer's parameters; 0 is an empty or ZeRO-3 placeholder tensor, which matches anything
		params = {p.data_ptr() for o in list(self._all_opts) for g in o.param_groups for p in g["params"]} - {
			0
		}

		def holds_params(v):
			if isinstance(v, torch.Tensor):
				return v.data_ptr() in params
			# every entry: a probe or late-layer LoRA trains only tensors far down the dict
			return isinstance(v, dict) and any(isinstance(x, torch.Tensor) and x.data_ptr() in params
				for x in v.values())  # fmt: skip

		def is_optim(v):
			if isinstance(v, list | tuple):
				return any(is_optim(x) for x in v)
			return isinstance(v, dict) and "state" in v and "param_groups" in v

		vals = list(obj.values())
		if holds_params(obj) or any(holds_params(v) for v in vals):
			ck["saves"] = ck.get("saves", 0) + 1
			ck["optimizer"] = ck.get("optimizer", False) or any(is_optim(v) for v in vals)
			keys = " ".join(str(k).lower() for k in obj)
			ck["scheduler"] = ck.get("scheduler", False) or "sched" in keys
			ck["rng"] = ck.get("rng", False) or "rng" in keys or "random" in keys or "seed" in keys
			ck["step"] = ck.get("step", False) or any(w in keys for w in ("step", "epoch", "iter"))

	def _weights_mark(self, model) -> tuple:
		"""Which weights generated: the active adapter(s), a fingerprint of a few trainable (else
		first) parameters, and how many optimizer steps had run (generations with training between
		them are not compared: a few small steps can leave greedy outputs the same). Not the model
		object: two loads of one checkpoint are the same weights. At most OUTPUT_WATCH calls."""
		adapter = None
		with suppress(Exception):  # transformers 4.x: active_adapter() raises with no adapter loaded
			if getattr(model, "_hf_peft_config_loaded", False):
				adapter = model.active_adapters()
			elif not callable(getattr(type(model), "active_adapter", None)):
				adapter = getattr(model, "active_adapter", None)  # PEFT's PeftModel: a property
		params = [p for p in model.parameters() if p.requires_grad][:4] or list(model.parameters())[:4]
		with torch.no_grad():
			fp = sum(float(p.detach().reshape(-1)[:256].float().sum()) for p in params)
		return str(adapter), fp, self._any_opt_steps

	def _watch_outputs(self, model, inputs, gen):
		"""Hygiene (case study, Oct 28): different weights generating identical outputs for the same
		inputs (a cache, or the adapter not switched). Keeps, per input row, what each set of weights
		generated."""
		if self._output_watch_calls >= OUTPUT_WATCH or not isinstance(inputs, torch.Tensor):
			return
		self._output_watch_calls += 1
		mark = self._weights_mark(model)
		for row_in, row_out in zip(inputs.tolist(), gen, strict=False):
			key = hash(tuple(row_in))
			seen = self._outputs.setdefault(key, {})
			if len(seen) < 8:
				seen[mark] = hash(tuple(row_out))

	def _identical_outputs(self) -> dict:
		"""Inputs generated by two sets of weights that differ (another adapter, or weights loaded
		more than 0.1% apart by fingerprint, with no training step between) with the same output."""
		same = differ = 0
		for seen in self._outputs.values():
			marks = list(seen.items())
			for i in range(len(marks)):
				for j in range(i + 1, len(marks)):
					(m1, o1), (m2, o2) = marks[i], marks[j]
					weights_differ = m1[2] == m2[2] and (
						m1[0] != m2[0] or abs(m1[1] - m2[1]) > 1e-3 * max(abs(m1[1]), abs(m2[1]), 1e-12)
					)
					if weights_differ:
						differ += 1
						same += o1 == o2
		return {"outputs_compared": differ, "outputs_identical": same} if differ else {}

	def _retrieval_timed(self, kind: str, fn, count):
		"""A retrieval stage (encode, index add, index train, search), timed with a sync either side,
		and how many items it took. Stages run outside the training step as often as inside it."""
		rec = self

		@functools.wraps(fn)
		def run(*args, **kwargs):
			depth = getattr(rec._retrieval_depth, "n", 0)
			if depth:
				return fn(*args, **kwargs)  # a stage inside another, on this thread (encode calling itself)
			rec._retrieval_depth.n = depth + 1
			try:
				rec._sync()
				t0 = rec._now()
				out = fn(*args, **kwargs)
				rec._sync()
				dt = rec._now() - t0
			finally:
				rec._retrieval_depth.n = depth
			n = None
			with suppress(Exception):
				n = int(count(args, kwargs))
			with rec._retrieval_lock:
				st = rec.retrieval.setdefault(kind, {"calls": 0, "items": 0, "seconds": 0.0})
				st["calls"] += 1
				st["seconds"] += dt
				st["items"] += n or 0
			return out

		return run

	def _hook_st_encode(self, mod):
		cls = getattr(mod, "SentenceTransformer", None)
		if cls is None or not hasattr(cls, "encode"):
			return

		def n(args, kwargs):
			x = args[1] if len(args) > 1 else kwargs.get("sentences", kwargs.get("inputs"))
			return 1 if isinstance(x, str) else len(x)

		self._patch(cls, "encode", lambda orig: self._retrieval_timed("encode", orig, n))

	def _hook_faiss(self, mod):
		"""faiss swaps each index class's add, train and search for a numpy wrapper at import; those
		wrappers are what the script calls, so each is timed by class (count: the vectors' rows)."""

		def rows(args, kwargs):
			x = args[1] if len(args) > 1 else kwargs.get("x")
			return x.shape[0]

		base = getattr(mod, "Index", None)
		for name in dir(mod):
			cls = getattr(mod, name, None)
			if not isinstance(cls, type) or not (
				(isinstance(base, type) and issubclass(cls, base)) or name.startswith(("Index", "GpuIndex"))
			):
				continue  # GPU indexes (index_cpu_to_gpu) are Index subclasses with their own wrappers
			for meth, kind in (("add", "index_add"), ("add_with_ids", "index_add"), ("train", "index_train"),
				("search", "search")):  # fmt: skip
				if meth in cls.__dict__:  # the class's own wrapper, not one inherited (timed once)
					with suppress(Exception):
						self._patch(
							cls, meth, lambda orig, kind=kind: self._retrieval_timed(kind, orig, rows)
						)

	def _hook_vllm(self, mod):
		rec = self
		llm = getattr(mod, "LLM", None)
		if llm is None:
			return

		def factory(orig):
			def generate(self_, *args, **kwargs):
				rec._maybe_stop()
				rec._gen_call_starts()
				script_cap = probe_cap = None
				if rec.generate_sample:
					args, kwargs = _sample_prompts(args, kwargs, rec.generate_sample)
					args, kwargs, script_cap, probe_cap = _cap_new_tokens(
						args, kwargs, SERVING_PROBE_MAX_TOKENS
					)
				rec._sync()
				t0 = rec._now()
				rec._generating += 1  # beam_search calls generate once a step: those inner calls don't count
				try:
					outs = orig(self_, *args, **kwargs)
				finally:
					rec._generating -= 1
				dt = rec._now() - t0
				with suppress(Exception):
					sp = kwargs.get("sampling_params", args[1] if len(args) > 1 else None)
					sp = sp[0] if isinstance(sp, list) and sp else sp
					cap = script_cap if probe_cap else getattr(sp, "max_tokens", None)
					lengths, trunc = [], []
					for o in outs:
						for c in o.outputs:
							lengths.append(len(c.token_ids))
							reason = getattr(c, "finish_reason", None)  # needs no cap to read
							# stopped at the probe's cap, not the script's: how it would have ended isn't known
							cut = probe_cap and reason == "length" and len(c.token_ids) >= probe_cap
							trunc.append(
								None if cut else (reason == "length" if reason is not None else None)
							)
					rec.gens.append({"seconds": dt, "lengths": lengths, "truncated": trunc, "cap": cap,
						"probe_cap": probe_cap, "backend": "vllm"})  # fmt: skip
				rec.acc["generate"] += dt
				if not rec._generating:
					rec._gen_calls_done += 1
				if rec.generate_sample:
					raise StopProbe  # the sampled call is the serving probe's measurement: done
				return outs

			return generate

		self._patch(llm, "generate", factory)

	def _hook_trainer(self, mod):
		rec = self
		tr = getattr(mod, "Trainer", None)
		if tr is None:
			return

		def factory(orig):
			def log(self_, logs, *args, **kwargs):
				with suppress(Exception):
					rec.trainer_logs.append(
						{k: float(v) for k, v in dict(logs).items() if isinstance(v, int | float)}
					)
				return orig(self_, logs, *args, **kwargs)

			return log

		self._patch(tr, "log", factory)

		def plan_factory(orig):
			# Trainer builds its LR schedule once with the run's total optimizer steps (epochs x loader
			# batches / accumulation, or args.max_steps), before the first step: the real run's length.
			# create_scheduler is on every path (create_optimizer_and_scheduler, DeepSpeed, FSDP)
			def create(self_, num_training_steps, *args, **kwargs):
				with suppress(Exception):
					rec.run_length = {
						"planned_steps": int(num_training_steps),
						"epochs": float(self_.args.num_train_epochs) if self_.args.max_steps <= 0 else None,
					}
				return orig(self_, num_training_steps, *args, **kwargs)

			return create

		if hasattr(tr, "create_scheduler"):
			self._patch(tr, "create_scheduler", plan_factory)

	def _hook_lightning(self, mod):
		"""Lightning's Trainer: the run's length, as HF's create_scheduler gives it (max_steps, or
		max_epochs for the loader's batches to multiply)."""
		rec = self
		tr = getattr(mod, "Trainer", None)
		if tr is None:
			return

		def factory(orig):
			def fit(self_, *args, **kwargs):
				with suppress(Exception):
					steps, epochs = getattr(self_, "max_steps", -1), getattr(self_, "max_epochs", None)
					# Lightning's default with neither set is 1000 epochs, and an early stop or a time
					# limit ends the run before its plan: no planned length then, rather than a wrong one
					stops_early = any(
						k.__name__ in ("EarlyStopping", "Timer")
						for c in getattr(self_, "callbacks", [])
						for k in type(c).__mro__
					)  # fmt: skip (a subclass counts)
					if stops_early:
						rec.run_length = {}  # the plan is a ceiling, not the run's length
					elif steps and steps > 0:
						rec.run_length = {"planned_steps": int(steps), "epochs": None}
					elif epochs and epochs > 0 and epochs != 1000:
						rec.run_length = {"planned_steps": None, "epochs": float(epochs)}
				return orig(self_, *args, **kwargs)

			return fit

		self._patch(tr, "fit", factory)

	def _hook_grpo(self, mod):
		"""TRL's GRPOTrainer: the reference model's log-probs (computed with no gradient) and the
		reward functions, timed as their own phases; each reward call's values kept for pass rate
		per prompt. Generation is already timed by the generate hooks (rollout)."""
		rec = self
		tr = getattr(mod, "GRPOTrainer", None)
		if tr is None:
			return

		inside: set = set()  # kinds being timed now, so nested seams don't count twice

		def timed(kind, fn):
			@functools.wraps(fn)  # transformers reads forward's signature (logits_to_keep)
			def run(*args, **kwargs):
				if kind in inside:
					return fn(*args, **kwargs)
				inside.add(kind)
				rec._sync()
				t0 = rec._now()
				try:
					return fn(*args, **kwargs)
				finally:
					rec._sync()
					rec.acc[kind] += rec._now() - t0
					inside.discard(kind)

			return run

		def no_grad_call():
			return not torch.is_grad_enabled() or torch.is_inference_mode_enabled()

		def logps_factory(orig):
			# later TRL: a method, called with no gradient for the reference; the policy's old
			# log-probs (num_iterations > 1) go through it too, with the policy as the model
			def logps(self_, *args, **kwargs):
				if not no_grad_call():
					out = orig(self_, *args, **kwargs)  # the policy's own forward: part of the update
					rec._note_entropy(out, args[2] if len(args) > 2 else kwargs.get("attention_mask"))
					return out
				ref = getattr(self_, "ref_model", None)
				model = args[0] if args else kwargs.get("model")
				if ref is not None:
					is_ref = model is ref
				else:  # PEFT: the policy with its adapter off; only when there is a KL term at all
					is_ref = (
						float(getattr(self_, "beta", 0) or 0) > 0
						and int(getattr(self_, "num_iterations", 1) or 1) == 1
					)
				if not is_ref:
					return orig(self_, *args, **kwargs)
				return timed("reference", orig)(self_, *args, **kwargs)

			return logps

		if hasattr(tr, "_get_per_token_logps"):
			self._patch(tr, "_get_per_token_logps", logps_factory)
		if hasattr(tr, "_move_model_to_vllm"):  # later TRL: the weight sync is a method
			self._patch(tr, "_move_model_to_vllm", lambda orig: timed("weight_sync", orig))

		def policy_forward(fn):
			# TRL 0.14 with PEFT has no ref_model: the reference is the policy with its adapter
			# off, under inference_mode. Generation runs with no_grad, not inference_mode.
			@functools.wraps(fn)
			def forward(*args, **kwargs):
				if torch.is_inference_mode_enabled() and not rec._generating:
					return timed("reference", fn)(*args, **kwargs)
				return fn(*args, **kwargs)

			return forward

		def wrap_reward(i, fn):
			@functools.wraps(fn)  # TRL logs each reward under its function's __name__
			def reward(*args, **kwargs):
				out = timed("reward", fn)(*args, **kwargs)
				with suppress(Exception):
					vals = [float(v) for v in out][:4096]
					if len(rec.reward_calls) < 512:
						rec.reward_calls.append({"func": i, "values": vals})
				return out

			return reward

		def init_factory(orig):
			def init(self_, *args, **kwargs):
				orig(self_, *args, **kwargs)
				with suppress(Exception):
					rec.group_size = int(getattr(self_, "num_generations", 0)) or None
				with suppress(Exception):
					# TRL 0.14: the reference log-probs are a closure inside compute_loss, so the
					# reference model's own forward is the seam (or the policy's, with PEFT)
					ref = getattr(self_, "ref_model", None)
					if isinstance(ref, nn.Module):
						ref.forward = timed("reference", ref.forward)
					elif isinstance(getattr(self_, "model", None), nn.Module):
						self_.model.forward = policy_forward(self_.model.forward)
				with suppress(Exception):
					# TRL 0.14 (no _get_per_token_logps): the policy's own forward in compute_loss returns the
					# completion's logits, so the exact entropy is read there (A's item 14; PM, Nov 1)
					if not hasattr(tr, "_get_per_token_logps") and isinstance(
						getattr(self_, "model", None), nn.Module
					):
						eos = getattr(getattr(self_, "processing_class", None), "eos_token_id", None)
						self_.model.register_forward_hook(rec._entropy_hook(eos), with_kwargs=True)
				with suppress(Exception):
					# TRL 0.14 with use_vllm: the sync is inline in compute_loss, through the vLLM
					# model's load_weights (main process only)
					llm = getattr(self_, "llm", None)
					if llm is not None and not hasattr(tr, "_move_model_to_vllm"):
						m = llm.llm_engine.model_executor.driver_worker.model_runner.model
						m.load_weights = timed("weight_sync", m.load_weights)
				with suppress(Exception):
					funcs = list(self_.reward_funcs)
					for i, fn in enumerate(funcs):
						if isinstance(fn, nn.Module):
							fn.forward = timed("reward", fn.forward)  # a reward model: timed, not read
						elif callable(fn):
							funcs[i] = wrap_reward(i, fn)
					self_.reward_funcs = funcs

			return init

		self._patch(tr, "__init__", init_factory)

	def _hook_training_args(self, mod):
		ta = getattr(mod, "TrainingArguments", None)
		if ta is None:
			return

		def factory(orig):
			def post_init(self_, *args, **kwargs):
				# probes log every step and never phone home (wandb, comet, hub)
				self_.logging_steps = 1
				with suppress(Exception):
					self_.logging_strategy = "steps"
				self_.report_to = "none"
				self_.push_to_hub = False
				self_.disable_tqdm = True
				return orig(self_, *args, **kwargs)

			return post_init

		self._patch(ta, "__post_init__", factory)

	def _sample_loop(self):
		try:
			import psutil

			proc = psutil.Process()
		except Exception:
			proc = None
		read_power = gpu_power_reader() if self.cuda else None
		while not self._sampler_stop.wait(SAMPLE_EVERY_S):
			if proc is not None:
				with suppress(Exception):
					rss = proc.memory_info().rss + sum(
						c.memory_info().rss for c in proc.children(recursive=True)
					)
					self._rss_peak = max(self._rss_peak, rss)
			if read_power is not None:
				with suppress(Exception):
					w = read_power()
					if w is not None and math.isfinite(w) and 0 < w < 5_000:
						self._power.append(w)

	# ------------------------------------------------------------- model discovery (once)

	def _discover(self, opt):
		t0 = self._now()
		opt_params = {id(p) for g in opt.param_groups for p in g["params"]}
		mods = [o for o in gc.get_objects() if isinstance(o, nn.Module)]
		owners = [m for m in mods if any(id(p) in opt_params for p in m.parameters(recurse=True))]
		children = set()
		for m in owners:
			for c in m.modules():
				if c is not m:
					children.add(id(c))
		self._roots = [m for m in owners if id(m) not in children]
		params = [p for r in self._roots for p in r.parameters()]
		buffers = [b for r in self._roots for b in r.buffers()]
		self._weights = _unique_bytes(params + buffers)
		uniq = {id(p): p for p in params}
		self._params_total = sum(p.numel() for p in uniq.values())
		self._params_trainable = sum(p.numel() for p in uniq.values() if p.requires_grad)
		head, width = None, 0
		for r in self._roots:
			for m in r.modules():
				w = getattr(m, "weight", None)
				if (
					isinstance(m, nn.Linear)
					and isinstance(w, torch.Tensor)
					and w.dim() == 2
					and w.shape[0] > width
				):
					head, width = m, w.shape[0]
		if head is not None and width >= 256:

			def hook(_m, _inp, out):
				if not isinstance(out, torch.Tensor):
					return
				if self.eval_active:
					if self.evals and self.evals[-1]["grad_enabled"] is None:
						self.evals[-1]["grad_enabled"] = bool(torch.is_grad_enabled() and out.requires_grad)
					return
				t0 = self._now()
				self._head_out["bytes"] = max(self._head_out["bytes"] or 0, _bytes(out))
				self._head_out["dtype"] = out.dtype
				with torch.no_grad():
					norm = torch.linalg.vector_norm(out.detach(), dim=-1).float().mean()
					self._head_out["norm"] = norm if self._clean else float(norm)
				self.acc["overhead"] += self._now() - t0

			self._head_handles.append(head.register_forward_hook(hook))
		trainable = [p for p in uniq.values() if p.requires_grad and p.numel() <= 4_000_000]
		self._sample_params = sorted(trainable, key=lambda p: -p.numel())[:4]
		self.acc["overhead"] += self._now() - t0

	# ------------------------------------------------------------- optimizer hooks

	def _pre(self, opt, args, kwargs):
		with suppress(TypeError):
			self._all_opts.add(opt)  # every optimizer, for what a checkpoint holds (a GAN's G and D)
		if self.opt is None:
			self.opt = opt
			self._discover(opt)
		if opt is not self.opt:
			if self.lr_range:
				self._range_note_other(opt)
			return
		self._maybe_stop()
		if self.lr_range:
			self._set_range_lr(opt)
		self._sync()
		self._t_pre = self._now()
		if self._data_end_bwd:
			# between the last backward and the step is not forward time; a window a fetch started
			# stays open, for step(closure) loops (Lightning, LBFGS) that run forward inside the step
			self._data_end = None
		self._acc_at_pre = (self.acc["forward"], self.acc["backward"], self.acc["data"],
			self.acc["generate"] + self.acc["reference"] + self.acc["reward"] + self.acc["weight_sync"])  # fmt: skip
		# update-to-weight needs the weights before the step; a small sample, and not on timed steps
		sampled = self._range_sample(opt) if self.lr_range else self._sample_params
		self._sample = [] if self._clean else [(p, p.detach().clone()) for p in sampled]
		self._sync()
		self._overhead_pre = self._now() - self._t_pre

	def _range_stall_check(self):
		"""After a backward in a range test: has the ramp's optimizer stopped stepping? Two causes
		are read, nothing else (a script may run many backwards per step on purpose): a GradScaler
		skipping every step (fp16 overflow at a high rate never calls step(), so the ramp would wait
		for the timeout), and the script stepping another optimizer over the same weights (it
		rebuilt its optimizer: the ramp no longer sets the rate). Either stops the probe."""
		most = max((s["micro"] for s in self.steps), default=0)
		limit = max(RANGE_STALL_MICRO, RANGE_STALL_FACTOR * most) if self.steps else RANGE_STALL_FIRST
		if self.acc["micro"] < limit:
			return
		if self._range_overlap_steps:
			self.lr_range_replaced = True
			self._stop_pending = True
		elif list(self._scalers):
			self.lr_range_stalled = len(self.steps)
			self._stop_pending = True

	def _range_sample(self, opt) -> list:
		"""The weights whose update the range test reads (probes/lr_range.py `follows_rate`): from the
		optimizer whose rate it sets, not the model's (another optimizer's weights move at that one's
		rate). The largest up to RANGE_SAMPLE_MAX elements, else the smallest; never one whose
		values are all zero at the start (LoRA's B), whose update the review found grows with the
		weight it has built, not the rate."""
		if self._range_params is None:
			self._range_params = []
			with suppress(Exception):
				ps = {id(p): p for g in opt.param_groups for p in g["params"] if p.requires_grad}
				live = [p for p in ps.values() if bool(p.detach().abs().amax() > 0)]
				small = [p for p in live if p.numel() <= RANGE_SAMPLE_MAX]
				if small:
					pick = sorted(small, key=lambda p: -p.numel())
				else:
					pick = sorted(live, key=lambda p: p.numel())
				self._range_params = pick[:RANGE_SAMPLE]
		return self._range_params

	def _range_note_other(self, opt):
		"""Another optimizer stepped during a range test, noted as it steps (the script's optimizers
		are gone by the summary). One holding a weight (an `nn.Parameter`) the ramp's optimizer
		doesn't is another optimizer (Muon beside AdamW, a GAN's two, even with a shared encoder):
		those weights moved at the script's own rates, so the curve isn't the ramp's alone. One over
		plain tensors (an attack's perturbation) isn't counted. One over only the ramp's weights is a
		wrapper or a replacement: counted towards the stall rule, which tells them apart."""
		if self.opt is None:
			return
		with suppress(Exception):
			mine = {id(p) for g in self.opt.param_groups for p in g["params"]}
			theirs = [p for g in opt.param_groups for p in g["params"]]
			if any(isinstance(p, nn.Parameter) and id(p) not in mine for p in theirs):
				self._range_others.add(id(opt))
			elif theirs and all(id(p) in mine for p in theirs):
				self._range_overlap_steps += 1

	def range_other_optimizers(self) -> int:
		return len(self._range_others)

	def _set_range_lr(self, opt):
		"""The range test's rate for this step on every param group, each group keeping its ratio to
		the largest rate the script gave (a LoRA+ or layer-wise setup keeps its shape). The ratio is
		read from `initial_lr` when a scheduler set it: a warm-up from 0 makes every group's first
		rate 0 (A's review, Nov 2)."""
		from preflight.probes.lr_range import lr_at

		def rate(g) -> float:
			for key in ("initial_lr", "lr"):
				with suppress(Exception):
					v = float(g.get(key))
					if v > 0:
						return v
			return 0.0

		try:
			groups = opt.param_groups
			scales = self._lr_scales
			if scales is None:
				base = [rate(g) for g in groups]
				top = max(base, default=0.0)
				scales = [b / top if top > 0 else 1.0 for b in base]
				if top > 0:  # a hand-written warm-up from exactly 0: read the ratios at its next step
					self._lr_scales = scales
			lr = lr_at(len(self.steps), int(self.lr_range["steps"]), float(self.lr_range["start"]),
				float(self.lr_range["end"]))  # fmt: skip
			for i, g in enumerate(groups):
				v = lr * (scales[i] if i < len(scales) else 1.0)
				if torch.is_tensor(g.get("lr")):
					g["lr"].fill_(v)
				else:
					g["lr"] = v
		except Exception as exc:  # the test is extra: the probe runs on at the script's own rate
			self.lr_range_error = f"{type(exc).__name__}: {exc}"[:300]
			self.lr_range = None

	def _range_diverged(self, step: dict):
		"""Stop at the step the range test diverges (probes/lr_range.py `Divergence`, the rule its
		reading uses): past it the curve says nothing more, and some scripts stop themselves on a
		NaN. A loss that isn't finite reaches here as None with the step's non-finite count."""
		if self._lr_div is None:
			from preflight.probes.lr_range import Divergence

			self._lr_div = Divergence()
		if self._lr_div.add(step.get("loss"), bool(step.get("nonfinite"))):
			self._stop_pending = True

	def _grad_norm_of(self, opt):
		"""Global grad norm, as a 0-d tensor (no sync). Read after the step: with a closure (Lightning,
		LBFGS) the gradients only exist once optimizer.step() has run the closure."""
		grads = [p.grad for g in opt.param_groups for p in g["params"] if p.grad is not None]
		if not grads:
			return None, 0
		by_dev: dict = {}
		for g in grads:
			by_dev.setdefault((g.device, g.dtype), []).append(g.detach())
		# per-tensor norms in each grad's own dtype, then one small fp32 reduction: no full-size copies
		norms = [n.float() for group in by_dev.values() for n in torch._foreach_norm(group)]
		total = torch.linalg.vector_norm(torch.stack([n.to(norms[0].device) for n in norms]))
		return total, _unique_bytes(grads)

	def _post(self, opt, args, kwargs):
		self._any_opt_steps += 1  # any optimizer: training between two generations (_identical_outputs)
		if opt is not self.opt:
			return
		clean, kind = self._clean, self._kind()
		self._sync()
		t_post = self._now()
		event = None
		if self.cuda and torch.cuda.is_initialized():
			event = torch.cuda.Event(enable_timing=True)
			event.record()  # the step boundary, timed on the device: no host sync needed
		t0 = t_post
		grad_norm, grad_bytes = self._grad_norm_of(opt)
		if grad_norm is not None and not clean:
			grad_norm = float(grad_norm)
		utw = upd = None
		if self._sample:
			ratios, sizes = [], []
			for p, before in self._sample:
				d = float(torch.linalg.vector_norm((p.detach() - before).float()))
				sizes.append(d)
				w = torch.linalg.vector_norm(before.float())
				if w > 0:
					ratios.append(d / float(w))
			utw = statistics.fmean(ratios) if ratios else None
			# the range test reads the update's own size: a ratio to a weight the updates built goes flat
			upd = sum(sizes) if self.lr_range and sizes else None
			self._sample = []
		if self.lr_range:
			self._range_overlap_steps = 0
		opt_state = _unique_bytes(
			t for s in opt.state.values() for t in s.values() if isinstance(t, torch.Tensor)
		)
		peak = reserved = used = reserved_now = None
		if self.cuda and torch.cuda.is_initialized():
			peak = max(torch.cuda.max_memory_allocated(), self._eval_peak_stash)
			reserved = torch.cuda.max_memory_reserved()
			with suppress(Exception):
				# B (Nov 1): the allocator's reserve read at the same instant as the whole device, so
				# what isn't PyTorch's (NCCL, cuBLAS workspaces, the CUDA context) is their difference
				reserved_now = torch.cuda.memory_reserved()
				free, total = torch.cuda.mem_get_info()
				used = total - free  # the whole device: PyTorch, NCCL, cuBLAS workspaces, other processes
			torch.cuda.reset_peak_memory_stats()
			self._eval_peak_stash = 0
		scale = None
		for s in list(self._scalers):
			with suppress(Exception):
				# timed steps keep the scale tensor and read it after the probe (get_scale() syncs)
				sc = getattr(s, "_scale", None)
				scale = (sc.detach().clone() if sc is not None else None) if clean else float(s.get_scale())
		lr = opt.param_groups[0].get("lr")
		overhead = self._overhead_pre + (self._now() - t0) + self.acc["overhead"]
		f0, b0, d0, g0 = self._acc_at_pre
		a = self.acc
		rl_inner = a["generate"] + a["reference"] + a["reward"] + a["weight_sync"]
		inner = (a["forward"] - f0) + (a["backward"] - b0) + (a["data"] - d0) + (rl_inner - g0)
		opt_time = max(0.0, (t_post - self._t_pre) - self._overhead_pre - inner)
		start = self._last_post if self._last_post is not None else (self._first_fetch or self._t_start)
		wall = max(1e-9, t_post - start - overhead - a["io"] - a["eval"])
		self.steps.append(
			{
				"kind": kind,
				"wall": wall,
				"event": event,
				"data": a["data"],
				"forward": max(0.0, a["forward"] - rl_inner),
				"rollout": a["generate"],
				"reference": a["reference"],
				"reward": a["reward"],
				"weight_sync": a["weight_sync"],
				"backward": a["backward"],
				"optimizer": opt_time,
				"loss": a["loss"] if a["loss_n"] else None,
				"loss_t": a["loss_t"],
				"scales_t": a["scales_t"],
				"micro": a["micro"],
				"tokens": a["tokens"],
				"real": a["real"],
				"samples": a["samples"],
				"padded_len": a["padded_len"] or None,
				"nonfinite": a["nonfinite"],
				"grad_norm": grad_norm,
				"utw": utw,
				"range_upd": upd,
				"grad_bytes": grad_bytes,
				"opt_bytes": opt_state,
				"peak": peak,
				"reserved": reserved,
				"device_used": used,
				"reserved_now": reserved_now,
				"lr": float(lr)
				if isinstance(lr, int | float) or (torch.is_tensor(lr) and not clean)
				else None,
				"logit_norm": self._head_out["norm"],
				"attn_max": a["attn_max"],
				"loss_scale": scale,
				"io": a["io"],
				"io_bytes": a["io_bytes"],
				"eval_s": a["eval"],
				"profiled": self._prof is not None,
			}
		)
		self._reset_acc()
		n = len(self.steps)
		if kind == "timed" and n == self.warmup + self.timed and not self.alternate_sync:
			# drain what the timed steps queued, so the first phase step doesn't carry it as "other"
			self._hard_sync()
		if self.lr_range:
			self._range_diverged(self.steps[-1])
		self.stage = f"step {n}"
		self._progress()
		self._last_post = self._now()
		if self._prof is not None:
			self._prof.step()
			self._prof_steps += 1
			if self._prof_steps >= PROFILE_STEPS:
				self._prof.stop()
				self._stop_pending = True
		elif n >= self.target:
			if self.profile and self.trace_path:
				self._start_profiler()
			else:
				self._stop_pending = True

	def _resolve_deferred(self):
		"""After the probe: read what timed steps left on the device, and time them from their events."""
		if self._resolved:
			return
		self._resolved = True
		if self.cuda and torch.cuda.is_initialized():
			with suppress(Exception):
				torch.cuda.synchronize()
		prev = None
		for st in self.steps:
			ev = st.pop("event", None)
			timed_by_events = st["kind"] == "timed" or (self.alternate_sync and st["kind"] == "phase")
			if timed_by_events and ev is not None and prev is not None:
				with suppress(Exception):
					# device time between boundaries includes host stalls (the GPU idles while the
					# loader works), so it is the step's wall time without any sync inside the step
					st["wall"] = max(1e-9, prev.elapsed_time(ev) / 1000.0 - st["io"] - st["eval_s"])
			prev = ev
			losses = st.pop("loss_t", [])
			scales = st.pop("scales_t", [])
			if losses:
				vals = []
				for lt, sc in zip(losses, scales, strict=False):
					v = self._scalar(lt)
					for x in sc:
						xv = self._scalar(x)
						if v is not None and xv:
							v /= xv
					vals.append(v)
				good = [
					v for v in vals if v is not None and v == v and v not in (float("inf"), float("-inf"))
				]
				st["nonfinite"] += len(vals) - len(good)
				st["loss"] = sum(good) if good else None
			for key in ("grad_norm", "logit_norm", "loss_scale"):
				st[key] = self._scalar(st[key])

	def _start_profiler(self):
		from torch.profiler import ProfilerActivity, profile, schedule

		acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if self.cuda else [])
		path = self.trace_path

		rec = self

		def ready(p):
			p.export_chrome_trace(path)
			try:
				rec._kernels = rec._kernel_summary(p)
			except Exception as exc:  # the table is a bonus; never fail a probe over it
				rec._kernels = {"error": f"{type(exc).__name__}: {exc}"[:300]}

		# shapes and FLOPs feed B's per-kernel table; this window is never timed, so the overhead is free
		self._prof = profile(
			activities=acts,
			schedule=schedule(wait=1, warmup=1, active=PROFILE_ACTIVE),
			on_trace_ready=ready,
			record_shapes=True,
			with_flops=True,
			with_stack=False,
		)
		self._prof.start()

	def _kernel_summary(self, prof) -> dict:
		from .kernels import kernel_table

		dtype = {"bfloat16": "bf16", "float16": "fp16"}.get(
			str(self._head_out["dtype"]).replace("torch.", ""), "fp32"
		)
		device = ("rocm" if getattr(torch.version, "hip", None) else "cuda") if self.cuda else "cpu"
		tf32 = bool(
			torch.get_float32_matmul_precision() != "highest" or torch.backends.cuda.matmul.allow_tf32
		)
		return kernel_table(prof, steps=PROFILE_ACTIVE, device=device, dtype=dtype, allow_tf32=tf32)

	# ------------------------------------------------------------- results

	def summary(self) -> dict:
		if self.eval_active:
			self._end_eval()
		self._resolve_deferred()
		timed = [s for s in self.steps if s["kind"] == "timed"]
		steps = list(self.steps)
		# the phase split: synchronised phase steps on a GPU; on CPU nothing runs asynchronously, so the
		# timed steps' own phase timings are exact
		phased = [s for s in steps if s["kind"] == "phase"] if self.cuda else timed
		if self.cuda and not phased:
			phased = [s for s in steps if s["kind"] == "warmup"][1:]
		out: dict = {"steps": len(steps), "num_workers": self.num_workers, "stage": self.stage}
		out["train_loader_batches"] = self.train_loader_batches
		# the dataset the training loader draws from: examples, pairs, images, or a NodeLoader's input
		# nodes per epoch, which B's engine divides by the batch for the run's length (D, Nov 1)
		n = None
		for src in (self._train_sampler, self._train_dataset):  # the sampler's draw, else the dataset
			with suppress(Exception):
				n = len(src) if src is not None else None
			if isinstance(n, int) and n > 0:
				out["train_dataset_len"] = float(n)
				break
		out.update(self._identical_outputs())
		known = [t for g in self.gens for t in g["truncated"] if t is not None]
		if self.gens and known:  # how many sequences truncation_rate is over (those whose ending is known)
			out["generation_ending_known"] = float(len(known))
		probe_caps = [g["probe_cap"] for g in self.gens if g.get("probe_cap")]
		if probe_caps:  # the serving probe stopped each reply at this many tokens, not the script's cap
			out["serving_probe_max_tokens"] = float(max(probe_caps))
			capped = [g for g in self.gens if g.get("probe_cap")]
			n = sum(len(g["truncated"]) for g in capped)
			# the share of replies still running at the cap, for B's extrapolation past it
			cut = sum(1 for g in capped for t in g["truncated"] if t is None)
			if n:
				out["serving_probe_capped_fraction"] = cut / n
		if self.checkpoints.get("saves"):
			ck = self.checkpoints
			out["checkpoint_saves_seen"] = ck["saves"]
			out.update(
				{
					f"checkpoint_has_{k}": float(bool(ck.get(k)))
					for k in ("optimizer", "scheduler", "rng", "step")
				}
			)
		for kind, st in self.retrieval.items():
			out[f"{kind}_items"] = st["items"]
			out[f"{kind}_seconds"] = st["seconds"]
			if st["seconds"] > 0 and st["items"]:
				out[f"{kind}_items_per_s"] = st["items"] / st["seconds"]
		if self.lens_stats:
			# the mean over the batches seen, and the largest graph batch (the memory one)
			keys = {k for st in self.lens_stats for k in st}
			out.update({k: statistics.fmean(st[k] for st in self.lens_stats if k in st) for k in keys})
			if any("graph_nodes_per_batch" in st for st in self.lens_stats):
				out["graph_nodes_max"] = max(st.get("graph_nodes_per_batch", 0) for st in self.lens_stats)
			if "images_per_batch" in out:
				# the model's layout, from its conv weights (scripts convert the batch after the fetch)
				with suppress(Exception):
					convs = [
						p for r in self._roots for p in r.parameters() if p.dim() == 4 and p.shape[-1] > 1
					]
					if convs:
						# every weight channels-last; at least one not also plain-contiguous (a depthwise
						# (C, 1, k, k) weight is both, so it can't tell)
						out["channels_last"] = float(all(p.is_contiguous(memory_format=torch.channels_last) for p in convs)
							and any(not p.is_contiguous() for p in convs))  # fmt: skip
		out.update({k: v for k, v in self.run_length.items() if v is not None})
		out["host_rss_peak_bytes"] = self._rss_peak or None
		out["train_peak_before_eval_bytes"] = self._train_peak_before_eval or None
		if self._kernels is not None:
			out["kernels"] = self._kernels
		if self._roots:
			try:
				out["arch"] = model_shape(self._roots, sdpa_seen=self._sdpa_seen)
			except Exception as exc:  # shape is a nice-to-have; never fail the probe over it
				out["arch"] = {"error": f"{type(exc).__name__}: {exc}"[:300]}
		if self._power:
			out["power"] = {
				"mean_w": statistics.fmean(self._power),
				"max_w": max(self._power),
				"samples": len(self._power),
			}
		out.update(self._eval_summary())
		out.update(self._gen_summary())
		out.update(self._rl_summary())
		saves = [s for s in steps if s["io"] > 0]
		if saves:
			out["checkpoint"] = {
				"saves": len(saves),
				"seconds_mean": statistics.fmean(s["io"] for s in saves),
				"bytes_mean": int(statistics.fmean(s["io_bytes"] for s in saves)),
			}
		if not steps:
			return out
		if timed:
			walls = sorted(s["wall"] for s in timed)
			mean = statistics.fmean(walls)
			keys = ["data", "forward", "backward", "optimizer"] + [
				k for k in ("rollout", "reference", "reward", "weight_sync") if any(s.get(k) for s in phased)
			]
			by_phase = {}
			if phased:
				ref = statistics.fmean(s["wall"] for s in phased)
				phases = {k: statistics.fmean(s[k] for s in phased) for k in keys}
				phases["other"] = max(0.0, ref - sum(phases.values()))
				total = sum(phases.values()) or 1.0
				by_phase = {k: v / total for k, v in phases.items()}
			out["step_time"] = {
				"n_steps": len(timed),
				"mean_s": mean,
				"p50_s": statistics.median(walls),
				"p95_s": _quantile(walls, 0.95),
				"first_step_s": steps[0]["wall"],
				"by_phase": by_phase,
				"unprofiled": True,
			}
			tokens = statistics.fmean(s["tokens"] for s in timed)
			samples = statistics.fmean(s["samples"] for s in timed)
			out["throughput"] = {
				"tokens_per_s": tokens / mean if tokens else None,
				"samples_per_s": samples / mean if samples else None,
			}
			out["tokens_per_step"] = tokens
			out["samples_per_step"] = samples
			if by_phase:
				out["loader_wait_fraction"] = min(1.0, by_phase.get("data", 0.0))
		real = sum(s["real"] for s in steps)
		tok = sum(s["tokens"] for s in steps)
		samples_total = sum(s["samples"] for s in steps)
		out["padding_fraction"] = 1 - real / tok if tok else None
		out["tokens_per_sample_mean"] = real / samples_total if tok and samples_total else None
		gn = [s["grad_norm"] for s in steps]
		ma, run = [], []
		for v in gn:
			if v is not None and v == v:
				run.append(v)
			ma.append(statistics.fmean(run) if run else None)
		series = {
			"loss": [s["loss"] for s in steps],
			"grad_norm": [v if v is None or v == v else None for v in gn],
			"grad_norm_ma": ma,
			"update_to_weight": [s["utw"] for s in steps],
			"output_logit_norm": [s["logit_norm"] for s in steps],
			"lr": [s["lr"] for s in steps],
		}
		for key, src in (("max_attn_logit", "attn_max"), ("loss_scale", "loss_scale")):
			if any(s[src] is not None for s in steps):
				series[key] = [s[src] for s in steps]
		out["series"] = series
		out["nan_count"] = sum(s["nonfinite"] for s in steps) + sum(1 for v in gn if v is not None and v != v)
		scales = [s["loss_scale"] for s in steps if s["loss_scale"] is not None]
		out["loss_scale_events"] = sum(1 for a, b in itertools.pairwise(scales) if b < a)
		peaks = [s["peak"] for s in steps if s["peak"] is not None]
		out["memory"] = {
			"weights": self._weights,
			"gradients": max(s["grad_bytes"] for s in steps),
			"optimizer": max(s["opt_bytes"] for s in steps),
			"logits": self._head_out["bytes"],
			"peak": max(peaks) if peaks else None,
			"reserved": max((s["reserved"] for s in steps if s["reserved"] is not None), default=None),
			"peak_step": (1 + max(range(len(steps)), key=lambda i: steps[i]["peak"] or 0)) if peaks else None,
		}
		out["step_log"] = [
			{
				"index": i,
				"kind": st["kind"],
				"wall_s": st["wall"],
				"tokens": st["tokens"] or None,
				"real_tokens": st["real"] or None,
				"samples": st["samples"] or None,
				"padded_len": st["padded_len"],
				"micro_steps": st["micro"],
				"peak_allocated_bytes": st["peak"],
				"peak_reserved_bytes": st["reserved"],
				"device_used_bytes": st["device_used"],
				# until the schema has the field (asked of the PM), the step log leaves it out
				**({"reserved_bytes": st.get("reserved_now")} if _STEP_HAS_RESERVED else {}),
			}
			for i, st in enumerate(steps, 1)
		][:256]
		out["params_total"] = self._params_total
		out["params_trainable"] = self._params_trainable
		out["micro_steps_per_step"] = round(statistics.fmean(s["micro"] for s in steps), 3)
		out["head_dtype"] = (
			str(self._head_out["dtype"]).replace("torch.", "") if self._head_out["dtype"] else None
		)
		out["tf32"] = bool(
			torch.get_float32_matmul_precision() != "highest" or torch.backends.cuda.matmul.allow_tf32
		)
		return out

	def _eval_summary(self) -> dict:
		done = [e for e in self.evals if e.get("batches")]
		if not done:
			return {}
		e = done[0]
		peaks = [x["peak"] for x in done if x.get("peak") is not None]
		return {
			"eval": {
				"batches": e["batches"],
				"batch_size": e.get("batch_size"),
				"seconds_per_batch": statistics.fmean(x["seconds"] / x["batches"] for x in done),
				"peak_allocated_bytes": max(peaks) if peaks else None,
				"grad_enabled": e.get("grad_enabled"),
				"loader_batches": e.get("loader_batches"),
				"memory_by_batch": [int(x) for x in e.get("memory_by_batch", [])][:64],
			}
		}

	def _gen_summary(self) -> dict:
		if not self.gens:
			return {}
		lengths = [n for g in self.gens for n in g["lengths"]]
		trunc = [t for g in self.gens for t in g["truncated"] if t is not None]
		caps = [g["cap"] for g in self.gens if g["cap"]]
		return {
			"generation": {
				"calls": len(self.gens),
				"seconds_total": sum(g["seconds"] for g in self.gens),
				"sequences": len(lengths),
				"new_tokens_mean": statistics.fmean(lengths) if lengths else None,
				"new_tokens_p95": _quantile(lengths, 0.95) if lengths else None,
				"new_tokens_max": max(lengths) if lengths else None,
				"max_new_tokens": int(max(caps)) if caps else None,
				# over the sequences whose ending is known (a cap seen, or vLLM's finish_reason)
				"truncation_rate": sum(trunc) / len(trunc) if trunc else None,
				"backend": self.gens[-1]["backend"],
			}
		}

	def _gen_call_starts(self) -> None:
		"""An eval probe ends as a generate call starts once it has measured generate_calls of them:
		the script's work on the last batch is done and measured, the next batch isn't started."""
		if self.generate_calls and self._gen_calls_done >= self.generate_calls and not self._generating:
			raise StopProbe

	def _entropy_hook(self, eos_token_id):
		"""A forward hook for TRL 0.14's policy: on the update's forward (gradients on, not generating, with
		num_logits_to_keep), the mean entropy of the policy over the completion tokens TRL trains on (up to and
		including the first EOS, as its completion_mask), from the logits themselves: exact, not a sample. Read under
		no_grad in chunks of positions, so the extra memory is one chunk's fp32 softmax."""
		rec = self

		def hook(module, args, kwargs, output):
			if len(rec._entropy_exact) >= ENTROPY_MC_MAX or rec._generating or not torch.is_grad_enabled():
				return None
			keep = kwargs.get("num_logits_to_keep") or kwargs.get("logits_to_keep")
			ids = args[0] if args else kwargs.get("input_ids")
			logits = getattr(output, "logits", None)
			if not keep or ids is None or logits is None:
				return None
			with suppress(Exception), torch.no_grad():
				# TRL 0.14 asks for num_logits_to_keep = K + 1 and drops the last: the completion is K tokens (review, Nov 1)
				k = int(keep) - 1  # the hook is only on TRL 0.14's GRPO policy, which always asks for K + 1
				rec._entropy_exact.append((len(rec.steps), completion_entropy(logits, ids, k, eos_token_id)))
			return None

		return hook

	def _note_entropy(self, logps, attention_mask) -> None:
		"""The policy's mean -log p over the completion tokens it was trained on: on sampled tokens a
		Monte Carlo estimate of its entropy, biased by the sampling temperature (TRL's default 0.9),
		top-p/top-k and the EOS cut. Kept on the device, so the step gains no sync; read at the end.
		Only TRL versions with a _get_per_token_logps method reach it: in TRL 0.14 (the probe image's)
		that's a closure inside compute_loss, so a 0.14 run has no entropy (known limit)."""
		if len(self._entropy_mc) >= ENTROPY_MC_MAX:
			return
		with suppress(Exception), torch.no_grad():
			lp = logps.detach()
			mask = (
				attention_mask[:, -lp.shape[1] :].to(lp.dtype)
				if attention_mask is not None
				else torch.ones_like(lp)
			)
			self._entropy_mc.append((len(self.steps), -(lp * mask).sum() / mask.sum().clamp(min=1)))

	def _rl_summary(self) -> dict:
		logs = self.trainer_logs
		gen = self._gen_summary().get("generation")

		def col(kind):
			for key in TRL_KEYS[kind]:
				vals = [entry[key] for entry in logs if key in entry]
				if vals:
					return vals
			return []

		rl = {}
		if col("truncation"):
			rl["truncation_rate"] = statistics.fmean(col("truncation"))
		elif gen and gen["truncation_rate"] is not None:
			rl["truncation_rate"] = gen["truncation_rate"]
		if col("zero_spread"):
			rl["zero_spread_fraction"] = statistics.fmean(col("zero_spread"))
		for name, kind in (
			("kl", "kl"),
			("entropy", "entropy"),
			("clip_fraction", "clip"),
			("reward_mean", "reward"),
		):
			if col(kind):
				rl[name] = col(kind)
		if "entropy" not in rl and self._entropy_exact:
			# TRL 0.14 logs no entropy: the policy's own, from its logits on the update's forward
			by_step_x: dict[int, list[float]] = {}
			with suppress(Exception):
				for step, t in self._entropy_exact:
					v = float(t)
					if math.isfinite(v):
						by_step_x.setdefault(step, []).append(v)
			if by_step_x:
				rl["entropy"] = [statistics.fmean(v) for _, v in sorted(by_step_x.items())]
				self.rl_entropy_exact = True
		if "entropy" not in rl and self._entropy_mc:
			# TRL logged no entropy: the estimate from the policy's own log-probs, one per optimizer step
			by_step: dict[int, list[float]] = {}
			with suppress(Exception):
				for step, t in self._entropy_mc:
					v = float(t)
					if math.isfinite(v):
						by_step.setdefault(step, []).append(v)
			if by_step:
				rl["entropy"] = [statistics.fmean(v) for _, v in sorted(by_step.items())]
				self.rl_entropy_estimated = True
		groups = reward_groups(self.reward_calls, self.group_size)
		counts = groups.pop("_counts", {})
		rl.update(groups)
		if gen and gen["sequences"]:
			rl["response_length"] = {
				"mean": gen["new_tokens_mean"],
				"p95": gen["new_tokens_p95"],
				"max": gen["new_tokens_max"],
				"n": gen["sequences"],
			}
		elif col("len_mean"):
			m = col("len_mean")
			mx = col("len_max") or m
			rl["response_length"] = {"mean": statistics.fmean(m), "p95": max(mx), "max": max(mx), "n": len(m)}
		if self.rl_entropy_estimated:
			counts = {**counts, "rl_entropy_estimated": 1.0}  # rl.entropy is the Monte Carlo estimate
		if self.rl_entropy_exact:
			counts = {**counts, "rl_entropy_exact": 1.0}  # rl.entropy from the policy's logits (TRL 0.14)
		return (
			{"rl": rl} if rl else {}
		) | counts  # counts: top-level extras (RLSignals has no field for them)


def completion_entropy(logits, input_ids, keep: int, eos_token_id=None):
	"""Mean entropy (nats) of the next-token distributions over a batch's completion tokens, as a device scalar.

	`logits` is TRL 0.14's (B, keep + 1, V) from model(input_ids, num_logits_to_keep=keep + 1); the last position
	predicts past the sequence and is dropped, as TRL drops it. The completion is input_ids' last `keep` tokens, and
	only the tokens up to and including the first EOS count (TRL's completion_mask). Chunked over positions so the
	fp32 softmax scratch stays near ENTROPY_CHUNK_BYTES."""
	# position i predicts token i + 1: the K completion tokens are predicted by positions -K-1 .. -2 (review, Nov 1)
	z = logits[:, :-1, :] if logits.shape[1] == keep + 1 else logits[:, -keep - 1 : -1, :]
	b, k, v = z.shape
	comp = input_ids[:, -k:]
	if eos_token_id is not None:
		is_eos = comp == eos_token_id
		first = torch.where(is_eos.any(1), is_eos.int().argmax(1), torch.full((b,), k, device=comp.device))
		mask = (torch.arange(k, device=comp.device).unsqueeze(0) <= first.unsqueeze(1)).to(torch.float32)
	else:
		mask = torch.ones(b, k, device=comp.device)
	# three chunk-sized fp32 buffers live at once (the upcast, log p, p * log p in place): each a third of the budget
	chunk = max(1, int(ENTROPY_CHUNK_BYTES // max(1, 3 * b * v * 4)))
	total = torch.zeros((), device=z.device, dtype=torch.float32)
	for i in range(0, k, chunk):
		logp = torch.log_softmax(z[:, i : i + chunk, :].float(), dim=-1)
		h = -logp.exp().mul_(logp).sum(-1)  # (B, chunk)
		total = total + (h * mask[:, i : i + chunk]).sum()
	return total / mask.sum().clamp(min=1)


def detect_env() -> dict:
	import platform
	from importlib import metadata

	libs = {}
	for name in (
		"torch",
		"transformers",
		"peft",
		"trl",
		"accelerate",
		"lightning",
		"vllm",
		"datasets",
		"sentence-transformers",
		"torch-geometric",
		"timm",
		"deepspeed",
		"bitsandbytes",
		"flash-attn",
	):
		with suppress(metadata.PackageNotFoundError):
			libs[name] = metadata.version(name)
	env = {"libraries": libs, "python": platform.python_version()}
	if torch.cuda.is_available():
		props = torch.cuda.get_device_properties(0)
		hip = getattr(torch.version, "hip", None)
		name, unknown = canonical_gpu_name(
			props.name,
			props.total_memory,
			getattr(props, "multi_processor_count", None),
			arch=getattr(props, "gcnArchName", None),
		)
		env["gpu_device_name"] = props.name  # as torch reports it: MIG slices are looked up by it
		if unknown:
			env["gpu_unresolved"] = unknown  # not in B's table: kept as reported, filed for B
		env.update(
			backend="rocm" if hip else "cuda",
			gpu={
				"name": name,
				"count": torch.cuda.device_count(),
				"memory_bytes": props.total_memory,
				"vendor": "amd" if hip else "nvidia",
				"compute_capability": None if hip else f"{props.major}.{props.minor}",
			},
		)
	elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
		env.update(backend="mps", gpu={"name": "Apple MPS", "count": 1, "vendor": "apple"})
	else:
		env.update(backend="cpu", gpu=None)
	return env


def mfu(summary: dict, env: dict) -> tuple[float | None, float | None]:
	"""(flops per step, MFU). Transformer rule: forward 2N + activation-grad 2N + weight-grad 2N_trainable per token."""
	tokens = summary.get("tokens_per_step")
	n, nt = summary.get("params_total"), summary.get("params_trainable")
	st = summary.get("step_time")
	if not tokens or not n or not st:
		return None, None
	flops = tokens * (4 * n + 2 * (nt or 0))
	dtype = summary.get("head_dtype") or "float32"
	prec = {"bfloat16": "bf16", "float16": "fp16"}.get(dtype, "fp32")
	if prec == "fp32" and summary.get("tf32"):
		prec = "tf32"
	gpu = env.get("gpu") or {}
	top = peak(gpu.get("name"), prec, gpu.get("memory_bytes"), device_name=env.get("gpu_device_name"))
	if not top:
		return flops, None
	# the median step, as B projects run time from p50_s (probe-plan D1)
	util = flops / (st.get("p50_s") or st["mean_s"]) / top
	# above 1 means the precision guess was wrong (mixed kernels); say nothing rather than something false
	return flops, util if util <= 1.0 else None


def write_json(path: str | Path, data: dict) -> None:
	Path(path).write_text(json.dumps(data, default=str))
