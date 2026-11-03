"""B's prediction API: predict, build_grids, sensitivity (CONTRACT.md, "Prediction API").

Pure and Pyodide-safe: reads a FlightRecord, returns schema objects, no I/O. The
browser's PyodideProvider calls `predict` on every knob change, and `build_grids`
precomputes the sliders the GridProvider reads without Python.

What each lens gets:
- GPU lenses: memory.peak, risk.oom, time.step, time.total, cost.total, plus risk.wall_cap
  with a scheduler cap and risk.preemption_loss with a preemption rule.
- API lenses: cost.total, tokens.total, time.total, memory.context_fill, risk.runaway.

Every estimate carries a 90% interval or an honest abstention. The settings behind
each number are in docs/decisions/engine-settings.md; `fill_predictions` is the call
A's scripts/make_golden.py makes to replace the golden fixtures.
"""

from __future__ import annotations

import contextvars
import functools
import itertools
import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from preflight.engine import (
	agent_cost,
	conformal,
	divergence,
	gpus,
	knobs,
	memory,
	prices,
	rl,
	serving,
	steptime,
)
from preflight.engine import oom as oom_model
from preflight.schema.encoding import MAX_GRID_CELLS, choose_scale, encode_f16, encode_mask
from preflight.schema.prediction import (
	Estimate,
	EstimateRate,
	Grid,
	GridAxis,
	KnobImpact,
	Objective,
	Prediction,
	Predictions,
	Target,
)
from preflight.schema.record import CostBasis, CostLine, FlightRecord, JobCost, Knob, ModelCard, Range

LEVEL = 0.9  # cited: contract, 90% intervals
API_LENSES = frozenset({"agent"})
TRAIN_LENSES = frozenset({"finetune", "pretrain", "rl", "retrieval", "gnn", "vision"})
UNMODELLED_WIDEN = 1.5  # guess: band widening when a knob the model doesn't capture moves; replaced by corpus runs varying those knobs (target Oct 31)
OK_PROBE = ("ok", "cached")
MODEL_ID_TOTAL = "time-total@0.1.0"
MODEL_ID_COST = "cost-gpu-hours@0.1.0"
MODEL_ID_AGENT_TIME = "agent-time-concurrency@0.1.0"
MODEL_ID_CONTEXT = "agent-context-fill@0.1.0"
MODEL_ID_PREEMPT = "preemption-poisson@0.2.0"  # 0.2.0 (Oct 31): interruptions during redone work count
MODEL_ID_RUN_COST = "run-cost@0.1.0"

_TARGETS = {
	"memory.peak": ("bytes", "memory"),
	"risk.oom": ("probability", "risk"),
	"risk.divergence": ("probability", "risk"),
	"time.step": ("s", None),
	"time.total": ("s", "time"),
	"cost.total": ("usd", "cost"),
	# Chiya, Nov 1: the quantity the dollars come from, so the browser recomputes cost.total at a user's own rate with
	# no round trip; the interval is the quantity's, and a user's rate has none of its own
	"cost.gpu_hours": ("gpu_hours", "cost"),
	"risk.wall_cap": ("probability", "risk"),
	"risk.preemption_loss": ("s", "risk"),
	"tokens.total": ("tokens", None),
	"memory.context_fill": ("fraction", "memory"),
	"risk.runaway": ("probability", "risk"),
	"rl.zero_spread": ("fraction", "risk"),
	"rl.truncation": ("fraction", None),
	"time.rollout": ("s", None),
	"serving.throughput": ("tokens/s", "time"),
	"serving.ttft_p95": ("s", None),
	"serving.tpot": ("s", None),
	"serving.kv_tokens": ("tokens", "memory"),
}

# Features the caller switches on from its flags (the engine reads no files). Off by default.
# flags.json engine.rl: zero-spread groups, rollout time and truncation on the RL lens;
# engine.serving: throughput, time to first token, time per token and KV capacity on the serving lens
FEATURES = {
	"rl": False,
	"serving": False,
	"steptime_cross_gpu": True,  # Chiya, Oct 31: on, with the rough band shown past +-50%
	"divergence": False,  # flags.json engine.divergence: off until the sweep's held-out report meets its row
	"carry_free_knobs": False,  # flags.json engine.carry_free_knobs: after the freeze (Nov 2), off until checked
	"context_same_stack": False,  # flags.json engine.context_same_stack: after the freeze (Nov 2), off until checked
	"memory_reserved": False,  # flags.json engine.memory_reserved: Nov 3, off until the PM turns it on
	"eval_passes": False,  # flags.json engine.eval_passes: Nov 3, off until checked
}  # the defaults when the caller passes no flags (scripts, tests); keep them the same as flags.json
# the flags.json key for each feature: fill_predictions(record, flags=flag) reads them for that call
FLAG_OF = {
	"rl": "engine.rl",
	"serving": "engine.serving",
	"steptime_cross_gpu": "engine.steptime_cross_gpu",
	"divergence": "engine.divergence",
	"carry_free_knobs": "engine.carry_free_knobs",
	"context_same_stack": "engine.context_same_stack",
	"memory_reserved": "engine.memory_reserved",
	"eval_passes": "engine.eval_passes",
}
_FEATURES_NOW: contextvars.ContextVar[dict[str, bool] | None] = contextvars.ContextVar(
	"preflight_features", default=None
)


def feature(name: str) -> bool:
	"""Whether a feature is on: the caller's flags for this fill, else FEATURES."""
	now = _FEATURES_NOW.get()
	return (now if now is not None else FEATURES)[name]


def features_from(flag: Callable[[str], bool]) -> dict[str, bool]:
	"""Every feature from the caller's flag reader (a missing flag reads as off)."""
	return {name: bool(flag(key)) for name, key in FLAG_OF.items()}


RL_TARGETS = ("rl.zero_spread", "time.rollout", "rl.truncation")
SERVING_TARGETS = ("serving.throughput", "serving.ttft_p95", "serving.tpot", "serving.kv_tokens")


# ---------------------------------------------------------------- small helpers


MEMORY_TERMS = (
	"weights",
	"master_weights",
	"frozen_models",
	"context",
	"gradients",
	"optimizer",
	"comm_buffers",
	"activations",
	"logits",
	"kv_cache",
	"generation_engine",
	"eval_spike",
)  # fmt: skip  (the twelve memory.<term> ids C's ring groups)


def _meta(tid: str) -> tuple[str, str | None]:
	"""(unit, readout) for a target id, including the per-term and per-phase families."""
	if tid in _TARGETS:
		return _TARGETS[tid]
	if tid.startswith("memory.") and tid[7:] in MEMORY_TERMS:
		return "bytes", None
	if tid.startswith("time.phase."):
		return "s", None
	raise KeyError(f"unknown target {tid!r}")


def _target(tid: str) -> Target:
	unit, readout = _meta(tid)
	return Target(id=tid, label_key=f"target.{tid}", unit=unit, readout=readout)


def _abstain(unit: str, model_id: str, why: str) -> Estimate:
	return Estimate(
		point=None,
		lo=None,
		hi=None,
		unit=unit,
		status="abstain",
		model_id=model_id,
		source="exact",
		reason=why,
	)


def _est(point, lo, hi, unit, status, model_id, reason=None) -> Estimate:
	if point is None or not all(math.isfinite(v) for v in (point, lo, hi)):
		return _abstain(unit, model_id, reason or "the estimate isn't a usable number")
	lo, hi = min(lo, point), max(hi, point)
	return Estimate(
		point=point,
		lo=lo,
		hi=hi,
		unit=unit,
		status=status,
		model_id=model_id,
		source="exact",
		reason=reason or None,
	)


def _api_lens(record: FlightRecord) -> bool:
	lens = record.job.lens
	backend = record.env.backend if record.env else None
	return lens in API_LENSES or (lens in ("eval", "serving") and backend == "api")


def targets_for(record: FlightRecord) -> list[Target]:
	"""The targets this record's lens shows, in readout order."""
	if _api_lens(record):
		ids = ["cost.total", "time.total", "memory.context_fill", "risk.runaway", "tokens.total"]
	else:
		ids = ["memory.peak", "time.total", "cost.total", "cost.gpu_hours", "risk.oom", "time.step"]
		if _has_shape(record):  # the per-term split needs the analytical model
			ids += [f"memory.{t}" for t in MEMORY_TERMS]
		ids += [f"time.phase.{ph}" for ph in _probe_phases(record)]
		if feature("rl") and record.job.lens == "rl":
			ids += list(RL_TARGETS)
		if feature("divergence") and record.job.lens != "serving":  # training runs have a loss to watch
			ids.append("risk.divergence")
		if feature("serving") and record.job.lens == "serving":
			ids += list(SERVING_TARGETS)
		c = record.job.constraints
		if c.wall_time_cap_s:
			ids.append("risk.wall_cap")
		if c.preemption.kind != "none":
			ids.append("risk.preemption_loss")
	return [_target(t) for t in ids]


def _has_shape(record: FlightRecord) -> bool:
	try:
		knobs.setup_from(_script_map(record), _current(record))
	except (ValueError, AttributeError):
		return False
	return True


def _probe_phases(record: FlightRecord) -> list[str]:
	names = {
		ph
		for p in record.probes
		if p.status in OK_PROBE and p.step_time
		for ph, v in p.step_time.by_phase.items()
		if v
	}
	order = list(steptime.PHASES)
	return sorted(names, key=lambda n: (order.index(n) if n in order else len(order), n))


def _current(record: FlightRecord) -> dict:
	return {k.id: k.current for k in record.knobs}


ENGINE_WHAT_IFS = frozenset({"gpu_type", "gpus"})  # accepted even when a record predates their knobs
WORKLOAD_KNOBS = frozenset({"max_steps", "epochs", "n_examples", "n_tasks"})  # define the job; not levers
TRUNCATING_KNOBS = frozenset(
	{"seq_len", "max_completion_len", "max_new_tokens", "max_prompt_len", "image_size", "context_limit"}
)


_SIZES = "batch_size seq_len grad_accum gpus eval_batch_size encode_batch_size image_size max_steps n_tasks concurrency max_turns context_limit"
SIZE_KNOBS = frozenset(_SIZES.split())


def _values(record: FlightRecord, overrides: dict | None) -> dict:
	"""Current knob values with the overrides applied; a value of the wrong kind is the caller's bug."""
	vals = _current(record)
	kinds = {k.id: k.kind for k in record.knobs}
	for kid, v in (overrides or {}).items():
		if kid not in vals and kid not in ENGINE_WHAT_IFS:
			raise ValueError(f"unknown knob {kid!r}: the record declares {sorted(vals)}")
		kind = kinds.get(kid)
		if kind in ("int", "float") and (
			isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v)
		):
			raise ValueError(f"knob {kid!r} takes a number, got {v!r}")
		if kind == "bool" and not isinstance(v, bool):
			raise ValueError(f"knob {kid!r} takes true or false, got {v!r}")
		vals[kid] = v
	return vals


def _invalid(values: dict) -> str | None:
	"""Values the engine can't predict at, as a reason every target abstains with."""
	for kid in sorted(SIZE_KNOBS & set(values)):
		v = values[kid]
		if v is not None and not isinstance(v, bool) and isinstance(v, int | float) and v <= 0:
			return f"{kid} must be at least 1"
	for kid in _PRECISION_KNOBS:
		v = values.get(kid)
		if isinstance(v, str) and v not in ("mixed", "unknown") and _precision({kid: v}) is None:
			return (
				f"precision {v!r} isn't one the engine models (quantised weights use the quantization knob)"
			)
	return None


def _probe_values(record: FlightRecord, probe) -> dict:
	"""The knob values a probe actually ran at: the record's current values, overridden by the probe's."""
	return {**_current(record), **probe.knobs}


def _same(a, b) -> bool:
	if isinstance(a, int | float) and isinstance(b, int | float) and not isinstance(a, bool):
		return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
	return a == b


# run length and learning rate don't change one step's memory or time; the GPU is checked separately
_MATCH_IGNORES = frozenset(
	{
		"gpu_type",
		"max_steps",
		"epochs",
		"n_examples",
		"checkpoint_every",
		"eval_every",
		"lr",
		"warmup",
		"seed",
	}
)


# After the freeze (Nov 2; C's grid probes: lr 2e-4 cells abstained): knobs that change neither a step's memory nor its
# time nor the run's length, only what the run learns. The engine carries step time and memory across them, as it does
# for the learning rate, and the grids say so (Grid.independent_of), under engine.carry_free_knobs. Kept out: kl_beta
# (at 0 TRL loads no reference model), lora_dropout (at 0 PEFT drops the dropout and its saved mask), temperature and
# loss_type (they change how long generations run). Each one's knob spec must be free of memory, time and cost
# (tests/test_engine_free_knobs.py).
FREE_KNOBS = frozenset(
	{
		"lr",
		"lr_schedule",
		"warmup",
		"warmup_steps",
		"weight_decay",
		"grad_clip",
		"lora_alpha",
		"clip_low",
		"clip_high",
		"seed",
	}
)
# the readouts a grid's target family follows, for Grid.independent_of; a target not here lists no knob
_GRID_READOUTS = {"memory": {"memory"}, "time": {"time"}, "cost": {"time", "cost"}}


def _ignored() -> frozenset:
	"""Knobs a probe match and the off-probe check pass over: run length and the learning rate always, the other
	free knobs when engine.carry_free_knobs is on."""
	return _MATCH_IGNORES | FREE_KNOBS if feature("carry_free_knobs") else _MATCH_IGNORES


def independent_of(record: FlightRecord, target_id: str, axes: list[str]) -> list[str]:
	"""The record's knobs this target doesn't depend on, off the grid's axes: the grid answers at any value of them.
	Empty unless engine.carry_free_knobs is on."""
	if not feature("carry_free_knobs"):
		return []
	family = target_id.split(".", 1)[0]  # memory.*, time.*, cost.*; risk.oom follows memory
	need = _GRID_READOUTS.get(family) or ({"memory"} if target_id == "risk.oom" else None)
	if not need:
		return []
	return sorted(
		k.id
		for k in record.knobs
		if k.id in FREE_KNOBS and k.id not in axes and not (need & knobs.knob_spec(k.id).affects)
	)


MEDIAN_SE = math.sqrt(math.pi / 2)  # cited: a sample median's standard error is sqrt(pi/2) x sigma / sqrt(k)
LENGTH_DRAWS = 4000  # derived: 4,000 simulated steps put the padded-length summaries within about 1%
TOKEN_PHASES = (
	"forward",
	"backward",
)  # the phases whose time follows the tokens in a batch; the rest stay fixed
_PPF_A = (
	-3.969683028665376e01,
	2.209460984245205e02,
	-2.759285104469687e02,
	1.383577518672690e02,
	-3.066479806614716e01,
	2.506628277459239e00,
)
_PPF_B = (
	-5.447609879822406e01,
	1.615858368580409e02,
	-1.556989798598866e02,
	6.680131188771972e01,
	-1.328068155288572e01,
)
_PPF_C = (
	-7.784894002430293e-03,
	-3.223964580411365e-01,
	-2.400758277161838e00,
	-2.549732539343734e00,
	4.374664141464968e00,
	2.938163982698783e00,
)
_PPF_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00)


def _poly(coef, x):
	out = np.zeros_like(x)
	for c in coef:
		out = out * x + c
	return out


def _norm_ppf(p: np.ndarray) -> np.ndarray:
	"""The normal quantile, vectorised (cited: Acklam's rational approximation, relative error about 1e-9)."""
	p = np.clip(np.asarray(p, dtype=float), 1e-15, 1 - 1e-15)
	out = np.empty_like(p)
	lo, hi = p < 0.02425, p > 1 - 0.02425
	mid = ~(lo | hi)
	q = np.sqrt(-2 * np.log(p[lo]))
	out[lo] = _poly(_PPF_C, q) / (_poly(_PPF_D, q) * q + 1)
	q = np.sqrt(-2 * np.log(1 - p[hi]))
	out[hi] = -_poly(_PPF_C, q) / (_poly(_PPF_D, q) * q + 1)
	q = p[mid] - 0.5
	r = q * q
	out[mid] = _poly(_PPF_A, r) * q / (_poly(_PPF_B, r) * r + 1)
	return out


@functools.lru_cache(maxsize=256)
def _padded_step(
	median: float, mean: float, cap: float | None, rows: int, accum: int
) -> tuple[float, float, float]:
	"""(median, mean, log spread) of a step's padded tokens per row, when every micro-batch of `rows` rows (all ranks
	together: a step waits for the slowest) pads to its longest row and a step sums `accum` micro-batches. Rows are
	log-normal with the dataset's own median and mean (Chiya, Nov 1: time follows the median and the mean, not the
	tail, which sizes peak memory), cut at seq_len. A micro-batch's longest row is drawn from the quantile of
	U^(1/rows), so the cost doesn't grow with the batch; the result is cached for knob moves."""
	s = math.sqrt(2 * math.log(mean / median)) if mean > median > 0 else 0.0
	n = max(
		200, LENGTH_DRAWS // max(1, accum)
	)  # about LENGTH_DRAWS quantiles in all, whatever the accumulation
	u = np.random.default_rng(0).random((n, max(1, accum)))
	logs = math.log(median) + s * _norm_ppf(u ** (1.0 / max(1, rows)))
	if cap:
		logs = np.minimum(logs, math.log(cap))
	step = np.exp(logs).mean(axis=1)  # tokens per row, over the step's micro-batches
	return float(np.median(step)), float(np.mean(step)), float(np.std(np.log(step)))


def _dataset_lengths(p) -> tuple[float, float] | None:
	ex = p.extras or {}
	med, mean = ex.get("dataset_p50_tokens"), ex.get("dataset_mean_tokens")  # A's names, Nov 1
	return (float(med), float(mean)) if med and mean else None


def _probe_rows(p) -> list[float]:
	"""Padded tokens per row on each of the probe's timed steps (its step log)."""
	return [x.tokens / x.samples for x in (p.step_log or []) if x.kind == "timed" and x.tokens and x.samples]


def _length_factor(record: FlightRecord, p, medians: bool | None = None) -> float | None:
	"""How much longer the full run's batches are than the ones this probe saw: median against median from the
	probe's step log, or mean against mean (`medians=False`, or no step log): the dataset's padded length per row
	at the probe's own batch, seq_len and accumulation, over the probe's padded length
	(tokens_per_sample_mean / (1 - padding_fraction)). Applied to the probe's forward and backward time before the
	step model fits, so each probe carries its own padding and a knob move that lands on another probe doesn't jump
	(review, Nov 1). Per-rank rows: the probe's tokens and timing are one rank's view. None when the dataset's
	lengths or the probe's padding aren't known, or the rows are packed."""
	data, lens = p.data, _dataset_lengths(p)
	pv = _probe_values(record, p)
	if (
		lens is None or not data or not data.tokens_per_sample_mean or data.padding_fraction is None
		or data.padding_fraction >= 1 or data.packing or pv.get("packing") or record.job.lens == "rl"
	):  # fmt: skip
		return None
	cap = pv.get("seq_len")
	run_med, run_mean, _ = _padded_step(
		*lens, float(cap) if cap else None, int(pv.get("batch_size") or 1), int(pv.get("grad_accum") or 1)
	)
	# like with like (reviews, Nov 1): the probe's p50 is a median step, so the run's median padded length per row
	# over the median of the probe's own timed steps (step_log: padded tokens over rows); without a step log, mean
	# over mean, which is right when the probe's batches have the run's shape
	timed = _probe_rows(p)
	if timed and medians is not False:
		return run_med / float(np.median(timed))
	return run_mean / (float(data.tokens_per_sample_mean) / (1 - float(data.padding_fraction)))


@dataclass(frozen=True)
class _LengthTerm:
	total_scale: float  # the run's mean step against its median step (1 without them)
	step_sigma: float  # added to the step's log sigma: the probe's few-step median against the run's median
	total_sigma: float  # replaces it for a total, a sum over many steps
	reason: str
	intact: bool = (
		True  # False once later terms rescale the step's sigma; the total then keeps the step's band
	)


def _length_term(
	record: FlightRecord,
	values: dict,
	match,
	steps_hint: float | None = None,
	breakdown: dict | None = None,
	corrected: bool = True,
) -> _LengthTerm | None:
	"""Batches that vary in length, as time sees them (Chiya, Nov 1). The run's median step follows the median padded
	length (each probe's own correction, _length_factor, before the step model fits) and its total the mean: the
	total scales by the mean step over the median at the query's own batch, seq_len and accumulation. The probe's
	p50 is the median of k steps, so its sampling error, sqrt(pi/2) x the probe's own step spread / sqrt(k), stays on
	the step. A total sums many steps: with the dataset's lengths known its length error is the per-step spread /
	sqrt(steps); without them the run's mean comes from the probe's k steps (spread / sqrt(k)). The 99.9th
	percentile and the longest row stay out of time. RL rollouts have their own length model (engine/rl.py), and
	packed rows don't pad."""
	if record.job.lens == "rl":
		return None
	probes = [p for p in record.probes if p.status in OK_PROBE and p.step_time and p.step_time.p50_s]
	p = match if match in probes else (probes[0] if probes else None)
	if p is None or not p.step_time.n_steps:
		return None
	st_, k = p.step_time, p.step_time.n_steps
	seen = math.log(st_.p95_s / st_.p50_s) / steptime.Z90 if st_.p95_s and st_.p95_s > st_.p50_s else 0.0
	lens = _dataset_lengths(p)
	pv = _probe_values(record, p)
	packed = bool(values.get("packing")) or bool(pv.get("packing")) or bool(p.data and p.data.packing)
	if lens is None or packed or not corrected or _length_factor(record, p) is None:
		if seen <= 0:
			return None
		why = f"batches vary in length (the probe's {k} timed steps), so a full run's median step can differ from the probe's {k}-step median"
		return _LengthTerm(1.0, MEDIAN_SE * seen / math.sqrt(k), seen / math.sqrt(k), why)
	cap = values.get("seq_len") or pv.get("seq_len")
	batch = int(values.get("batch_size") or pv.get("batch_size") or 1)
	accum = int(values.get("grad_accum") or pv.get("grad_accum") or 1)
	run_med, run_mean, spread = _padded_step(*lens, float(cap) if cap else None, batch, accum)
	phases = breakdown or st_.by_phase or {}  # the query's own step (its corrected phases), not the probe's
	total_phase = sum(phases.values())
	# "step" is the whole step when the probe had no phase split, and the factor scaled it whole
	share = (
		sum(phases.get(ph, 0.0) for ph in (*TOKEN_PHASES, "step")) / total_phase if total_phase > 0 else 1.0
	)
	total_scale = (1 + share * (run_mean / run_med - 1)) if run_med > 0 else 1.0
	why = (
		f"batches vary in length: the run's median step follows the dataset's median padded length ({run_med:.0f} "
		f"tokens a row here) and its total the mean ({run_mean:.0f})"
	)
	return _LengthTerm(
		total_scale, MEDIAN_SE * seen / math.sqrt(k), spread / math.sqrt(steps_hint or 1.0), why
	)


def _matching_probe(record: FlightRecord, values: dict, ok=OK_PROBE):
	"""A probe whose every knob equals the query's: the number is a measurement, not a prediction."""
	ignored = _ignored()
	for p in record.probes:
		if p.status in ok:
			pv = _probe_values(record, p)
			if all(_same(pv.get(k), v) for k, v in values.items() if k not in ignored):
				return p
	return None


# D's walkthrough, Nov 3: knobs whose effect the engine computes from the layer shape (heads, MLP width, per-layer
# activations). On a script map with only a parameter count (GenericArch) they have no analytical effect, so a move
# would read as no change at all; count them as unmodelled there, as for any knob the engine doesn't model.
SHAPE_DEPENDENT = {
	"memory": frozenset({"lora_r", "eval_no_grad", "gradient_checkpointing", "attn"}),
	# review, Nov 3: the step-time model applies checkpointing's recompute without a layer shape
	"time": frozenset({"lora_r", "eval_no_grad", "attn"}),
}


# knobs.py marks these modelled for time, but the step-time model never reads them: it reads the work per step
# (batch, sequence, accumulation, image size, a GNN's fan-outs: knobs.work_units), precision, checkpointing, the
# GPU, GPUs and nodes. Moved off every probe, the step was "carried" by the model while its point stayed put (nanoGPT's n_layer 12 to 24: 0.955 s to
# 0.939 s). So such a move is said as unmodelled: held at its probed effect, wider, extrapolated (B, Nov 3: a bug
# fix found writing the knobs guide; memory is unchanged)
STEP_TIME_UNREAD = frozenset(
	{
		"n_layer",
		"d_model",
		"n_head",
		"hidden_channels",
		"gnn_layers",
		"lora_r",
		"quantization",
		"optimizer",
		"attn",
		"eval_no_grad",
		"num_workers",
		"prefetch_factor",
		"tensor_parallel",
		"pipeline_parallel",
		"use_vllm",
		"kv_cache_dtype",
		"gpu_memory_utilization",
	}
)  # fmt: skip (the last three: the serving simulator's, off)


def _generic_shape(record: FlightRecord) -> bool:
	try:
		arch, _setup, _shape = knobs.setup_from(_script_map(record), _current(record))
	except (ValueError, AttributeError):
		return False
	return isinstance(arch, memory.GenericArch)


def _unmodelled_moves(record: FlightRecord, values: dict, readout: str) -> list[str]:
	"""Knobs that change this readout, aren't modelled for it, and sit where no probe ran."""
	out = []
	generic = None
	for kid, v in values.items():
		spec = knobs.knob_spec(kid)
		if readout not in spec.affects or kid == "gpu_type":
			continue
		unread = readout == "time" and kid in STEP_TIME_UNREAD
		if readout in spec.modelled and not unread:
			if kid not in SHAPE_DEPENDENT.get(readout, ()):
				continue
			if generic is None:
				generic = _generic_shape(record)
			if not generic:
				continue
		probed = [_probe_values(record, p).get(kid) for p in record.probes if p.status in OK_PROBE]
		if probed and not any(_same(v, pv) for pv in probed):
			out.append(kid)
	return out


# knobs the calibration axis covers, and run-length knobs that don't change a step's memory or time
SHAPE_KNOBS = frozenset(
	{
		"batch_size",
		"seq_len",
		"encode_batch_size",
		"image_size",
		"max_model_len",
		"max_prompt_len",
		"max_completion_len",
	}
)
RUN_KNOBS = frozenset(
	{
		"max_steps",
		"epochs",
		"n_examples",
		"checkpoint_every",
		"eval_every",
		"lr",
		"warmup",
		"seed",
		"gpu_type",
		"gpus",
	}
)


def _off_probe(record: FlightRecord, values: dict, also_ok: frozenset = frozenset()) -> list[str]:
	"""Knobs other than batch and sequence whose value no probe ran (Review, Oct 27: such estimates are extrapolated)."""
	runs = [_probe_values(record, p) for p in record.probes if p.status in OK_PROBE]
	if not runs:
		return []
	out = []
	ignored = _ignored()
	for kid, v in values.items():
		if (
			kid in SHAPE_KNOBS
			or kid in RUN_KNOBS
			or kid in also_ok
			or kid.startswith("fanout.")
			or kid in ignored
		):
			continue
		if not any(_same(v, pv.get(kid)) for pv in runs):
			out.append(kid)
	return sorted(out)


CONTEXT_MAX_BYTES = (
	2 * 2**30
)  # guess: a real context is 0.25-1 GiB; a gap above 2 GiB is another process on the GPU
CONTEXT_AGREE_BYTES = 64 * 2**20  # guess: two steps within 64 MiB of each other agree on the context


_BACKEND_OF = {"nvidia": "cuda", "amd": "rocm"}


def _same_stack(record: FlightRecord, gpu: str) -> bool:
	"""Whether a move to `gpu` keeps the probes' vendor and software stack: the same vendor, and the record's own
	backend the one that vendor runs (CUDA for NVIDIA, ROCm for AMD). The script, its image and its libraries go with
	it, so the memory outside the allocator is the probes' own measured context, not a guess (Chiya, Nov 2: on the zoo,
	moving an L4 record to the A10 swapped a measured 0.27 GB for the guessed 0.81 GB)."""
	src = _env_gpu(record)
	if src is None or record.env is None:
		return False
	try:
		a, b = gpus.get_gpu(src), gpus.get_gpu(gpu)
	except KeyError:
		return False
	return a.vendor == b.vendor and record.env.backend == _BACKEND_OF.get(b.vendor)


def _context_bytes(record: FlightRecord, gpu: str | None = None) -> float:
	"""The device memory outside PyTorch's allocator (CUDA context, cuBLAS workspaces, other libraries), measured from
	the probes' timed steps: whole-device use (cudaMemGetInfo, at the step boundary) less the step's reserved peak.
	The largest gap that a second step agrees with (reviews, Nov 1): when the allocator hands memory back between the
	peak and the boundary (GRPO frees its generation cache) a step's gap reads low or negative, so the steps that
	released nothing show the whole context; another process's one-step spike on a shared GPU has no partner. C, Nov 1: a guessed 0.75 GiB on a GRPO probe's reserved peak read 23.8 GB on a
	23.7 GB L4 for a run that fit. The guess stays where no probe logged whole-device use, and on a GPU of another
	vendor or stack; with engine.context_same_stack on (after the freeze, Nov 2), a move within the probes' vendor and
	stack keeps their measured context (`_same_stack`)."""
	moved = gpu is not None and gpu != _env_gpu(record)
	if moved and not (feature("context_same_stack") and _same_stack(record, gpu)):
		return float(memory.DEFAULT_CONTEXT_BYTES)
	scale = _context_scale(record, gpu) if moved else 1.0
	gaps = [
		x.device_used_bytes - x.peak_reserved_bytes
		for p in record.probes
		if p.status in OK_PROBE
		for x in (p.step_log or [])
		if x.kind == "timed" and x.device_used_bytes and x.peak_reserved_bytes
	]
	gaps = sorted((g for g in gaps if 0 < g <= CONTEXT_MAX_BYTES), reverse=True)
	if not gaps:
		return float(memory.DEFAULT_CONTEXT_BYTES)
	if len(gaps) == 1:
		return float(gaps[0]) * scale
	# the largest gap that a second step agrees with: a release after the peak only lowers a step's gap, and a
	# co-tenant's one-step spike on a shared GPU has no partner (review, Nov 1)
	for i, g in enumerate(gaps[:-1]):
		if g - gaps[i + 1] <= CONTEXT_AGREE_BYTES:
			return float(g) * scale
	return float(np.median(gaps)) * scale


def _context_scale(record: FlightRecord, gpu: str) -> float:
	"""On a bigger card of the same stack, the measured context scaled up by the ratio of SMs (CUs on AMD), never
	down: part of the CUDA context is reserved per SM (each SM's thread stacks), so an L4's 58 SMs read low for an
	H100's 132 [guess: the share that grows with SMs isn't measured; scaling all of it is the safe side]. Review, Nov 2."""
	try:
		src, tgt = gpus.get_gpu(_env_gpu(record)), gpus.get_gpu(gpu)
	except (KeyError, TypeError):
		return 1.0
	if not src.compute_units or not tgt.compute_units:
		return 1.0
	return max(1.0, tgt.compute_units / src.compute_units)


def _capacity(record: FlightRecord, gpu: str) -> int:
	"""Usable bytes on this GPU: the probe's reported total when the probe ran on it, else the table's."""
	g = record.env.gpu if record.env else None
	if g is not None and g.memory_bytes and gpu == _env_gpu(record):
		return int(g.memory_bytes)
	return gpus.get_gpu(gpu).capacity_bytes()


def _script_map(record: FlightRecord):
	"""The script map, with the parameter count a probe measured when the map itself doesn't give one (Nov 1: A's
	A10 GRPO record names Qwen2.5-0.5B-Instruct but no size, while its probe counted 494,032,768 parameters)."""
	sm = record.job.script_map
	model = getattr(sm, "model", None)
	if sm is None or model is None or getattr(model, "param_count", None):
		return sm
	counted = [
		float(p.extras["params_total"])
		for p in record.probes
		if p.status in OK_PROBE and p.extras and p.extras.get("params_total")
	]
	if not counted:
		return sm
	return sm.model_copy(update={"model": model.model_copy(update={"param_count": int(max(counted))})})


def _env_gpu(record: FlightRecord) -> str | None:
	"""The probe's GPU: a table name, or an unlisted fallback built from the probe's own memory and SM count."""
	if record.env and record.env.gpu and record.env.gpu.name:
		g = record.env.gpu
		try:
			return gpus.resolve(g.name, g.memory_bytes)
		except KeyError:
			if g.memory_bytes:
				cu = next(
					(
						p.extras.get("gpu_compute_units")
						for p in record.probes
						if p.extras.get("gpu_compute_units")
					),
					None,
				)
				return gpus.unlisted_name(g.name, g.memory_bytes, int(cu) if cu else None)
			return None
	return None


def _gpu_move(record: FlightRecord, values: dict) -> str | None:
	"""The GPU asked for, or None when gpu_type names the GPU the probes ran on (no move).

	Records from A's harness carry a gpu_type knob whose current value is env.gpu.name (runner/pipeline.py),
	so the probed GPU must never read as a move, whether named as the probe reported it (an unlisted card would
	otherwise lose its step time) or by another name for the same card. Compared with the GPU the probes
	measured, not the knob's current value, so a knob left stale after a step-up still predicts for the GPU
	the dial shows.
	"""
	name = values.get("gpu_type")
	if not (isinstance(name, str) and name):
		return None
	g = record.env.gpu if record.env else None
	if g is not None and name == g.name:
		return None
	probed = _env_gpu(record)
	try:
		if probed is not None and gpus.resolve(name) == probed:
			return None
	except KeyError:
		pass  # a name the table doesn't know is a move to an unknown GPU, and abstains downstream
	return name


def _target_gpu(record: FlightRecord, values: dict) -> str | None:
	name = _gpu_move(record, values)
	if name:
		try:
			return gpus.resolve(name)
		except KeyError:
			return None
	return _env_gpu(record)


def _n_gpus(values: dict, record: FlightRecord) -> int:
	n = values.get("gpus")
	if n is None and record.env and record.env.gpu:
		n = record.env.gpu.count
	return max(1, int(n or 1))


# ---------------------------------------------------------------- memory


@dataclass(frozen=True)
class _MemoryResult:
	peak: Estimate
	oom: Estimate
	detail: memory.MemoryEstimate | None = None
	terms: dict | None = None  # memory.<term> estimates at the peak moment


def probe_ran_eval(p) -> bool:
	"""Whether this probe ran an eval pass, so its peak can include one. One rule for the engine and C's checks
	(Chiya, Oct 30): A's `extras.eval_off` first (set when a point is re-run with eval off after an eval OOM),
	then whether the probe recorded eval stats."""
	off = (p.extras or {}).get("eval_off")
	# records store extras as numbers, so A's marker arrives as 1.0 or 0.0: any finite number, non-zero is off
	if isinstance(off, (int, float)) and math.isfinite(off):
		return off == 0
	return p.eval is not None


def _longest_seen(p) -> float | None:
	"""The longest padded row this probe processed in any logged step (warm-up included: its memory peak covers
	the whole probe)."""
	rows = [s.padded_len for s in (p.step_log or []) if s.padded_len]
	return float(max(rows)) if rows else None


def _dataset_longest(record: FlightRecord) -> tuple[float, float, int | None, int | None] | None:
	"""The training set's longest row as A's recorder measured it (Probe.extras dataset_max_tokens), with the seq_len
	it was cut at, the rows scanned and the rows in the set: from the probe that cut least, since each probe reads
	rows at its own knobs."""
	best = None
	for p in record.probes:
		ex = p.extras or {}
		mx = ex.get("dataset_max_tokens") if p.status in OK_PROBE else None
		if not mx:
			continue
		cut = float(_probe_values(record, p).get("seq_len") or 0)
		rows, total = ex.get("dataset_rows_scanned"), ex.get("dataset_rows_total")
		if best is None or cut > best[1] or (cut == best[1] and float(mx) > best[0]):
			best = (float(mx), cut, int(rows) if rows else None, int(total) if total else None)
	return best


def _past_scan_cut(record: FlightRecord, values: dict) -> str | None:
	"""C, Nov 3 (D's walkthrough item 6): a seq_len above the one the dataset was scanned at. Each probe's scan cuts
	rows at its own seq_len, so when the longest row reached the cut, rows past it aren't measured: time holds them at
	the cut (the median and mean barely move when they are rare), and says so rather than reading as no change."""
	if record.job.lens not in TOKEN_LENSES:
		return None
	known = _dataset_longest(record)
	q = float(values.get("seq_len") or 0)
	if known is None or not q:
		return None
	mx, cut, _rows, _total = known
	if not cut or q <= cut or mx < cut:
		return None
	sm = record.job.script_map
	if values.get("packing") or ("packing" not in values and sm and sm.data and sm.data.packing):
		return None  # packed rows don't pad: the length doesn't come from the scan
	ok = [p for p in record.probes if p.status in OK_PROBE]
	p50 = max((float((p.extras or {}).get("dataset_p50_tokens") or 0) for p in ok), default=0.0)
	if p50 >= cut:
		return (
			None  # rows fill the cut (fixed-length blocks): the step scales with seq_len, not with the scan
		)
	p999 = max((float((p.extras or {}).get("dataset_p999_tokens") or 0) for p in ok), default=0.0)
	rare = (
		f"; 99.9% of rows are {p999:,.0f} tokens or fewer, so the median and mean barely move"
		if p999 and p999 < cut
		else ""
	)
	return (
		f"seq_len {q:,.0f} is past the {cut:,.0f} the dataset was scanned at, and its longest rows reached that cut, so "
		f"rows past {cut:,.0f} tokens aren't measured and are held at {cut:,.0f}{rare}; a batch with one is slower"
	)


def _query_seq(record: FlightRecord, values: dict) -> tuple[float | None, str | None]:
	"""The length a memory query is sized at, and a note saying why. A full run meets the training set's longest
	row, which a few probe batches don't see (review, Oct 30), so the tail sets peak memory (Chiya, Nov 1: p99.9 and
	max belong to peak memory). Only a scan of every row sizes a query below seq_len, at min(seq_len, longest row);
	otherwise it's sized at seq_len, and the readout says why (C, Nov 1: the LoRA zoo record's probes saw 517
	tokens, a query was sized at 1,024, and no eval fix could clear the OOM that length implies)."""
	if record.job.lens not in TOKEN_LENSES:
		return None, None
	q = float(values.get("seq_len") or 0)
	known = _dataset_longest(record)
	if known is not None:
		mx, cut, rows, total = known
		if not (rows and total and rows >= total):
			# review, Nov 1: the recorder scans at most 200,000 rows and stops after 10 s, so a partial scan's longest
			# row may not be the set's
			scanned = f"{rows:,} of {total:,}" if rows and total else "part of the"
			return None, (
				f"sized for rows of {q:,.0f} tokens, the seq_len: the recorder scanned {scanned} training rows, so "
				"the set's longest row isn't known"
			)
		over = f" over all {rows:,} training rows"
		if cut and mx >= cut:  # the probe's own cut hid the longest row: rows reach at least the cut
			if q and q > cut:
				return None, (
					f"sized for rows of {q:,.0f} tokens, the seq_len: rows reach at least {cut:,.0f}{over}, and past "
					"that isn't measured"
				)
			return (
				None,
				f"sized for rows of {q:,.0f} tokens, the seq_len: the training set has rows that long{over}",
			)
		seq = min(q, mx) if q else mx
		return seq, (
			f"sized for the training set's longest row, {seq:,.0f} tokens (measured{over}), which a full run meets; "
			"an eval set's rows aren't measured, so eval is sized the same"
		)
	seen = [x for x in (_longest_seen(p) for p in record.probes if p.status in OK_PROBE) if x]
	if q and seen and max(seen) < 0.9 * q:
		return None, (
			f"sized for rows of {q:,.0f} tokens, the seq_len, as a full run may meet them: the probes' rows reached "
			f"{max(seen):,.0f}, and the dataset's longest row isn't recorded"
		)
	return None, None


def _padded_seq(record: FlightRecord, values: dict, probe=None) -> float | None:
	"""The sequence length activations were sized by: a probe's longest padded row (cut by its seq_len), from its
	step log; a query's from _query_seq (the dataset's longest row when it's recorded, else the seq_len setting)."""
	if probe is None:
		return _query_seq(record, values)[0]
	if record.job.lens not in TOKEN_LENSES:
		return None
	longest = _longest_seen(probe)
	sq = float(_probe_values(record, probe).get("seq_len") or 0)
	if longest is None:
		return None
	return min(longest, sq) if sq else longest


def _sized(shape, values: dict, seq: float | None) -> tuple[knobs.Shape, dict]:
	"""The shape and knob values with seq_len replaced by the padded length rows really reach."""
	if seq is None:
		return shape, values
	s = max(1, round(seq))
	return replace(shape, seq=s), {**values, "seq_len": s}


# Chiya, Nov 3: the reserved pool. The probes measure the allocator's reserved peak, but a short probe can stop before
# the pool settles. The corpus margin (reserved over allocated, by family and size, fitted on the
# calibration split by scripts/fit_reserved_margin.py into `_reserved_margin.py`, never the zoo) raises a prediction only where the probes' own
# ratio is below it: the point to the family's median, the upper end to its 90th percentile.
_MARGIN_FAMILY = {
	"finetune": "hf",
	"rl": "hf",
	"pretrain": "gpt",
	"vision": "cnn",
	"retrieval": "encoder",
}
# review, Nov 3: a training-step margin says nothing about inference: vLLM preallocates its pool, and an eval's batches
# aren't training steps
_MARGIN_SKIP = ("serving", "eval")
_MARGIN_SIZES = ((1e7, "<10M"), (1e8, "<100M"), (1e9, "<1B"), (float("inf"), ">=1B"))


def _margins() -> dict:
	from preflight.engine._reserved_margin import MARGINS

	return MARGINS


def _margin(lens: str, n_params: float | None) -> tuple[float, float, str]:
	"""(median, q90, what it is) for this lens and size; a lens with no corpus family takes the widest family."""
	t = _margins()
	fam = _MARGIN_FAMILY.get(lens)
	if fam is None:
		m50 = max(c["median"] for c in t["by_family"].values())
		m90 = max(c["q90"] for c in t["by_family"].values())
		return m50, m90, f"no corpus family for {lens}, so the widest family's"
	if n_params:
		size = next(name for top, name in _MARGIN_SIZES if n_params < top)
		c = t["by_size"].get(f"{fam}|{size}")
		if c:
			return c["median"], c["q90"], f"{fam} models {size}"
	c = t["by_family"][fam]
	return c["median"], c["q90"], f"{fam} models"


def _probe_pool_ratio(record: FlightRecord) -> float | None:
	"""Reserved over allocated at the probe with the largest reserved peak (the one the prediction leans on)."""
	best = None
	for p in record.probes:
		m = p.memory
		usable = p.status in OK_PROBE and m and m.peak_reserved_bytes and m.peak_allocated_bytes
		if usable and (best is None or m.peak_reserved_bytes > best.peak_reserved_bytes):
			best = m
	return best.peak_reserved_bytes / best.peak_allocated_bytes if best else None


def _with_reserved_margin(
	record: FlightRecord, est: memory.MemoryEstimate, ctx: float, capacity: float, n_params: float | None
) -> memory.MemoryEstimate:
	if (
		not feature("memory_reserved")
		or record.job.lens in _MARGIN_SKIP
		or est.status in ("abstain", "measured")
		or not est.point
	):
		return est
	ratio = _probe_pool_ratio(record)
	if ratio is None:
		return est
	m50, m90, what = _margin(record.job.lens, n_params)
	f50, f90 = max(1.0, m50 / ratio), max(1.0, m90 / ratio)
	if f90 <= 1.0:
		return est
	alloc = max(0.0, est.point - ctx)
	point = ctx + alloc * f50
	hi = max(est.hi or point, ctx + alloc * f90)
	why = (
		f"the probes' reserved pool is {ratio:.2f}x their allocated peak, under the corpus's {m50:.2f}x median and "
		f"{m90:.2f}x 90th percentile for {what}, so the point and the upper end are raised to them"
	)
	p = memory.p_oom(point, est.sigma_log, capacity)
	return replace(est, point=point, hi=hi, p_oom=p, oom_status=memory.oom_status(p),
		reason="; ".join(x for x in (est.reason, why) if x))  # fmt: skip


def _memory(record: FlightRecord, values: dict, eval_off: bool = False) -> _MemoryResult:
	mid = memory.MODEL_ID
	gpu = _target_gpu(record, values)
	if gpu is None:
		why = "the GPU isn't known, so there's no capacity to compare against; pick a GPU"
		return _MemoryResult(_abstain("bytes", mid, why), _abstain("probability", mid, why))
	capacity = _capacity(record, gpu)
	ctx = _context_bytes(record, gpu)  # measured from the probes' whole-device use where they logged it
	oom_probe = _matching_probe(record, values, ok=("oom",))
	if oom_probe is not None and gpu == _env_gpu(record):  # an OOM on one GPU says nothing about a bigger one
		why = f"probe {oom_probe.id} ran out of memory at these settings"
		est = memory.MemoryEstimate(None, None, None, "bytes", "abstain", mid, 1.0, "fails", 0.0, None, why)  # type: ignore[arg-type]
		oom = _est(1.0, 1.0, 1.0, "probability", "measured", mid, f"fails: {why}")
		return _MemoryResult(_abstain("bytes", mid, why), oom, est)
	moves = _unmodelled_moves(record, values, "memory")
	if moves:
		why = f"{', '.join(moves)} changes memory in a way the engine doesn't model yet; run a probe here"
		return _MemoryResult(_abstain("bytes", mid, why), _abstain("probability", mid, why))
	sm = _script_map(record)
	lens = record.job.lens
	try:
		arch, setup, shape = knobs.setup_from(sm, values)
	except (ValueError, AttributeError) as exc:
		res = _empirical_memory(record, values, capacity, str(exc))
		_seq, seq_note = _query_seq(record, values)  # review, Nov 1: the no-shape path is sized the same way
		if seq_note and res.peak.status != "abstain":
			res = replace(res, peak=_with_note(res.peak, seq_note), oom=_with_note(res.oom, seq_note))
		return res
	shape, sized_values = _sized(shape, values, _padded_seq(record, values))
	if eval_off:
		setup = replace(setup, eval_batch=None)  # predict_as_run: the probe ran this point without eval
	mprobes = []
	for p in record.probes:
		if p.status not in OK_PROBE or not p.memory or not p.memory.peak_reserved_bytes:
			continue
		pv = _probe_values(record, p)
		try:
			pa, ps, psh = knobs.setup_from(sm, pv)
		except (ValueError, AttributeError):
			continue
		psh, pv = _sized(psh, pv, _padded_seq(record, pv, probe=p))
		if not probe_ran_eval(p):
			ps = replace(ps, eval_batch=None)  # the probe never ran eval, so its peak can't include it
		an = memory.breakdown(pa, ps, psh.batch, psh.seq, capacity_bytes=capacity).allocator_peak
		mprobes.append(
			memory.MemoryProbe(knobs.work_units(lens, pv), float(p.memory.peak_reserved_bytes), an)
		)
	match = _matching_probe(record, values)
	measured = False
	if match is not None and mprobes:
		# a probe at these knobs measured the peak only if it saw the moment that sets it
		moment = memory.breakdown(arch, setup, shape.batch, shape.seq, capacity_bytes=capacity).peak_moment
		# review, Nov 1: and only on the probe's own GPU; a GPU move is a prediction, whatever the knobs match
		measured = gpu == _env_gpu(record) and (
			moment == "train" or (moment == "eval" and probe_ran_eval(match))
		)
	est = memory.predict_peak(
		arch,
		setup,
		shape.batch,
		shape.seq,
		probes=mprobes or None,
		capacity_bytes=capacity,
		context_bytes=ctx,
		work_units=knobs.work_units(lens, sized_values),
		measured=measured,
		base_half_width=conformal.TABLE.lookup("memory.peak", gpu, lens).half_width
		if conformal.TABLE.lookup("memory.peak", gpu, lens)
		else None,
	)
	if (
		est.status == "predicted"
		and est.breakdown is not None
		and est.breakdown.peak_moment != "train"
		and not any(probe_ran_eval(p) for p in record.probes if p.status in OK_PROBE)
	):
		why = f"the peak comes at {est.breakdown.peak_moment}, which no probe ran; run a probe with {est.breakdown.peak_moment} in it"
		est = replace(est, status="extrapolated", reason="; ".join(x for x in (est.reason, why) if x))
	est = _with_reserved_margin(
		record, est, ctx, capacity, arch.param_count() if hasattr(arch, "param_count") else None
	)
	cell = conformal.TABLE.lookup("memory.peak", gpu, lens)
	if est.status != "abstain" and est.point and cell is not None and cell.residuals:
		# P(OOM) from held-out residuals of the allocator peak (context is added outside the residual)
		alloc = est.point - ctx
		pool = conformal.TABLE.cells.get(("memory.peak", "", ""))
		p_emp, pooled = oom_model.pooled_p_oom(
			cell.residuals,
			pool.residuals if pool is not None and pool is not cell else None,
			alloc,
			capacity - ctx,
			memory.FITS_BELOW,
		)
		if pooled:
			who = " ".join(x for x in (cell.key[1], cell.key[2]) if x)
			oom_source = f"from all {pool.n} held-out runs; none of the {cell.n} {who} runs went over"
			floor = 0.5 / (len(pool.residuals) + 1)
		else:
			oom_source = f"from {cell.n} held-out runs"
			floor = 0.5 / (
				len(cell.residuals) + 1
			)  # cited: empirical_p_oom's half-count, no run over the margin
		est = replace(est, p_oom=p_emp, oom_status=memory.oom_status(p_emp))
	else:
		oom_source = "simulated band"
		floor = None
	off = _off_probe(record, values)
	if est.status in ("predicted", "measured") and off:
		why = f"{', '.join(off)} {'is' if len(off) == 1 else 'are'} off the probed values; the model carries it there"
		est = replace(est, status="extrapolated", reason="; ".join(x for x in (est.reason, why) if x))
	res = _from_memory_estimate(est, mid, oom_source, floor)
	if (
		cell is not None and mprobes and res.peak.status not in ("abstain", "measured")
	):  # no probes: the prior set it
		res = replace(res, peak=_with_note(res.peak, _band_note(cell)))
	if est.status != "abstain" and est.breakdown is not None:
		stack = dict(est.breakdown.stack)
		if est.breakdown.peak_moment == "eval":
			# C's ring shows eval as its own wedge: the training moment's terms plus the eval spike
			train = memory.breakdown(
				arch,
				replace(setup, eval_batch=None),
				shape.batch,
				shape.seq,
				context_bytes=ctx,  # Nov 1: the measured context, as the point has
				capacity_bytes=capacity,
			)
			stack = {**train.stack, "eval_spike": est.breakdown.allocator_peak - train.allocator_peak}
		res = replace(res, terms=_term_estimates(stack, res.peak))
	_seq, seq_note = _query_seq(record, values)
	if seq_note and res.peak.status != "abstain":
		res = replace(res, peak=_with_note(res.peak, seq_note), oom=_with_note(res.oom, seq_note))
	probed = any(
		p.status in OK_PROBE for p in record.probes
	)  # review: a skeleton has no GPU before any probe
	if probed and _env_gpu(record) is None and res.peak.status != "abstain":
		# Nov 1: A's A10 GRPO record carried no GPU, and the readout said "tight, 69%" for the knob's L4 unannounced
		note = f"the probe's GPU isn't in the record, so this is for {gpu} from the GPU knob, with a guessed CUDA context"
		res = replace(res, peak=_with_note(res.peak, note), oom=_with_note(res.oom, note))
	disagree = _faketensor_disagreements(arch, setup)
	if disagree and res.peak.status != "abstain":
		# PM, Nov 1: a term the FakeTensor trace disagrees with is named, never hidden
		case, worse = disagree
		listed = ", ".join(f"{t} {r:.3g}x" for t, r in worse.items())
		peak_note = (
			f"the analytical split disagrees with a FakeTensor trace of {case} on {listed} (engine / trace)"
		)
		terms = dict(res.terms or {})
		for t, r in worse.items():
			if f"memory.{t}" in terms:
				terms[f"memory.{t}"] = _with_note(
					terms[f"memory.{t}"],
					f"the analytical {t} term is {r:.3g}x a FakeTensor trace of {case}"
					+ ("; the probes calibrate the total" if probed else ""),
				)
		res = replace(res, peak=_with_note(res.peak, peak_note), terms=terms or res.terms)
	return res


def _faketensor_disagreements(arch, setup) -> tuple[str, dict[str, float]] | None:
	"""(the traced case, {term: engine / trace}) for the terms that miss by more than 20% on the zoo model this
	record runs (scripts/faketensor_memory_check.py), matched on its shape, method, checkpointing and optimizer
	family; None for any other model (review, Nov 1: a Qwen case said nothing about another model or optimizer)."""
	from preflight.engine import _faketensor_check as ft

	if not isinstance(arch, memory.TransformerArch):
		return None
	opt = str(setup.optimizer).lower()
	family = "adafactor" if "adafactor" in opt else "adamw" if opt.startswith("adam") else None
	key = {
		"model": [arch.n_layers, arch.d_model, arch.vocab_size],
		"method": setup.method,
		"checkpointing": setup.grad_checkpointing != "none",
		"optimizer": family,
	}
	for case, row in ft.DATA["cases"].items():
		if row["key"] == key:
			worse = {t: d["median_ratio"] for t, d in row["terms"].items() if not d["agrees"]}
			return (case, worse) if worse else None
	return None


# the terms that grow with batch and length absorb what the model misses; the rest are set by the model's size
FLEXIBLE_TERMS = ("activations", "logits", "eval_spike")


def _term_estimates(stack: dict, peak: Estimate) -> dict:
	"""memory.<term> at the peak, adding up to the calibrated peak, with its relative band.

	C, Nov 1: scaling every term to the peak made a GRPO run's weights read 9.9 GB for a 494M-parameter model (5x),
	when the probe measured 2.0 GB. Weights, gradients, optimizer state, the context and the other size-set terms
	keep their modelled values; activations, logits and the eval spike take the rest of the peak, in proportion,
	or activations alone when the model has none. Only when the size-set terms already exceed the peak are all of
	them scaled down together."""
	vals = {t: max(0.0, stack.get(t, 0.0)) for t in MEMORY_TERMS}
	total = sum(vals.values())
	if not total or peak.status == "abstain":
		return {}
	lo_f, hi_f = peak.lo / peak.point, peak.hi / peak.point
	status = "predicted" if peak.status == "measured" else peak.status  # the split is modelled, not measured
	fixed = sum(v for t, v in vals.items() if t not in FLEXIBLE_TERMS)
	flex = sum(vals[t] for t in FLEXIBLE_TERMS)
	rest = peak.point - fixed
	if rest >= 0:
		why = (
			"weights, gradients, optimizer and caches as the model sizes them; activations, logits and any eval spike "
			"share the rest of the calibrated peak"
		)
		for t in FLEXIBLE_TERMS:
			vals[t] = rest * vals[t] / flex if flex else (rest if t == "activations" else 0.0)
	else:
		why = "split by the analytical model, scaled down to the calibrated peak"
		vals = {t: v * peak.point / total for t, v in vals.items()}
	return {
		f"memory.{t}": _est(v, v * lo_f, v * hi_f, "bytes", status, memory.MODEL_ID, why)
		for t, v in vals.items()
	}


def _pct(p: float) -> str:
	"""2.5%, 40%, 97.5%: one decimal only where it isn't zero."""
	return f"{round(p * 100, 1):g}%"


def _from_memory_estimate(
	est: memory.MemoryEstimate, mid: str, oom_source: str = "simulated band", floor: float | None = None
) -> _MemoryResult:
	"""`floor` is the lowest P(OOM) the held-out runs can give (half a count in n + 1), when they set it."""
	if est.status == "abstain":
		return _MemoryResult(
			_abstain("bytes", mid, est.reason), _abstain("probability", mid, est.reason), est
		)
	peak = _est(est.point, est.lo, est.hi, "bytes", est.status, mid, est.reason)
	# PM, Nov 1: every reason leads with the S2b word itself (fits below 1%, fails at 80% or more, tight between),
	# never a second scale ("tight: out of memory unlikely" read as two verdicts)
	word = est.oom_status
	# Chiya, Oct 31: a number only where the reliability diagram verifies it; in between, the 5-95% range
	# the bins verify the empirical P(OOM) from held-out residuals; a simulated band's P(OOM) is never verified
	empirical = oom_source != "simulated band"
	bins = conformal.TABLE.oom_bins if empirical else ()
	shown = oom_model.shown_as_number(est.p_oom, bins)
	evidence = oom_model.end_evidence(est.p_oom, bins)
	if evidence is not None:
		# Chiya, Oct 31: an end bin that doesn't verify shows what happened to similar runs, not a probability
		k, n = evidence
		why = f"{word}: {k} of {n} similar runs ran out of memory ({oom_source})"
		lo, hi = conformal.wilson(k, n)
	elif shown:
		why = f"{word}: P(OOM) {_pct(est.p_oom)} ({oom_source})"
		if word == "tight" and est.p_oom < oom_model.UNCERTAIN[0]:
			at_floor = floor is not None and est.p_oom <= floor + 1e-9
			why += (
				f"; that is the lowest these runs can show, and fits needs below {_pct(memory.FITS_BELOW)}"
				if at_floor
				else f"; fits needs below {_pct(memory.FITS_BELOW)}"
			)
		lo = hi = est.p_oom
	else:
		why = (
			f"{word}: P(OOM) between 5% and 95% ({oom_source}); a chance in that range isn't verified on real "
			"outcomes yet"
		)
		lo, hi = oom_model.UNCERTAIN
	if est.breakdown is not None and est.breakdown.peak_moment != "train":
		why += f"; the peak comes at {est.breakdown.peak_moment}, not during training steps"
	# a probe that fitted measured the peak, not an OOM: P(OOM) is "measured" only from an OOM probe
	oom_status = "predicted" if est.status == "measured" else est.status
	oom = _est(est.p_oom, lo, hi, "probability", oom_status, mid, why)
	return _MemoryResult(peak, oom, est)


def _empirical_memory(
	record: FlightRecord, values: dict, capacity: float, why_no_shape: str
) -> _MemoryResult:
	"""No model shape: memory as a line in work units through the probes (no analytical prior)."""
	mid = memory.MODEL_ID
	lens = record.job.lens
	pts = [
		memory.MemoryProbe(
			knobs.work_units(
				lens, _sized(knobs.Shape(), _probe_values(record, p), _padded_seq(record, {}, probe=p))[1]
			),
			float(p.memory.peak_reserved_bytes),
			0.0,
		)
		for p in record.probes
		if p.status in OK_PROBE and p.memory and p.memory.peak_reserved_bytes
	]
	shape_knobs = {"batch_size", "seq_len", "encode_batch_size", "image_size", "max_model_len"}
	moved = [
		kid
		for kid, v in values.items()
		if kid not in shape_knobs
		and not kid.startswith("fanout.")
		and "memory" in knobs.knob_spec(kid).affects
		and _sampled_differs(record, kid, v)
	]
	if moved:
		why = f"no model shape, so the probes' line can't follow {', '.join(moved)}; run a probe here"
		return _MemoryResult(_abstain("bytes", mid, why), _abstain("probability", mid, why))
	if len({p.tokens for p in pts}) < 2:
		why = f"no model shape ({why_no_shape}) and fewer than two probe sizes to fit a line through"
		return _MemoryResult(_abstain("bytes", mid, why), _abstain("probability", mid, why))
	cal = memory.Calibration.fit(pts)
	x = knobs.work_units(lens, _sized(knobs.Shape(), values, _padded_seq(record, values))[1])
	measured = _matching_probe(record, values) is not None and _target_gpu(record, values) == _env_gpu(record)
	cell = conformal.TABLE.lookup("memory.peak", _target_gpu(record, values), lens)
	h = (
		memory.HALF_WIDTH_CALIBRATED
		if measured
		else cal.half_width(x, base=cell.half_width if cell else None)
	)
	status = "measured" if measured else ("extrapolated" if cal.is_below(x) else "predicted")
	if not measured and (x > cal.range()[1] or h > memory.ABSTAIN_HALF_WIDTH):
		why = f"{x:.0f} work units is too far from the probes for a line fit; run a probe here"
		return _MemoryResult(_abstain("bytes", mid, why), _abstain("probability", mid, why))
	ctx = _context_bytes(record, _target_gpu(record, values))
	point = cal.correct(x, 0.0) + ctx
	if point <= 0:
		why = "the line through the probes goes below zero here; run a probe here"
		return _MemoryResult(_abstain("bytes", mid, why), _abstain("probability", mid, why))
	p = memory.p_oom(point, h / memory.Z90, capacity)
	hi = point * math.exp(h)
	if cal.is_below(x):  # the same floor as the analytical path (memory.predict_peak)
		hi = max(hi, cal.correct(cal.min_tokens, 0.0) + ctx)
	est = memory.MemoryEstimate(
		point, point * math.exp(-h), hi, "bytes", status, mid, p, memory.oom_status(p),  # type: ignore[arg-type]
		h / memory.Z90, None, "; ".join(x for x in (
			"fitted to the probes alone: the script map has no model shape",
			_band_note(cell) if cell is not None and not measured else None,
		) if x),
	)  # fmt: skip
	est = _with_reserved_margin(record, est, ctx, capacity, None)
	return _from_memory_estimate(est, mid)


def _band_note(cell: conformal.Cell) -> str:
	return f"band from held-out runs: {cell.label()}"


def _with_note(est: Estimate, note: str) -> Estimate:
	return est.model_copy(update={"reason": "; ".join(x for x in (est.reason, note) if x)})


# ---------------------------------------------------------------- step time

# Chiya, Nov 2: the MI300X corpus slice (295 configs, `b-amd-grouped`), scored once from the NVIDIA rows of the same
# configs; no refit on it, so a move across vendors still abstains, now saying what was measured
AMD_CROSS = (
	"measured on the MI300X (295 corpus configs, Nov 2): from NVIDIA rows of the same configs, 22% within 20% (NVIDIA "
	"torch 2.8.0 to MI300X torch 2.5.1+rocm6.2: card and software differ), 254 of 1,147 pairs, and inside its band on "
	"411 (36%), reading about 0.63x the measured step"
)


@dataclass(frozen=True)
class _StepResult:
	est: Estimate
	sigma: float  # log sigma of the step time, for totals and tail probabilities
	first_extra_s: float = 0.0  # compile and warm-up, paid once
	phases: dict | None = None  # seconds per phase, scaled to the step's point
	length: _LengthTerm | None = (
		None  # varying batch lengths: the total uses the mean, with its own length error
	)


def _step_x(lens: str, values: dict) -> float:
	"""Work per optimizer step from the settings: work units per micro-batch times accumulation."""
	return knobs.work_units(lens, values) * max(1, int(values.get("grad_accum", 1) or 1))


# A probe was cut by seq_len when its longest padded row reached the limit (rows are padded to the longest in
# their batch and never past seq_len). guess: without row lengths, a mean padded length at this share of the limit
# counts as cut; replaced by the step log's longest row whenever A's recorder has one (it does since Oct 28)
TRUNCATED_SHARE = 0.95
TOKEN_LENSES = ("finetune", "pretrain")  # RL work also counts completions, which the dataloader's tokens miss


def _measured_tokens(p) -> tuple[float, float] | None:
	"""(tokens, samples) one optimizer step of this probe processed, from A's recorder (Probe.extras)."""
	ex = p.extras or {}
	t, n = ex.get("tokens_per_step"), ex.get("samples_per_step")
	ok = all(
		isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0 for v in (t, n)
	)
	return (float(t), float(n)) if ok else None


def _token_points(record: FlightRecord) -> list[tuple[float, float, float, float, bool]] | None:
	"""(batch, accumulation, padded tokens per sample, seq_len, truncated) for every timed probe, or None unless
	the measured-token axis applies: a token lens, a batch_size knob, every timed probe reporting tokens (mixing
	measured and set work on one axis would bend the fit) and at least one probe the limit didn't cut."""
	if record.job.lens not in TOKEN_LENSES or all(k.id != "batch_size" for k in record.knobs):
		return None
	pts = []
	for p in record.probes:
		if p.status not in OK_PROBE or not p.step_time or not p.step_time.p50_s:
			continue
		m = _measured_tokens(p)
		pv = _probe_values(record, p)
		b = pv.get("batch_size")
		if m is None or not isinstance(b, (int, float)) or b <= 0:
			return None
		tokens, samples = m
		sq = float(pv.get("seq_len") or 0)
		longest = _longest_seen(p)
		cut = (longest >= sq) if longest is not None else (tokens / samples >= TRUNCATED_SHARE * sq)
		pts.append((float(b), samples / float(b), tokens / samples, sq, bool(sq) and cut))
	if not pts or all(trunc for *_, trunc in pts):
		return None  # the data may run longer than any limit seen: the settings are the axis
	return pts


def _on_token_axis(record: FlightRecord) -> bool:
	"""One switch for both sides of the step-time fit: probes and queries use measured tokens, or neither does."""
	return _token_points(record) is not None


def _step_work(record: FlightRecord, values: dict) -> float:
	"""Tokens one optimizer step will process at these settings (Chiya, Oct 30: scale by measured tokens, not by
	seq_len). Samples per step times padded tokens per sample, which A's probes measure: rows are padded to the
	longest in their batch, so the per-sample figure follows batch size, and seq_len only cuts it when the rows
	reach it. On the settings' axis when _on_token_axis says so, for probes and queries alike."""
	pts = _token_points(record)
	if pts is None:
		return _step_x(record.job.lens, values)
	b_q = float(values.get("batch_size") or 1)
	seq_q = float(values.get("seq_len") or 0) or math.inf
	# accumulation: what the probes measured (samples / batch); a declared grad_accum knob moves it by the ratio of
	# the query's value to the knob's current one, so a knob that disagrees with what ran (a DeepSpeed config) can't
	# put the query on a different scale from the probes
	measured = statistics.median(a for _, a, *_ in pts)
	accum = measured
	if any(k.id == "grad_accum" for k in record.knobs):
		cur = float(_current(record).get("grad_accum") or 1) or 1.0
		accum = measured * float(values.get("grad_accum") or cur) / cur
	by_batch: dict[float, list[float]] = {}
	for b, _, u, _, trunc in pts:
		if not trunc:
			by_batch.setdefault(b, []).append(u)
	free = sorted((b, statistics.fmean(us)) for b, us in by_batch.items())  # probes at one batch: their mean
	# padded tokens per sample at this batch: linear in log2(batch) between probed batches, flat beyond them
	if b_q <= free[0][0]:
		u_q = free[0][1]
	elif b_q >= free[-1][0]:
		u_q = free[-1][1]
	else:
		u_q = free[-1][1]
		for (b0, u0), (b1, u1) in itertools.pairwise(free):
			if b0 <= b_q <= b1:
				t = 0.0 if b1 == b0 else (math.log2(b_q) - math.log2(b0)) / (math.log2(b1) - math.log2(b0))
				u_q = u0 + t * (u1 - u0)
				break
	# a probe cut by its seq_len still bounds the rows from below: at its batch, rows are at least that long
	floor = max(
		(u for b, _, u, sq, trunc in pts if trunc and b == b_q and seq_q >= sq),
		default=0.0,
	)
	return b_q * accum * min(seq_q, max(u_q, floor))


def _probe_work(record: FlightRecord, p) -> float:
	"""A probe's place on the step-time axis: the tokens it measured when the record's probes all report them."""
	if _on_token_axis(record):
		m = _measured_tokens(p)
		if m is not None:
			return m[0]
	return _step_x(record.job.lens, _probe_values(record, p))


_PRECISION_KNOBS = ("precision", "dtype", "torch_dtype", "ptdtype")


def _precision(values: dict) -> str | None:
	"""The compute dtype a precision knob names; None when it says mixed, unknown or nothing usable."""
	for k in _PRECISION_KNOBS:
		v = values.get(k)
		if isinstance(v, str) and v not in ("mixed", "unknown"):
			try:
				return gpus.normalise_dtype(v)
			except ValueError:
				return None
	return None


def _dtype_ratio(
	gpu: str, src: str, tgt: str, compute_share: float = 0.7, tf32: bool = False
) -> float | None:
	"""Step-time ratio for a precision change on one GPU: compute share by peak, the rest by bytes.

	fp32 matmuls run at the TF32 peak when the probe had TF32 on (the kernel table says so)."""
	spec = gpus.get_gpu(gpu)
	ps, pt = spec.peak_flops(src, allow_tf32=tf32), spec.peak_flops(tgt, allow_tf32=tf32)
	if not ps or not pt:
		return None
	bs, bt = memory.dtype_bytes(src), memory.dtype_bytes(tgt)
	return compute_share * ps / pt + (1 - compute_share) * bt / bs


def _step(record: FlightRecord, values: dict) -> _StepResult:
	lens = record.job.lens
	sprobes, current_probe = [], None
	# every timed probe corrected, or none: a fit mixing corrected and uncorrected probes jumps between them
	timed = [p for p in record.probes if p.status in OK_PROBE and p.step_time and p.step_time.p50_s]
	# and all with one estimator: medians only if every timed probe has a step log (review, Nov 1)
	medians = all(_probe_rows(p) for p in timed)
	factors = {p.id: _length_factor(record, p, medians) for p in timed}
	if any(f is None for f in factors.values()):
		factors = {}
	for p in record.probes:
		st_ = p.step_time
		if p.status not in OK_PROBE or not st_ or not st_.p50_s:
			continue
		phases = steptime.phases_seconds(st_.by_phase, st_.p50_s) or {"step": st_.p50_s}
		f = factors.get(p.id)
		if f is not None:  # the run's batches, not the probe's: on the phases that follow the tokens
			phases = {k: v * f if k in TOKEN_PHASES or k == "step" else v for k, v in phases.items()}
		sprobes.append(steptime.StepProbe(_probe_work(record, p), phases, p.id))
		if p.kernels and current_probe is None:
			current_probe = p
	mid = steptime.MODEL_ID_SAME
	if not sprobes:
		return _StepResult(_abstain("s", mid, "no probe timed a step yet"), 0.0)
	model = steptime.SameGpuModel.fit(sprobes, constant_phases=steptime.CONSTANT_PHASES)
	cell = conformal.TABLE.lookup("time.step", _env_gpu(record), lens)
	if cell is not None:  # held-out conformal half-width replaces the simulated same-GPU sigma
		model = replace(model, sigma=cell.half_width / steptime.Z90)
	probe_gc = {bool(_probe_values(record, p).get("gradient_checkpointing", False)) for p in record.probes}
	recompute = bool(values.get("gradient_checkpointing")) and probe_gc == {False}
	match = _matching_probe(record, values)
	src_gpu, tgt_gpu = _env_gpu(record), _target_gpu(record, values)
	requested = _gpu_move(record, values)
	if requested and tgt_gpu is None:
		return _StepResult(_abstain("s", mid, f"{requested!r} isn't a GPU the table knows"), 0.0)
	if requested and src_gpu is None and tgt_gpu is not None:
		why = f"the probes ran without a known GPU (CPU, MPS or an unlisted card), so their times don't carry to {tgt_gpu}"
		return _StepResult(_abstain("s", steptime.MODEL_ID_CROSS, why), 0.0)
	same_gpu = tgt_gpu == src_gpu
	if not same_gpu and (gpus.is_unlisted(src_gpu) or gpus.is_unlisted(tgt_gpu)):
		why = f"{src_gpu if gpus.is_unlisted(src_gpu) else tgt_gpu} isn't in the GPU table, so time can't be carried across GPUs"
		return _StepResult(_abstain("s", steptime.MODEL_ID_CROSS, why), 0.0)
	est = model.predict(
		_step_work(record, values),
		recompute=recompute,
		measured_probe=match.id if match and same_gpu else None,
	)
	if est.status == "abstain":
		return _StepResult(_abstain("s", mid, est.reason), est.sigma_log)
	point, sigma, status, reasons = est.point, est.sigma_log, est.status, [est.reason] if est.reason else []
	lt = _length_term(record, values, match, _total_steps(record, values)[0], est.breakdown, bool(factors))
	if lt is not None:
		sigma = math.hypot(sigma, lt.step_sigma)
		reasons.append(lt.reason)
		f = factors.get(match.id) if match is not None else None
		if (
			status == "measured" and f is not None and abs(f - 1) > 0.01
		):  # the run's batches aren't the probe's
			status = "predicted"

	# precision moved away from what the probes ran
	q_prec = _precision(values)
	x_q = _step_work(record, values)
	by_dist = sorted(
		(
			abs(math.log(max(_probe_work(record, p), 1e-9) / max(x_q, 1e-9))),
			i,
			_precision(_probe_values(record, p)),
		)
		for i, p in enumerate(p for p in record.probes if p.status in OK_PROBE)
	)
	p_precs = {pr for _, _, pr in by_dist}
	if q_prec and p_precs and q_prec not in p_precs and gpus.is_unlisted(src_gpu):
		why = f"{src_gpu} isn't in the GPU table, so a precision change can't be scaled"
		return _StepResult(_abstain("s", steptime.MODEL_ID_CROSS, why), sigma)
	if q_prec and p_precs and q_prec not in p_precs and src_gpu:
		src_prec = next((pr for _, _, pr in by_dist if pr), None)  # the nearest probe's, deterministically
		tf32 = (
			bool(current_probe.kernels.allow_tf32)
			if current_probe is not None and current_probe.kernels
			else False
		)
		tf32 = bool(values.get("allow_tf32", values.get("tf32", tf32)))
		ratio = _dtype_ratio(src_gpu, src_prec, q_prec, tf32=tf32) if src_prec else None
		if ratio is None:
			why = f"{q_prec} has no fast path on {src_gpu}"
			return _StepResult(_abstain("s", steptime.MODEL_ID_CROSS, why), sigma)
		point *= ratio
		sigma = math.hypot(sigma, steptime.SIGMA_SAME_VENDOR)
		status = "extrapolated"
		reasons.append(f"scaled from the probes' {src_prec} to {q_prec} by the GPU's peak and bytes")
		mid = steptime.MODEL_ID_CROSS

	# another GPU than the probes ran on
	rough = False
	if not same_gpu:
		cross = None
		if current_probe is not None and current_probe.step_time and current_probe.step_time.p50_s:
			cross = steptime.predict_cross_gpu_from_probe(
				current_probe.kernels, current_probe.step_time.p50_s, src_gpu, tgt_gpu
			)
		# Chiya, Oct 31 (option b): with engine.steptime_cross_gpu on, a GPU move takes the held-out cross-GPU band,
		# shown past +-50% as rough; off, K1's simulated band and the usual abstain
		xcell = (
			conformal.TABLE.lookup("time.step.cross", tgt_gpu, None)
			if feature("steptime_cross_gpu")
			else None
		)
		if feature("steptime_cross_gpu") and gpus.get_gpu(src_gpu).vendor != gpus.get_gpu(tgt_gpu).vendor:
			# Chiya, Oct 31: abstain, labelled; Nov 2: the AMD slice measured the move and K1 misses it (AMD_CROSS)
			if tgt_gpu == "MI300X":  # the one AMD card measured; another AMD card falls to "not measured"
				why = (
					f"not shown across vendors: {tgt_gpu} is another vendor, and the move was {AMD_CROSS}, so it "
					"isn't a usable estimate; run a probe there"
				)
			else:
				why = (
					f"not shown across vendors: {tgt_gpu} is another vendor, and this move isn't measured "
					"(only NVIDIA to the MI300X is); run a probe there"
				)
			return _StepResult(_abstain("s", steptime.MODEL_ID_CROSS, why), sigma)
		if cross is not None and cross.status != "abstain":
			ratio = cross.point / current_probe.step_time.p50_s
			rough = bool(xcell)
			# Chiya, Oct 31: K1's band from held-out corpus pairs, never narrower than the simulated one
			cross_sigma = max(cross.sigma_log, xcell.half_width / steptime.Z90) if xcell else cross.sigma_log
			sigma = math.hypot(sigma, cross_sigma)
			if xcell:
				reasons.append(
					f"cross-GPU band from {xcell.n} held-out corpus pairs ({xcell.groups} architectures)"
				)
			if cross.reason:
				reasons.append(cross.reason)
		else:
			if cross is not None and "fast path" in (cross.reason or ""):
				return _StepResult(_abstain("s", steptime.MODEL_ID_CROSS, cross.reason), sigma)
			phases = {k: v for k, v in (est.breakdown or {}).items()}
			try:
				scaled = steptime.scale_phases_by_class(phases, src_gpu, tgt_gpu, dtype=q_prec or "bf16")
			except ValueError as exc:
				return _StepResult(_abstain("s", steptime.MODEL_ID_CROSS, str(exc)), sigma)
			ratio = sum(scaled.values()) / max(sum(phases.values()), 1e-12)
			class_sigma = steptime.cross_sigma(src_gpu, tgt_gpu) * steptime.SIGMA_SECONDARY_SPECS
			# review, Oct 31: never narrower than the held-out band either, since the card says held_out
			rough = bool(xcell)
			sigma = math.hypot(
				sigma, max(class_sigma, xcell.half_width / steptime.Z90) if xcell else class_sigma
			)
			status = "extrapolated"
			reasons.append("no kernel table on the probe, so phases are scaled by class")
		point *= ratio
		mid = steptime.MODEL_ID_CROSS
		if status == "measured":
			status = "predicted"

	# a different GPU count than the probes: data parallel adds the gradient all-reduce
	world = _n_gpus(values, record)
	probed_world = _probed_world(record)
	if world != probed_world:
		if probed_world != 1:
			why = f"the probes ran on {probed_world} GPUs; another count isn't modelled from there"
			return _StepResult(_abstain("s", steptime.MODEL_ID_COMM, why), sigma)
		grad_bytes = _grad_bytes(record, values)
		if grad_bytes is None:
			why = "the trainable parameter count isn't known, so the gradient all-reduce can't be sized"
			return _StepResult(_abstain("s", steptime.MODEL_ID_COMM, why), sigma)
		nodes = int(values.get("nodes") or 1)
		if nodes > 1:
			if world % nodes:
				why = f"{world} GPUs don't split evenly across {nodes} nodes"
				return _StepResult(_abstain("s", steptime.MODEL_ID_COMM, why), sigma)
			multi = steptime.multi_node_step(
				point, grad_bytes, nodes, world // nodes, tgt_gpu or src_gpu, _strategy(record, values),
				float(values.get("inter_node_gbps") or steptime.INTER_NODE_GBPS),
			)  # fmt: skip
			comm_sigma = steptime.SIGMA_MULTINODE
		else:
			multi = steptime.multi_gpu_step(
				point, grad_bytes, world, tgt_gpu or src_gpu, _strategy(record, values)
			)
			comm_sigma = steptime.SIGMA_COMM
		if multi.status == "abstain":
			return _StepResult(_abstain("s", steptime.MODEL_ID_COMM, multi.reason), sigma)
		# each part keeps its own error: the measured step its sigma, the estimated all-reduce its own
		comm = multi.breakdown.get("comm_exposed", 0.0)
		sigma = math.hypot(point * sigma, comm * comm_sigma) / multi.point
		lt = replace(lt, intact=False) if lt is not None else None
		point, status, mid = multi.point, "extrapolated", steptime.MODEL_ID_COMM
		reasons.append(
			multi.reason if nodes > 1 else f"{world} GPUs: the gradient all-reduce is estimated, not measured"
		)

	off = [
		k
		for k in _off_probe(record, values, also_ok=frozenset({"grad_accum"}))
		if k not in _PRECISION_KNOBS and k not in STEP_TIME_UNREAD
	]  # an unread knob is said below, as unmodelled for time
	if off and status in ("predicted", "measured"):
		status = "extrapolated"
		reasons.append(
			f"{', '.join(off)} {'is' if len(off) == 1 else 'are'} off the probed values; the model carries it there"
		)

	past = _past_scan_cut(record, values)
	if past and status in ("predicted", "measured"):
		status = "extrapolated"
		reasons.append(past)

	moves = [m for m in _unmodelled_moves(record, values, "time") if m != "gpus"]
	if moves:
		sigma *= UNMODELLED_WIDEN
		lt = replace(lt, intact=False) if lt is not None else None
		status = "extrapolated" if status != "abstain" else status
		reasons.append(f"{', '.join(moves)} isn't modelled for time yet; held at its probed effect")
	h = memory.Z90 * sigma
	if rough and h > memory.ABSTAIN_HALF_WIDTH:
		# Chiya, Oct 31 (option b): the held-out cross-GPU band is shown past +-50%, labelled, beside the probe offer
		reasons.append("rough, no probe on this GPU; run a probe here")
		status = "extrapolated"
	elif h > memory.ABSTAIN_HALF_WIDTH:
		why = "; ".join([*reasons, "the range would be wider than +-50%; run a probe here"])
		return _StepResult(_abstain("s", mid, why), sigma)
	first = 0.0
	for p in record.probes:
		if p.status in OK_PROBE and p.step_time and p.step_time.first_step_s and p.step_time.p50_s:
			first = max(first, p.step_time.first_step_s - p.step_time.p50_s)
	bd = {k: v for k, v in (est.breakdown or {}).items() if v > 0}
	total = sum(bd.values())
	phases = {k: v * point / total for k, v in bd.items()} if total else None
	return _StepResult(
		_est(point, point * math.exp(-h), point * math.exp(h), "s", status, mid, "; ".join(reasons) or None),
		sigma,
		first,
		phases,
		lt,
	)


def _probed_world(record: FlightRecord) -> int:
	counts = {_n_gpus(_probe_values(record, p), record) for p in record.probes if p.status in OK_PROBE}
	return min(counts) if counts else 1


def _grad_bytes(record: FlightRecord, values: dict) -> float | None:
	"""Gradient bytes one step all-reduces: trainable parameters in the compute dtype."""
	n = None
	for p in record.probes:
		if p.status in OK_PROBE and p.extras.get("params_trainable"):
			n = p.extras["params_trainable"]
			break
	if n is None:
		try:
			arch, setup, _ = knobs.setup_from(_script_map(record), values)
			n = memory.static_terms(arch, setup)["gradients"] / memory.dtype_bytes(setup.compute_dtype)
		except (ValueError, AttributeError, KeyError):
			return None
	return float(n) * memory.dtype_bytes(_precision(values) or "bf16")


def _strategy(record: FlightRecord, values: dict) -> str:
	sm = record.job.script_map
	dist = (sm.training.distributed if sm and sm.training else "none") or "none"
	return {"fsdp": "fsdp", "deepspeed": "zero3", "ddp": "ddp"}.get(dist, "ddp")


# ---------------------------------------------------------------- run totals


def checkpoint_every(record: FlightRecord, knob_values: dict | None = None) -> float | None:
	"""Steps between the script's checkpoint saves at these knob values, or None when it doesn't save."""
	values = _values(record, knob_values)
	tr = record.job.script_map.training if record.job.script_map else None
	every = values.get("checkpoint_every") or (tr.checkpoint_every if tr else None)
	return float(every) if every else None


def total_steps(record: FlightRecord, knob_values: dict | None = None) -> tuple[float | None, str]:
	"""Optimizer steps in the whole run at these knob values, or None and why (for the campaign planner)."""
	return _total_steps(record, _values(record, knob_values))


def _probe_dataset_len(record: FlightRecord, values: dict) -> float | None:
	"""The training set's rows per epoch as the probes' loaders saw them (A's `train_dataset_len`), for a record
	whose script map has none (C, Nov 1: the live LoRA record, written before the runner backfilled it). The same
	rules as the runner's backfill (runner/pipeline.py `_with_dataset_size`, review, Nov 1): no patched probe (a
	patch may subsample), an oom probe only past its first step (before it the latest loader may be an eval one), ok
	before cached before oom. Not when the query packs differently from the probe: packing changes the row count."""
	rank = {"ok": 0, "cached": 1, "oom": 2}
	ps = sorted(
		(
			p
			for p in record.probes
			if p.status in rank
			and not (p.extras or {}).get("patched")
			and (p.status != "oom" or (p.steps or 0) >= 1)
			and (p.extras or {}).get("train_dataset_len")
		),
		key=lambda p: rank[p.status],
	)
	if not ps:
		return None
	p = ps[0]
	pv = _probe_values(record, p)
	sm = record.job.script_map
	packs = values.get("packing") if "packing" in values else bool(sm and sm.data and sm.data.packing)
	if "packing" in values and bool(packs) != bool(pv.get("packing", packs)):
		return None
	if packs and values.get("seq_len") != pv.get("seq_len", values.get("seq_len")):
		return None
	return float(p.extras["train_dataset_len"])


_PACKING_MOVED = (
	"the run's length isn't known: the probes counted the dataset's rows at their own packing and seq_len, and "
	"packing changes the row count"
)


def _total_steps(record: FlightRecord, values: dict) -> tuple[float | None, str]:
	sm = record.job.script_map
	tr = sm.training if sm else None
	data = sm.data if sm else None
	steps = values.get("max_steps") or (tr.max_steps if tr else None)
	if steps:
		return float(steps), ""
	epochs = values.get("epochs") or (tr.epochs if tr else None) or 1.0
	n = values.get("n_examples") or (data.n_examples if data else None)
	if not n:
		n = _probe_dataset_len(record, values)
		if not n and _probe_dataset_len(
			record, {k: v for k, v in values.items() if k not in ("packing", "seq_len")}
		):
			return None, _PACKING_MOVED
	if not n:
		return None, "the run's length isn't known: no max_steps, and no dataset size to divide by the batch"
	per_step = (
		float(values.get("batch_size", 1) or 1)
		* float(values.get("grad_accum", 1) or 1)
		* _n_gpus(values, record)
	)
	return math.ceil(float(n) * float(epochs) / per_step), ""


def _overheads(record: FlightRecord, values: dict, steps: float) -> tuple[float, list[str]]:
	"""Seconds outside the timed steps: checkpoint saves and eval passes the probes measured."""
	sm = record.job.script_map
	tr = sm.training if sm else None
	extra, notes = 0.0, []
	every = values.get("checkpoint_every") or (tr.checkpoint_every if tr else None)
	ck = next((p.checkpoint for p in record.probes if p.checkpoint and p.checkpoint.seconds_mean), None)
	if every and ck:
		extra += math.floor(steps / float(every)) * ck.seconds_mean
	elif every:
		notes.append("checkpoint time not measured, so not included")
	ev_every = values.get("eval_every") or (tr.eval_every if tr else None)
	ev = next((p for p in record.probes if p.eval and p.eval.seconds_per_batch), None)
	if ev_every and ev:
		pass_s = ev.extras.get("eval_pass_s") or (ev.eval.batches * ev.eval.seconds_per_batch)
		extra += math.floor(steps / float(ev_every)) * pass_s
		if "eval_pass_s" not in ev.extras:
			notes.append("eval passes counted at the probe's capped size")
	return extra, notes


@dataclass(frozen=True)
class _TimeResult:
	total: Estimate
	step: Estimate
	sigma: float
	phases: dict | None = None
	first_s: float = 0.0  # compile and warm-up, paid once before the first step
	steps: float | None = None


# Chiya, Nov 3: an eval by command (lm_eval's loglikelihood tasks) has no optimizer steps; A's eval probe (probes.eval_units)
# times its scoring batches instead. Total = the probe's time outside its scoring batches (load, tokenizing, lm_eval's
# batch-size search; the warm-up batches count as passes) + scoring passes x the per-pass time, passes = the eval's
# requests / the rows a batch holds. Only at the probe's own GPU and knobs: nothing here scales to another.
EVAL_PASS_SIGMA = 0.30  # guess: log-space, as serving.SIGMA_UNCALIBRATED; no eval run has calibrated it
MODEL_ID_EVAL = "eval-passes@0.1.0"


def _eval_passes_total(record: FlightRecord, moved: bool = False) -> Estimate:
	if moved:
		why = "the eval's passes are timed at the probe's GPU and settings only; run a probe at these"
		return _abstain("s", MODEL_ID_EVAL, why)
	probe = next(
		(p for p in record.probes if p.status in OK_PROBE and (p.extras or {}).get("eval_forward_batches")),
		None,
	)
	if probe is None:
		return _abstain(
			"s", MODEL_ID_EVAL, "no eval probe timed a scoring batch; run a probe with probes.eval_units"
		)
	x = probe.extras
	per, rows, n = x.get("eval_forward_s_p50"), x.get("eval_forward_rows_mean"), x.get("eval_forward_batches")
	requests = x.get("eval_requests_total")
	if not per or not rows or not requests:
		why = "the probe timed scoring batches but didn't record how many requests the eval has, so the passes aren't known"
		return _abstain("s", MODEL_ID_EVAL, why)
	passes = math.ceil(requests / rows)
	fixed = max(0.0, (probe.duration_s or 0.0) - (n + EVAL_WARMUP_BATCHES) * per)
	pt = passes * per
	h = memory.Z90 * EVAL_PASS_SIGMA
	why = (
		f"{passes:,} scoring passes ({requests:,.0f} requests at {rows:.0f} a batch) x {per:.2f} s a pass (the probe's "
		f"{n:.0f} batches, p50) + {fixed:.0f} s outside the scoring batches; an eval that scores its longest requests first "
		"(lm_eval does) makes the probe's batches the slowest, so this leans long"
	)
	return _est(
		pt + fixed, pt * math.exp(-h) + fixed, pt * math.exp(h) + fixed, "s", "predicted", MODEL_ID_EVAL, why
	)


EVAL_WARMUP_BATCHES = (
	2  # cited: recorder/runtime.py, probes.eval_units runs two scoring batches before the measured ones
)


def _time(record: FlightRecord, values: dict) -> _TimeResult:
	step = _step(record, values)
	if feature("rl") and record.job.lens == "rl" and step.est.status != "abstain":
		step = _with_rollout(record, values, step)
	if step.est.status == "abstain":
		return _TimeResult(_abstain("s", MODEL_ID_TOTAL, step.est.reason), step.est, step.sigma)
	steps, why = _total_steps(record, values)
	if steps is None:
		return _TimeResult(_abstain("s", MODEL_ID_TOTAL, why), step.est, step.sigma)
	extra, notes = _overheads(record, values, steps)
	fixed = extra + step.first_extra_s
	lt = step.length
	tsig = step.sigma
	# a sum over many steps: the length error of a total, not of a median (only while nothing rescaled the sigma)
	if lt is not None and lt.intact:
		tsig = math.sqrt(max(0.0, step.sigma**2 - lt.step_sigma**2) + lt.total_sigma**2)
	h = memory.Z90 * tsig
	pt = steps * step.est.point * (lt.total_scale if lt is not None else 1.0)
	reason = "; ".join(x for x in [step.est.reason, *notes] if x) or None
	total = _est(
		pt + fixed,
		pt * math.exp(-h) + fixed,
		pt * math.exp(h) + fixed,
		"s",
		_total_status(step.est),
		MODEL_ID_TOTAL,
		reason,
	)
	return _TimeResult(total, step.est, step.sigma, step.phases, step.first_extra_s, steps)


def _total_status(step: Estimate) -> str:
	return "predicted" if step.status == "measured" else step.status


def _gpu_price(record: FlightRecord, gpu: str) -> gpus.GpuPrice | None:
	provider = record.env.provider if record.env else "modal"
	return gpus.price(gpu, provider) or gpus.on_demand_reference(gpu)


def _cost(record: FlightRecord, values: dict, time: Estimate) -> Estimate:
	if time.status == "abstain":
		return _abstain("usd", MODEL_ID_COST, time.reason or "no run time to price")
	gpu = _target_gpu(record, values)
	pr = _gpu_price(record, gpu) if gpu else None
	if pr is None:
		return _abstain("usd", MODEL_ID_COST, f"no price for {gpu or 'this GPU'} in the table")
	rate = pr.usd_per_hour * _n_gpus(values, record) / 3600.0
	why = f"{gpu} at ${pr.usd_per_hour:.2f}/h ({pr.provider}, {pr.as_of})"
	est = _est(time.point * rate, time.lo * rate, time.hi * rate, "usd", time.status, MODEL_ID_COST, why)
	row = prices.gpu_rate(gpu, pr.provider) or prices.market_reference(gpu)
	return _rated(est, "cost.gpu_hours", pr.usd_per_hour, row.id if row else None)


def _rated(
	est: Estimate | None, quantity: str, usd_per_unit: float | None, row: str | None, unit: str = "gpu_hours"
):
	"""Contract 0.6.0: a cost estimate says the quantity and rate it is, so a client re-prices it at the visitor's own
	rate without parsing the reason. Abstentions and unpriced estimates carry none."""
	if est is None or est.status == "abstain" or usd_per_unit is None:
		return est
	return est.model_copy(
		update={
			"rate": EstimateRate(
				quantity=quantity, usd_per_unit=float(usd_per_unit), unit=unit, price_row=row
			)
		}
	)


def _gpu_hours(record: FlightRecord, values: dict, time: Estimate) -> Estimate:
	"""GPU-hours of the run: time.total x the GPUs it holds. cost.total is this x the table's $/GPU-hour, so a user's
	own rate gives dollars in the browser from this alone (C, Nov 1)."""
	if time.status == "abstain":
		return _abstain("gpu_hours", MODEL_ID_COST, time.reason or "no run time to count GPU-hours from")
	if _target_gpu(record, values) is None:  # a CPU or unknown device: no GPU-hours to price at any rate
		return _abstain(
			"gpu_hours", MODEL_ID_COST, "the run has no GPU the table knows, so there are no GPU-hours"
		)
	n = _n_gpus(values, record) / 3600.0
	why = f"{_n_gpus(values, record)} GPU{'s' if _n_gpus(values, record) != 1 else ''} for the run's time"
	return _est(time.point * n, time.lo * n, time.hi * n, "gpu_hours", time.status, MODEL_ID_COST, why)


def _tail_prob(time: Estimate, sigma: float, limit: float) -> float:
	if sigma <= 0:
		return float(time.point > limit)
	z = (math.log(limit) - math.log(time.point)) / sigma
	return 0.5 * math.erfc(z / math.sqrt(2))


def _wall_cap(record: FlightRecord, time: _TimeResult) -> Estimate:
	cap = record.job.constraints.wall_time_cap_s
	if time.total.status == "abstain":
		return _abstain("probability", MODEL_ID_TOTAL, time.total.reason)
	p = _tail_prob(time.total, time.sigma, cap)
	why = f"P(the run outlasts the {cap / 3600:.0f} h job limit) {p:.0%}"
	if p >= 0.5:
		why += "; plan checkpoints and resubmits"
	return _est(p, p, p, "probability", time.total.status, MODEL_ID_TOTAL, why)


def _preemption(record: FlightRecord, values: dict, time: _TimeResult) -> Estimate:
	rule = record.job.constraints.preemption
	if time.total.status == "abstain":
		return _abstain("s", MODEL_ID_PREEMPT, time.total.reason)
	if rule.kind != "poisson" or not rule.mtti_s:
		why = "the cluster's interruption rate isn't known; set a mean time between interruptions"
		if rule.policy:
			why = f"{rule.policy}: {why}"
		return _abstain("s", MODEL_ID_PREEMPT, why)
	step = time.step.point or 0.0
	every = values.get("checkpoint_every") or (
		record.job.script_map.training.checkpoint_every
		if record.job.script_map and record.job.script_map.training
		else None
	)
	# no checkpoints (or fewer than one per run): an interruption loses everything so far
	interval = min(float(every) * step, time.total.point) if every else time.total.point
	# guess: a requeue plus reload takes at least a minute; replaced by A's measured first-step time per job
	restart = (rule.grace_s or 0.0) + 60.0
	m = rule.mtti_s

	def loss(t):
		"""Expected seconds lost: the run is n pieces of length L, each retried from its checkpoint until one attempt
		outlasts it, so failures per piece are geometric and interruptions during redone work count too. Exactly
		n((M + R)(e^(L/M) - 1) - L); for L much shorter than M, that is about t/M x (L/2 + R) (Young/Daly)."""
		if t <= 0:
			return 0.0
		piece = min(interval, t)
		x = piece / m
		if x > 50:  # interruptions come far faster than any piece finishes: it never ends
			return math.inf
		return math.ceil(t / piece) * ((m + restart) * math.expm1(x) - piece)

	why = (
		f"about {time.total.point / m:.1f} interruptions, each redoing the work since the last checkpoint"
		if every
		else f"about {time.total.point / m:.1f} interruptions, and with no checkpoints each one restarts the run"
	)
	return _est(
		loss(time.total.point),
		loss(time.total.lo),
		loss(time.total.hi),
		"s",
		time.total.status,
		MODEL_ID_PREEMPT,
		why,
	)


# ---------------------------------------------------------------- API lenses (agents)


def _sampled_model(record: FlightRecord) -> str | None:
	"""The model the agent probe ran: an untagged task belongs to it, not to whatever is chosen now."""
	for p in record.probes:
		if p.status in OK_PROBE and isinstance(p.knobs.get("model"), str):
			return p.knobs["model"]
	cur = _current(record).get("model")
	return cur if isinstance(cur, str) else None


def _tasks_for(record: FlightRecord, values: dict):
	s = record.agent_sample
	if s is None:
		return None, []
	model = values.get("model")
	default = _sampled_model(record)
	tasks = [t for t in s.tasks if model is None or (t.model or default) == model]
	return s, tasks


def _cap_turns(tasks, max_turns: int | None):
	"""Tasks as they'd run under a turn cap: cut at the cap (input by the recorded context, the rest pro rata).

	Returns the tasks, how many the cap cuts, and how many hit a lower cap in the sample (so their real
	cost past it is unknown when the cap is raised)."""
	if not max_turns:
		return tasks, 0, 0
	out, cut, hit = [], 0, 0
	for t in tasks:
		if t.turns > max_turns:
			cut += 1
			share = max_turns / t.turns
			ctx = t.context_tokens_by_turn
			in_share = sum(ctx[:max_turns]) / sum(ctx) if ctx and sum(ctx) > 0 else share
			out.append(
				t.model_copy(
					update={
						"turns": max_turns,
						"input_tokens": int(t.input_tokens * in_share),
						"cached_input_tokens": int(t.cached_input_tokens * in_share),
						"output_tokens": int(t.output_tokens * share),
						"reasoning_tokens": int(t.reasoning_tokens * share),
						"latency_s": t.latency_s * share,
						"cost_usd": None if t.cost_usd is None else t.cost_usd * in_share,
						"context_tokens_by_turn": list(ctx[:max_turns]),
						"stop_reason": "max_turns",
					}
				)
			)
		else:
			hit += t.stop_reason == "max_turns" and t.turns < max_turns
			out.append(t)
	return out, cut, hit


def _mark(e: Estimate, why: str) -> Estimate:
	"""An estimate the knobs pushed past what the sample shows: extrapolated, with the reason."""
	if e.status == "abstain":
		return e
	reason = "; ".join(x for x in (e.reason, why) if x)
	return e.model_copy(
		update={"status": "extrapolated" if e.status != "abstain" else e.status, "reason": reason}
	)


def _agent(record: FlightRecord, values: dict) -> dict[str, Estimate]:
	s, tasks = _tasks_for(record, values)
	ids = ["cost.total", "time.total", "memory.context_fill", "risk.runaway", "tokens.total"]
	if s is None:
		return {
			t: _abstain(_meta(t)[0], agent_cost.MODEL_ID, "no task sample yet; run the agent probe")
			for t in ids
		}
	model = values.get("model")
	if model is not None and not tasks:
		why = f"no sampled task ran on {model}; sample it first"
		return {t: _abstain(_meta(t)[0], agent_cost.MODEL_ID, why) for t in ids}
	pop = int(values.get("n_tasks") or s.population or 0) or None
	cap = values.get("max_turns")
	capped = frozenset(
		t.task_id for t in tasks if cap and t.turns > cap
	)  # stopped by the what-if, not the recorder
	tasks, cut, hit = _cap_turns(tasks, cap)
	sub = s.model_copy(update={"tasks": tasks})
	tot = agent_cost.agent_totals(sub, population=pop, capped_ids=capped)
	out = {}
	cost = tot["cost.total"]
	if "prompt_caching" in values and _sampled_differs(record, "prompt_caching", values["prompt_caching"]):
		out["cost.total"] = _abstain(
			"usd",
			agent_cost.MODEL_ID,
			"prompt-cache pricing needs the API price table; sample with caching on first",
		)
	else:
		out["cost.total"] = cost.to_estimate()
	out["tokens.total"] = tot["tokens.total"].to_estimate()
	rr = tot["risk.runaway"]
	out["risk.runaway"] = (
		_est(rr.point, rr.lo, rr.hi, "probability", rr.status, rr.model_id, rr.reason)
		if rr.status != "abstain"
		else _abstain("probability", rr.model_id, rr.reason)
	)
	out["time.total"] = _agent_time(s, tasks, pop, values)
	out["memory.context_fill"] = _context_fill(tasks, values)
	notes = []
	if cut:
		notes.append(
			f"{cut} of {len(tasks)} sampled tasks would stop at {values['max_turns']} turns, unfinished"
		)
	if hit:
		notes.append(f"{hit} sampled tasks hit a lower turn cap, so what they'd spend past it is unknown")
	moves = [
		m
		for m in _unmodelled_moves(record, values, "cost")
		if m not in ("prompt_caching", "max_turns", "n_tasks")
	]
	if moves:
		notes.append(f"{', '.join(moves)} isn't modelled for agents yet; held at its sampled effect")
	for why in notes:
		for tid in ("cost.total", "tokens.total", "time.total", "risk.runaway"):
			out[tid] = _mark(out[tid], why)
	return out


def _sampled_differs(record: FlightRecord, kid: str, v) -> bool:
	probed = [_probe_values(record, p).get(kid) for p in record.probes if p.status in OK_PROBE]
	return bool(probed) and not any(_same(v, pv) for pv in probed)


def _agent_time(sample, tasks, pop, values) -> Estimate:
	"""Wall clock: task-seconds over concurrency, or the rate limits, whichever binds."""
	mid = MODEL_ID_AGENT_TIME
	conc = max(1, int(values.get("concurrency", 1) or 1))
	lat = agent_cost.estimate_total([t.latency_s for t in tasks], pop, "s")
	if lat.status == "abstain":
		return _abstain("s", mid, lat.reason)
	bounds = [(lat.point / conc, lat.lo / conc, lat.hi / conc, f"{conc} tasks at a time")]
	if sample.rate_limit_tpm:
		tok = agent_cost.estimate_total([t.input_tokens + t.output_tokens for t in tasks], pop, "tokens")
		if tok.status != "abstain":
			k = 60.0 / sample.rate_limit_tpm
			bounds.append(
				(tok.point * k, tok.lo * k, tok.hi * k, f"the {sample.rate_limit_tpm:,.0f} tokens/min limit")
			)
	if sample.rate_limit_rpm:
		req = agent_cost.estimate_total([t.turns for t in tasks], pop, "requests")
		if req.status != "abstain":
			k = 60.0 / sample.rate_limit_rpm
			bounds.append(
				(
					req.point * k,
					req.lo * k,
					req.hi * k,
					f"the {sample.rate_limit_rpm:,.0f} requests/min limit",
				)
			)
	point, lo, hi, why = max(bounds, key=lambda b: b[0])
	lo = max(b[1] for b in bounds)
	hi = max(b[2] for b in bounds)
	return _est(point, lo, hi, "s", "predicted", mid, f"bound by {why}")


def _context_fill(tasks, values) -> Estimate:
	"""Share of the context limit the fullest tasks reach (95th percentile of each task's peak)."""
	mid = MODEL_ID_CONTEXT
	limit = values.get("context_limit")
	peaks = [max(t.context_tokens_by_turn) for t in tasks if t.context_tokens_by_turn]
	if not limit or len(peaks) < 3:
		why = "no context limit set" if not limit else "too few tasks recorded their context by turn"
		return _abstain("fraction", mid, why)
	arr = np.asarray(peaks, dtype=float) / float(limit)
	if arr.max() > 1.0:
		why = f"at {limit:,} tokens, {np.mean(arr > 1.0):.0%} of sampled tasks would overflow the context"
		return _abstain("fraction", mid, why)
	rng = np.random.default_rng(0)
	boots = np.quantile(arr[rng.integers(0, arr.size, (2000, arr.size))], 0.95, axis=1)
	lo, hi = (float(v) for v in np.quantile(boots, [0.05, 0.95]))
	point = float(np.quantile(arr, 0.95))
	overflowed = sum(1 for t in tasks if t.stop_reason == "context_overflow")
	status, why = "predicted", None
	if overflowed:
		status = "extrapolated"
		why = f"{overflowed} sampled task{'s' if overflowed != 1 else ''} hit the limit, so their real need is higher"
	return _est(point, lo, max(hi, point), "fraction", status, mid, why)


# ---------------------------------------------------------------- run cost (contract 0.2.0, gap 21)


def _scale(
	e: Estimate, k: float, unit: str, mid: str = MODEL_ID_RUN_COST, why: str | None = None
) -> Estimate:
	if e.status == "abstain":
		return _abstain(unit, mid, e.reason or "nothing to price")
	return _est(e.point * k, e.lo * k, e.hi * k, unit, _total_status(e), mid, why or e.reason)


def _sum(parts: list[Estimate], unit: str, why: str | None = None) -> Estimate:
	"""Lows and highs added line by line (contract 0.2.0): a deliberately cautious total."""
	if not parts:
		return _abstain(unit, MODEL_ID_RUN_COST, "nothing billed")
	bad = next((e for e in parts if e.status == "abstain"), None)
	if bad is not None:
		return _abstain(unit, MODEL_ID_RUN_COST, bad.reason or "a line can't be priced")
	order = ("measured", "predicted", "extrapolated")
	status = max((_total_status(e) for e in parts), key=order.index)
	return _est(
		sum(e.point for e in parts),
		sum(e.lo for e in parts),
		sum(e.hi for e in parts),
		unit,
		status,
		MODEL_ID_RUN_COST,
		why,
	)


def _gpu_row(
	gpu: str, provider: str, region: str | None, billing: str
) -> tuple[prices.GpuRate | None, str | None]:
	"""The price row for this GPU, and a note when it isn't the one asked for."""
	kind = "spot" if billing == "spot" else "on_demand"
	row = prices.gpu_rate(gpu, provider, region, kind)
	if row is not None:
		return row, None
	if kind == "spot":
		row = prices.gpu_rate(gpu, provider, region, "on_demand")
		if row is not None:
			return row, f"no spot rate listed for {gpu} on {provider}, so the on-demand rate is used"
	row = prices.market_reference(gpu)
	if row is not None:
		where = f"{provider} ({region})" if region else provider
		what = "spot or on-demand rate" if kind == "spot" else "rate"
		return (
			row,
			f"no {where} {what} for {gpu}; the cheapest listed on-demand rate ({row.provider}, {row.region}) is used",
		)
	return None, None


def _peak_moment(record: FlightRecord, values: dict, mem: _MemoryResult) -> str:
	"""When the peak comes: from the memory estimate, else the analytical breakdown (an OOM probe carries none)."""
	if mem.detail is not None and mem.detail.breakdown is not None:
		return mem.detail.breakdown.peak_moment
	gpu = _target_gpu(record, values)
	try:
		arch, setup, shape = knobs.setup_from(_script_map(record), values)
		shape, _ = _sized(shape, values, _padded_seq(record, values))
		cap = _capacity(record, gpu) if gpu else None
		return memory.breakdown(arch, setup, shape.batch, shape.seq, capacity_bytes=cap).peak_moment
	except (ValueError, AttributeError, KeyError):
		return "train"


def _crash_point(record: FlightRecord, values: dict, moment: str, steps: float) -> tuple[float, str]:
	"""Steps run before an out-of-memory error stops the job, and where that is in words."""
	if moment != "eval":
		return 1.0, "the first step"
	sm = record.job.script_map
	tr = sm.training if sm else None
	every = values.get("eval_every") or (tr.eval_every if tr else None)
	if every and float(every) < steps:
		return float(every), f"the first eval ({float(every):,.0f} steps in)"
	if every:
		return steps, f"the eval at the end of the run ({steps:,.0f} steps in)"
	epochs = float(values.get("epochs") or (tr.epochs if tr else None) or 1.0)
	n = max(1.0, math.ceil(steps / max(epochs, 1.0)))  # no step interval: eval at the end of each epoch
	return n, f"the first eval, at the end of the first epoch ({n:,.0f} steps in)"


def _at_risk(
	record: FlightRecord,
	values: dict,
	mem: _MemoryResult,
	tm: _TimeResult,
	usd_per_s: float | None,
	why_rate: str,
	unit: str = "usd",
) -> Estimate:
	"""P(OOM) x the cost up to the crash, in dollars or (with usd_per_s = GPUs / 3600) GPU-hours. Its range
	spans the P(OOM) bin's edges (S7), since the bin is all the screen claims about the probability until the
	reliability check passes."""
	oom = mem.oom
	if oom.status == "abstain":
		return _abstain(unit, MODEL_ID_RUN_COST, oom.reason or "no out-of-memory estimate")
	if tm.step.status == "abstain" or tm.steps is None:
		return _abstain(unit, MODEL_ID_RUN_COST, tm.total.reason or "no step time, so no cost up to a crash")
	if usd_per_s is None:
		return _abstain(unit, MODEL_ID_RUN_COST, why_rate)
	moment = _peak_moment(record, values, mem)
	n, where = _crash_point(record, values, moment, tm.steps)
	h = memory.Z90 * tm.sigma
	step = tm.step.point
	# checkpoint saves and eval passes up to the crash; the eval that crashes is counted whole, a cautious upper end
	fixed = tm.first_s + _overheads(record, values, n)[0]
	c_pt = (fixed + n * step) * usd_per_s
	c_lo = (fixed + n * step * math.exp(-h)) * usd_per_s
	c_hi = (fixed + n * step * math.exp(h)) * usd_per_s
	p = oom.point
	if oom.status == "measured":
		p_lo = p_hi = p
	else:
		edges = next((lo, hi) for lo, hi, label in oom_model.BINS if label == oom_model.bin_label(p))
		p_lo, p_hi = edges[0], min(edges[1], 1.0)
	if oom.status == "measured":
		why = f"fails: a probe ran out of memory at these settings, so the run stops at {where}; {why_rate}"
	else:
		why = (
			f"{memory.oom_status(p)}: P(OOM) between {_pct(p_lo)} and {_pct(p_hi)} x the cost up to {where}; "
			f"{why_rate}"
		)
	order = ("predicted", "extrapolated")
	status = max(
		("predicted" if oom.status == "measured" else oom.status, _total_status(tm.step)), key=order.index
	)
	return _est(p * c_pt, p_lo * c_lo, p_hi * c_hi, unit, status, MODEL_ID_RUN_COST, why)


def _basis(record: FlightRecord, provider: str | None, billing: str | None, region: str | None) -> CostBasis:
	prior = record.cost_basis
	if provider is None:
		provider = (
			prior.provider
			if prior
			else (record.env.provider if record.env and record.env.provider else "modal")
		)
	return CostBasis(
		price_table=f"prices@{prices.PRICE_TABLE_VERSION}",
		as_of=prices.PRICE_TABLE_VERSION,
		provider=provider,
		region=region or (prior.region if prior else None),
		billing=billing or (prior.billing if prior else "on_demand"),
		includes=["gpu", "tokens"],
	)


def _api_cost(record: FlightRecord, values: dict, basis: CostBasis) -> JobCost:
	est = _agent(record, values)
	tokens, usd = est["tokens.total"], est["cost.total"]
	model = values.get("model") or _sampled_model(record)
	row = prices.token_rate(model) if model else None
	rate = (
		usd.point / tokens.point
		if usd.status != "abstain" and tokens.status != "abstain" and tokens.point
		else None
	)
	line = CostLine(
		kind="tokens",
		label_key="cost.line.tokens",
		quantity=tokens,
		rate_usd=rate,  # blended USD per token over the sampled tasks' input and output
		usd=usd,
		price_row=row.id if row else None,
	)
	return JobCost(lines=[line], total_usd=_sum([usd], "usd", usd.reason))


def run_cost(
	record: FlightRecord,
	knob_values: dict | None = None,
	provider: str | None = None,
	billing: str | None = None,
	region: str | None = None,
) -> tuple[CostBasis, JobCost]:
	"""The run's cost breakdown and where its rates come from (contract 0.2.0). Preflight's own spend is
	`record.costs`, never here; money at risk sits beside the total, never in it."""
	record = _fit_view(record)
	basis = _basis(record, provider, billing, region)
	values = _values(record, knob_values)
	why = _invalid(values)
	if why:
		return basis, JobCost(total_usd=_abstain("usd", "input-check@0.1.0", why))
	if _api_lens(record):
		return basis, _api_cost(record, values, basis)
	mem, tm = _memory(record, values), _time(record, values)
	gpu, n = _target_gpu(record, values), _n_gpus(values, record)
	own = basis.billing == "own_cluster"
	row, note = _gpu_row(gpu, basis.provider, basis.region, basis.billing) if gpu else (None, None)
	if row is None:
		why_rate = f"no price for {gpu or 'this GPU'} in the table"
	else:
		why_rate = f"{gpu} at ${row.usd_per_hour:.2f}/h ({row.provider}, {row.as_of})"
		if note:
			why_rate += f"; {note}"
	if own and row is not None:
		why_rate = f"at the {row.provider} on-demand rate, ${row.usd_per_hour:.2f}/h, since your cluster bills in GPU-hours"
	per_s = row.usd_per_hour * n / 3600.0 if row else None

	def line(label: str, seconds: Estimate) -> CostLine:
		gpus_why = f"{n} x {gpu}" if gpu else None
		hours = _scale(
			seconds,
			n / 3600.0,
			"gpu_hours",
			why="; ".join(x for x in (gpus_why, seconds.reason) if x) or None,
		)
		billed = not own and row is not None
		if billed:
			usd = _rated(
				_scale(hours, row.usd_per_hour, "usd", why=why_rate),
				"line.quantity",
				row.usd_per_hour,
				row.id,
			)
		elif not own:  # billed in dollars, but this GPU has no price: say so on the line, not just the total
			usd = _abstain("usd", MODEL_ID_RUN_COST, why_rate)
		else:
			usd = None
		return CostLine(
			kind="gpu",
			label_key=label,
			quantity=hours,
			rate_usd=row.usd_per_hour if billed else None,
			usd=usd,
			price_row=row.id if row else None,
		)

	lines = [line("cost.line.gpu", tm.total)]
	if record.job.constraints.preemption.kind != "none":
		loss = _preemption(record, values, tm)
		if loss.status != "abstain":
			lines.append(line("cost.line.gpu_redo", loss))  # work redone after interruptions
	hours = _sum([x.quantity for x in lines], "gpu_hours")
	if own:
		total = None
		market = (
			_rated(
				_scale(hours, row.usd_per_hour, "usd", why=why_rate),
				"job_cost.gpu_hours",
				row.usd_per_hour,
				row.id,
			)
			if row
			else _abstain("usd", MODEL_ID_RUN_COST, why_rate)
		)
	else:
		total = _sum([x.usd for x in lines if x.usd is not None], "usd", why_rate)
		total = _rated(
			total, "job_cost.gpu_hours", row.usd_per_hour if row else None, row.id if row else None
		)
		market = None
	if not own and basis.billing == "spot" and len(lines) == 1:
		why_spot = (
			"spot: the interruption rate isn't known, so redone work isn't counted; set it under constraints"
		)
		total = (
			total.model_copy(update={"reason": "; ".join(x for x in (total.reason, why_spot) if x)})
			if total
			else total
		)
	at_risk = _at_risk(record, values, mem, tm, per_s, why_rate)
	if row is not None:  # GPU-hours at risk x the same rate, so the browser re-prices it with the total
		at_risk = _rated(at_risk, "job_cost.at_risk_gpu_hours", row.usd_per_hour, row.id)
	# contract 0.3.0: the same risk in GPU-hours, shown first on an own cluster (Chiya, Oct 27); it needs no price
	hours_why = f"{n} x {gpu}" if gpu else "GPU-hours"
	at_risk_hours = _at_risk(record, values, mem, tm, n / 3600.0, hours_why, unit="gpu_hours")
	return basis, JobCost(
		lines=lines,
		total_usd=total,
		gpu_hours=hours,
		market_equivalent_usd=market,
		at_risk_gpu_hours=at_risk_hours,
		at_risk_usd=at_risk,
	)


# ---------------------------------------------------------------- the three calls


def predict_as_run(record: FlightRecord, probe) -> Prediction:
	"""The prediction for a probe's own setting as that probe ran it: with no eval pass when it ran none
	(probe_ran_eval), so a held-out check compares like with like. For C's coverage checks."""
	record = _fit_view(record)
	return _predict_values(
		record, _values(record, _probe_values(record, probe)), eval_off=not probe_ran_eval(probe)
	)


def _fit_view(record: FlightRecord) -> FlightRecord:
	"""The record as the fits see it: without the learning-rate range test's probe (D's ask, from A's review, Nov 2).
	`p1-lr` runs at p1's knobs with its measured fields cleared, so it would count twice as an ok probe at p1's settings;
	it stays on the record, where its curve is read."""
	if not any(
		any((k == "lr_range" or k.startswith("lr_range.")) for k in (p.extras or {})) for p in record.probes
	):
		return record
	keep = [
		p
		for p in record.probes
		if not any((k == "lr_range" or k.startswith("lr_range.")) for k in (p.extras or {}))
	]
	return record.model_copy(update={"probes": keep})


def predict(record: FlightRecord, knob_values: dict | None = None) -> Prediction:
	"""Every target for this lens at these knob values (others at their current values)."""
	record = _fit_view(record)
	return _predict_values(record, _values(record, knob_values))


def _predict_values(record: FlightRecord, values: dict, eval_off: bool = False) -> Prediction:
	why = _invalid(values)
	if why:
		return Prediction({t.id: _abstain(t.unit, "input-check@0.1.0", why) for t in targets_for(record)})
	if _api_lens(record):
		return Prediction(_agent(record, values))
	mem = _memory(record, values, eval_off=eval_off)
	tm = _time(record, values)
	if feature("eval_passes") and record.job.lens == "eval" and tm.total.status == "abstain":
		moved = _target_gpu(record, values) != _env_gpu(record) or any(
			values.get(k) != v for k, v in _current(record).items()
		)
		tm = replace(tm, total=_eval_passes_total(record, moved), sigma=EVAL_PASS_SIGMA)
	out = {
		"memory.peak": mem.peak,
		"risk.oom": mem.oom,
		"time.step": tm.step,
		"time.total": tm.total,
		"cost.total": _cost(record, values, tm.total),
		"cost.gpu_hours": _gpu_hours(record, values, tm.total),
	}
	out.update(mem.terms or {})
	if tm.phases and tm.step.status != "abstain":
		h = memory.Z90 * tm.sigma
		for ph, v in tm.phases.items():
			out[f"time.phase.{ph}"] = _est(
				v, v * math.exp(-h), v * math.exp(h), "s", tm.step.status, tm.step.model_id
			)
	tcell = conformal.TABLE.lookup("time.step", _env_gpu(record), record.job.lens)
	if tcell is not None:  # _step takes the probe GPU's held-out band, and cross-GPU time builds on it
		for tid in [t for t in out if t in ("time.step", "time.total") or t.startswith("time.phase.")]:
			if out[tid].status not in ("abstain", "measured"):
				out[tid] = _with_note(out[tid], _band_note(tcell))
	c = record.job.constraints
	if c.wall_time_cap_s:
		out["risk.wall_cap"] = _wall_cap(record, tm)
	if c.preemption.kind != "none":
		out["risk.preemption_loss"] = _preemption(record, values, tm)
	if feature("rl") and record.job.lens == "rl":
		out.update(_rl_targets(record, values))
	if feature("divergence") and record.job.lens != "serving":
		out["risk.divergence"] = _divergence(record, values)
	if feature("serving") and record.job.lens == "serving" and not _QUIET.get():
		out.update(_serving_targets(record, values))
	note = _image_note(record)
	if note:
		out = {k: v if v.status == "abstain" else _with_note(v, note) for k, v in out.items()}
	return Prediction(out)


IMAGE_OVERRIDES = "probe.image_overrides"  # runner/pipeline.py's finding: probes.torch_new's overridden pins


def _image_note(record: FlightRecord) -> str | None:
	"""The overridden pins, as the runner's finding says them ("measured on transformers 4.52.3; your code pins
	4.46.3"), for every estimate's reason (B's review of probes.torch_new, Nov 3: the PM's fix)."""
	f = next((f for f in record.findings if f.id == IMAGE_OVERRIDES and f.source == "rule"), None)
	return (f.title[0].lower() + f.title[1:]) if f is not None and f.title else None


# the serving simulation takes tens of milliseconds; grids and rankings that don't read it skip it. A context
# variable, so a fill running in another thread (the pipeline runs one per job) never sees this one's setting
_QUIET = contextvars.ContextVar("preflight_quiet_serving", default=False)


class _quiet_serving:
	def __init__(self, on: bool):
		self.on = on
		self.token = None

	def __enter__(self):
		self.token = _QUIET.set(_QUIET.get() or self.on)

	def __exit__(self, *exc):
		_QUIET.reset(self.token)


# ---------------------------------------------------------------- self-hosted serving (engine/serving.py)


def _serve_probe(record: FlightRecord, values: dict | None = None):
	"""A probe that generated: the one at the query's settings when there is one, else the first. Its requests are the
	workload, and its wall time calibrates the simulation."""
	ps = [
		p
		for p in record.probes
		if p.status in OK_PROBE
		and p.generation
		and p.generation.sequences
		and p.generation.seconds_total
		and p.generation.new_tokens_mean
	]
	if values is not None:
		match = _matching_probe(record, values)
		if match is not None and match in ps:
			return match
	return ps[0] if ps else None


def _serve_setup(record: FlightRecord, values: dict) -> tuple[serving.ServeSetup | None, str]:
	try:
		arch, setup, _ = knobs.setup_from(_script_map(record), values)
	except (ValueError, AttributeError) as exc:
		return None, f"no model shape to size the KV cache: {exc}"
	if not isinstance(arch, memory.TransformerArch):
		return None, "the serving model needs a transformer's layers and heads"
	gpu = _target_gpu(record, values)
	if gpu is None or gpus.is_unlisted(gpu):
		return None, f"{gpu or 'this GPU'} isn't in the GPU table"
	weights = memory.static_terms(arch, setup)["weights"]
	kv = "fp8" if str(values.get("kv_cache_dtype", "")).startswith("fp8") else setup.compute_dtype
	return serving.ServeSetup(
		arch=arch,
		gpu=gpu,
		weight_bytes=weights,
		kv_dtype=kv,
		compute_dtype=setup.compute_dtype,
		tensor=setup.parallel.tensor,
		mem_util=float(values.get("gpu_memory_utilization") or 0.9),
		max_num_seqs=int(values.get("max_num_seqs") or 256),
		max_batched_tokens=int(values.get("max_batched_tokens") or 8192),
		max_model_len=int(values["max_model_len"]) if values.get("max_model_len") else None,
		capacity_bytes=float(_capacity(record, gpu)),
	), ""


def _serve_workload(
	record: FlightRecord, values: dict, p, probe_cap: float | None = None
) -> tuple[list[list[serving.Request]], bool]:
	"""(batches, online): the probe's requests, in as many back-to-back calls as the probe made, each with the
	probe's output lengths and a prompt length from the data (else the knob's cap). With a `request_rate` knob, the
	same requests arrive at that rate instead (online serving)."""
	g = p.generation
	pv = _probe_values(record, p)
	dat = p.data
	prompt = (
		(dat.tokens_per_sample_mean if dat else None)
		or values.get("max_prompt_len")
		or pv.get("max_prompt_len")
		or 512
	)
	p95 = (dat.tokens_per_sample_p95 if dat and dat.tokens_per_sample_mean else None) or None
	cap = probe_cap or values.get("max_new_tokens") or g.max_new_tokens
	rate = values.get("request_rate")
	calls = max(1, int(g.calls or 1))
	reqs = serving.workload(
		int(g.sequences),
		float(prompt),
		float(g.new_tokens_mean),
		float(p95) if p95 else None,
		float(g.new_tokens_p95) if g.new_tokens_p95 else None,
		rate_per_s=float(rate) if rate else None,
		max_output=int(cap) if cap else None,
		output_fit=_censored_fit(p),
	)
	if rate:
		return [reqs], True
	size = math.ceil(len(reqs) / calls)
	return [reqs[i : i + size] for i in range(0, len(reqs), size)], False


# guess: a probe whose outputs hit its cap this often hasn't seen how long they run uncapped (per call, not per
# response: the longest output in a call sets its pace)
TRUNCATED_CALL = 0.05
# guess: past this share at the cap, the probe hasn't seen the middle of its lengths, so no fit carries them past it
MAX_CENSORED = 0.5


def _probe_cap(p) -> tuple[float | None, float | None]:
	"""(the cap the probe's replies stopped at, the share of them still running there). A's serving probe (branch
	a-serving-cap, Nov 1) stops every reply at `extras.serving_probe_max_tokens` (1,024) and keeps the script's own
	cap in `generation.max_new_tokens` (4,096 on zoo/serving_vllm); a reply cut at the probe's cap counts as ending
	unknown, so the share still running is `extras.serving_probe_capped_fraction`, else the sequences whose ending
	isn't known (`extras.generation_ending_known`). Without the extra, the script's cap and `truncation_rate`."""
	g, ex = p.generation, (p.extras or {})
	cap = ex.get("serving_probe_max_tokens")
	script = float(g.max_new_tokens) if g.max_new_tokens else None
	# review, Nov 1: a script capped below the probe's cap stops there itself, with truncation_rate saying how often
	if cap and (script is None or float(cap) <= script):
		share = ex.get("serving_probe_capped_fraction")
		if share is None and g.sequences:
			# A writes generation_ending_known only when some ending is known: missing, every reply was cut. With
			# the two caps equal a cut reply may count either way, so the known ones that hit the cap count too
			known = ex.get("generation_ending_known")
			if known is None:
				share = 1.0
			else:
				unknown = max(0.0, g.sequences - float(known))
				share = min(1.0, (unknown + float(g.truncation_rate or 0.0) * float(known)) / g.sequences)
		return float(cap), (float(share) if share is not None else None)
	return script, g.truncation_rate


def _censored_fit(p) -> tuple[float, float] | None:
	"""The probe's output lengths as a log-normal fitted through its cap, when some replies were still running
	there (PM, Nov 1: A's vLLM probe caps new tokens at 1,024); None when none were, so the mean and p95 stand."""
	g = p.generation
	cap, t = _probe_cap(p)
	if not cap or not t or t > MAX_CENSORED or not g.new_tokens_mean:
		return None
	return serving.censored_lognormal(float(g.new_tokens_mean), cap, float(t))


def _serving_blocker(record: FlightRecord, values: dict, p) -> str | None:
	g = p.generation
	cap_p, t = _probe_cap(p)
	cap_q = values.get("max_new_tokens") or g.max_new_tokens
	if cap_p and cap_q and float(cap_q) > float(cap_p):
		per_call = max(1, round(g.sequences / max(g.calls, 1)))
		hit = 1 - (1 - t) ** per_call if t is not None else None
		if hit is not None and hit > TRUNCATED_CALL and _censored_fit(p) is not None:
			return None  # carried past the cap by the fit; the readout says so (_censored_note)
		if hit is None or hit > TRUNCATED_CALL:
			seen = f"{t:.0%} of" if t is not None else "an unknown share of"
			return (
				f"{seen} the probe's outputs hit its {float(cap_p):g}-token cap, so how long they run past it wasn't "
				"measured; probe at the higher cap"
			)
	return None


def _censored_note(values: dict, p) -> str | None:
	"""When outputs past the probe's cap come from the fit rather than a measurement."""
	g = p.generation
	cap_q = values.get("max_new_tokens") or g.max_new_tokens
	cap_p, t = _probe_cap(p)
	fit = _censored_fit(p)
	if fit is None or not cap_q or not cap_p or float(cap_q) <= cap_p:
		return None
	k = round(float(t) * g.sequences)
	return (
		f"{k} of the probe's {g.sequences} outputs were still running at its {cap_p:,.0f}-token cap; past it, lengths "
		f"come from a log-normal fitted through the cap (median {math.exp(fit[0]):,.0f} tokens), not a measurement"
	)


_SERVE_CAL: dict = {}  # (record id, probe id, setup) -> Calibration: the probe's simulation runs once per record


def _serving_targets(record: FlightRecord, values: dict) -> dict:
	mid = serving.MODEL_ID
	setup, why = _serve_setup(record, values)
	p = _serve_probe(record, values)
	if setup is None or p is None:
		why = why or "no probe generated, so there is no workload to simulate or calibrate on"
		return {t: _abstain(_meta(t)[0], mid, why) for t in SERVING_TARGETS}
	blocked = _serving_blocker(record, values, p)
	if blocked:
		return {t: _abstain(_meta(t)[0], mid, blocked) for t in SERVING_TARGETS}
	pv = _probe_values(record, p)
	probe_setup, _ = _serve_setup(record, pv)
	backend = p.generation.backend
	if (
		backend != "vllm"
	):  # an HF generate probe doesn't batch like vLLM: its wall time can't scale a vLLM simulation
		cal = serving.Calibration(
			source=f"uncalibrated: the probe generated with {backend or 'an unknown backend'}, not vLLM"
		)
	else:
		g = p.generation  # the measurements are in the key: a refilled probe with new numbers recalibrates
		key = (
			record.record_id,
			p.id,
			probe_setup or setup,
			g.calls,
			g.sequences,
			g.seconds_total,
			g.new_tokens_mean,
			g.new_tokens_p95,
			g.max_new_tokens,
			g.truncation_rate,
			_probe_cap(p),
		)
		if key not in _SERVE_CAL:
			if len(_SERVE_CAL) > 64:
				_SERVE_CAL.clear()
			_SERVE_CAL[key] = serving.calibrate(
				probe_setup or setup,
				# the probe's own replies stopped at its cap, so its simulated replies do too
				_serve_workload(record, pv, p, probe_cap=_probe_cap(p)[0])[0],
				float(p.generation.seconds_total),
			)
		cal = _SERVE_CAL[key]
		same_hw = probe_setup is not None and (
			probe_setup.gpu,
			probe_setup.tensor,
			probe_setup.compute_dtype,
		) == (
			setup.gpu,
			setup.tensor,
			setup.compute_dtype,
		)
		if not same_hw:  # the scale carries over, but one probe on other hardware doesn't narrow the band
			cal = replace(cal, sigma=serving.SIGMA_UNCALIBRATED, source=cal.source + ", on other hardware")
	batches, online = _serve_workload(record, values, p)
	note = _censored_note(values, p)
	if note:  # the tail past the cap is a fit: the band stays as wide as an uncalibrated one
		cal = replace(cal, sigma=max(cal.sigma, serving.SIGMA_UNCALIBRATED), source=f"{cal.source}; {note}")
	f = serving.forecast(setup, batches, cal, online)
	if f is None:
		why = f"the model doesn't fit: no room for a KV cache on {setup.gpu} at {setup.mem_util:.0%} of its memory"
		return {t: _abstain(_meta(t)[0], mid, why) for t in SERVING_TARGETS}
	status = "predicted" if _matching_probe(record, values) is p and backend == "vllm" else "extrapolated"
	if note:
		status = "extrapolated"
	out = {}
	for tid, r, unit in (
		("serving.throughput", f.throughput, "tokens/s"),
		("serving.ttft_p95", f.ttft_p95, "s"),
		("serving.tpot", f.tpot_p50, "s"),
	):
		out[tid] = _est(r.point, r.lo, r.hi, unit, status, mid, r.reason)
	kv = float(f.kv_capacity_tokens)
	out["serving.kv_tokens"] = _est(
		kv,
		kv,
		kv,
		"tokens",
		"predicted",
		mid,
		f"KV pool after the weights at {setup.mem_util:.0%} of the card; at most {f.peak_running} requests ran at once",
	)
	return out


# ---------------------------------------------------------------- RL post-training (engine/rl.py)


def _rl_probe(record: FlightRecord, values: dict | None = None):
	"""The probe that measured a rollout: the one at the query's settings when there is one, else the one whose
	sequences per call are nearest the query's."""
	ps = [
		p
		for p in record.probes
		if p.status in OK_PROBE and p.generation and p.generation.calls and p.generation.sequences
	]
	if not ps or values is None:
		return ps[0] if ps else None
	match = _matching_probe(record, values)
	if match is not None and match in ps:
		return match
	b = float(values.get("batch_size") or 1)
	return min(ps, key=lambda p: abs(math.log(float(_probe_values(record, p).get("batch_size") or 1) / b)))


# guess: a probe whose calls hit its cap this often (any response in the call reaching it) hasn't seen how long
# they run uncapped
TRUNCATED_PROBE = 0.05


def _probe_truncation(p) -> float | None:
	return (p.rl.truncation_rate if p.rl else None) or (
		p.generation.truncation_rate if p.generation else None
	)


def _lengths(p) -> rl.LengthSample | None:
	r = p.rl.response_length if p.rl else None
	if r is not None and r.mean and r.p95:
		return rl.LengthSample(r.mean, r.p95, r.max, r.n)
	g = p.generation
	if g and g.new_tokens_mean and g.new_tokens_p95:
		return rl.LengthSample(
			g.new_tokens_mean, g.new_tokens_p95, g.new_tokens_max or g.new_tokens_p95, g.sequences
		)
	return None


def _rollout_query(record: FlightRecord, values: dict, p) -> tuple[int, int, float | None, float | None]:
	"""(sequences per call at the probe, at the query, cap at the probe, cap at the query).

	Sequences per call follow the batch. In TRL's GRPO the batch counts completions, so group size moves how many
	prompts share them, not how many are generated (_group_size_meaning)."""
	pv = _probe_values(record, p)
	n_p = max(1, round(p.generation.sequences / p.generation.calls))
	a, b = pv.get("batch_size"), values.get("batch_size")
	n_q = max(1, round(n_p * float(b) / float(a))) if a and b else n_p
	cap_p = p.generation.max_new_tokens or pv.get("max_completion_len")
	cap_q = values.get("max_completion_len") or cap_p
	return n_p, n_q, (float(cap_p) if cap_p else None), (float(cap_q) if cap_q else None)


def _measured_rollout_s(p) -> float | None:
	st = p.step_time
	if not st or not st.p50_s:
		return None
	ph = steptime.phases_seconds(st.by_phase, st.p50_s)
	r = ph.get("rollout", 0.0) + ph.get("generate", 0.0)
	return r or None


def _rollout_blocker(record: FlightRecord, values: dict, p) -> str | None:
	"""Why the rollout can't be carried to these settings, or None."""
	pv = _probe_values(record, p)
	g_p, g_q = pv.get("group_size"), values.get("group_size")
	fw = record.job.framework or (record.job.script_map.framework if record.job.script_map else None)
	if g_p and g_q and float(g_p) != float(g_q) and fw != "trl":
		return (
			"whether the batch counts prompts or completions depends on the trainer, so a group-size change "
			"can't be timed here; probe at this group size"
		)
	cap_p = p.generation.max_new_tokens or pv.get("max_completion_len")
	cap_q = values.get("max_completion_len") or cap_p
	trunc = _probe_truncation(p)
	# the longest response in a call sets its pace, so what matters is how often a whole call hit the cap
	per_call = max(1, round(p.generation.sequences / max(p.generation.calls, 1)))
	call_hit = 1 - (1 - trunc) ** per_call if trunc is not None else None
	if cap_p and cap_q and float(cap_q) > float(cap_p) and (call_hit is None or call_hit > TRUNCATED_PROBE):
		seen = f"{trunc:.0%} of" if trunc is not None else "an unknown share of"
		return (
			f"{seen} the probe's responses hit its {float(cap_p):g}-token cap, so how long they run past it wasn't "
			"measured; probe at the higher cap"
		)
	return None


def _rollout(record: FlightRecord, values: dict) -> Estimate:
	mid = rl.MODEL_ID_ROLLOUT
	p = _rl_probe(record, values)
	if p is None:
		return _abstain("s", mid, "no probe timed a rollout (generation stats missing)")
	measured, sample = _measured_rollout_s(p), _lengths(p)
	if measured is None:
		return _abstain("s", mid, "the probe's step has no rollout phase to scale")
	if sample is None:
		return _abstain("s", mid, "the probe recorded no response lengths")
	blocked = _rollout_blocker(record, values, p)
	if blocked:
		return _abstain("s", mid, blocked)
	n_p, n_q, cap_p, cap_q = _rollout_query(record, values, p)
	iv = rl.rollout_seconds(measured, sample, n_p, cap_p, n_q, cap_q)
	same = n_q == n_p and cap_q == cap_p
	return _est(iv.point, iv.lo, iv.hi, "s", "predicted" if same else "extrapolated", mid, iv.reason)


def _with_rollout(record: FlightRecord, values: dict, step: _StepResult) -> _StepResult:
	"""Swap the step's rollout phase, which the per-phase line scales by tokens, for the batch-maximum model, which
	follows group size and the completion cap as well."""
	ph = dict(step.phases or {})
	linear = ph.get("rollout", 0.0) + ph.get("generate", 0.0)
	ro = _rollout(record, values)
	if (
		linear
		and ro.status == "abstain"
		and ro.reason
		and ("group-size" in ro.reason or "past it" in ro.reason)
	):
		# the step's biggest phase can't be carried to these settings, so the step can't either
		return replace(step, est=_abstain("s", step.est.model_id, f"rollout: {ro.reason}"))
	if not linear or ro.status == "abstain":
		return step
	rest = max(step.est.point - linear, 0.0)
	h = memory.Z90 * step.sigma
	point = rest + ro.point
	lo, hi = rest * math.exp(-h) + ro.lo, rest * math.exp(h) + ro.hi
	for k in ("rollout", "generate"):
		if k in ph:
			ph[k] = ro.point * ph[k] / linear
	sigma = max(step.sigma, math.log(hi / point) / memory.Z90 if point > 0 else step.sigma)
	note = f"rollout: {ro.reason}"
	est = _est(
		point,
		lo,
		hi,
		"s",
		step.est.status if ro.status == "predicted" else "extrapolated",
		step.est.model_id,
		"; ".join(x for x in (step.est.reason, note) if x),
	)
	return replace(step, est=est, sigma=sigma, phases=ph)


def _rl_signals(record: FlightRecord, values: dict, p) -> str | None:
	"""What the probe's TRL logs say about the run: KL to the reference and where its trend goes by the last step,
	the policy's entropy, the mean reward. None when the probe logged none of them."""
	if p is None or not p.rl:
		return None
	steps_total, _ = _total_steps(record, values)
	seen = int(p.steps or 0) or len(p.rl.kl) or len(p.rl.reward_mean)
	parts = [
		rl.kl_trend(p.rl.kl, seen, steps_total),
		rl.entropy_trend(
			p.rl.entropy,
			bool((p.extras or {}).get("rl_entropy_estimated")),
			bool((p.extras or {}).get("rl_entropy_exact")),
		),
		rl.reward_trend(p.rl.reward_mean),
	]
	if not p.rl.entropy:
		parts.append("no policy entropy was logged")
	parts = [x for x in parts if x]
	return "training signals: " + "; ".join(parts) if len(parts) > (0 if p.rl.entropy else 1) else None


def _rl_targets(record: FlightRecord, values: dict) -> dict:
	out = {"time.rollout": _rollout(record, values)}
	p = _rl_probe(record, values) or next((q for q in record.probes if q.status in OK_PROBE and q.rl), None)
	g = values.get("group_size")
	mid = rl.MODEL_ID_ZERO
	if p is None or not p.rl or not p.rl.pass_rate_per_prompt:
		out["rl.zero_spread"] = _abstain("fraction", mid, "the probe recorded no per-prompt pass rates")
	elif not g:
		out["rl.zero_spread"] = _abstain("fraction", mid, "no group size: set the group_size knob")
	elif len(p.rl.pass_rate_per_prompt) < rl.MIN_PROMPTS:
		n = len(p.rl.pass_rate_per_prompt)
		why = f"only {n} prompts with a pass rate; the share needs {rl.MIN_PROMPTS} to say anything about the rest"
		out["rl.zero_spread"] = _abstain("fraction", mid, why)
	else:
		k = p.rl.group_size or (_probe_values(record, p).get("group_size"))
		iv = rl.zero_spread(p.rl.pass_rate_per_prompt, int(g), int(k) if k else None)
		why = iv.reason
		if p.rl.zero_spread_fraction is not None and k:
			why += f"; the probe measured {p.rl.zero_spread_fraction:.0%} at its group size of {int(k)}"
		status = "predicted" if k and int(g) == int(k) else "extrapolated"
		out["rl.zero_spread"] = _est(iv.point, iv.lo, iv.hi, "fraction", status, mid, why)
	signals = _rl_signals(record, values, p)
	if signals and "rl.zero_spread" in out:
		# A's item 14 (PM, Nov 1): the probe's logged KL, entropy and reward, on the RL lens's risk readout
		out["rl.zero_spread"] = _with_note(out["rl.zero_spread"], signals)
	sample = _lengths(p) if p is not None else None
	cap = values.get("max_completion_len") or (
		p.generation.max_new_tokens if p is not None and p.generation else None
	)
	if sample is None or not cap:
		why = (
			"no response lengths recorded" if sample is None else "no completion cap: set max_completion_len"
		)
		out["rl.truncation"] = _abstain("fraction", rl.MODEL_ID_ROLLOUT, why)
	else:
		cap_p = p.generation.max_new_tokens if p.generation else None
		measured = _probe_truncation(p)
		if cap_p and float(cap) > float(cap_p) and (measured is None or measured > TRUNCATED_PROBE):
			out["rl.truncation"] = _abstain(
				"fraction",
				rl.MODEL_ID_ROLLOUT,
				_rollout_blocker(record, {**values, "group_size": None}, p)
				or "the probe's lengths were capped",
			)
		else:
			share, lo, hi = rl.truncation_band(sample, float(cap))
			why = f"share of responses reaching the {float(cap):g}-token cap, from the probe's length distribution"
			if measured is not None:
				why += f"; the probe measured {measured:.0%}"
			if cap_p and float(cap) > float(cap_p) and measured is not None:
				# past the probe's cap the tail wasn't seen: at most the share that hit the old cap
				share, lo, hi = min(share, measured), 0.0, measured
				why += f" at its {float(cap_p):g}-token cap, so at most that at a higher one"
			same = cap_p is not None and float(cap) == float(cap_p)
			out["rl.truncation"] = _est(
				share, lo, hi, "fraction", "predicted" if same else "extrapolated", rl.MODEL_ID_ROLLOUT, why
			)
	return out


def axis_values(knob: Knob, n: int = 6) -> list:
	"""Slider stops for one knob: log-spaced powers of two for wide integer ranges, the current value always in."""
	if knob.kind == "bool":
		return [False, True]
	if knob.kind == "categorical":
		vals = list(knob.domain.values) or [knob.current]
		return vals if knob.current in vals else [*vals, knob.current]
	lo = knob.domain.min if knob.domain.min is not None else knob.current
	hi = knob.domain.max if knob.domain.max is not None else knob.current
	lo, hi = float(lo), float(hi)
	if hi <= lo:
		return [knob.current]
	if knob.kind == "int":
		if lo > 0 and hi / lo >= 8:
			pts = [2.0**e for e in range(math.ceil(math.log2(lo)), math.floor(math.log2(hi)) + 1)]
			if len(pts) > n:
				idx = np.unique(np.round(np.linspace(0, len(pts) - 1, n)).astype(int))
				pts = [pts[i] for i in idx]
		else:
			pts = list(np.unique(np.round(np.linspace(lo, hi, n))))
		vals = sorted({int(v) for v in pts} | {int(knob.current)})
	else:
		if knob.domain.log and lo <= 0:
			lo = hi / 1000 if hi > 0 else lo  # a log axis can't start at zero
		log = lo > 0 and (knob.domain.log or hi / lo >= 100)
		pts = np.geomspace(lo, hi, n) if log else np.linspace(lo, hi, n)
		vals = sorted({float(f"{v:.4g}") for v in pts} | {float(knob.current)})
	return vals


_GRID_TARGETS = ("memory.peak", "risk.oom", "time.total", "cost.total", "cost.gpu_hours")
_AGENT_GRID_TARGETS = ("cost.total", "time.total", "memory.context_fill", "risk.runaway", "tokens.total")
# derived: three halvings and doublings around the current value, the span A's probes cover
LOCAL_FACTORS = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)  # fmt: skip


def local_values(knob: Knob) -> list:
	"""Slider stops around the current value: halvings and doublings inside the domain, current included."""
	if knob.kind in ("bool", "categorical"):
		return axis_values(knob)
	cur = float(knob.current)
	if cur <= 0:
		return axis_values(knob)
	lo = knob.domain.min if knob.domain.min is not None else -math.inf
	hi = knob.domain.max if knob.domain.max is not None else math.inf
	vals = {cur * f for f in LOCAL_FACTORS if lo <= cur * f <= hi}
	if knob.kind == "int":
		return sorted({round(v) for v in vals} | {int(knob.current)})
	return sorted({float(f"{v:.4g}") for v in vals} | {float(knob.current)})


def probed_values(record: FlightRecord, knob: Knob) -> list:
	"""Every value of this knob a probe ran at (ok, cached, or out of memory: each one measured)."""
	out = []
	for p in record.probes:
		if p.status in (*OK_PROBE, "oom"):
			v = _probe_values(record, p).get(knob.id)
			if v is not None and not any(_same(v, w) for w in out):
				out.append(v)
	return out


def _with_probed(record: FlightRecord, knobs_: list[Knob], axes: list[list]) -> list[list]:
	"""Each axis with every probed value of its knob and its current value in (PM, Nov 2: the pretrain sample probed
	batch 8 and 12, but its batch grid's ladder around 12 skipped 8, so the probed setting read "not trusted").
	Inside the cell budget: when the extra stops push a grid over it, the stops farthest from the current value that
	no probe ran at go first."""
	out = []
	for k, ax in zip(knobs_, axes, strict=True):
		if k.kind in ("bool", "categorical"):
			out.append(ax)
			continue
		keep = [*probed_values(record, k), k.current]
		lo, hi = k.domain.min, k.domain.max
		extra = [
			(int(v) if k.kind == "int" else float(v))
			for v in keep
			if isinstance(v, int | float) and not isinstance(v, bool)
			and (k.kind != "int" or float(v).is_integer())  # review: int(8.5) is a value no probe ran at
			and (lo is None or v >= lo) and (hi is None or v <= hi)
		]  # fmt: skip
		vals = list(ax)
		for v in extra:
			if not any(_same(v, w) for w in vals):
				vals.append(v)
		out.append(sorted(vals))

	def droppable(j: int) -> list:
		protected = [*probed_values(record, knobs_[j]), knobs_[j].current]
		return [v for v in out[j] if not any(_same(v, w) for w in protected)]

	while math.prod(len(a) for a in out) > MAX_GRID_CELLS:
		# drop one unprobed stop from the longest axis that still has one (review: not just the longest, which may
		# be all probed values), the stop farthest (in log) from the current value
		open_ = [j for j in range(len(out)) if knobs_[j].kind not in ("bool", "categorical") and droppable(j)]
		if not open_:
			break
		i = max(open_, key=lambda j: len(out[j]))
		k = knobs_[i]
		cands = droppable(i)
		cur = (
			float(k.current)
			if isinstance(k.current, int | float) and not isinstance(k.current, bool)
			else 0.0
		)

		def dist(v, cur=cur):
			try:
				return abs(math.log(float(v) / cur)) if v > 0 and cur > 0 else abs(float(v) - cur)
			except (TypeError, ValueError):
				return 0.0

		drop = max(cands, key=dist)
		out[i] = out[i][: out[i].index(drop)] + out[i][out[i].index(drop) + 1 :]
	return out


def build_grids(
	record: FlightRecord, knob_ids: list[str], targets: list[str] | None = None, local: bool = False
) -> list[Grid]:
	"""Predictions precomputed over the knobs' slider stops, one grid per target (row-major over the axes)."""
	if not 1 <= len(knob_ids) <= 4:
		raise ValueError("a grid takes one to four knobs")
	axes_knobs = [record.knob(k) for k in knob_ids]
	per = 8 if len(knob_ids) <= 2 else 5
	axes = [local_values(k) if local else axis_values(k, per) for k in axes_knobs]
	axes = _with_probed(record, axes_knobs, axes)
	cells = math.prod(len(a) for a in axes)
	if cells > MAX_GRID_CELLS:
		raise ValueError(f"{cells} cells is over the inline limit of {MAX_GRID_CELLS}")
	keep = _AGENT_GRID_TARGETS if _api_lens(record) else _GRID_TARGETS
	tids = targets or [t.id for t in targets_for(record) if t.id in keep]
	vals: dict[str, dict[str, list]] = {
		t: {"point": [], "lo": [], "hi": [], "abstain": [], "model": []} for t in tids
	}
	quiet = not set(tids) & set(SERVING_TARGETS)
	for combo in np.ndindex(*[len(a) for a in axes]):
		knob_set = {k.id: axes[i][j] for i, (k, j) in enumerate(zip(axes_knobs, combo, strict=True))}
		with _quiet_serving(quiet):
			pred = predict(record, knob_set).root
		for t in tids:
			e = pred.get(t)
			v = vals[t]
			if e is None or e.status == "abstain":
				v["point"].append(0.0), v["lo"].append(0.0), v["hi"].append(0.0), v["abstain"].append(True)
			else:
				(
					v["point"].append(e.point),
					v["lo"].append(e.lo),
					v["hi"].append(e.hi),
					v["abstain"].append(False),
				)
			v["model"].append(e.model_id if e is not None else "")
	fixed = {k.id: k.current for k in record.knobs if k.id not in knob_ids}
	grid_axes = [
		GridAxis(
			knob_id=k.id,
			values=a,
			scale="categorical" if k.kind in ("categorical", "bool") else ("log" if _is_log(a) else "linear"),
		)
		for k, a in zip(axes_knobs, axes, strict=True)
	]
	out = []
	for t in tids:
		v = vals[t]
		pt, lo, hi = (np.asarray(v[x], dtype=np.float64) for x in ("point", "lo", "hi"))
		scale = choose_scale(pt, lo, hi)
		models = [m for m, a in zip(v["model"], v["abstain"], strict=True) if m and not a] or [
			m for m in v["model"] if m
		]
		out.append(
			Grid(
				id=f"g.{t}.{'_'.join(knob_ids)}",
				target_id=t,
				unit=_meta(t)[0],
				axes=grid_axes,
				point=encode_f16(pt, scale),
				lo=encode_f16(lo, scale),
				hi=encode_f16(hi, scale),
				abstain=encode_mask(np.asarray(v["abstain"], dtype=bool)),
				model_id=max(set(models), key=models.count) if models else "none",
				scale=scale,
				fixed=fixed,
				independent_of=independent_of(record, t, knob_ids) or None,
			)
		)
	return out


def _is_log(values: list) -> bool:
	nums = [float(v) for v in values if isinstance(v, int | float) and not isinstance(v, bool)]
	return len(nums) >= 3 and nums[0] > 0 and nums[-1] / nums[0] >= 8


# ---------------------------------------------------------------- sensitivity


Metric = Callable[[Prediction, dict], float | None]


def _objective_fn(record: FlightRecord, objective: Objective) -> tuple[Metric, str]:
	"""The number to minimise for this objective, from a prediction and the knob values behind it."""
	api = _api_lens(record)

	def pick(tid) -> Metric:
		def f(pred, _values):
			e = pred.root.get(tid)
			return None if e is None or e.status == "abstain" else e.point

		return f

	kind = objective.kind
	if kind == "custom" and objective.target_id:
		return pick(objective.target_id), objective.target_id
	if objective.weights:

		def blend(pred, values):
			total = 0.0
			for tid, w in objective.weights.items():
				v = pick(tid)(pred, values)
				if v is None:
					return None
				total += w * v
			return total

		return blend, "weighted blend"
	if kind == "max_throughput":
		if api:
			return pick("time.total"), "time.total"

		def seconds_per_unit(pred, values):  # minimising this maximises throughput
			e = pred.root.get("time.step")
			if e is None or e.status == "abstain" or not e.point:
				return None
			return e.point / max(_step_work(record, values), 1e-12)

		return seconds_per_unit, "time.step per unit of work"
	table = {
		"min_time": "time.total",
		"min_cost": "cost.total",
		"fit_memory": "memory.context_fill" if api else "risk.oom",
		"min_risk": "risk.runaway" if api else "risk.oom",
		"max_quality_per_usd": "cost.total",
	}
	tid = table.get(kind, "time.total")
	return pick(tid), tid


def _feasible(record: FlightRecord, pred: Prediction) -> bool:
	oom = pred.root.get("risk.oom")
	return not (oom is not None and oom.status != "abstain" and oom.point >= memory.FAILS_AT)


def _noise(pred: Prediction, tid: str) -> float:
	"""Relative half-width of the objective's own band: moves smaller than this are not a signal."""
	e = pred.root.get(tid)
	if e is None or e.status == "abstain" or not e.point or e.unit == "probability":
		return 0.0
	return (e.hi - e.lo) / (2 * abs(e.point))


# derived: P(OOM) from a cell of n runs sits at (over + 0.5) / (n + 1) (engine/oom.py), so the smallest cell
# (conformal.MIN_CELL runs) can't place it closer than 0.5 / (n + 1) = 0.05 to 0 or 1
OOM_FLOOR = 0.5 / (conformal.MIN_CELL + 1)


def sensitivity(
	record: FlightRecord,
	objective: Objective,
	pinned: list[str] | set[str] | None = None,
	knobs: dict | None = None,
) -> list[KnobImpact]:
	"""Which knobs move the objective most across their range (one at a time, others at current values).

	Impact is the change across the range relative to the current value, capped at 1 (a probability
	objective uses the change itself). Knobs that move it less than its own band are left out.
	"""
	_, tid = _objective_fn(record, objective)
	with _quiet_serving(tid not in SERVING_TARGETS):
		return _sensitivity(record, objective, pinned, knobs)


def _sensitivity(
	record: FlightRecord,
	objective: Objective,
	pinned: list[str] | set[str] | None = None,
	knobs: dict | None = None,
) -> list[KnobImpact]:
	f, tid = _objective_fn(record, objective)
	start = dict(knobs or {})  # rank from this setting (C's provider passes the live one)
	pinned = set(pinned or ())
	base_values = _values(record, start)
	base_pred = predict(record, start)
	base = f(base_pred, base_values)
	prob = _TARGETS.get(tid, ("", None))[0] == "probability"
	noise = _noise(base_pred, "time.step" if tid.startswith("time.step") else tid)
	# when the current settings already fail, every neighbour may too; rank by the objective anyway
	check = _feasible(record, base_pred)
	out = []
	for k in record.knobs:
		if k.id in WORKLOAD_KNOBS or k.id in pinned:
			continue
		stops = axis_values(k, 6)
		if len(stops) < 2:
			continue
		scored = []
		for v in stops:
			q = {**start, k.id: v}
			pred = predict(record, q)
			y = f(pred, _values(record, q))
			if y is not None and (not check or _feasible(record, pred)):
				peak = pred.root.get("memory.peak")
				scored.append((v, y, peak.point if peak is not None and peak.point else math.inf))
		if len(scored) < 2:
			continue
		ys = [y for _, y, _ in scored]
		# P(OOM) is ranked between the smallest cell's floor and ceiling, so a lever doesn't outrank another only
		# because its range reached a cell with more runs and a lower floor; other probabilities are sampled rates
		ranked = [min(max(y, OOM_FLOOR), 1 - OOM_FLOOR) for y in ys] if tid == "risk.oom" else ys
		spread = max(ranked) - min(ranked)
		ref = abs(base) if base else max(abs(y) for y in ys) or 1.0
		impact = min(1.0, spread) if prob else min(1.0, spread / ref)
		if impact <= noise or impact == 0:
			continue
		best_v, _, best_peak = min(zip(scored, ranked, strict=True), key=lambda s: (s[1], s[0][2]))[0]
		direction = "either"
		if k.kind in ("int", "float"):
			xs = np.asarray([float(v) for v, _, _ in scored])
			r = np.corrcoef(xs, np.asarray(ys))[0, 1] if np.ptp(xs) > 0 and np.ptp(ys) > 0 else 0.0
			if r <= -0.5:
				direction = "increase"
			elif r >= 0.5:
				direction = "decrease"
		why = f"{tid} from {min(ys):.3g} to {max(ys):.3g} across the range; best at {best_v}"
		if k.id in TRUNCATING_KNOBS:
			why += "; shorter cuts what the model sees, so check quality before taking it"
		out.append(
			(
				KnobImpact(knob_id=k.id, impact=impact, direction=direction, best_value=best_v, reason=why),
				best_peak,
			)
		)
	# ties (two levers both reaching P(OOM)'s floor, say) go first to the lever that doesn't cut what the model
	# sees, then to the one leaving more memory headroom
	out.sort(key=lambda o: (-round(o[0].impact, 6), o[0].knob_id in TRUNCATING_KNOBS, o[1]))
	return [r for r, _ in out]


# ---------------------------------------------------------------- the whole predictions block


DIVERGENCE_INPUTS = ("lr", "warmup", "warmup_steps", "lr_schedule", "grad_clip")


def _divergence_model():
	try:
		from preflight.engine._divergence_model import MODEL
	except ImportError:  # not fitted yet: the readout abstains and says so
		return None
	return MODEL


def _divergence(record: FlightRecord, values: dict) -> Estimate:
	"""P(the run diverges later), from the longest probe's stability series at the probe's settings."""
	mid = divergence.MODEL_ID
	model = _divergence_model()
	if model is None:  # review, Nov 1: the verdict comes first, whatever the knobs or the probe
		return _abstain("probability", mid, divergence.warn(None, {}, {})["reason"])
	if _off_probe(record, values, also_ok=frozenset({"grad_accum"})):
		return _abstain(
			"probability", mid, "the warning reads the probe's own steps; probe this setting to see it"
		)
	probes = [
		p for p in record.probes if p.status in OK_PROBE and p.stability and p.stability.series.get("loss")
	]
	if not probes:
		return _abstain("probability", mid, "no probe logged a loss series")
	p = max(probes, key=lambda q: len(q.stability.series["loss"]))
	# review, Nov 2: the classifier reads the learning rate, the warm-up, the schedule and the clipping, which the free
	# knobs carry past for memory and time; a move of any of them away from the probe's own value isn't the probe's risk
	pv = _probe_values(record, p)
	moved = [k for k in DIVERGENCE_INPUTS if k in values and not _same(values[k], pv.get(k))]
	if moved:
		return _abstain(
			"probability",
			mid,
			f"the warning reads the probe's own {', '.join(moved)}; probe this setting to see it",
		)
	cfg = {**{kv.id: kv.current for kv in record.knobs}, **p.knobs}
	cfg.setdefault("precision", cfg.get("dtype"))
	w = divergence.warn(model, p.stability.series, cfg)
	if w["status"] == "abstain":
		return _abstain("probability", mid, w["reason"])
	return _est(w["point"], w["point"], w["point"], "probability", "predicted", mid, w["reason"])


_CARDS = {
	divergence.MODEL_ID: (
		"classifier",
		"Not used: the warning comes from the threshold rules (D6). Held out on 720 corpus runs (377 on L4, 343 on "
		"A10; 44 diverged): AUROC 0.66 (95% 0.56 to 0.76) against the 0.85 target, lead 138 steps (3 to 269: the point estimate clears 100, the interval doesn't), "
		"73 of 676 stable runs alarmed. Why: 37 of the 44 divergences are fp16 blow-ups, and 15 come in the first 30 steps, "
		"before the 25-step window can say much. Held out by settings family (precision, qk-norm, init), not by "
		"architecture: every sweep run is the same small GPT",
	),
	serving.MODEL_ID: (
		"simulation",
		"KV-cache capacity, prefill and decode rooflines and a continuous-batching scheduler (vLLM, Vidur); one scale "
		"factor fitted on the probe's own requests; not yet checked against a full vLLM run (validation row: "
		"tokens/sec within 15%)",
	),
	rl.MODEL_ID_ZERO: (
		"bootstrap",
		"p^G + (1-p)^G per prompt, each pass rate with a Jeffreys posterior, prompts resampled (DAPO, Dr. GRPO); "
		"not yet checked against a full RL run (validation row: GRPO zoo script)",
	),
	rl.MODEL_ID_ROLLOUT: (
		"bootstrap",
		"the expected longest capped response in a batch, bootstrapped from the probe's lengths (HybridFlow); "
		"not yet checked against a full RL run (validation row: rollout time within 20%)",
	),
	memory.MODEL_ID: ("analytical", "Korthikanti-style accounting, calibrated by the probes (E3); S1, S2a-d"),
	steptime.MODEL_ID_SAME: (
		"analytical",
		"per-phase linear fits on the probes; S4b below the smallest probe",
	),
	steptime.MODEL_ID_CROSS: ("analytical", "per-kernel roofline replay (K1), class fallback (K4); S4"),
	MODEL_ID_TOTAL: ("analytical", "steps x step time plus measured checkpoint and eval overheads"),
	MODEL_ID_COST: ("rule", "GPU-hours x the dated price in engine/prices"),
	MODEL_ID_RUN_COST: (
		"rule",
		"run cost lines as quantity x a dated price-table rate, totals added line by line; money at risk is "
		"P(OOM) over its bin's edges x the cost up to the crash (S7)",
	),
	MODEL_ID_PREEMPT: (
		"simulation",
		"Poisson interruptions, half a checkpoint interval redone each (Young/Daly)",
	),
	agent_cost.MODEL_ID: ("bootstrap", "bootstrap plus Clopper-Pearson runaway margin; S6, S6b"),
	MODEL_ID_AGENT_TIME: ("bootstrap", "task-seconds over concurrency, or the rate limits"),
	MODEL_ID_CONTEXT: ("bootstrap", "95th percentile of each task's peak context over the limit"),
}


CALIBRATION = "simulated"  # Review, Oct 27: until real held-out runs replace the simulated constants


_CARD_TARGET = {
	memory.MODEL_ID: "memory.peak",
	steptime.MODEL_ID_SAME: "time.step",
	steptime.MODEL_ID_CROSS: "time.step.cross",
}


# the models whose own predictions the corpus scores: memory and same-GPU step time leave-one-architecture-out, the
# cross-GPU replay leave-one-GPU-out (Chiya, Oct 31: installed)
_CARD_SCORED = frozenset({memory.MODEL_ID, steptime.MODEL_ID_SAME, steptime.MODEL_ID_CROSS})

# Chiya, Oct 31: what a held-out band does and doesn't cover yet, said on the card itself
_CARD_CAVEAT = {
	steptime.MODEL_ID_SAME: "Calibrated on corpus models, not users' scripts; checked on unseen scripts Nov 2.",
	steptime.MODEL_ID_CROSS: (
		"Rough, no probe on this GPU: the band comes from corpus pairs, scored on GPUs left out, and is shown past "
		"+-50% with a run-a-probe offer beside it. Calibrated on corpus models' op tables, not users' kernel tables; "
		"checked on unseen scripts Nov 2. A move to another vendor's GPU abstains: "
		+ AMD_CROSS
		+ ". The pairs share "
		"architectures: clustered by architecture, the 95% interval on coverage is 90% to 96%."
	),
}


def _calibration(mid: str, record: FlightRecord | None = None) -> tuple[str, float | None]:
	"""("held_out", None) when the conformal table has a cell this record's band really uses (its GPU and lens, or
	the merged cells it falls back to), else ("simulated", None). Without a record: any cell for the target."""
	target = _CARD_TARGET.get(mid)
	if not target:
		return CALIBRATION, None
	if mid == steptime.MODEL_ID_CROSS and not feature("steptime_cross_gpu"):
		return CALIBRATION, None  # the held-out band is installed but not switched on
	if record is not None:
		cell = conformal.TABLE.lookup(target, _env_gpu(record), record.job.lens)
		return ("held_out", None) if cell is not None else (CALIBRATION, None)
	cells = [c for k, c in conformal.TABLE.cells.items() if k[0] == target]
	return ("held_out", None) if cells else (CALIBRATION, None)


def _cards(ids: set[str], record: FlightRecord | None = None) -> list[ModelCard]:
	"""Model cards. With a record, a card says held_out only when that record's own band comes from held-out runs:
	a fine-tune's step time keeps its simulated band, so its card says so, whatever other lenses have."""
	out = []
	for mid in sorted(ids):
		kind, notes = _CARDS.get(mid, ("analytical", None))
		calib, _ = _calibration(mid, record)
		target = _CARD_TARGET.get(mid)
		extra: dict = {}
		if calib == "held_out" and target and mid in _CARD_SCORED:
			t = conformal.TABLE
			extra["training_data"] = f"{t.built_from}; table {t.version}"
			if target in t.coverage and t.coverage[target][1]:
				h, n = t.coverage[target]
				extra["coverage"] = h / n
			extra["per_gpu_coverage"] = {g: h / n for g, (h, n) in t.holdout.get(target, {}).items() if n}
			notes = f"{notes or ''}. {_CARD_CAVEAT.get(mid, '')} {t.describe(target)}".strip(". ")
			if mid == steptime.MODEL_ID_CROSS:
				# Chiya, Oct 31: the A100 holdout on the card too; it scored same-GPU step time, and the card says so
				a100 = [v for g, v in t.holdout.get("time.step", {}).items() if g.upper().startswith("A100")]
				h, n = (sum(v[i] for v in a100) for i in (0, 1))
				if n:
					notes += f" A100 holdout (same-GPU step time, scored once): {h} of {n} inside the band."
		if mid == divergence.MODEL_ID:
			# scored on held-out sweep runs and not used (Chiya, Nov 1); its card says so, not "simulated"
			calib = "held_out, not used"
			extra["training_data"] = (
				"corpus divergence sweep, 720 runs on L4 and A10 (remote-runs b-divergence-export-2, Nov 1); "
				"report: corpus/reports/divergence_nov1.json"
			)
			extra["heldout_error"] = {"auroc": 0.657, "lead_steps": 138.5, "false_alarm_rate": 73 / 676}
		out.append(
			ModelCard(
				id=mid,
				kind=kind,
				version=mid.split("@")[-1],
				training_data=extra.pop(
					"training_data",
					"none held out yet; bands sized by simulation (docs/decisions/engine-settings.md)",
				),
				notes=f"calibration: {calib}. {notes or ''}".strip(),
				**extra,
			)
		)
	return out


def _probed_ranges(record: FlightRecord) -> list[Knob]:
	out = []
	for k in record.knobs:
		vals = [
			_probe_values(record, p).get(k.id)
			for p in record.probes
			if p.status in OK_PROBE and k.kind in ("int", "float")
		]
		nums = [float(v) for v in vals if isinstance(v, int | float) and not isinstance(v, bool)]
		out.append(k.model_copy(update={"probed_range": Range(min=min(nums), max=max(nums))}) if nums else k)
	return out


# cited: below 5%, where P(OOM) shows as a number (Chiya, Oct 31). Such a cell reads "tight" unless it is below 1%
# (S2b), so the guide offers it as "measure these first", never as a fix to apply (PM, Nov 1)
CLEARS_P_OOM = oom_model.UNCERTAIN[0]
MAX_CLEARING_PAIRS = 3  # guess: enough for the guide to offer a choice; each pair adds one local 2-D grid
# moves that can only add memory are never tried; bools are tried at the value that saves memory
_SAVES_AT = {"eval_no_grad": True, "gradient_checkpointing": True, "packing": False}
# review, Nov 1: the guide offers GPUs on their own, and a GPU swap "clears" any run (a B200 did on the LoRA record,
# so no pair was ever searched); eval_every doesn't move the peak; parallelism saves memory going up, not tried
_NOT_A_CLEARING_KNOB = {"gpu_type", "gpus", "eval_every", "tensor_parallel", "pipeline_parallel"}
CLEARING_GRID_SUFFIX = ".clears"  # pair grids' ids end with this, apart from a record's own 2-D grids


def _saving_values(k: Knob) -> list:
	"""The local values of a memory knob that could lower the peak: smaller sizes, the memory-saving side of a
	switch, any other category."""
	if k.kind == "bool":
		v = _SAVES_AT.get(k.id)
		return [v] if v is not None and v != k.current else []
	if k.kind == "categorical":
		return [v for v in local_values(k) if v != k.current]
	return [v for v in local_values(k) if float(v) < float(k.current)]


def _p_oom(record: FlightRecord, values: dict) -> float | None:
	e = predict(record, values).root.get("risk.oom")
	return e.point if e is not None and e.status != "abstain" else None


def clearing_pairs(record: FlightRecord, max_pairs: int = MAX_CLEARING_PAIRS) -> list[tuple[str, str]]:
	"""Pairs of memory knobs that together clear an out-of-memory risk no single move clears (Chiya, Nov 1: the
	guide had nothing to offer on the LoRA zoo record, where eval batch 4 with seq_len 512 reads 17.5 GiB at P(OOM)
	2.5%, tight, but every single move still fails). Empty when the current setting already clears, or one move does.
	Ranked by how few steps from the current setting the nearest clearing cell is."""
	base = _p_oom(record, {})
	if base is None or base <= CLEARS_P_OOM:
		return []
	mem = [
		k
		for k in record.knobs
		if k.id not in WORKLOAD_KNOBS
		and k.id not in _NOT_A_CLEARING_KNOB
		and "memory" in knobs.knob_spec(k.id).affects
		and _saving_values(k)
	]
	singles = {k.id: _saving_values(k) for k in mem}
	for k in mem:
		for v in singles[k.id]:
			p = _p_oom(record, {k.id: v})
			if p is not None and p <= CLEARS_P_OOM:
				return []  # one move clears: the 1-D grids already show it
	found = []
	for a, b in itertools.combinations(mem, 2):
		best = None
		for i, va in enumerate(singles[a.id]):
			for j, vb in enumerate(singles[b.id]):
				p = _p_oom(record, {a.id: va, b.id: vb})
				if p is not None and p <= CLEARS_P_OOM:
					# steps from current: the largest saving values sit nearest the current one
					steps = (len(singles[a.id]) - i if a.kind in ("int", "float") else 1) + (
						len(singles[b.id]) - j if b.kind in ("int", "float") else 1
					)
					best = steps if best is None else min(best, steps)
		if best is not None:
			found.append((best, a.id, b.id))
	return [(a, b) for _, a, b in sorted(found)[:max_pairs]]


def predictions_for(record: FlightRecord, max_axes: int = 2) -> Predictions:
	"""Targets, estimates at the current knobs, and grids over the gridded knobs (two at a time)."""
	record = _fit_view(record)
	targets = targets_for(record)
	est = predict(record).root
	gridded = [k.id for k in record.knobs if k.gridded]
	grids = build_grids(record, gridded[:max_axes]) if gridded else []
	# Review, Oct 27: a 1-D grid for every knob around the current point, so any single move reads instantly
	for k in record.knobs:
		grids += build_grids(record, [k.id], local=True)
	# Chiya, Nov 1: when no single move clears the risk of running out of memory, local 2-D grids over the pairs
	# that do, so the guide can offer a combined move it reads off the record (agreed with C in from-b.md)
	for a, b in clearing_pairs(record):
		grids += [
			g.model_copy(update={"id": g.id + CLEARING_GRID_SUFFIX})
			for g in build_grids(record, [a, b], local=True)
		]
	return Predictions(
		targets=targets, estimates={t.id: est[t.id] for t in targets if t.id in est}, grids=grids
	)


def _vendor(gpu: str) -> str | None:
	try:
		return gpus.get_gpu(gpu).vendor
	except KeyError:
		return None


def calibration_summary() -> dict:
	"""What the coverage self-test page shows from the corpus: each target's leave-one-architecture-out coverage and
	the cross-GPU holdout, both with Wilson 95% intervals, and P(OOM)'s reliability bins. Plain data for C's page."""
	t = conformal.TABLE

	def row(hits: int, n: int) -> dict:
		lo, hi = conformal.wilson(hits, n)
		return {"hits": hits, "n": n, "rate": hits / n if n else None, "wilson95": [lo, hi]}

	targets = sorted({k[0] for k in t.cells} | set(t.coverage))
	vendors = {v for k in t.cells if k[1] for v in [_vendor(k[1])] if v}
	return {
		"built_from": t.built_from,
		"targets": {
			tid: {
				"coverage": row(*t.coverage[tid]) if tid in t.coverage else None,
				"holdout": {g: row(h, n) for g, (h, n) in sorted(t.holdout.get(tid, {}).items())},
				"lenses": sorted({k[2] for k in t.cells if k[0] == tid and k[2]}),
			}
			for tid in targets
		},
		"oom_bins": [
			{
				"lo": b[0],
				"hi": b[1],
				"n": b[2],
				"ooms": b[4] if len(b) > 4 else None,
				"agrees": b[3],
				"verified": b[2] >= oom_model.MIN_SHOWN and b[3],
				# what the engine shows: a number at an end only once that bin verifies (else its counts), and in the
				# middle only once every middle bin does (else "uncertain (5-95%)")
				"shown_as_number": (
					(b[2] >= oom_model.MIN_SHOWN and b[3]) or len(b) <= 4
					if b[1] <= oom_model.UNCERTAIN[0] + 1e-9 or b[0] >= oom_model.UNCERTAIN[1] - 1e-9
					else oom_model.middle_verified(t.oom_bins)
				),
			}
			for b in t.oom_bins
		],
		"oom_middle_verified": oom_model.middle_verified(t.oom_bins),
		# Chiya, Nov 1: shown on every surface; Nov 2: the MI300X slice measured, not calibrated on (no refit)
		"vendors_measured": sorted(vendors),
		"amd": None if "amd" in vendors else AMD_CROSS,
	}


def current_models() -> dict[str, str]:
	"""Every model the engine can put on a record, name -> its id now (gap 15: records pin the id they used)."""
	from preflight.campaign import planner  # the planner's card rides on records too

	ids = [
		memory.MODEL_ID,
		steptime.MODEL_ID_SAME,
		steptime.MODEL_ID_CROSS,
		steptime.MODEL_ID_COMM,
		agent_cost.MODEL_ID,
		rl.MODEL_ID_ZERO,
		rl.MODEL_ID_ROLLOUT,
		serving.MODEL_ID,
		planner.MODEL_ID,
		MODEL_ID_TOTAL,
		MODEL_ID_COST,
		MODEL_ID_AGENT_TIME,
		MODEL_ID_CONTEXT,
		MODEL_ID_PREEMPT,
		MODEL_ID_RUN_COST,
	]
	return {i.split("@")[0]: i for i in ids}


def stale_models(record: FlightRecord) -> list[tuple[str, str]]:
	"""(the id a record's numbers came from, the id the engine has now) for every model that has moved on since.

	A shared record keeps the numbers it was predicted with; this is how the page says "predicted with
	memory-analytical@0.1.0; the current model is 0.2.0" instead of silently showing different numbers."""
	now = current_models()
	used = {m.id for m in record.models} | {e.model_id for e in record.predictions.estimates.values()}
	out = []
	for mid in sorted(used):
		name = mid.split("@")[0]
		if name in now and now[name] != mid:
			out.append((mid, now[name]))
	# the held-out bands are versioned too: a card names the table it was banded with
	tables = {
		m.training_data.rsplit("; table ", 1)[1]
		for m in record.models
		if m.training_data and "; table " in m.training_data
	}
	for v in sorted(tables - {conformal.TABLE.version}):
		out.append((f"conformal-table@{v}", f"conformal-table@{conformal.TABLE.version}"))
	return out


def fill_predictions(record: FlightRecord, *, flags: Callable[[str], bool] | None = None) -> FlightRecord:
	"""The record with B's parts filled: predictions, model cards, knobs' probed ranges. Returns a copy.

	`flags` is the caller's flag reader (the runner's `flag`): with it, engine.rl, engine.serving,
	engine.steptime_cross_gpu and engine.divergence decide this fill; without it, FEATURES does.
	This is the call scripts/make_golden.py makes in place of the fixture predictions.
	"""
	if flags is not None:
		token = _FEATURES_NOW.set(features_from(flags))
		try:
			return fill_predictions(record)
		finally:
			_FEATURES_NOW.reset(token)
	preds = predictions_for(record)
	used = {e.model_id for e in preds.estimates.values()} | {g.model_id for g in preds.grids}
	used.discard("none")
	used |= {MODEL_ID_RUN_COST}
	others = [m for m in record.models if m.id not in used and not m.id.startswith("golden-fixture")]
	basis, cost = run_cost(record)
	return record.model_copy(
		update={
			"predictions": preds,
			"models": [*others, *_cards(used, record)],
			"knobs": _probed_ranges(record),
			"cost_basis": basis,
			"job_cost": cost,
		}
	)
