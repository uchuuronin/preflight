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

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from preflight.engine import agent_cost, gpus, knobs, memory, steptime
from preflight.schema.encoding import MAX_GRID_CELLS, choose_scale, encode_f16, encode_mask
from preflight.schema.prediction import (
	Estimate,
	Grid,
	GridAxis,
	KnobImpact,
	Objective,
	Prediction,
	Predictions,
	Target,
)
from preflight.schema.record import FlightRecord, Knob, ModelCard, Range

LEVEL = 0.9
API_LENSES = frozenset({"agent"})
TRAIN_LENSES = frozenset({"finetune", "pretrain", "rl", "retrieval", "gnn", "vision"})
UNMODELLED_WIDEN = 1.5  # time sigma multiplier when a time knob the model doesn't capture has moved
OK_PROBE = ("ok", "cached")
MODEL_ID_TOTAL = "time-total@0.1.0"
MODEL_ID_COST = "cost-gpu-hours@0.1.0"
MODEL_ID_AGENT_TIME = "agent-time-concurrency@0.1.0"
MODEL_ID_CONTEXT = "agent-context-fill@0.1.0"
MODEL_ID_PREEMPT = "preemption-poisson@0.1.0"

_TARGETS = {
	"memory.peak": ("bytes", "memory"),
	"risk.oom": ("probability", "risk"),
	"time.step": ("s", None),
	"time.total": ("s", "time"),
	"cost.total": ("usd", "cost"),
	"risk.wall_cap": ("probability", "risk"),
	"risk.preemption_loss": ("s", "risk"),
	"tokens.total": ("tokens", None),
	"memory.context_fill": ("fraction", "memory"),
	"risk.runaway": ("probability", "risk"),
}


# ---------------------------------------------------------------- small helpers


def _target(tid: str) -> Target:
	unit, readout = _TARGETS[tid]
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
		ids = ["memory.peak", "time.total", "cost.total", "risk.oom", "time.step"]
		c = record.job.constraints
		if c.wall_time_cap_s:
			ids.append("risk.wall_cap")
		if c.preemption.kind != "none":
			ids.append("risk.preemption_loss")
	return [_target(t) for t in ids]


def _current(record: FlightRecord) -> dict:
	return {k.id: k.current for k in record.knobs}


ENGINE_WHAT_IFS = frozenset({"gpu_type", "gpus"})  # accepted even when a record predates their knobs
WORKLOAD_KNOBS = frozenset({"max_steps", "epochs", "n_examples", "n_tasks"})  # define the job; not levers
TRUNCATING_KNOBS = frozenset(
	{"seq_len", "max_completion_len", "max_new_tokens", "max_prompt_len", "image_size", "context_limit"}
)


def _values(record: FlightRecord, overrides: dict | None) -> dict:
	vals = _current(record)
	for kid, v in (overrides or {}).items():
		if kid not in vals and kid not in ENGINE_WHAT_IFS:
			raise ValueError(f"unknown knob {kid!r}: the record declares {sorted(vals)}")
		vals[kid] = v
	return vals


def _probe_values(record: FlightRecord, probe) -> dict:
	"""The knob values a probe actually ran at: the record's current values, overridden by the probe's."""
	return {**_current(record), **probe.knobs}


def _same(a, b) -> bool:
	if isinstance(a, int | float) and isinstance(b, int | float) and not isinstance(a, bool):
		return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)
	return a == b


def _matching_probe(record: FlightRecord, values: dict, ok=OK_PROBE):
	"""A probe whose every knob equals the query's: the number is a measurement, not a prediction."""
	for p in record.probes:
		if p.status in ok:
			pv = _probe_values(record, p)
			if all(_same(pv.get(k), v) for k, v in values.items() if k != "gpu_type"):
				return p
	return None


def _unmodelled_moves(record: FlightRecord, values: dict, readout: str) -> list[str]:
	"""Knobs that change this readout, aren't modelled for it, and sit where no probe ran."""
	out = []
	for kid, v in values.items():
		spec = knobs.knob_spec(kid)
		if readout not in spec.affects or readout in spec.modelled or kid == "gpu_type":
			continue
		probed = [_probe_values(record, p).get(kid) for p in record.probes if p.status in OK_PROBE]
		if probed and not any(_same(v, pv) for pv in probed):
			out.append(kid)
	return out


def _env_gpu(record: FlightRecord) -> str | None:
	if record.env and record.env.gpu and record.env.gpu.name:
		try:
			return gpus.resolve(record.env.gpu.name, record.env.gpu.memory_bytes)
		except KeyError:
			return None
	return None


def _target_gpu(record: FlightRecord, values: dict) -> str | None:
	name = values.get("gpu_type")
	if isinstance(name, str) and name:
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


def _memory(record: FlightRecord, values: dict) -> _MemoryResult:
	mid = memory.MODEL_ID
	gpu = _target_gpu(record, values)
	if gpu is None:
		why = "the GPU isn't known, so there's no capacity to compare against; pick a GPU"
		return _MemoryResult(_abstain("bytes", mid, why), _abstain("probability", mid, why))
	capacity = gpus.get_gpu(gpu).capacity_bytes()
	oom_probe = _matching_probe(record, values, ok=("oom",))
	if oom_probe is not None:
		why = f"probe {oom_probe.id} ran out of memory at these settings"
		est = memory.MemoryEstimate(None, None, None, "bytes", "abstain", mid, 1.0, "fails", 0.0, None, why)  # type: ignore[arg-type]
		oom = _est(1.0, 1.0, 1.0, "probability", "measured", mid, why)
		return _MemoryResult(_abstain("bytes", mid, why), oom, est)
	moves = _unmodelled_moves(record, values, "memory")
	if moves:
		why = f"{', '.join(moves)} changes memory in a way the engine doesn't model yet; run a probe here"
		return _MemoryResult(_abstain("bytes", mid, why), _abstain("probability", mid, why))
	sm = record.job.script_map
	lens = record.job.lens
	try:
		arch, setup, shape = knobs.setup_from(sm, values)
	except (ValueError, AttributeError) as exc:
		return _empirical_memory(record, values, capacity, str(exc))
	mprobes = []
	for p in record.probes:
		if p.status not in OK_PROBE or not p.memory or not p.memory.peak_reserved_bytes:
			continue
		pv = _probe_values(record, p)
		try:
			pa, ps, psh = knobs.setup_from(sm, pv)
		except (ValueError, AttributeError):
			continue
		if p.eval is None:
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
		measured = moment == "train" or (moment == "eval" and match.eval is not None)
	est = memory.predict_peak(
		arch,
		setup,
		shape.batch,
		shape.seq,
		probes=mprobes or None,
		capacity_bytes=capacity,
		work_units=knobs.work_units(lens, values),
		measured=measured,
	)
	if (
		est.status == "predicted"
		and est.breakdown is not None
		and est.breakdown.peak_moment != "train"
		and not any(p.eval is not None for p in record.probes if p.status in OK_PROBE)
	):
		why = f"the peak comes at {est.breakdown.peak_moment}, which no probe ran; run a probe with {est.breakdown.peak_moment} in it"
		est = replace(est, status="extrapolated", reason="; ".join(x for x in (est.reason, why) if x))
	return _from_memory_estimate(est, mid)


def _from_memory_estimate(est: memory.MemoryEstimate, mid: str) -> _MemoryResult:
	if est.status == "abstain":
		return _MemoryResult(
			_abstain("bytes", mid, est.reason), _abstain("probability", mid, est.reason), est
		)
	peak = _est(est.point, est.lo, est.hi, "bytes", est.status, mid, est.reason)
	words = {"fits": "fits", "tight": "tight", "fails": "won't fit"}[est.oom_status]
	why = f"{words}: P(out of memory) {est.p_oom:.0%}"
	if est.breakdown is not None and est.breakdown.peak_moment != "train":
		why += f"; the peak comes at {est.breakdown.peak_moment}, not during training steps"
	oom = _est(est.p_oom, est.p_oom, est.p_oom, "probability", est.status, mid, why)
	return _MemoryResult(peak, oom, est)


def _empirical_memory(
	record: FlightRecord, values: dict, capacity: float, why_no_shape: str
) -> _MemoryResult:
	"""No model shape: memory as a line in work units through the probes (no analytical prior)."""
	mid = memory.MODEL_ID
	lens = record.job.lens
	pts = [
		memory.MemoryProbe(
			knobs.work_units(lens, _probe_values(record, p)), float(p.memory.peak_reserved_bytes), 0.0
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
	x = knobs.work_units(lens, values)
	measured = _matching_probe(record, values) is not None
	h = memory.HALF_WIDTH_CALIBRATED if measured else cal.half_width(x)
	status = "measured" if measured else ("extrapolated" if cal.is_below(x) else "predicted")
	if not measured and (x > cal.range()[1] or h > memory.ABSTAIN_HALF_WIDTH):
		why = f"{x:.0f} work units is too far from the probes for a line fit; run a probe here"
		return _MemoryResult(_abstain("bytes", mid, why), _abstain("probability", mid, why))
	point = cal.correct(x, 0.0) + memory.DEFAULT_CONTEXT_BYTES
	if point <= 0:
		why = "the line through the probes goes below zero here; run a probe here"
		return _MemoryResult(_abstain("bytes", mid, why), _abstain("probability", mid, why))
	p = memory.p_oom(point, h / memory.Z90, capacity)
	est = memory.MemoryEstimate(
		point, point * math.exp(-h), point * math.exp(h), "bytes", status, mid, p, memory.oom_status(p),  # type: ignore[arg-type]
		h / memory.Z90, None, "fitted to the probes alone: the script map has no model shape",
	)  # fmt: skip
	return _from_memory_estimate(est, mid)


# ---------------------------------------------------------------- step time


@dataclass(frozen=True)
class _StepResult:
	est: Estimate
	sigma: float  # log sigma of the step time, for totals and tail probabilities
	first_extra_s: float = 0.0  # compile and warm-up, paid once


def _step_x(lens: str, values: dict) -> float:
	"""Work per optimizer step: work units per micro-batch times accumulation."""
	return knobs.work_units(lens, values) * max(1, int(values.get("grad_accum", 1) or 1))


def _precision(values: dict) -> str | None:
	for k in ("precision", "dtype", "torch_dtype", "ptdtype"):
		if k in values and isinstance(values[k], str):
			return gpus.normalise_dtype(values[k]) if values[k] not in ("mixed", "unknown") else None
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
	for p in record.probes:
		st_ = p.step_time
		if p.status not in OK_PROBE or not st_ or not st_.p50_s:
			continue
		pv = _probe_values(record, p)
		phases = steptime.phases_seconds(st_.by_phase, st_.p50_s) or {"step": st_.p50_s}
		sprobes.append(steptime.StepProbe(_step_x(lens, pv), phases, p.id))
		if p.kernels and current_probe is None:
			current_probe = p
	mid = steptime.MODEL_ID_SAME
	if not sprobes:
		return _StepResult(_abstain("s", mid, "no probe timed a step yet"), 0.0)
	model = steptime.SameGpuModel.fit(sprobes, constant_phases=steptime.CONSTANT_PHASES)
	probe_gc = {bool(_probe_values(record, p).get("gradient_checkpointing", False)) for p in record.probes}
	recompute = bool(values.get("gradient_checkpointing")) and probe_gc == {False}
	match = _matching_probe(record, values)
	src_gpu, tgt_gpu = _env_gpu(record), _target_gpu(record, values)
	same_gpu = src_gpu is None or tgt_gpu is None or src_gpu == tgt_gpu
	est = model.predict(
		_step_x(lens, values), recompute=recompute, measured_probe=match.id if match and same_gpu else None
	)
	if est.status == "abstain":
		return _StepResult(_abstain("s", mid, est.reason), est.sigma_log)
	point, sigma, status, reasons = est.point, est.sigma_log, est.status, [est.reason] if est.reason else []

	# precision moved away from what the probes ran
	q_prec = _precision(values)
	p_precs = {_precision(_probe_values(record, p)) for p in record.probes if p.status in OK_PROBE}
	if q_prec and p_precs and q_prec not in p_precs and src_gpu:
		src_prec = next(iter(p_precs - {None}), None)
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
	if not same_gpu:
		cross = None
		if current_probe is not None and current_probe.step_time and current_probe.step_time.p50_s:
			cross = steptime.predict_cross_gpu_from_probe(
				current_probe.kernels, current_probe.step_time.p50_s, src_gpu, tgt_gpu
			)
		if cross is not None and cross.status != "abstain":
			ratio = cross.point / current_probe.step_time.p50_s
			sigma = math.hypot(sigma, cross.sigma_log)
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
			sigma = math.hypot(sigma, steptime.cross_sigma(src_gpu, tgt_gpu) * steptime.SIGMA_SECONDARY_SPECS)
			status = "extrapolated"
			reasons.append("no kernel table on the probe, so phases are scaled by class")
		point *= ratio
		mid = steptime.MODEL_ID_CROSS
		if status == "measured":
			status = "predicted"

	moves = _unmodelled_moves(record, values, "time")
	if moves:
		sigma *= UNMODELLED_WIDEN
		status = "extrapolated" if status != "abstain" else status
		reasons.append(f"{', '.join(moves)} isn't modelled for time yet; held at its probed effect")
	h = memory.Z90 * sigma
	if h > memory.ABSTAIN_HALF_WIDTH:
		why = "; ".join([*reasons, "the range would be wider than +-50%; run a probe here"])
		return _StepResult(_abstain("s", mid, why), sigma)
	first = 0.0
	for p in record.probes:
		if p.status in OK_PROBE and p.step_time and p.step_time.first_step_s and p.step_time.p50_s:
			first = max(first, p.step_time.first_step_s - p.step_time.p50_s)
	return _StepResult(
		_est(point, point * math.exp(-h), point * math.exp(h), "s", status, mid, "; ".join(reasons) or None),
		sigma,
		first,
	)


# ---------------------------------------------------------------- run totals


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


def _time(record: FlightRecord, values: dict) -> _TimeResult:
	step = _step(record, values)
	if step.est.status == "abstain":
		return _TimeResult(_abstain("s", MODEL_ID_TOTAL, step.est.reason), step.est, step.sigma)
	steps, why = _total_steps(record, values)
	if steps is None:
		return _TimeResult(_abstain("s", MODEL_ID_TOTAL, why), step.est, step.sigma)
	extra, notes = _overheads(record, values, steps)
	fixed = extra + step.first_extra_s
	h = memory.Z90 * step.sigma
	pt = steps * step.est.point
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
	return _TimeResult(total, step.est, step.sigma)


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
	return _est(time.point * rate, time.lo * rate, time.hi * rate, "usd", time.status, MODEL_ID_COST, why)


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
	interval = float(every) * step if every else time.total.point  # no checkpoints: lose everything so far
	restart = (rule.grace_s or 0.0) + 60.0  # requeue and reload, at least a minute
	lost_each = interval / 2 + restart  # Young/Daly: on average half an interval is redone

	def loss(t):
		return t / rule.mtti_s * lost_each

	why = f"about {time.total.point / rule.mtti_s:.1f} interruptions, each redoing half a checkpoint interval"
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


def _tasks_for(record: FlightRecord, values: dict):
	s = record.agent_sample
	if s is None:
		return None, []
	model = values.get("model")
	tasks = [t for t in s.tasks if model is None or t.model in (None, model)]
	return s, tasks


def _agent(record: FlightRecord, values: dict) -> dict[str, Estimate]:
	s, tasks = _tasks_for(record, values)
	ids = ["cost.total", "time.total", "memory.context_fill", "risk.runaway", "tokens.total"]
	if s is None:
		return {
			t: _abstain(_TARGETS[t][0], agent_cost.MODEL_ID, "no task sample yet; run the agent probe")
			for t in ids
		}
	model = values.get("model")
	if model is not None and not tasks:
		why = f"no sampled task ran on {model}; sample it first"
		return {t: _abstain(_TARGETS[t][0], agent_cost.MODEL_ID, why) for t in ids}
	pop = int(values.get("n_tasks") or s.population or 0) or None
	sub = s.model_copy(update={"tasks": tasks})
	tot = agent_cost.agent_totals(sub, population=pop)
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


# ---------------------------------------------------------------- the three calls


def predict(record: FlightRecord, knob_values: dict | None = None) -> Prediction:
	"""Every target for this lens at these knob values (others at their current values)."""
	values = _values(record, knob_values)
	if _api_lens(record):
		return Prediction(_agent(record, values))
	mem = _memory(record, values)
	tm = _time(record, values)
	out = {
		"memory.peak": mem.peak,
		"risk.oom": mem.oom,
		"time.step": tm.step,
		"time.total": tm.total,
		"cost.total": _cost(record, values, tm.total),
	}
	c = record.job.constraints
	if c.wall_time_cap_s:
		out["risk.wall_cap"] = _wall_cap(record, tm)
	if c.preemption.kind != "none":
		out["risk.preemption_loss"] = _preemption(record, values, tm)
	return Prediction(out)


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
		pts = (
			np.geomspace(lo, hi, n)
			if (knob.domain.log or (lo > 0 and hi / lo >= 100))
			else np.linspace(lo, hi, n)
		)
		vals = sorted({float(f"{v:.4g}") for v in pts} | {float(knob.current)})
	return vals


_GRID_TARGETS = ("memory.peak", "risk.oom", "time.total", "cost.total")


def build_grids(record: FlightRecord, knob_ids: list[str], targets: list[str] | None = None) -> list[Grid]:
	"""Predictions precomputed over the knobs' slider stops, one grid per target (row-major over the axes)."""
	if not 1 <= len(knob_ids) <= 4:
		raise ValueError("a grid takes one to four knobs")
	axes_knobs = [record.knob(k) for k in knob_ids]
	per = 8 if len(knob_ids) <= 2 else 5
	axes = [axis_values(k, per) for k in axes_knobs]
	cells = math.prod(len(a) for a in axes)
	if cells > MAX_GRID_CELLS:
		raise ValueError(f"{cells} cells is over the inline limit of {MAX_GRID_CELLS}")
	tids = targets or [t.id for t in targets_for(record) if t.id in _GRID_TARGETS or _api_lens(record)]
	vals: dict[str, dict[str, list]] = {
		t: {"point": [], "lo": [], "hi": [], "abstain": [], "model": []} for t in tids
	}
	for combo in np.ndindex(*[len(a) for a in axes]):
		knob_set = {k.id: axes[i][j] for i, (k, j) in enumerate(zip(axes_knobs, combo, strict=True))}
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
				unit=_TARGETS[t][0],
				axes=grid_axes,
				point=encode_f16(pt, scale),
				lo=encode_f16(lo, scale),
				hi=encode_f16(hi, scale),
				abstain=encode_mask(np.asarray(v["abstain"], dtype=bool)),
				model_id=max(set(models), key=models.count) if models else "none",
				scale=scale,
				fixed=fixed,
			)
		)
	return out


def _is_log(values: list) -> bool:
	nums = [float(v) for v in values if isinstance(v, int | float) and not isinstance(v, bool)]
	return len(nums) >= 3 and nums[0] > 0 and nums[-1] / nums[0] >= 8


# ---------------------------------------------------------------- sensitivity


def _objective_fn(
	record: FlightRecord, objective: Objective
) -> tuple[Callable[[Prediction], float | None], str]:
	api = _api_lens(record)

	def pick(tid):
		def f(pred):
			e = pred.root.get(tid)
			return None if e is None or e.status == "abstain" else e.point

		return f

	kind = objective.kind
	if kind == "custom" and objective.target_id:
		return pick(objective.target_id), objective.target_id
	if objective.weights:

		def blend(pred):
			total = 0.0
			for tid, w in objective.weights.items():
				v = pick(tid)(pred)
				if v is None:
					return None
				total += w * v
			return total

		return blend, "weighted blend"
	table = {
		"min_time": "time.total",
		"min_cost": "cost.total",
		"fit_memory": "memory.context_fill" if api else "risk.oom",
		"min_risk": "risk.runaway" if api else "risk.oom",
		"max_quality_per_usd": "cost.total",
	}
	if kind == "max_throughput":
		if api:
			return pick("time.total"), "time.total"

		def neg_tput(pred):
			e = pred.root.get("time.step")
			return None if e is None or e.status == "abstain" or not e.point else e.point

		return neg_tput, "time.step"
	tid = table.get(kind, "time.total")
	return pick(tid), tid


def _feasible(record: FlightRecord, pred: Prediction) -> bool:
	oom = pred.root.get("risk.oom")
	return not (oom is not None and oom.status != "abstain" and oom.point >= memory.FAILS_AT)


def sensitivity(record: FlightRecord, objective: Objective) -> list[KnobImpact]:
	"""Which knobs move the objective most across their range (one at a time, others at current values)."""
	f, tid = _objective_fn(record, objective)
	base_pred = predict(record)
	base = f(base_pred)
	# when the current settings already fail, every neighbour may too; rank by the objective anyway
	check = _feasible(record, base_pred)
	rows = []
	for k in record.knobs:
		if k.id in WORKLOAD_KNOBS:
			continue
		stops = axis_values(k, 6)
		if len(stops) < 2:
			continue
		scored = []
		for v in stops:
			pred = predict(record, {k.id: v})
			y = f(pred)
			if y is not None and (not check or _feasible(record, pred)):
				scored.append((v, y))
		if len(scored) < 2:
			continue
		ys = [y for _, y in scored]
		spread = max(ys) - min(ys)
		ref = abs(base) if base else max(abs(y) for y in ys) or 1.0
		best_v, _ = min(scored, key=lambda s: s[1])
		direction = "either"
		if k.kind in ("int", "float"):
			xs = np.asarray([float(v) for v, _ in scored])
			r = np.corrcoef(xs, np.asarray(ys))[0, 1] if np.ptp(xs) > 0 and np.ptp(ys) > 0 else 0.0
			if r <= -0.5:
				direction = "increase"
			elif r >= 0.5:
				direction = "decrease"
		why = f"{tid} from {min(ys):.3g} to {max(ys):.3g} across the range; best at {best_v}"
		if k.id in TRUNCATING_KNOBS:
			why += "; shorter cuts what the model sees, so check quality before taking it"
		rows.append((k.id, spread / ref, direction, best_v, why))
	top = max((r[1] for r in rows), default=0.0) or 1.0
	out = [
		KnobImpact(knob_id=kid, impact=min(1.0, imp / top), direction=d, best_value=bv, reason=why)
		for kid, imp, d, bv, why in rows
	]
	return sorted(out, key=lambda r: -r.impact)


# ---------------------------------------------------------------- the whole predictions block


_CARDS = {
	memory.MODEL_ID: ("analytical", "Korthikanti-style accounting, calibrated by the probes (E3); S1, S2a-d"),
	steptime.MODEL_ID_SAME: (
		"analytical",
		"per-phase linear fits on the probes; S4b below the smallest probe",
	),
	steptime.MODEL_ID_CROSS: ("analytical", "per-kernel roofline replay (K1), class fallback (K4); S4"),
	MODEL_ID_TOTAL: ("analytical", "steps x step time plus measured checkpoint and eval overheads"),
	MODEL_ID_COST: ("rule", "GPU-hours x the dated price in engine/gpus.py"),
	MODEL_ID_PREEMPT: (
		"simulation",
		"Poisson interruptions, half a checkpoint interval redone each (Young/Daly)",
	),
	agent_cost.MODEL_ID: ("bootstrap", "bootstrap plus Clopper-Pearson runaway margin; S6, S6b"),
	MODEL_ID_AGENT_TIME: ("bootstrap", "task-seconds over concurrency, or the rate limits"),
	MODEL_ID_CONTEXT: ("bootstrap", "95th percentile of each task's peak context over the limit"),
}


def _cards(ids: set[str]) -> list[ModelCard]:
	out = []
	for mid in sorted(ids):
		kind, notes = _CARDS.get(mid, ("analytical", None))
		out.append(ModelCard(id=mid, kind=kind, version=mid.split("@")[-1], notes=notes))
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


def predictions_for(record: FlightRecord, max_axes: int = 2) -> Predictions:
	"""Targets, estimates at the current knobs, and grids over the gridded knobs (two at a time)."""
	targets = targets_for(record)
	est = predict(record).root
	gridded = [k.id for k in record.knobs if k.gridded]
	grids = build_grids(record, gridded[:max_axes]) if gridded else []
	return Predictions(
		targets=targets, estimates={t.id: est[t.id] for t in targets if t.id in est}, grids=grids
	)


def fill_predictions(record: FlightRecord) -> FlightRecord:
	"""The record with B's parts filled: predictions, model cards, knobs' probed ranges. Returns a copy.

	This is the call scripts/make_golden.py makes in place of the fixture predictions.
	"""
	preds = predictions_for(record)
	used = {e.model_id for e in preds.estimates.values()} | {g.model_id for g in preds.grids}
	used.discard("none")
	others = [m for m in record.models if m.id not in used and not m.id.startswith("golden-fixture")]
	return record.model_copy(
		update={"predictions": preds, "models": [*others, *_cards(used)], "knobs": _probed_ranges(record)}
	)
