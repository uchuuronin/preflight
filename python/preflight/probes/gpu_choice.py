"""Which GPU a probe runs on when the visitor hasn't chosen, and what to try after an OOM.

Policy from scripts/mc_kernels_gpu.py, 40,000 simulated jobs, 20% of them with
no size the code gives away (docs/decisions/probe-plan.md G, L, U). Every figure
below is simulated: the job mix, memory needs and estimate error are assumptions,
so the numbers rank the policies and say nothing yet about real jobs:

  G  always L4                         72.8% measured   $0.030 a job
     always A100-80GB                  94.2%            $0.099
     always H100                       94.2%            $0.149
     size-aware, 1.3x, one step up     93.0%            $0.059   <- chosen
     size-aware, 1.15x, one step up    93.0%            $0.057
  L  top A100-80GB (or + H100)         93.2%            $0.055
     + RTX-PRO-6000 96 GB              93.8%            $0.057
     + H200 141 GB                     95.7%            $0.063   <- chosen
     + H200 + B200 180 GB              96.4%            $0.067
  U  unknown size: one step up         95.7%            $0.064
     unknown size: walk the ladder     97.6%            $0.068   <- chosen

The cheapest GPU whose usable memory is at least MARGIN times the estimated
need; after a GPU OOM, the next size up: once when the size was estimated,
up the whole ladder when it wasn't. Jobs that need more than one H200 wait
for multi-GPU probes. The estimate here is coarse (parameters × bytes per
parameter for the kind of job, plus activations); B's memory model replaces
it once engine/api.py lands. GPU names are the canonical names in B's table
(engine/gpus.py).
"""

import logging
import re
from typing import NamedTuple

from preflight.harness.static import FOUR_BIT
from preflight.schema.script_map import ScriptMap

log = logging.getLogger(__name__)

MARGIN = 1.3  # simulated: probe-plan G (scripts/mc_kernels_gpu.py); zoo runs on Modal replace it (Oct 28-29)
# Top of the ladder (L): H100 holds no more than A100-80GB, so it never wins a memory choice;
# H200 lifts measured jobs from 93.2% to 95.7% for $0.008 more a job.
LADDER = ["L4", "L40S", "A100-SXM-80GB", "H200-SXM-141GB"]
# usable GiB when B's table isn't importable (B's usable_capacity_gib() as of 14ef490)
_FALLBACK_USABLE_GIB = {
	"T4": 14.74,
	"L4": 21.45,
	"A10": 22.08,
	"L40S": 44.16,
	"A100-SXM-40GB": 39.39,
	"A100-SXM-80GB": 79.15,
	"RTX-PRO-6000-96GB": 88.32,
	"H100-SXM-80GB": 79.18,
	"H200-SXM-141GB": 139.59,
	"B200-SXM-180GB": 178.2,
	"B300-SXM-288GB": 285.12,
}
# T4 is cheaper than L4 but has no bf16 tensor cores, so a bf16 probe there would time an
# emulated path that says little about the GPU the job really runs on; L4 is the floor.
_FALLBACK_MODAL = (
	"T4",
	"L4",
	"A10",
	"L40S",
	"A100-SXM-40GB",
	"A100-SXM-80GB",
	"RTX-PRO-6000-96GB",
	"H100-SXM-80GB",
	"H200-SXM-141GB",
	"B200-SXM-180GB",
	"B300-SXM-288GB",
)
# Modal list rates when B's table isn't importable (engine/gpus.py _MODAL, 2026-07)
_FALLBACK_USD_PER_HOUR = {
	"T4": 0.59,
	"L4": 0.80,
	"A10": 1.10,
	"L40S": 1.95,
	"A100-SXM-40GB": 2.10,
	"A100-SXM-80GB": 2.50,
	"RTX-PRO-6000-96GB": 3.03,
	"H100-SXM-80GB": 3.95,
	"H200-SXM-141GB": 4.54,
	"B200-SXM-180GB": 6.25,
	"B300-SXM-288GB": 7.10,
}
# an OOM only clears on a GPU with clearly more memory: A100-80GB and H100 differ by 0.03 GiB
BIGGER = 1.05  # derived: A100-80GB and H100 differ by 0.03 GiB usable; any real step up is >40% more
ACTIVATION_GB = (
	4.0  # guess: median probe activations; replaced by the zoo runs' measured activations (Oct 28-29)
)
# bytes of GPU memory per parameter, by kind of job. derived: bf16 weights 2 B; full training adds
# fp32 master 4, Adam moments 8 and grads 2; 4-bit weights 0.5 B plus quantisation constants
BYTES_PER_PARAM = {
	"lora": 2.3,  # bf16 weights plus adapter state
	"qlora": 0.75,  # 4-bit weights plus adapter state (int8 LoRA counts as lora: a safe overestimate)
	"full": 16.0,  # bf16 weights + fp32 master + Adam moments + grads
	"infer": 2.1,  # bf16 weights plus a little KV/workspace
}


def usable_gb(gpu: str) -> float:
	try:
		from preflight.engine.gpus import get_gpu

		return get_gpu(gpu).usable_capacity_gib() * 2**30 / 1e9
	except Exception:
		return _FALLBACK_USABLE_GIB.get(gpu, 0.0) * 2**30 / 1e9


def usd_per_hour(gpu: str) -> float:
	try:
		from preflight.engine.gpus import price

		p = price(gpu)
		if p is not None:
			return p.usd_per_hour
	except Exception:
		pass
	return _FALLBACK_USD_PER_HOUR.get(gpu, float("inf"))


def offered_gpus() -> list[str]:
	"""GPUs a visitor can pick: Modal's list from B's table, else the same list as of Oct 2026."""
	try:
		from preflight.engine.gpus import MODAL_GPU

		return list(MODAL_GPU)
	except ImportError:
		return list(_FALLBACK_MODAL)


MAX_NAME = (
	64  # derived: the longest canonical GPU name is 18 characters; visitor text, bounded before any regex
)


# Modal's GPU strings, for when B's table isn't importable (engine/gpus.py MODAL_GPU at 14ef490);
# "H100!" pins a real H100 (plain "H100" may be upgraded to an H200)
_FALLBACK_MODAL_STRINGS = {"T4": "T4", "L4": "L4", "A10": "A10", "L40S": "L40S", "A100-SXM-40GB": "A100-40GB",
	"A100-SXM-80GB": "A100-80GB", "RTX-PRO-6000-96GB": "RTX-PRO-6000", "H100-SXM-80GB": "H100!",
	"H200-SXM-141GB": "H200", "B200-SXM-180GB": "B200", "B300-SXM-288GB": "B300"}  # fmt: skip


def modal_gpu(gpu: str) -> str:
	"""The string Modal's gpu= takes for a canonical GPU name."""
	try:
		from preflight.engine.gpus import MODAL_GPU

		table = MODAL_GPU
	except ImportError:
		table = _FALLBACK_MODAL_STRINGS
	name = canonical(gpu)
	if name not in table:
		raise ValueError(f"{gpu!r} is not a GPU Modal offers")
	return table[name]


def canonical(gpu: str) -> str:
	if len(gpu) > MAX_NAME:
		return gpu[:MAX_NAME]
	try:
		from preflight.engine.gpus import resolve

		return resolve(gpu)
	except Exception:
		return gpu


# bounded digit runs and a left boundary, so a long digit string can't backtrack quadratically
_SIZE = re.compile(r"(?<![a-z0-9.])(\d{1,5}(?:\.\d{1,3})?)\s?([bm])(?![a-z])", re.I)
_MOE = re.compile(r"(?<![\d.])(\d{1,3})\s?x\s?(\d{1,4}(?:\.\d{1,3})?)\s?b(?![a-z])", re.I)


def params_from_name(name: str | None) -> float | None:
	"""Parameter count from a model id: Qwen2.5-1.5B, Llama-3.1-8B, gpt2-125M, Mixtral-8x7B (≈46.7B total)."""
	if not name:
		return None
	base = name.rsplit("/", 1)[-1][:128]
	moe = _MOE.search(base)
	if moe:
		experts, size = int(moe.group(1)), float(moe.group(2))
		return experts * size * 1e9 * 0.834  # shared attention: Mixtral 8x7B is 46.7B, not 56B
	sizes = [float(v) * (1e9 if u.lower() == "b" else 1e6) for v, u in _SIZE.findall(base)]
	return max(sizes) if sizes else None


def _kind(sm: ScriptMap) -> str:
	m = sm.model
	if sm.lens in ("serving", "eval"):
		return "infer"
	if m is not None and m.lora is not None:
		return "qlora" if m.quantization in FOUR_BIT else "lora"
	return "full"


def estimate_need_gb(sm: ScriptMap) -> float:
	"""Coarse GPU memory need before any probe; 0 when nothing says how big the model is."""
	if sm.lens == "agent":
		return 0.0
	m = sm.model
	params = (m.param_count if m else None) or params_from_name(m.hf_id or m.name if m else None)
	if not params:
		return 0.0
	return params * BYTES_PER_PARAM[_kind(sm)] / 1e9 + ACTIVATION_GB + 1.0  # + CUDA context


CEILING_ENV = (
	"PREFLIGHT_GPU_CEILING"  # set on the public demo: L40S (Chiya, Nov 1; A100-SXM-80GB before, never H200)
)
SAFE_CEILING = "L40S"  # the demo's own cap (runner/modal_app.py GPU_CEILING, Nov 1)


def ceiling() -> str | None:
	"""The biggest GPU this deployment probes on, or None for no cap."""
	import os

	raw = os.environ.get(CEILING_ENV, "").strip()
	if not raw:
		return None
	cap = canonical(raw)
	if usable_gb(cap) <= 0:  # a mistyped cap still caps: the demo's own, never "no GPU at all"
		log.error("%s=%r is not a known GPU; capping at %s", CEILING_ENV, raw[:64], SAFE_CEILING)
		return SAFE_CEILING
	return cap


def under_ceiling(gpu: str, cap: str | None = None) -> bool:
	"""Whether gpu is within the cap: no more memory and no dearer an hour (an L40S cap keeps the
	A100-40GB out, though its 40 GB is under the L40S's 48; Chiya, Nov 1)."""
	cap = cap if cap is not None else ceiling()
	if cap is None:
		return True
	g = canonical(gpu)
	return usable_gb(g) <= usable_gb(cap) and usd_per_hour(g) <= usd_per_hour(cap)


def _pool(allowed: list[str] | None) -> list[str]:
	pool = list(dict.fromkeys(canonical(g) for g in allowed)) if allowed else list(LADDER)
	cap = ceiling()
	capped = [g for g in pool if under_ceiling(g, cap)]
	# allowed GPUs all above the cap (the visitor's run is H200-only): probe on the cap and let
	# the cross-GPU model carry the numbers over
	if cap is None:
		return pool
	return capped or [cap]


def choose_probe_gpu(need_gb: float, allowed: list[str] | None = None, *, margin: float = MARGIN) -> str:
	"""The cheapest GPU (allowed ones if the visitor set any) with need × margin usable; else the largest."""
	pool = _pool(allowed)
	fits = [g for g in pool if usable_gb(g) >= need_gb * margin]
	if fits:
		return min(fits, key=lambda g: (usd_per_hour(g), usable_gb(g)))
	# nothing holds it: the largest, and among GPUs within BIGGER of that the cheapest (H100 holds
	# 0.03 GiB more than A100-80GB, which is no reason to pay 58% more for the same OOM)
	top = max(usable_gb(g) for g in pool)
	return min((g for g in pool if usable_gb(g) * BIGGER >= top), key=usd_per_hour)


def next_gpu(current: str, allowed: list[str] | None = None) -> str | None:
	"""After a GPU OOM: the cheapest GPU with more usable memory, or None when there is nothing bigger."""
	pool = _pool(allowed)
	cur = canonical(current)
	have = usable_gb(cur)
	if have <= 0:  # a GPU the table doesn't know: nothing says which way is up, so don't guess
		return None
	bigger = [g for g in pool if g != cur and usable_gb(g) >= have * BIGGER]
	return min(bigger, key=lambda g: (usd_per_hour(g), usable_gb(g))) if bigger else None


def escalates(status: str | None, error_code: str | None) -> bool:
	"""Whether a failed probe is worth a try on a bigger GPU: a training-step GPU OOM only.

	Not host memory (a bigger GPU doesn't help), and not an eval OOM: that one is about the eval
	batch, which the visitor can change, so the probe re-runs with eval off on the same GPU and
	the eval memory per batch says where it would have run out.
	"""
	return status == "oom" and error_code not in ("oom.host", "oom.eval")


# shown on the record; the dispatcher's own error (app ids, URLs) stays in the server log
STEP_UP_FAILED = (
	"The larger GPU was not available, so this shows the out-of-memory result on the smaller one."
)


class Ladder(NamedTuple):
	result: object  # the ProbeResult the record keeps
	tried: list[str]  # GPUs the probe ran on, in order; the record says when it stepped up
	step_up_error: str | None  # the bigger GPU could not run the probe (no capacity, cancelled)


def max_steps_up(need_gb: float) -> int:
	"""Steps up after GPU OOMs: one when the size was estimated, the whole ladder when it wasn't (U)."""
	return 1 if need_gb > 0 else len(LADDER) - 1


def probe_on_ladder(
	run, first: str, allowed: list[str] | None = None, *, steps: int = 1, may_step_up=None
) -> Ladder:
	"""run(gpu) -> ProbeResult. Runs on `first`; after a GPU OOM, on the next size up, up to `steps` times.

	steps comes from max_steps_up(). With a size estimate, one step: the
	simulation (G, U) puts more steps at under 0.1 points for the extra probe
	time. With no estimate (a model built in the script), the policy started on
	the cheapest GPU blind, and walking the ladder measures 97.8% of those jobs
	against 88.2% for one step, for $0.004 more a job on average. A job that
	still runs out needs more than one GPU. may_step_up(gpu) -> bool lets the
	caller refuse a further run (probe budget spent, job cancelled). If a run
	on a bigger GPU fails to start, the last measured result stands: an OOM on
	the smaller GPU is still an answer.
	"""
	tried = [first]
	result = run(first)
	current = first
	for _ in range(max(0, steps)):
		p = result.probe
		if not escalates(p.status, p.error_code):
			break
		bigger = next_gpu(current, allowed)
		if bigger is None or (may_step_up is not None and not may_step_up(bigger)):
			break
		tried.append(bigger)
		try:
			result = run(bigger)
		except Exception:  # the dispatcher's error, not the visitor's: keep the measured OOM
			log.warning("probe step-up to %s failed", bigger, exc_info=True)
			return Ladder(result, tried, STEP_UP_FAILED)
		current = bigger
	return Ladder(result, tried, None)
