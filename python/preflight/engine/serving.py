"""Self-hosted serving: KV-cache capacity, prefill and decode rooflines, and a continuous-batching scheduler simulation
(main plan, Generation and serving: vLLM, Vidur, DistServe).

KV cache. Each token holds K and V for every layer: 2 x layers x KV heads x head dim x bytes, split by tensor
parallel. Capacity is what is left of `gpu_memory_utilization` of the card once the weights and a workspace are in.

Rooflines. One scheduler iteration runs a prefill chunk and one decode token for every running sequence in one forward
pass. It is compute-bound when there are many tokens: 2 FLOPs per active parameter per token, plus attention over the
context. It is memory-bound when there are few: the weights are read once, and every running sequence's KV is read.
The iteration takes the larger of the two, plus a fixed scheduling overhead.

Scheduler. A discrete-event simulation of vLLM-style continuous batching with chunked prefill. Each iteration, every
running sequence decodes one token, and the rest of the `max_batched_tokens` budget prefills waiting requests first
come, first served. A request is admitted only while its prompt and full output fit the free KV cache. This is more
conservative than vLLM, which admits on the prompt alone and preempts later, so peak concurrency is a lower bound.

Calibration. Without a probe, the two efficiencies and the overhead are guesses, with a wide band. With A's vLLM probe,
the probe's own workload is simulated, and the ratio of measured to simulated time scales every prediction (one
factor, fitted on one run), with a narrower band. Pure Python and numpy, no I/O: it runs in Pyodide.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from preflight.engine import gpus
from preflight.engine.memory import TransformerArch, dtype_bytes

MODEL_ID = "serving-sim@0.1.0"
Z90 = 1.6448536269514722
EFF_BW = (
	0.75  # guess: decode reaches about 75% of HBM bandwidth (vLLM's decode kernels; revisited on A's probes)
)
EFF_FLOPS = 0.55  # guess: prefill reaches about 55% of dense peak
OVERHEAD_S = 0.003  # guess: scheduling and sampling per iteration, ~3 ms in vLLM V1 at small batches
WORKSPACE_BYTES = 1.5 * 2**30  # guess: activations, CUDA graphs and the sampler outside the KV pool
SIGMA_UNCALIBRATED = 0.30  # guess: log-space sigma with no probe on this model and GPU
SIGMA_CALIBRATED = 0.10  # guess: one probe fixes the scale; what's left is how the workload moves
MAX_REQUESTS = 4000  # a simulation bigger than this is sampled down: the rates settle long before


@dataclass(frozen=True)
class ServeSetup:
	arch: TransformerArch
	gpu: str
	weight_bytes: float  # per GPU, after tensor parallel and quantisation
	kv_dtype: str = "bf16"
	compute_dtype: str = "bf16"
	tensor: int = 1
	mem_util: float = 0.9
	max_num_seqs: int = 256
	max_batched_tokens: int = 8192
	max_model_len: int | None = None
	capacity_bytes: float | None = None  # the card's usable memory; default from the GPU table
	eff_bw: float = EFF_BW
	eff_flops: float = EFF_FLOPS
	overhead_s: float = OVERHEAD_S


@dataclass(frozen=True)
class Request:
	arrival_s: float
	prompt: int
	output: int


@dataclass(frozen=True)
class SimResult:
	seconds: float  # first arrival to last token
	output_tokens: int
	throughput: float  # output tokens per second
	ttft: np.ndarray  # per request: arrival to first output token
	tpot: np.ndarray  # per request: seconds per output token after the first
	latency: np.ndarray  # per request: arrival to last token
	peak_running: int
	kv_capacity_tokens: int
	iterations: int
	rejected: int = 0  # requests whose prompt plus output never fit the cache
	notes: tuple[str, ...] = ()


# ---------------------------------------------------------------- KV cache


def kv_bytes_per_token(arch: TransformerArch, kv_dtype: str = "bf16", tensor: int = 1) -> float:
	"""2 (K and V) x layers x KV heads x head dim x bytes, per token, per GPU. Tensor parallel splits the KV heads,
	but never below one per GPU: with more GPUs than KV heads, vLLM replicates them."""
	per_gpu = max(arch.kv_heads / max(tensor, 1), 1.0)
	return 2.0 * arch.n_layers * per_gpu * arch.head_dim * dtype_bytes(kv_dtype)


def kv_capacity_tokens(s: ServeSetup) -> int:
	"""Tokens the KV pool holds: the card's share for vLLM, less the weights and a workspace."""
	cap = s.capacity_bytes or gpus.get_gpu(s.gpu).usable_capacity_gib() * 2**30
	free = cap * s.mem_util - s.weight_bytes - WORKSPACE_BYTES
	return max(0, int(free / kv_bytes_per_token(s.arch, s.kv_dtype, s.tensor)))


# ---------------------------------------------------------------- rooflines


def iteration_seconds(
	s: ServeSetup, prefill_tokens: int, prefill_ctx: float, decode_seqs: int, kv_tokens: int
) -> float:
	"""One forward pass: `prefill_tokens` new prompt tokens attending over `prefill_ctx` tokens of context on average,
	plus one token for each of `decode_seqs` sequences, which together hold `kv_tokens` of KV cache."""
	spec = gpus.get_gpu(s.gpu)
	a = s.arch
	t = max(s.tensor, 1)
	tokens = prefill_tokens + decode_seqs
	params = a.active_linear_params() / t
	attn_flops = 4.0 * a.n_layers * a.n_heads * a.head_dim / t * (prefill_tokens * prefill_ctx + kv_tokens)
	flops = 2.0 * params * tokens + attn_flops
	peak = spec.peak_flops(s.compute_dtype) or spec.peak_flops("bf16") or 1e12
	kvb = kv_bytes_per_token(a, s.kv_dtype, s.tensor)
	moved = s.weight_bytes + kvb * (kv_tokens + tokens)  # read the weights and the cache, write the new KV
	compute = flops / (peak * s.eff_flops)
	memory = moved / (spec.bandwidth_bytes() * s.eff_bw)
	return max(compute, memory) + s.overhead_s


def prefill_seconds(s: ServeSetup, prompt: int) -> float:
	return iteration_seconds(s, prompt, prompt / 2, 0, 0)


def decode_seconds(s: ServeSetup, batch: int, ctx: int) -> float:
	return iteration_seconds(s, 0, 0, batch, batch * ctx)


# ---------------------------------------------------------------- the scheduler


def simulate(s: ServeSetup, requests: list[Request]) -> SimResult:
	"""Continuous batching with chunked prefill, iteration by iteration."""
	reqs = sorted(requests, key=lambda r: r.arrival_s)
	notes = []
	if len(reqs) > MAX_REQUESTS:
		step = len(reqs) / MAX_REQUESTS
		reqs = [reqs[int(i * step)] for i in range(MAX_REQUESTS)]
		notes.append(f"simulated {MAX_REQUESTS} of the requests, evenly spaced")
	cap = kv_capacity_tokens(s)
	n = len(reqs)
	first = np.full(n, np.nan)
	done = np.full(n, np.nan)
	limit = s.max_model_len or 10**9
	# an output stops at the model length; a prompt that reaches it is refused
	reqs = [Request(r.arrival_s, r.prompt, min(r.output, max(1, limit - r.prompt))) for r in reqs]
	need = [r.prompt + r.output for r in reqs]
	too_long = {i for i in range(n) if reqs[i].prompt >= limit}
	rejected = {i for i in range(n) if need[i] > cap} | too_long
	if too_long:
		notes.append(f"{len(too_long)} prompts are longer than max_model_len ({limit}) and are refused")
	if rejected - too_long:
		notes.append(
			f"{len(rejected - too_long)} requests are longer than the whole KV cache ({cap} tokens) and never run"
		)
	waiting = [i for i in range(n) if i not in rejected]
	waiting.reverse()  # pop() from the end: first come, first served
	prefilled = dict.fromkeys(range(n), 0)
	generated = dict.fromkeys(range(n), 0)
	running: list[int] = []  # admitted, prefill done or in progress
	kv_used = 0
	clock = reqs[0].arrival_s if reqs else 0.0
	start = clock
	iters, peak = 0, 0
	arrivals = [r.arrival_s for r in reqs]
	while waiting or running:
		# admit arrived requests while their whole sequence fits the cache and a slot is free
		while (
			waiting
			and arrivals[waiting[-1]] <= clock
			and len(running) < max(1, s.max_num_seqs)
			and kv_used + need[waiting[-1]] <= cap
		):
			i = waiting.pop()
			running.append(i)
			kv_used += need[i]
		if not running:
			clock = max(clock, arrivals[waiting[-1]])
			continue
		decode = [i for i in running if prefilled[i] >= reqs[i].prompt]
		budget = max(0, s.max_batched_tokens - len(decode))
		chunks, ctx = {}, 0.0
		for i in running:
			if budget <= 0:
				break
			left = reqs[i].prompt - prefilled[i]
			if left > 0:
				take = min(left, budget)
				chunks[i] = take
				ctx += take * (prefilled[i] + take / 2)
				budget -= take
		pf = sum(chunks.values())
		kv_tokens = sum(prefilled[i] + generated[i] for i in decode)
		dt = iteration_seconds(s, pf, ctx / pf if pf else 0.0, len(decode), kv_tokens)
		clock += dt
		iters += 1
		peak = max(peak, len(running))
		for i, take in chunks.items():
			prefilled[i] += take
			if prefilled[i] >= reqs[i].prompt and generated[i] == 0:
				generated[i] = 1  # the prefill's last position samples the first token
				first[i] = clock
		for i in decode:
			generated[i] += 1
			if math.isnan(first[i]):
				first[i] = clock
		finished = [i for i in running if prefilled[i] >= reqs[i].prompt and generated[i] >= reqs[i].output]
		for i in finished:
			running.remove(i)
			kv_used -= need[i]
			done[i] = clock
	ok = np.array([i not in rejected for i in range(n)], dtype=bool)
	arr = np.asarray(arrivals)
	out_tokens = int(sum(r.output for i, r in enumerate(reqs) if ok[i]))
	span = max(clock - start, 1e-12)
	outs = np.asarray([max(r.output - 1, 1) for r in reqs], dtype=float)
	return SimResult(
		seconds=span,
		output_tokens=out_tokens,
		throughput=out_tokens / span,
		ttft=(first - arr)[ok],
		tpot=((done - first) / outs)[ok],
		latency=(done - arr)[ok],
		peak_running=peak,
		kv_capacity_tokens=cap,
		iterations=iters,
		rejected=len(rejected),
		notes=tuple(notes),
	)


def simulate_batches(s: ServeSetup, batches: list[list[Request]]) -> SimResult:
	"""Batches sent one after another, each starting when the last ends: how a probe that made several generate calls
	ran them. Times inside each batch are relative to its own start."""
	results = [simulate(s, b) for b in batches if b]
	if not results:
		raise ValueError("no requests")
	if len(results) == 1:
		return results[0]
	return SimResult(
		seconds=sum(r.seconds for r in results),
		output_tokens=sum(r.output_tokens for r in results),
		throughput=sum(r.output_tokens for r in results) / max(sum(r.seconds for r in results), 1e-12),
		ttft=np.concatenate([r.ttft for r in results]),
		tpot=np.concatenate([r.tpot for r in results]),
		latency=np.concatenate([r.latency for r in results]),
		peak_running=max(r.peak_running for r in results),
		kv_capacity_tokens=results[0].kv_capacity_tokens,
		iterations=sum(r.iterations for r in results),
		rejected=sum(r.rejected for r in results),
		notes=tuple(dict.fromkeys(n for r in results for n in r.notes)),
	)


# ---------------------------------------------------------------- workloads


def lengths(
	mean: float, p95: float, size: int, rng: np.random.Generator, cap: int | None = None
) -> np.ndarray:
	"""Request lengths: a log-normal through the mean and the 95th percentile (rl.lognormal_from), at least 1."""
	from preflight.engine.rl import lognormal_from

	fit = lognormal_from(mean, max(p95, mean * 1.0001))
	x = rng.lognormal(fit[0], fit[1], size) if fit else np.full(size, mean)
	if cap:
		x = np.minimum(x, cap)
	return np.maximum(1, np.rint(x)).astype(int)


def workload(
	n: int,
	prompt_mean: float,
	output_mean: float,
	prompt_p95: float | None = None,
	output_p95: float | None = None,
	rate_per_s: float | None = None,
	max_output: int | None = None,
	seed: int = 0,
) -> list[Request]:
	"""n requests: all at once (an offline batch, as an eval sweep sends them) or Poisson at `rate_per_s`."""
	rng = np.random.default_rng(seed)
	p = lengths(prompt_mean, prompt_p95 or prompt_mean * 2, n, rng)
	o = lengths(output_mean, output_p95 or output_mean * 2, n, rng, max_output)
	t = np.cumsum(rng.exponential(1 / rate_per_s, n)) if rate_per_s else np.zeros(n)
	return [Request(float(a), int(b), int(c)) for a, b, c in zip(t, p, o, strict=True)]


# ---------------------------------------------------------------- calibration and readouts


@dataclass(frozen=True)
class Calibration:
	scale: float = 1.0  # measured / simulated seconds on the probe's own workload
	sigma: float = SIGMA_UNCALIBRATED
	source: str = "uncalibrated: guessed efficiencies"


def calibrate(s: ServeSetup, probe_batches: list[list[Request]], measured_s: float) -> Calibration:
	"""One scale factor from A's vLLM probe: its measured wall time over the simulated one for the same batches."""
	n = sum(len(b) for b in probe_batches)
	sim = simulate_batches(s, probe_batches)
	if sim.seconds <= 0 or measured_s <= 0 or sim.output_tokens == 0:
		return Calibration()
	scale = measured_s / sim.seconds
	return Calibration(scale, SIGMA_CALIBRATED, f"calibrated on the probe's {n} requests (x{scale:.2f})")


@dataclass(frozen=True)
class Readout:
	point: float
	lo: float
	hi: float
	reason: str


@dataclass(frozen=True)
class ServingForecast:
	throughput: Readout  # output tokens per second
	ttft_p95: Readout
	tpot_p50: Readout
	kv_capacity_tokens: int
	peak_running: int
	notes: tuple[str, ...] = field(default_factory=tuple)


def forecast(
	s: ServeSetup, batches: list[list[Request]], cal: Calibration | None = None, online: bool = False
) -> ServingForecast | None:
	"""Throughput and latencies for these batches; None when nothing can run (the model doesn't fit the card)."""
	cal = cal or Calibration()
	r = simulate_batches(s, batches)
	if r.output_tokens == 0 or r.kv_capacity_tokens == 0:
		return None
	n = sum(len(b) for b in batches)
	h = Z90 * cal.sigma
	mode = (
		"arriving at the request rate"
		if online
		else "sent at once, so time to first token is how long the batch takes to drain"
	)
	why = f"{n} requests {mode}, through a continuous-batching simulation; {cal.source}"

	def band(v: float, unit_why: str, slower_is_bigger: bool) -> Readout:
		v = v * cal.scale if slower_is_bigger else v / cal.scale
		return Readout(v, v * math.exp(-h), v * math.exp(h), f"{unit_why}; {why}")

	ttft = float(np.quantile(r.ttft, 0.95)) if r.ttft.size else math.nan
	tpot = float(np.quantile(r.tpot, 0.5)) if r.tpot.size else math.nan
	return ServingForecast(
		throughput=band(r.throughput, "output tokens per second", False),
		ttft_p95=band(ttft, "95th percentile time to first token", True),
		tpot_p50=band(tpot, "median seconds per output token", True),
		kv_capacity_tokens=r.kv_capacity_tokens,
		peak_running=r.peak_running,
		notes=r.notes,
	)
