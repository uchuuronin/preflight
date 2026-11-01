"""The versioned price table behind every dollar Preflight shows (contract: "Run cost", gap 21).

One row per provider and GPU, model or storage tier, and region, each with a source and an
as-of date. New tools, models or providers add rows here, never formulas to the UI. The table
is in USD; D adds regional rows and converts at display time.

Rows whose figures come from a third-party tracker rather than the provider's own page carry
confidence "secondary", and the UI can say so.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PRICE_TABLE_VERSION = "2026-11-01"  # a date: CostBasis.as_of reads it. Nov 1: three secondary rows checked; Glimmer self-hosted, measured
DEFAULT_REGION = "us"

Kind = Literal["on_demand", "spot", "own_cluster"]
Confidence = Literal["provider", "secondary", "contract", "measured"]


@dataclass(frozen=True)
class GpuRate:
	gpu: str  # engine.gpus table name
	provider: str  # same ids as Env.provider: modal, amd_devcloud, ...
	usd_per_hour: float
	as_of: str
	source: str
	region: str = DEFAULT_REGION
	kind: Kind = "on_demand"
	preemptible: bool = False  # can be taken back mid-run (spot, Modal's serverless containers)
	confidence: Confidence = "provider"

	@property
	def id(self) -> str:
		return f"gpu/{self.provider}/{self.gpu}/{self.region}/{self.kind}"


@dataclass(frozen=True)
class TokenRate:
	model: str  # the id the knob and AgentTask.model use
	provider: str
	input_per_m: float
	output_per_m: float
	as_of: str
	source: str
	cached_input_per_m: float | None = None  # None: cached input billed as input
	reasoning_per_m: float | None = None  # None: reasoning billed as output, inside output tokens
	region: str = DEFAULT_REGION
	confidence: Confidence = "provider"

	@property
	def id(self) -> str:
		return f"tokens/{self.provider}/{self.model}/{self.region}"

	def usd(self, input_tokens: float, output_tokens: float, cached_input_tokens: float = 0.0) -> float:
		"""Dollars for one call or task. Cached tokens are a part of the input, priced at the cached rate."""
		cached = min(max(cached_input_tokens, 0.0), input_tokens)
		cached_rate = self.input_per_m if self.cached_input_per_m is None else self.cached_input_per_m
		return (
			(input_tokens - cached) * self.input_per_m
			+ cached * cached_rate
			+ output_tokens * self.output_per_m
		) / 1e6


@dataclass(frozen=True)
class StorageRate:
	provider: str
	usd_per_gb_month: float
	as_of: str
	source: str
	region: str = DEFAULT_REGION
	confidence: Confidence = "provider"

	@property
	def id(self) -> str:
		return f"storage/{self.provider}/{self.region}"


# ---------------------------------------------------------------- GPUs

_MODAL_SOURCE = "Modal list rate, per-second billing, base region"
_MODAL = (
	("T4", 0.59),
	("L4", 0.80),
	("A10", 1.10),
	("L40S", 1.95),
	("A100-SXM-80GB", 2.50),
	("RTX-PRO-6000-96GB", 3.03),
	("H100-SXM-80GB", 3.95),
	("H200-SXM-141GB", 4.54),
	("B200-SXM-180GB", 6.25),
	("B300-SXM-288GB", 7.10),
)
GPU_RATES: tuple[GpuRate, ...] = (
	*(GpuRate(g, "modal", p, "2026-07", _MODAL_SOURCE, preemptible=True) for g, p in _MODAL),
	GpuRate(
		"A100-SXM-40GB",
		"modal",
		2.10,
		"2026-11",
		"modal.com/pricing, checked Nov 1 2026: 'Nvidia A100, 40 GB' at $0.000583/s ($2.10/h; the page doesn't say SXM "
		"or PCIe, and single-GPU specs are the same)",
		preemptible=True,
	),
	GpuRate(
		"MI300X",
		"amd_devcloud",
		1.99,
		"2026-11",
		"Chiya, Nov 1: the team's AMD droplet bills $1.99/h for its whole uptime (was the contract's $2.59, gap 8)",
		confidence="provider",
	),
)

# ---------------------------------------------------------------- LLM and API tokens

TOKEN_RATES: tuple[TokenRate, ...] = (
	TokenRate(
		"muse-spark",
		"meta",
		1.25,
		4.25,
		"2026-09",
		"eesel.ai, Muse Spark 1.3 pricing (Sep 8 2026; standard xhigh endpoint, rates confirmed on Artificial "
		"Analysis); cached input is the stated 88% discount. Nov 1: Meta's own page (dev.meta.ai) couldn't be reached "
		"from here, so this row stays secondary",
		cached_input_per_m=0.15,
		confidence="secondary",
	),
	TokenRate(
		"muse-spark-open",
		"baseten",
		1.25,
		4.25,
		"2026-11",
		"baseten.co/library/muse-spark-open, checked Nov 1 2026: $1.25 input, $4.25 output, $0.15 cached input per 1M",
		cached_input_per_m=0.15,
	),
	TokenRate(
		"muse-glimmer",
		"qubrid",
		0.25,
		1.05,
		"2026-09",
		"qubrid.com/blog/muse-glimmer-pricing-api-cost-per-1m-tokens (Sep 3 2026), checked Nov 1 2026: Qubrid's own rate "
		"card, $0.25 input, $1.05 output, $0.03 cached input per 1M",
		cached_input_per_m=0.03,
	),
)

# ---------------------------------------------------------------- storage (dollars only where storage is paid)

# ---------------------------------------------------------------- models we serve ourselves


@dataclass(frozen=True)
class SelfHostedRate:
	"""A model Preflight serves itself, priced by the GPU time one call takes (Chiya, Nov 1): GPU-seconds per call
	at the GPU's own rate row, and the cold start a call pays when the endpoint has scaled to zero."""

	model: str
	gpu: str  # a GPU_RATES name
	gpu_seconds_per_call: float  # one call alone on the GPU, warm
	cold_start_s: float  # asked for to ready, after scaling to zero (no queue)
	call: str  # what one call was when it was measured
	as_of: str
	source: str
	provider: str = "modal"
	gpu_seconds_per_call_batched: float | None = None  # with several calls in flight
	idle_s: float = 0.0  # what the endpoint idles after its last call before scaling to zero
	confidence: Confidence = "measured"

	@property
	def id(self) -> str:
		return f"selfhosted/{self.provider}/{self.model}/{self.gpu}"

	def usd_per_gpu_second(self) -> float:
		r = gpu_rate(self.gpu, self.provider)
		if r is None:
			raise KeyError(f"no {self.provider} rate for {self.gpu}")
		return r.usd_per_hour / 3600

	def usd_per_call(self, batched: bool = False) -> float:
		s = (
			self.gpu_seconds_per_call_batched
			if batched and self.gpu_seconds_per_call_batched
			else self.gpu_seconds_per_call
		)
		return s * self.usd_per_gpu_second()

	def usd_cold(self) -> float:
		"""What a call after the endpoint scaled to zero adds: the cold start, and the idle tail before it scales down."""
		return (self.cold_start_s + self.idle_s) * self.usd_per_gpu_second()


_GLIMMER_CALL = (
	"one triage: 17,138 tokens in, about 300 out (reasoning low), the fixed prompt in glimmer/common.py"
)
SELF_HOSTED_RATES: tuple[SelfHostedRate, ...] = (
	SelfHostedRate(
		"muse-glimmer",
		"A100-SXM-40GB",
		12.8,
		286.0,
		_GLIMMER_CALL,
		"2026-11",
		"measured: remote-runs b-glimmer-smoke-a100 (GitHub run 41330014731, Nov 1): vLLM 0.28.0, FP8 copy of "
		"meta-models/Muse-Glimmer-30B@cec94fcf; median of 5 calls alone 12.8 s (38 tokens/s decoding, 4.6 s to the "
		"first token), 4 at once 27.7 s; cold start 286 s",
		gpu_seconds_per_call_batched=6.9,
		idle_s=120.0,
	),
	SelfHostedRate(
		"muse-glimmer",
		"L4",
		43.7,
		18.1,
		_GLIMMER_CALL,
		"2026-11",
		"measured: remote-runs b-glimmer-smoke-l4 (GitHub run 41328764340, Nov 1): llama.cpp build 12030, Meta's "
		"Q4_K_M GGUF@b44f1072; median of 5 calls 43.7 s (14.4 tokens/s decoding, 22.3 s to the first token); server "
		"ready 18 s after the container started (the L4 itself took 19 min to schedule, not counted). The own-GPU proof, "
		"not hosted",
	),
)

STORAGE_RATES: tuple[StorageRate, ...] = ()  # shown in GB against the quota until a paid tier is added


# ---------------------------------------------------------------- lookups


def gpu_rate(
	gpu: str, provider: str | None = None, region: str | None = None, kind: Kind = "on_demand"
) -> GpuRate | None:
	"""The row for this GPU from this provider (any region if none given), or None."""
	rows = [
		r
		for r in GPU_RATES
		if r.gpu == gpu
		and (provider is None or r.provider == provider)
		and (region is None or r.region == region)
		and r.kind == kind
	]
	return min(rows, key=lambda r: r.usd_per_hour) if rows else None


def market_reference(gpu: str) -> GpuRate | None:
	"""Cheapest on-demand row for this GPU: the grey "market equivalent" beside own-cluster GPU-hours."""
	return gpu_rate(gpu, kind="on_demand")


def token_rate(model: str, provider: str | None = None) -> TokenRate | None:
	m = model.lower()
	rows = [
		r
		for r in TOKEN_RATES
		if (r.model == m or m.startswith(r.model + "-")) and (provider is None or r.provider == provider)
	]
	return min(rows, key=lambda r: len(r.model) * -1) if rows else None  # the most specific id wins


def storage_rate(provider: str, region: str | None = None) -> StorageRate | None:
	for r in STORAGE_RATES:
		if r.provider == provider and (region is None or r.region == region):
			return r
	return None


def self_hosted_rate(model: str, gpu: str | None = None) -> SelfHostedRate | None:
	for r in SELF_HOSTED_RATES:
		if r.model == model and (gpu is None or r.gpu == gpu):
			return r
	return None


def rows() -> list:
	return [*GPU_RATES, *TOKEN_RATES, *STORAGE_RATES, *SELF_HOSTED_RATES]
