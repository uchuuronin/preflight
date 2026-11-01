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

PRICE_TABLE_VERSION = "2026-11-01"  # Nov 1: three secondary rows checked on the providers' own pages
DEFAULT_REGION = "us"

Kind = Literal["on_demand", "spot", "own_cluster"]
Confidence = Literal["provider", "secondary", "contract"]


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
	GpuRate("MI300X", "amd_devcloud", 2.59, "2026-10", "Preflight contract, gap 8", confidence="contract"),
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


def rows() -> list:
	return [*GPU_RATES, *TOKEN_RATES, *STORAGE_RATES]
