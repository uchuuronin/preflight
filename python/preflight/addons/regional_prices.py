"""D3: regional GPU prices, layered on B's dated price table (engine/gpus.py).

Modal charges a multiplier on its base rate when a Function or Sandbox pins a container
region: 1.15x for a broad region (us, eu, ap) and 1.75x for a narrow one (us-west, uk, ...);
with no region there is no multiplier. Source: Modal, "Region selection" (docs), read
2026-10-27. B's table stays the one source of base rates; this module only derives rows.
Pure, no I/O, so it runs in Pyodide like the engine.
"""

from __future__ import annotations

from dataclasses import replace

from preflight.engine.gpus import GpuPrice, price, prices

SOURCE_URL = "https://modal.com/docs/guide/region-selection"
AS_OF = "2026-10"

# [cited: Modal "Region selection" docs, read 2026-10-27]
MULTIPLIER: dict[str, float] = {"broad": 1.15, "narrow": 1.75}

MODAL_REGIONS: dict[str, str] = {
	**dict.fromkeys(("us", "eu", "ap"), "broad"),
	**dict.fromkeys(
		(
			"us-east",
			"us-central",
			"us-south",
			"us-west",
			"eu-west",
			"eu-north",
			"eu-south",
			"ap-northeast",
			"ap-southeast",
			"ap-south",
			"ap-melbourne",
			"jp",
			"au",
			"uk",
			"ca",
			"me",
			"sa",
			"af",
			"mx",
		),
		"narrow",
	),
}


def region_kind(region: str | None) -> str | None:
	"""'broad', 'narrow', or None for no region or one Modal doesn't list."""
	return MODAL_REGIONS.get(region or "")


def _derive(base: GpuPrice, region: str) -> GpuPrice | None:
	kind = region_kind(region)
	if kind is None:
		return None
	m = MULTIPLIER[kind]
	return replace(
		base,
		usd_per_hour=round(base.usd_per_hour * m, 4),
		as_of=AS_OF,
		source=f"{base.source}; region {region} ({kind}) x{m}, Modal region selection docs ({SOURCE_URL})",
	)


def regional_price(gpu: str, region: str | None, provider: str = "modal") -> GpuPrice | None:
	"""B's rate for this GPU and provider, with the region's multiplier applied.

	No region returns B's row unchanged. Only Modal has regional pricing here.
	"""
	try:
		base = price(gpu, provider)
	except KeyError:  # B's resolve() raises for a GPU name it doesn't know
		return None
	if base is None:
		return None
	if not region:
		return base
	if provider != "modal":
		return None
	return _derive(base, region)


def regional_prices(region: str) -> list[GpuPrice]:
	"""Every Modal GPU in B's table, priced for one region."""
	rows = (_derive(p, region) for p in prices("modal"))
	return [r for r in rows if r is not None]
