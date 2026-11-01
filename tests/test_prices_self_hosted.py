"""Glimmer self-hosted (Chiya, Nov 1): priced by GPU-seconds per call at the GPU's own rate row."""

import pytest

from preflight.engine import prices


def test_a_glimmer_call_is_priced_from_its_gpu_seconds_and_the_gpu_rate():
	r = prices.self_hosted_rate("muse-glimmer", "A100-SXM-40GB")
	assert r is not None and r.confidence == "measured" and r.source.startswith("measured:")
	hourly = prices.gpu_rate("A100-SXM-40GB", "modal").usd_per_hour
	assert r.usd_per_call() == pytest.approx(r.gpu_seconds_per_call * hourly / 3600)
	assert r.usd_per_call(batched=True) < r.usd_per_call()
	assert r.usd_cold() == pytest.approx((r.cold_start_s + r.idle_s) * hourly / 3600)


def test_every_self_hosted_row_has_a_gpu_rate_and_a_unique_id():
	ids = [r.id for r in prices.SELF_HOSTED_RATES]
	assert len(ids) == len(set(ids))
	for r in prices.SELF_HOSTED_RATES:
		assert prices.gpu_rate(r.gpu, r.provider) is not None, r.id
		assert r in prices.rows()


def test_the_price_table_version_is_a_date():
	"""CostBasis.as_of reads it (CI, Nov 1: a "2026-11-01b" version failed every cost basis)."""
	from datetime import date

	date.fromisoformat(prices.PRICE_TABLE_VERSION)
