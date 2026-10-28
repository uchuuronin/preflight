"""D3: regional GPU prices layered on B's dated price table (engine/gpus.py)."""

import pytest

from preflight.addons.regional_prices import (
	MODAL_REGIONS,
	MULTIPLIER,
	region_kind,
	regional_price,
	regional_prices,
)
from preflight.engine.gpus import price, prices


def test_modal_multipliers_match_modals_docs():
	assert MULTIPLIER == {"broad": 1.15, "narrow": 1.75}


@pytest.mark.parametrize(
	"region,kind",
	[
		("us", "broad"),
		("eu", "broad"),
		("ap", "broad"),
		("us-west", "narrow"),
		("uk", "narrow"),
		("jp", "narrow"),
	],
)
def test_region_kind(region, kind):
	assert region_kind(region) == kind


def test_unknown_region_is_none():
	assert region_kind("mars") is None
	assert region_kind("") is None


def test_no_region_means_bs_base_rate():
	base = price("H100-SXM-80GB")
	assert regional_price("H100-SXM-80GB", None) == base


def test_regional_rate_is_base_times_multiplier_with_its_own_source():
	base = price("H100-SXM-80GB")
	row = regional_price("H100-SXM-80GB", "us-west")
	assert row is not None and base is not None
	assert row.usd_per_hour == pytest.approx(base.usd_per_hour * 1.75)
	assert row.provider == "modal"
	assert row.gpu == base.gpu
	assert "us-west" in row.source and "1.75" in row.source
	assert row.as_of == "2026-10"
	assert row.preemptible == base.preemptible


def test_aliases_resolve_like_bs_table():
	assert regional_price("h100", "us") == regional_price("H100-SXM-80GB", "us")


def test_unknown_gpu_or_region_gives_none():
	assert regional_price("not-a-gpu", "us") is None
	assert regional_price("H100-SXM-80GB", "mars") is None


def test_only_modal_has_regional_rows():
	assert regional_price("MI300X", "us", provider="amd_devcloud") is None


def test_regional_prices_cover_every_modal_gpu_once():
	rows = regional_prices("eu")
	assert {r.gpu for r in rows} == {p.gpu for p in prices("modal")}
	assert len(rows) == len({r.gpu for r in rows})


def test_every_listed_region_is_broad_or_narrow():
	assert set(MODAL_REGIONS.values()) <= {"broad", "narrow"}
	assert len(MODAL_REGIONS) == 22
