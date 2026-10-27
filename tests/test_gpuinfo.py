"""Canonical GPU names and peak FLOP/s: B's table when it is importable, A's minimal one otherwise."""

import pytest

from preflight.recorder.gpuinfo import canonical_gpu_name, peak


def test_resolver_result_is_used():
	name, raw = canonical_gpu_name(
		"NVIDIA H100 80GB HBM3", 85e9, 132, resolver=lambda n, m, c: "H100-SXM-80GB"
	)
	assert (name, raw) == ("H100-SXM-80GB", None)


def test_unknown_gpu_keeps_the_raw_name_and_says_so():
	def unknown(n, m, c):
		raise KeyError(n)

	name, raw = canonical_gpu_name("Mystery Accelerator 9000", 16e9, 40, resolver=unknown)
	assert name == "Mystery Accelerator 9000" and raw == "Mystery Accelerator 9000"


def test_rocm_generic_name_falls_back_to_the_arch():
	seen = []

	def resolver(n, m, c):
		seen.append(n)
		if n == "gfx942":
			return "MI300X"
		raise KeyError(n)

	name, _ = canonical_gpu_name(
		"AMD Radeon Graphics", 192e9, 304, arch="gfx942:sramecc+:xnack-", resolver=resolver
	)
	assert name == "MI300X" and seen == ["AMD Radeon Graphics", "gfx942"]


def test_peak_prefers_bs_table_and_falls_back():
	assert peak("H100-SXM-80GB", "bf16", table=lambda g, p, m: 989e12) == 989e12
	assert peak("A100-SXM-80GB", "bf16", table=None) == 312e12  # A's minimal table


def test_with_bs_engine_when_merged():
	gpus = pytest.importorskip("preflight.engine.gpus")
	name, raw = canonical_gpu_name("NVIDIA L4", 23e9, 58)
	assert name == gpus.resolve("NVIDIA L4", 23e9, 58) and raw is None


def test_without_any_table_nothing_is_flagged():
	assert canonical_gpu_name("NVIDIA L4", 23e9, 58, resolver=None) == ("NVIDIA L4", None)


def test_mig_slices_keep_their_slice_with_bs_engine():
	gpus = pytest.importorskip("preflight.engine.gpus")
	raw = "NVIDIA A100-SXM4-80GB MIG 3g.40gb"
	name, flagged = canonical_gpu_name(raw, 40e9, 42)  # torch reports the slice's memory
	assert flagged is None and "MIG" in name and name.startswith("A100-SXM-80GB")
	full = gpus.get_gpu("A100-SXM-80GB").peak_flops("bf16")
	assert peak(name, "bf16", 40e9, device_name=raw) == pytest.approx(full * 3 / 7)


def test_bs_none_stands_for_a_missing_fast_path():
	pytest.importorskip("preflight.engine.gpus")
	assert peak("T4", "bf16") is None  # no bf16 tensor cores: B says None, A's table doesn't overrule it
