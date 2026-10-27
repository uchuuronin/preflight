import pytest

from preflight.engine import gpus

GIB = 2**30


def test_table_covers_both_vendors_and_many_types():
	assert len(gpus.list_gpus("nvidia")) >= 30
	assert len(gpus.list_gpus("amd")) >= 6


def test_every_row_is_internally_consistent():
	for name in gpus.list_gpus():
		s = gpus.get_gpu(name)
		assert s.memory_gib > 0 and s.bandwidth_gbps > 0 and s.compute_units > 0, name
		assert s.fp32 > 0, name
		for d in ("bf16", "fp16", "fp8"):
			v = getattr(s, d)
			assert v is None or v >= s.fp32 * 0.9, (name, d)
		assert s.confidence in ("datasheet", "derived", "secondary")


def test_h100_sxm_dense_bf16_and_bandwidth():
	h100 = gpus.get_gpu("H100-SXM-80GB")
	assert h100.peak_tflops("bf16") == pytest.approx(989.4, rel=0.01)
	assert h100.bandwidth_gbps == pytest.approx(3350)
	assert h100.memory_gib == 80


def test_mi300x_matches_rocm_docs():
	mi = gpus.get_gpu("MI300X")
	assert mi.peak_tflops("bf16") == pytest.approx(1307.4)
	assert mi.bandwidth_gbps == pytest.approx(5300)
	assert mi.compute_units == 304


@pytest.mark.parametrize(
	"device_name, memory_gib, cus, expected",
	[
		# NVIDIA names as torch.cuda.get_device_name reports them
		("NVIDIA H100 80GB HBM3", 79.6, None, "H100-SXM-80GB"),
		("NVIDIA H100 PCIe", 79.6, None, "H100-PCIe-80GB"),
		("NVIDIA H100 NVL", 93.0, None, "H100-NVL-94GB"),
		("NVIDIA H200", 139.8, None, "H200-SXM-141GB"),
		("NVIDIA H200 NVL", 139.8, None, "H200-NVL-141GB"),
		("NVIDIA GH200 480GB", 95.0, None, "GH200-96GB"),
		("NVIDIA GH200 144G HBM3e", 142.0, None, "GH200-144GB"),
		("NVIDIA A100-SXM4-80GB", 79.2, None, "A100-SXM-80GB"),
		("NVIDIA A100-SXM4-40GB", 39.4, None, "A100-SXM-40GB"),
		("NVIDIA A100-PCIE-40GB", 39.4, None, "A100-PCIe-40GB"),
		("NVIDIA A100 80GB PCIe", 79.2, None, "A100-PCIe-80GB"),
		("NVIDIA B200", 178.0, None, "B200-SXM-180GB"),
		("NVIDIA B300 SXM6 AC", 270.0, None, "B300-SXM-288GB"),
		("NVIDIA RTX PRO 6000 Blackwell Server Edition", 95.0, None, "RTX-PRO-6000-96GB"),
		("Tesla T4", 14.7, None, "T4"),
		("Tesla P100-PCIE-16GB", 15.9, None, "P100-16GB"),
		("Tesla V100-SXM2-32GB", 31.7, None, "V100-SXM2-32GB"),
		("Tesla V100-SXM2-16GB", 15.8, None, "V100-SXM2-16GB"),
		("Tesla V100-PCIE-16GB", 15.8, None, "V100-PCIe-16GB"),
		("Tesla M40 24GB", 22.4, None, "M40-24GB"),
		("NVIDIA GeForce GTX TITAN X", 11.9, None, "TITAN-X-Maxwell"),
		("NVIDIA TITAN X (Pascal)", 11.9, None, "TITAN-X-Pascal"),
		("NVIDIA TITAN V", 11.8, None, "TITAN-V"),
		("NVIDIA GeForce GTX 1080 Ti", 10.9, None, "GTX-1080-Ti"),
		("NVIDIA GeForce RTX 2080 Ti", 10.7, None, "RTX-2080-Ti"),
		("NVIDIA GeForce RTX 2080", 7.8, None, "RTX-2080"),
		("Quadro RTX 8000", 47.5, None, "Quadro-RTX-8000"),
		("NVIDIA RTX A4000", 15.7, None, "RTX-A4000"),
		("NVIDIA RTX A5000", 23.6, None, "RTX-A5000"),
		("NVIDIA RTX A6000", 47.5, None, "RTX-A6000"),
		("NVIDIA A40", 44.4, None, "A40"),
		("NVIDIA A16", 14.6, None, "A16"),
		("NVIDIA A10", 22.1, None, "A10"),
		("NVIDIA A10G", 22.1, None, "A10"),
		("NVIDIA GeForce RTX 3090", 23.7, None, "RTX-3090"),
		("NVIDIA L4", 22.0, None, "L4"),
		("NVIDIA L40", 44.4, None, "L40"),
		("NVIDIA L40S", 44.4, None, "L40S"),
		("NVIDIA GeForce RTX 4090", 23.6, None, "RTX-4090"),
		("NVIDIA GeForce RTX 5090", 31.4, None, "RTX-5090"),
		# AMD names as ROCm PyTorch reports them
		("AMD Instinct MI300X", 191.5, 304, "MI300X"),
		("AMD Instinct MI300X VF", 191.5, 304, "MI300X"),
		("AMD Instinct MI325X", 255.0, 304, "MI325X"),
		("AMD Instinct MI355X", 287.0, 256, "MI355X"),
		("AMD Instinct MI250X / MI250", 63.98, 110, "MI250X-GCD"),
		("AMD Instinct MI250X / MI250", 63.98, 104, "MI250-GCD"),
		("AMD Instinct MI210", 63.98, 104, "MI210"),
		("AMD Instinct MI100", 31.98, 120, "MI100"),
		("gfx942", 191.5, 304, "MI300X"),
		("gfx942", 255.0, 304, "MI325X"),
		("gfx90a", 63.98, 110, "MI250X-GCD"),
		# Modal strings and table names
		("a100-80gb", None, None, "A100-SXM-80GB"),
		("A100-40GB", None, None, "A100-SXM-40GB"),
		("h100!", None, None, "H100-SXM-80GB"),
		("RTX-PRO-6000", None, None, "RTX-PRO-6000-96GB"),
		("L40S", None, None, "L40S"),
	],
)
def test_resolves_device_rocm_and_modal_names(device_name, memory_gib, cus, expected):
	mem = int(memory_gib * GIB) if memory_gib else None
	assert gpus.resolve(device_name, memory_bytes=mem, compute_units=cus) == expected


def test_mig_slice_scales_to_its_share():
	full = gpus.get_gpu("A100-SXM-80GB")
	mig = gpus.get_gpu("NVIDIA A100-SXM4-80GB MIG 3g.40gb")
	assert mig.name.startswith("A100-SXM-80GB-MIG-3g")
	assert mig.peak_tflops("bf16") == pytest.approx(312 * 3 / 7)
	assert mig.bandwidth_gbps < full.bandwidth_gbps
	assert mig.nominal_bytes() == pytest.approx(40e9, rel=1e-6)


def test_unity_rtx_pro_mig_slice():
	mig = gpus.get_gpu("NVIDIA RTX PRO 6000 Blackwell Server Edition MIG 2g.48gb")
	assert mig.peak_tflops("bf16") == pytest.approx(500 * 2 / 4)


def test_unknown_gpu_raises_keyerror():
	with pytest.raises(KeyError):
		gpus.get_gpu("Voodoo 3")


def test_fp32_uses_tf32_only_when_allowed():
	a100 = gpus.get_gpu("A100-SXM-80GB")
	assert a100.peak_tflops("fp32") == pytest.approx(19.5)  # PyTorch's default: TF32 off
	assert a100.peak_tflops("fp32", allow_tf32=True) == pytest.approx(156)


def test_amd_fp32_ignores_tf32_by_default():
	mi = gpus.get_gpu("MI300X")
	assert mi.peak_tflops("fp32", allow_tf32=True) == pytest.approx(163.4)


@pytest.mark.parametrize(
	"name, dtype",
	[("T4", "bf16"), ("V100-SXM2-32GB", "bf16"), ("P100-16GB", "fp16"), ("A100-SXM-80GB", "fp8")],
)
def test_missing_fast_paths(name, dtype):
	assert not gpus.get_gpu(name).supports(dtype)


def test_ridge_point_is_flops_over_bytes():
	h100 = gpus.get_gpu("H100-SXM-80GB")
	assert h100.ridge_point("bf16") == pytest.approx(989.4e12 / 3350e9)


def test_peak_flops_is_a_drop_in_for_the_recorder():
	assert gpus.peak_flops("NVIDIA H100 80GB HBM3", "bf16") == pytest.approx(989.4e12)
	assert gpus.peak_flops("NVIDIA A100-SXM4-80GB", "tf32") == pytest.approx(156e12)
	assert gpus.peak_flops("NVIDIA A100-SXM4-80GB", "fp32") == pytest.approx(19.5e12)
	assert gpus.peak_flops("Tesla T4", "tf32") == pytest.approx(8.1e12)  # no TF32 on Turing: fp32 rate
	assert gpus.peak_flops("Tesla T4", "bf16") is None
	assert gpus.peak_flops("Apple MPS", "fp32") is None
	assert gpus.peak_flops(None, "bf16") is None


def test_modal_names_cover_every_priced_modal_gpu():
	for p in gpus.prices("modal"):
		assert p.gpu in gpus.MODAL_GPU
		assert gpus.resolve(gpus.MODAL_GPU[p.gpu]) == p.gpu


def test_price_lookup_has_source_and_date():
	p = gpus.price("H100-SXM-80GB", provider="modal")
	assert p.usd_per_hour == pytest.approx(3.95)
	assert p.source and p.as_of and p.preemptible


def test_missing_price_returns_none():
	assert gpus.price("RTX-4090", provider="modal") is None
	assert gpus.on_demand_reference("V100-SXM2-32GB") is None


def test_on_demand_reference_picks_cheapest_listing():
	assert gpus.on_demand_reference("MI300X").provider == "amd_devcloud"


def test_capacity_is_what_pytorch_reports_not_the_box():
	l4 = gpus.get_gpu("L4")
	assert l4.capacity_bytes() == pytest.approx(
		21.45 * GIB, rel=1e-3
	)  # A's golden record: 23,034,920,960 bytes
	assert l4.nominal_bytes() == 24 * GIB
	assert l4.usable_source == "reported"
	assert gpus.get_gpu("T4").capacity_bytes() < 15 * GIB


def test_estimated_capacity_by_memory_type():
	assert gpus.get_gpu("MI300X").usable_capacity_gib() == pytest.approx(192 * 0.99)
	assert gpus.get_gpu("A40").usable_capacity_gib() == pytest.approx(48 * 0.92)  # ECC on
	assert gpus.get_gpu("RTX-3090").usable_capacity_gib() == pytest.approx(24 * 0.985)
	assert gpus.get_gpu("A40").usable_source == "estimated"


def test_mig_bandwidth_follows_memory_slices():
	mig = gpus.get_gpu("NVIDIA A100-SXM4-80GB MIG 3g.40gb")
	assert mig.bandwidth_gbps == pytest.approx(2039 * 40e9 / (80 * GIB))


def test_normalise_dtype_accepts_torch_spellings():
	assert gpus.normalise_dtype("torch.bfloat16") == "bf16"
	assert gpus.normalise_dtype("float8_e4m3fnuz") == "fp8"
	with pytest.raises(ValueError):
		gpus.normalise_dtype("int2")


# ---------------------------------------------------------------- A's Oct 27 requests: MIG and bounded names


def test_mig_slice_memory_does_not_pick_the_smaller_card():
	assert gpus.resolve("NVIDIA A100-SXM4-80GB MIG 3g.40gb", int(40e9), 42) == "A100-SXM-80GB"
	assert gpus.resolve("NVIDIA A100-SXM4-40GB MIG 3g.20gb", int(20e9), 42) == "A100-SXM-40GB"


def test_mig_card_from_the_profile_when_the_name_has_no_size():
	assert gpus.resolve("NVIDIA A100 MIG 3g.40gb") == "A100-SXM-80GB"
	assert gpus.resolve("NVIDIA A100 MIG 7g.40gb") == "A100-SXM-40GB"
	assert gpus.resolve("NVIDIA A100 MIG 1g.5gb") == "A100-SXM-40GB"


def test_scaled_mig_name_round_trips():
	slice_ = gpus.get_gpu("NVIDIA A100-SXM4-80GB MIG 3g.40gb", int(40e9), 42)
	again = gpus.get_gpu(slice_.name)
	assert again.name == slice_.name and again.bf16 == pytest.approx(slice_.bf16)
	assert gpus.peak_flops(slice_.name, "bf16") == pytest.approx(312e12 * 3 / 7)


@pytest.mark.parametrize("name", ["1" * 40_000, "a100 " + "9" * 40_000 + "gb", "mig 3g." + "1" * 5_000])
def test_long_names_fail_fast(name):
	import time

	t = time.perf_counter()
	with pytest.raises(KeyError):
		gpus.resolve(name)
	assert gpus.peak_flops(name, "bf16") is None
	assert gpus.mig_profile(name) is None
	assert time.perf_counter() - t < 0.1


def test_sizes_in_names_still_resolve():
	assert gpus.resolve("a100-80gb") == "A100-SXM-80GB"
	assert gpus.resolve("NVIDIA A100-SXM4-40GB") == "A100-SXM-40GB"
	assert gpus.resolve("NVIDIA GH200 480GB") == "GH200-96GB"


@pytest.mark.parametrize(
	"name,card",
	[
		("NVIDIA GH200 MIG 1g.24gb", "GH200-96GB"),
		("NVIDIA GH200 MIG 1g.12gb", "GH200-96GB"),
		("NVIDIA A100 MIG 1g.10gb", "A100-SXM-80GB"),  # tie: the 80 GB card's smallest profile
		("NVIDIA A100 MIG 1g.20gb", "A100-SXM-80GB"),
		("NVIDIA H200 MIG 1g.18gb", "H200-SXM-141GB"),
		("NVIDIA A100-PCIE-40GB MIG 2g.10gb", "A100-PCIe-40GB"),
	],
)
def test_mig_profiles_pick_their_card(name, card):
	assert gpus.resolve(name, int(5e9)) == card
