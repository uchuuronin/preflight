import pytest

from preflight.engine import memory as m
from preflight.engine import steptime as st

GPT2 = m.TransformerArch(
	n_layers=12,
	d_model=768,
	n_heads=12,
	vocab_size=50257,
	d_ff=3072,
	tie_embeddings=True,
	bias=True,
	norm="layernorm",
	learned_positions=1024,
)


def test_training_flops_per_token_is_6n_plus_attention():
	linear = GPT2.param_breakdown()["linear"]
	expected = 6 * (linear + 768 * 50257) + 12 * 12 * 768 * 1024
	assert st.flops_per_token(GPT2, seq=1024, mode="train") == pytest.approx(expected)


def test_lora_skips_weight_gradients():
	full = st.flops_per_token(GPT2, 1024, mode="train")
	lora = st.flops_per_token(GPT2, 1024, mode="train_frozen_base")
	fwd = st.flops_per_token(GPT2, 1024, mode="forward")
	assert lora < full
	assert lora == pytest.approx(2 * fwd + 4 * 12 * 768 * 1024)


def test_full_recompute_adds_a_forward_in_hardware_flops_only():
	model = st.flops_per_token(GPT2, 1024, mode="train")
	hw = st.flops_per_token(GPT2, 1024, mode="train", recompute=True)
	assert hw == pytest.approx(model + st.flops_per_token(GPT2, 1024, mode="forward"))


def test_mfu():
	# 989.4 TFLOP/s peak; achieving half of it
	assert st.mfu(model_flops=989.4e12 * 0.5, seconds=1.0, gpu="H100", dtype="bf16") == pytest.approx(0.5)


def test_phase_fit_two_points_interpolates():
	fit = st.PhaseFit.fit([(1024, 0.10), (4096, 0.31)])
	assert fit.predict(1024) == pytest.approx(0.10)
	assert fit.predict(4096) == pytest.approx(0.31)
	assert fit.predict(2048) == pytest.approx(0.17)


def test_phase_fit_never_negative_intercept():
	fit = st.PhaseFit.fit([(1000, 0.05), (2000, 0.20)])  # straight line would cross below zero
	assert fit.intercept >= 0
	assert fit.slope >= 0


def test_phase_fit_single_point_is_proportional():
	fit = st.PhaseFit.fit([(1000, 0.2)])
	assert fit.predict(3000) == pytest.approx(0.6)


def test_constant_phase_stays_constant():
	fit = st.PhaseFit.fit([(1000, 0.02), (4000, 0.02)], constant=True)
	assert fit.predict(16000) == pytest.approx(0.02)


def test_compute_bound_kernel_scales_by_peak_flops():
	# big GEMM: 4096^3 * 2 flops, ~100 MB traffic -> compute-bound on both GPUs
	f = 2 * 4096**3
	k = st.KernelRow(
		name="sm80_gemm", seconds=f / 312e12 / 0.8, flops=f, bytes=3 * 4096 * 4096 * 2, dtype="bf16"
	)
	assert st.bound(k, "A100-80GB") == "compute"
	t = st.scale_kernel(k, "A100-80GB", "H100")
	assert t == pytest.approx(k.seconds * 312 / 989.4, rel=1e-6)


def test_memory_bound_kernel_scales_by_bandwidth():
	b = 1e9
	k = st.KernelRow(
		name="vectorized_elementwise", seconds=b / 2039e9 / 0.7, flops=b / 2, bytes=b, dtype="bf16"
	)
	assert st.bound(k, "A100-80GB") == "memory"
	assert st.scale_kernel(k, "A100-80GB", "H100") == pytest.approx(k.seconds * 2039 / 3350, rel=1e-6)


def test_unknown_cost_kernel_assumed_memory_bound():
	k = st.KernelRow(name="mystery", seconds=1e-3)
	assert st.scale_kernel(k, "A100-80GB", "H100") == pytest.approx(1e-3 * 2039 / 3350)


def test_tiny_kernel_is_latency_bound_and_unchanged():
	k = st.KernelRow(name="tiny", seconds=4e-6, flops=1e3, bytes=1e3, dtype="bf16")
	assert st.bound(k, "A100-80GB") == "latency"
	assert st.scale_kernel(k, "A100-80GB", "H100") == pytest.approx(4e-6)


def test_cross_gpu_step_keeps_host_gap():
	kernels = [
		st.KernelRow("gemm", 0.08, flops=2 * 4096**3 * 10, bytes=1e8, dtype="bf16"),
		st.KernelRow("elementwise", 0.02, flops=1e8, bytes=4e10 * 0.02 * 0.7, dtype="bf16"),
	]
	est = st.predict_cross_gpu(kernels, measured_step_s=0.12, src="A100-80GB", tgt="H100")
	assert est.breakdown["host"] == pytest.approx(0.02)
	assert est.point < 0.12
	assert est.lo < est.point < est.hi
	assert est.status == "predicted"


def test_cross_vendor_interval_is_wider():
	kernels = [st.KernelRow("gemm", 0.1, flops=2 * 4096**3 * 10, bytes=1e8, dtype="bf16")]
	nv = st.predict_cross_gpu(kernels, 0.1, "A100-80GB", "H100")
	amd = st.predict_cross_gpu(kernels, 0.1, "A100-80GB", "MI300X")
	assert amd.sigma_log > nv.sigma_log


def test_same_gpu_step_prediction_flags_extrapolation():
	probes = [
		st.StepProbe(
			tokens=2048, phases={"forward": 0.03, "backward": 0.06, "optimizer": 0.01, "data": 0.005}
		),
		st.StepProbe(
			tokens=8192, phases={"forward": 0.11, "backward": 0.22, "optimizer": 0.01, "data": 0.02}
		),
	]
	model = st.SameGpuModel.fit(probes, constant_phases={"optimizer"})
	mid = model.predict(4096)
	assert mid.status == "predicted"
	assert mid.breakdown["optimizer"] == pytest.approx(0.01)
	assert mid.point == pytest.approx(sum(mid.breakdown.values()))
	assert model.predict(32768).status == "abstain"
	assert model.predict(512).status == "abstain"
	assert model.predict(8192).status == "predicted"  # same work units is not a measurement by itself
	assert model.predict(8192, measured_probe="probe1").status == "measured"


# ---------------------------------------------------------------- phases from A's probes


def test_fractions_become_seconds_of_the_median_step():
	sec = st.phases_seconds(
		{"data": 0.04, "forward": 0.31, "backward": 0.55, "optimizer": 0.07, "idle": 0.03}, 0.74
	)
	assert sum(sec.values()) == pytest.approx(0.74)
	assert sec["backward"] == pytest.approx(0.55 * 0.74)


def test_every_schema_phase_has_a_scaling_rule():
	for name in (
		"data",
		"forward",
		"backward",
		"optimizer",
		"eval",
		"rollout",
		"reference",
		"reward",
		"update",
		"weight_sync",
		"idle",
		"other",
	):
		assert st.phase_scaling(name) in ("work", "generation", "eval", "constant", "comm")


def test_same_gpu_model_from_fractions_holds_optimizer_constant():
	probes = [
		st.StepProbe.from_fractions(
			2048, 0.74, {"forward": 0.31, "backward": 0.55, "optimizer": 0.10, "data": 0.04}
		),
		st.StepProbe.from_fractions(
			4096, 1.36, {"forward": 0.33, "backward": 0.58, "optimizer": 0.054, "data": 0.036}
		),
	]
	model = st.SameGpuModel.fit(probes)
	a, b = model.predict(3000), model.predict(3500)
	assert a.breakdown["optimizer"] == pytest.approx(b.breakdown["optimizer"])
	assert b.breakdown["forward"] > a.breakdown["forward"]


def test_recompute_adds_a_forward():
	probes = [
		st.StepProbe(2048, {"forward": 0.2, "backward": 0.4}),
		st.StepProbe(4096, {"forward": 0.4, "backward": 0.8}),
	]
	m = st.SameGpuModel.fit(probes)
	assert m.predict(3000, recompute=True).point == pytest.approx(m.predict(3000).point * 4 / 3)


# ---------------------------------------------------------------- kernel classes


@pytest.mark.parametrize(
	"name, cls",
	[
		("sm90_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize128x128x64", "gemm"),
		("ampere_bf16_s16816gemm_bf16_128x128_ldg8_f2f_stages_32x5_tn", "gemm"),
		("nvjet_hsh_128x256_64x4_2x1_v_bz_coopA_TNN", "gemm"),
		("Cijk_Alik_Bljk_BBS_BH_MT128x128x64_MI16x16x16x1", "gemm"),
		("void ck::kernel_gemm_xdl_cshuffle_v3<...>", "gemm"),
		("triton_tem_fused_mm_0", "gemm"),
		("aten::addmm", "gemm"),
		("flash_fwd_kernel<Flash_fwd_kernel_traits<128, 128, 64, 4>>", "attention"),
		("ck_tile::FmhaFwdKernel<...>", "attention"),
		("aten::_scaled_dot_product_flash_attention", "attention"),
		("implicit_convolve_sgemm", "conv"),
		("miopenSp3AsmConv_v21_1_3_gfx9_fp32_stride1", "conv"),
		("void at::native::(anonymous namespace)::vectorized_layer_norm_kernel<float>", "norm"),
		("void at::native::vectorized_elementwise_kernel<4, at::native::GeluCUDAKernelImpl>", "elementwise"),
		("triton_poi_fused_add_mul_0", "elementwise"),
		("triton_red_fused_sum_1", "reduction"),
		("void at::native::reduce_kernel<512, 1>", "reduction"),
		("void at::native::(anonymous namespace)::cunn_SoftMaxForward<8, float>", "softmax"),
		("ncclDevKernel_AllReduce_Sum_bf16_RING_LL", "comm"),
		("rcclKernel_AllReduce", "comm"),
		(
			"void at::native::(anonymous namespace)::multi_tensor_apply_kernel<FusedAdamWMathFunctor>",
			"optimizer",
		),
		("Memcpy HtoD (Pageable -> Device)", "copy"),
		("__amd_rocclr_copyBuffer", "copy"),
		("void at::native::index_select_small_index_kernel", "index"),
		("aten::nll_loss_forward", "loss"),
		("some_custom_kernel", "other"),
	],
)
def test_kernel_classifier_covers_cuda_rocm_triton_and_aten(name, cls):
	assert st.classify_kernel(name) == cls


def test_k4_fallback_scales_gemm_by_flops_and_elementwise_by_bandwidth():
	g = st.KernelRow("ampere_bf16_s16816gemm_bf16", 1e-3)
	e = st.KernelRow("vectorized_elementwise_kernel", 1e-3)
	assert st.scale_kernel(g, "A100-SXM-80GB", "H100-SXM-80GB") == pytest.approx(1e-3 * 312 / 989.4)
	assert st.scale_kernel(e, "A100-SXM-80GB", "H100-SXM-80GB") == pytest.approx(1e-3 * 2039 / 3350)


def test_comm_kernels_scale_with_the_link():
	k = st.KernelRow("ncclDevKernel_AllReduce_Sum_bf16_RING_LL", 1e-2)
	assert st.scale_kernel(k, "A100-SXM-80GB", "H100-SXM-80GB") == pytest.approx(1e-2 * 600 / 900)


def test_secondary_specs_widen_the_band():
	assert st.cross_sigma("A100-SXM-80GB", "RTX-PRO-6000-96GB") > st.cross_sigma(
		"A100-SXM-80GB", "H100-SXM-80GB"
	)


def test_phase_level_fallback_keeps_host_phases():
	out = st.scale_phases_by_class(
		{"forward": 0.2, "backward": 0.4, "data": 0.05, "rollout": 1.0}, "L4", "H100-SXM-80GB"
	)
	assert out["data"] == pytest.approx(0.05)
	assert out["forward"] < 0.2 and out["rollout"] < 1.0


# ---------------------------------------------------------------- multi-GPU


def test_ring_allreduce():
	t = st.allreduce_seconds(1e9, 8, "H100-SXM-80GB", efficiency=1.0, hop_latency_s=0)
	assert t == pytest.approx(2 * 7 / 8 * 1e9 / 450e9)
	assert st.allreduce_seconds(1e9, 1, "H100-SXM-80GB") == 0


def test_multi_gpu_step_is_extrapolated_and_slower_over_pcie():
	nv = st.multi_gpu_step(1.0, 14e9, 4, "A100-SXM-80GB")
	pc = st.multi_gpu_step(1.0, 14e9, 4, "A100-PCIe-80GB")
	assert nv.status == "extrapolated" and nv.reason
	assert pc.point > nv.point > 1.0


def test_moe_flops_use_active_params():
	moe = m.TransformerArch(
		n_layers=2,
		d_model=512,
		n_heads=8,
		vocab_size=1000,
		d_ff=1024,
		gated_mlp=True,
		n_experts=8,
		experts_per_token=2,
	)
	dense2 = m.TransformerArch(n_layers=2, d_model=512, n_heads=8, vocab_size=1000, d_ff=2048, gated_mlp=True)
	assert st.flops_per_token(moe, 128) == pytest.approx(st.flops_per_token(dense2, 128), rel=0.01)


def test_equal_work_unit_probes_keep_their_own_measurements():
	probes = [
		st.StepProbe(4096, {"forward": 0.2}, id="p2"),
		st.StepProbe(4096, {"forward": 0.3}, id="p3"),
		st.StepProbe(8192, {"forward": 0.4}, id="p1"),
	]
	m_ = st.SameGpuModel.fit(probes)
	assert m_.predict(4096, measured_probe="p2").point == pytest.approx(0.2)
	assert m_.predict(4096, measured_probe="p3").point == pytest.approx(0.3)


def test_bf16_onto_a_t4_abstains():
	k = [st.KernelRow("ampere_bf16_s16816gemm", 0.01, flops=1e12, bytes=1e9, dtype="bf16")]
	est = st.predict_cross_gpu(k, 0.012, "A100-SXM-80GB", "T4")
	assert est.status == "abstain" and "bf16" in est.reason


def test_fp32_kernels_assume_tf32_off_unless_recorded():
	off = st.KernelRow("sgemm", 0.01, flops=2e11, bytes=1e8, dtype="fp32")
	on = st.KernelRow("sgemm", 0.01, flops=2e11, bytes=1e8, dtype="fp32", allow_tf32=True)
	# V100 has no TF32; onto an A100 the replay is ~8x faster only when TF32 was on
	assert (
		st.scale_kernel(on, "V100-SXM2-32GB", "A100-SXM-80GB")
		< st.scale_kernel(off, "V100-SXM2-32GB", "A100-SXM-80GB") / 4
	)


def test_mig_slice_cannot_join_a_data_parallel_group():
	est = st.multi_gpu_step(1.0, 1e9, 2, "NVIDIA A100-SXM4-80GB MIG 3g.40gb")
	assert est.status == "abstain"


# ---------------------------------------------------------------- A's kernel tables (Oct 27 requests)

A_REPORTED_OTHER = [
	"aten::linalg_vector_norm", "aten::cat", "aten::stack", "aten::constant_pad_nd", "aten::slice_backward",
	"aten::neg", "aten::abs", "aten::sqrt", "aten::rsqrt", "aten::cos", "aten::eq", "aten::ne", "aten::gt",
	"aten::le", "aten::bitwise_and", "aten::masked_fill", "aten::lerp_", "aten::arange", "aten::linspace",
	"aten::randint", "aten::random_", "aten::scaled_dot_product_attention",
	"aten::_flash_attention_forward", "aten::_flash_attention_backward",
	"aten::_efficient_attention_forward", "aten::_efficient_attention_backward",
]  # fmt: skip


@pytest.mark.parametrize("name", A_REPORTED_OTHER)
def test_ops_a_saw_as_other_now_have_a_class(name):
	assert st.classify_kernel(name) != "other"


@pytest.mark.parametrize(
	"name,cls",
	[
		("aten::_flash_attention_forward", "attention"),
		("aten::scaled_dot_product_attention", "attention"),
		("aten::masked_fill_", "elementwise"),  # not a memset: it reads its input
		("aten::fill_", "memset"),
		("aten::lerp_", "elementwise"),
		("aten::ne", "elementwise"),
		("aten::neg", "elementwise"),
		("aten::empty", "host"),
		("aten::view", "host"),
		("aten::broadcast_tensors", "host"),  # a view, not a collective
		("c10d::broadcast_", "comm"),
		("aten::select_backward", "copy"),
		("aten::item", "host"),
	],
)
def test_classifier_edge_cases(name, cls):
	assert st.classify_kernel(name) == cls


def test_roll_up_rows_keep_their_class():
	assert st.classify_kernel("(tail) copy (6 ops)") == "copy"
	assert st.classify_kernel("(tail) attention (1 op)") == "attention"
	assert st.classify_kernel("(tail) bogus (3 ops)") == "other"
	assert st.classify_kernel("(unattributed device time)") == "other"


def test_host_ops_do_not_scale_with_the_gpu():
	k = st.KernelRow("aten::view", 2e-3, bytes=1e9)
	assert st.bound(k, "L4") == "latency"
	assert st.scale_kernel(k, "L4", "H100-SXM-80GB") == pytest.approx(2e-3)


def test_bytes_only_row_in_a_dtype_without_peak_does_not_raise():
	k = st.KernelRow("aten::arange", 1e-3, bytes=1e8, dtype="int64")
	assert st.scale_kernel(k, "L4", "H100-SXM-80GB") > 0


def test_same_gpu_replay_is_exact_even_when_bytes_overshoot():
	# 1e12 bytes can't move in 1 ms on any GPU; replaying on the same GPU must still return 1 ms
	k = st.KernelRow("vectorized_elementwise_kernel", 1e-3, bytes=1e12, dtype="bf16")
	assert st.scale_kernel(k, "L4", "L4") == pytest.approx(1e-3)


def _golden_tables():
	from pathlib import Path

	from preflight.schema import load_record

	root = Path(__file__).resolve().parents[1] / "schema" / "examples"
	for f in ("finetune.json", "pretrain.json"):
		rec = load_record((root / f).read_text())
		for p in rec.probes:
			if p.kernels:
				yield rec.env.gpu.name, p


def test_golden_kernel_tables_read_and_replay_exactly_on_their_own_gpu():
	seen = 0
	for src, probe in _golden_tables():
		rows = st.kernel_rows(probe.kernels)
		assert rows and all(r.klass in st.KERNEL_CLASSES for r in rows)
		assert all(r.flops is None or r.flops > 0 for r in rows)
		e = st.predict_cross_gpu_from_probe(probe.kernels, probe.step_time.p50_s, src, src)
		assert e.status == "predicted"
		assert e.point == pytest.approx(probe.step_time.p50_s, rel=1e-9)
		faster = st.predict_cross_gpu_from_probe(probe.kernels, probe.step_time.p50_s, src, "H100-SXM-80GB")
		assert faster.point < probe.step_time.p50_s
		seen += 1
	assert seen >= 2


def test_kernel_rows_reclassify_unknown_recorder_classes_and_take_tf32():
	row = {"name": "aten::mm", "kernel_class": "matmulish", "calls": 1.0, "device_s": 1e-3}
	row |= {"flops": 1e9, "bytes": 0, "dtype": "fp32"}
	table = {"rows": [row], "steps": 3, "backend": "cuda", "allow_tf32": True, "classified_by": None}
	(row,) = st.kernel_rows(table)
	assert row.klass is None and row.bytes is None and row.allow_tf32
	assert st.classify_kernel(row.name) == "gemm"


def test_cross_gpu_from_probe_abstains_without_a_usable_table():
	assert st.predict_cross_gpu_from_probe(None, 0.1, "L4", "H100-SXM-80GB").status == "abstain"
	row = {"name": "aten::mm", "calls": 1.0, "device_s": 1e-3, "dtype": "fp32"}
	cpu = {"rows": [row], "steps": 1, "backend": "cpu", "allow_tf32": False}
	e = st.predict_cross_gpu_from_probe(cpu, 0.1, "L4", "H100-SXM-80GB")
	assert e.status == "abstain" and "CPU" in e.reason


# ---------------------------------------------------------------- review fixes


@pytest.mark.parametrize(
	"name,cls",
	[
		("aten::addmm_", "gemm"),
		("aten::addbmm", "gemm"),
		("aten::addmv", "gemm"),
		("aten::baddbmm_", "gemm"),
		("aten::_int_mm", "gemm"),
		("aten::_weight_int4pack_mm", "gemm"),
		("aten::mv", "gemm"),
		("aten::dot", "gemm"),
		("record_param_comms", "comm"),
		("aten::conv_transpose2d", "conv"),
		("aten::expand_copy", "copy"),
		("aten::max_pool2d_with_indices", "reduction"),
		("aten::upsample_nearest2d", "copy"),
		("aten::_unsafe_index", "index"),
		("aten::glu", "elementwise"),
		("aten::add_", "elementwise"),
		("aten::silu_backward", "elementwise"),
	],
)
def test_review_classifier_cases(name, cls):
	assert st.classify_kernel(name) == cls


def test_classifier_cuts_long_names_before_matching():
	import time

	t = time.perf_counter()
	assert st.classify_kernel("ampere_" * 6000) in st.KERNEL_CLASSES
	assert time.perf_counter() - t < 0.05


def test_many_tiny_launches_are_latency_not_bandwidth():
	many = st.KernelRow("aten::add_", 10e-3, calls=2000)  # 5 us a launch
	assert st.bound(many, "L4") == "latency"
	assert st.scale_kernel(many, "L4", "H100-SXM-80GB") == pytest.approx(10e-3)
	big = st.KernelRow("aten::add_", 10e-3, calls=10)  # 1 ms a launch: bandwidth
	assert st.bound(big, "L4") == "memory"


def test_kernel_rows_validate_dict_tables():
	bad = {"rows": [{"name": "x", "calls": 1.0, "device_s": float("nan"), "dtype": "bf16"}],
		"steps": 1, "backend": "cuda", "allow_tf32": False}  # fmt: skip
	e = st.predict_cross_gpu_from_probe(bad, 0.1, "L4", "H100-SXM-80GB")
	assert e.status == "abstain" and "malformed" in e.reason
	neg = st.predict_cross_gpu([st.KernelRow("x", -1.0)], 0.1, "L4", "H100-SXM-80GB")
	assert neg.status == "abstain"
