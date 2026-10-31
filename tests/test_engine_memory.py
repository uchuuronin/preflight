import math

import pytest

from preflight.engine import memory as m

GPT2_SMALL = m.TransformerArch(
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
LLAMA2_7B = m.TransformerArch(
	n_layers=32,
	d_model=4096,
	n_heads=32,
	vocab_size=32000,
	d_ff=11008,
	tie_embeddings=False,
	gated_mlp=True,
	bias=False,
	norm="rmsnorm",
)


def test_gpt2_small_param_count():
	assert GPT2_SMALL.param_count() == 124_439_808


def test_llama2_7b_param_count():
	assert LLAMA2_7B.param_count() == 6_738_415_616


def test_activations_match_korthikanti_without_flash_or_checkpointing():
	arch = m.TransformerArch(n_layers=1, d_model=1024, n_heads=16, vocab_size=1000)
	setup = m.TrainingSetup(compute_dtype="bf16", flash_attention=False, dropout=True)
	s, b, h, a = 512, 4, 1024, 16
	expected = s * b * h * (34 + 5 * a * s / h)
	got = m.activation_bytes_per_layer(arch, setup, batch=b, seq=s)
	assert got == pytest.approx(expected, rel=1e-9)


def test_flash_attention_removes_quadratic_term():
	arch = m.TransformerArch(n_layers=1, d_model=1024, n_heads=16, vocab_size=1000)
	no_flash = m.TrainingSetup(flash_attention=False)
	flash = m.TrainingSetup(flash_attention=True)
	small = m.activation_bytes_per_layer(arch, flash, batch=1, seq=1024)
	big = m.activation_bytes_per_layer(arch, flash, batch=1, seq=2048)
	assert big == pytest.approx(2 * small, rel=0.01)
	assert m.activation_bytes_per_layer(arch, no_flash, 1, 2048) > big


def test_full_checkpointing_stores_far_less():
	none = m.TrainingSetup(grad_checkpointing="none")
	full = m.TrainingSetup(grad_checkpointing="full")
	a_none = m.activation_bytes(LLAMA2_7B, none, batch=1, seq=2048)
	a_full = m.activation_bytes(LLAMA2_7B, full, batch=1, seq=2048)
	assert a_full < a_none / 5


def test_full_finetune_fp32_adam_is_16_bytes_per_param():
	setup = m.TrainingSetup(param_dtype="fp32", compute_dtype="fp32", optimizer="adamw")
	terms = m.static_terms(GPT2_SMALL, setup)
	n = GPT2_SMALL.param_count()
	assert terms["weights"] + terms["gradients"] + terms["optimizer"] == pytest.approx(16 * n)


def test_mixed_precision_with_master_weights_is_16_bytes_per_param():
	setup = m.TrainingSetup(param_dtype="bf16", master_weights=True, optimizer="adamw")
	terms = m.static_terms(GPT2_SMALL, setup)
	total = sum(terms[k] for k in ("weights", "master_weights", "gradients", "optimizer"))
	assert total == pytest.approx(16 * GPT2_SMALL.param_count())


def test_lora_trains_only_adapters():
	setup = m.TrainingSetup(method="lora", lora_rank=16, lora_targets=("q", "v"), param_dtype="bf16")
	trainable = m.trainable_params(LLAMA2_7B, setup)
	assert trainable == 32 * 2 * 16 * (4096 + 4096)
	terms = m.static_terms(LLAMA2_7B, setup)
	assert terms["optimizer"] == pytest.approx(trainable * 8)  # fp32 adapters, two Adam states


def test_qlora_base_weights_are_about_half_a_byte():
	setup = m.TrainingSetup(method="qlora", lora_rank=8, param_dtype="nf4", compute_dtype="bf16")
	terms = m.static_terms(LLAMA2_7B, setup)
	linear = LLAMA2_7B.param_breakdown()["linear"]
	# 4.5 bits per quantised linear weight; embeddings and head stay in bf16 (~4.2 GB total)
	assert terms["weights"] < 0.65 * LLAMA2_7B.param_count()
	assert terms["weights"] > 0.5 * linear


def test_logits_term_scales_with_vocab_and_tokens():
	setup = m.TrainingSetup(compute_dtype="bf16")
	one = m.logits_bytes(LLAMA2_7B, setup, batch=1, seq=1024)
	assert one == pytest.approx(1024 * 32000 * (2 + 8))
	assert m.logits_bytes(LLAMA2_7B, setup, batch=2, seq=1024) == pytest.approx(2 * one)


def test_breakdown_peak_is_static_plus_dynamic():
	setup = m.TrainingSetup(param_dtype="bf16", master_weights=False)
	bd = m.breakdown(GPT2_SMALL, setup, batch=8, seq=1024)
	static = bd.terms["weights"] + bd.terms["optimizer"] + bd.terms.get("master_weights", 0)
	assert bd.allocator_peak > static + bd.terms["activations"]
	assert bd.total == pytest.approx(bd.allocator_peak + bd.terms["context"])


def test_optimizer_state_absent_before_first_step():
	setup = m.TrainingSetup()
	before = m.breakdown(GPT2_SMALL, setup, 1, 128, after_optimizer_step=False)
	assert before.terms["optimizer"] == 0


def test_eval_without_no_grad_costs_more_than_with():
	setup = m.TrainingSetup()
	with_ng = m.eval_breakdown(GPT2_SMALL, setup, eval_batch=64, seq=1024, no_grad=True)
	without = m.eval_breakdown(GPT2_SMALL, setup, eval_batch=64, seq=1024, no_grad=False)
	assert without.allocator_peak > with_ng.allocator_peak
	assert without.terms["activations"] > 10 * with_ng.terms["activations"]


def test_two_probe_calibration_reproduces_measurements():
	cal = m.Calibration.fit(
		[
			m.MemoryProbe(tokens=1024, measured_bytes=3.0e9, analytical_bytes=2.5e9),
			m.MemoryProbe(tokens=4096, measured_bytes=6.0e9, analytical_bytes=5.0e9),
		]
	)
	assert cal.correct(1024, 2.5e9) == pytest.approx(3.0e9)
	assert cal.correct(4096, 5.0e9) == pytest.approx(6.0e9)


def test_single_probe_calibration_is_constant_offset():
	cal = m.Calibration.fit([m.MemoryProbe(tokens=1024, measured_bytes=3.0e9, analytical_bytes=2.5e9)])
	assert cal.correct(8192, 7.0e9) == pytest.approx(7.5e9)


def test_predicted_range_is_the_smallest_to_twice_the_largest_probe():
	cal = m.Calibration.fit([m.MemoryProbe(1024, 3e9, 2.5e9), m.MemoryProbe(4096, 6e9, 5e9)])
	assert cal.range() == (1024, 8192)
	assert cal.in_range(8192) and not cal.in_range(8193) and not cal.in_range(1023)
	assert cal.is_below(512) and cal.halvings_below(512) == pytest.approx(1.0)


def test_band_grows_a_quarter_per_halving_below_the_smallest_probe():
	cal = m.Calibration.fit([m.MemoryProbe(1024, 3e9, 2.5e9), m.MemoryProbe(4096, 6e9, 5e9)])
	assert cal.half_width(1024) == pytest.approx(m.HALF_WIDTH_CALIBRATED)
	assert cal.half_width(256) == pytest.approx(m.HALF_WIDTH_CALIBRATED * 1.5)  # two halvings


def test_band_by_probe_count_follows_s2c():
	one = m.Calibration.fit([m.MemoryProbe(1024, 3e9, 2.5e9)])
	two = m.Calibration.fit([m.MemoryProbe(1024, 3e9, 2.5e9), m.MemoryProbe(4096, 6e9, 5e9)])
	assert one.half_width() == pytest.approx(m.HALF_WIDTH_ONE_PROBE)
	assert two.half_width() == pytest.approx(m.HALF_WIDTH_CALIBRATED)


def test_p_oom_monotone_and_bounded():
	cap = 80 * 2**30
	low = m.p_oom(40 * 2**30, 0.05, cap)
	edge = m.p_oom(78 * 2**30, 0.05, cap)
	high = m.p_oom(120 * 2**30, 0.05, cap)
	assert 0 <= low < 0.01
	assert low < edge < high
	assert high > 0.99


def test_status_words():
	assert m.oom_status(0.001) == "fits"
	assert m.oom_status(0.2) == "tight"
	assert m.oom_status(0.9) == "fails"


def test_generic_arch_uses_params_only_and_relies_on_calibration():
	arch = m.GenericArch(n_params=25_000_000)
	setup = m.TrainingSetup(param_dtype="fp32", compute_dtype="fp32")
	bd = m.breakdown(arch, setup, batch=32, seq=1)
	assert bd.terms["activations"] == 0
	assert bd.terms["weights"] == pytest.approx(4 * 25_000_000)


def test_predict_peak_end_to_end():
	setup = m.TrainingSetup(param_dtype="bf16")
	probes = []
	for b in (2, 8):
		an = m.breakdown(GPT2_SMALL, setup, batch=b, seq=1024).allocator_peak
		probes.append(m.MemoryProbe(tokens=b * 1024, measured_bytes=an * 1.1 + 3e8, analytical_bytes=an))
	est = m.predict_peak(GPT2_SMALL, setup, batch=12, seq=1024, probes=probes, gpu="L4")
	assert est.status == "predicted"
	assert est.lo < est.point < est.hi
	assert 0 <= est.p_oom <= 1
	far = m.predict_peak(GPT2_SMALL, setup, batch=24, seq=1024, probes=probes, gpu="L4")
	assert far.status == "abstain" and "outside the probed range" in far.reason
	low = m.predict_peak(GPT2_SMALL, setup, batch=1, seq=512, probes=probes, gpu="L4")
	assert low.status == "extrapolated" and "run a probe here" in low.reason
	assert math.log(low.hi / low.point) == pytest.approx(m.HALF_WIDTH_CALIBRATED * 1.5)  # 2 halvings below
	assert (low.hi - low.lo) / low.point > (est.hi - est.lo) / est.point


def test_far_below_the_probes_still_abstains():
	setup = m.TrainingSetup(param_dtype="bf16")
	probes = []
	for b in (4, 16):
		an = m.breakdown(GPT2_SMALL, setup, batch=b, seq=1024).allocator_peak
		probes.append(m.MemoryProbe(tokens=b * 1024, measured_bytes=an * 1.1 + 3e8, analytical_bytes=an))
	# the band reaches +-50% after (ln 1.5 / 0.107 - 1) / 0.25, about 11 halvings; this query is 12 below
	far = m.predict_peak(GPT2_SMALL, setup, batch=1, seq=1, probes=probes, gpu="L4")
	assert far.status == "abstain" and "too far below" in far.reason


def test_abstains_when_the_band_passes_fifty_percent():
	setup = m.TrainingSetup(param_dtype="bf16")
	an = m.breakdown(GPT2_SMALL, setup, batch=2, seq=1024).allocator_peak
	probes = [m.MemoryProbe(2048, an * 1.05, an)]
	est = m.predict_peak(GPT2_SMALL, setup, batch=64, seq=1024, probes=probes, gpu="L4")
	assert est.status == "abstain"
	assert est.point is None and "run a probe here" in est.reason
	prior = m.predict_peak(GPT2_SMALL, setup, batch=64, seq=1024, gpu="L4")
	assert prior.status == "predicted" and "not calibrated" in prior.reason


# ---------------------------------------------------------------- S1 / S2 decisions


def _three_probes(arch, setup, bs, s, scale=1.08, offset=3e8):
	out = []
	for b, q in ((bs, s), (bs // 2, s), (bs, s // 2)):
		an = m.breakdown(arch, setup, b, q).allocator_peak
		out.append(m.MemoryProbe(b * q, an * scale + offset, an))
	return out


def test_three_consistent_probes_give_the_calibrated_band():
	setup = m.TrainingSetup()
	cal = m.Calibration.fit(_three_probes(GPT2_SMALL, setup, 8, 1024, scale=1.0))
	assert cal.loo_log_error is not None and cal.loo_log_error < 1e-9
	assert cal.half_width(8 * 1024) == pytest.approx(m.HALF_WIDTH_CALIBRATED)


def test_disagreeing_probes_widen_the_band():
	setup = m.TrainingSetup()
	probes = _three_probes(GPT2_SMALL, setup, 8, 1024)
	probes[2] = m.MemoryProbe(probes[2].tokens, probes[2].measured_bytes * 1.4, probes[2].analytical_bytes)
	cal = m.Calibration.fit(probes)
	assert cal.half_width(8 * 1024) > m.HALF_WIDTH_CALIBRATED


def test_prior_band_without_probes():
	est = m.predict_peak(GPT2_SMALL, m.TrainingSetup(), 4, 512, gpu="L4")
	assert est.status == "predicted"
	assert est.hi / est.point == pytest.approx(__import__("math").exp(m.HALF_WIDTH_PRIOR))


def test_status_cutoffs_follow_the_monte_carlo():
	assert m.oom_status(0.009) == "fits"
	assert m.oom_status(0.011) == "tight"
	assert m.oom_status(0.79) == "tight"
	assert m.oom_status(0.80) == "fails"


def test_measured_point_keeps_a_narrow_band():
	setup = m.TrainingSetup()
	probes = _three_probes(GPT2_SMALL, setup, 8, 1024)
	est = m.predict_peak(GPT2_SMALL, setup, 8, 1024, probes=probes, gpu="A100-SXM-80GB", measured=True)
	assert est.status == "measured"
	assert est.point == pytest.approx(probes[0].measured_bytes + m.DEFAULT_CONTEXT_BYTES, rel=1e-6)


# ---------------------------------------------------------------- distributed


def test_zero_and_fsdp_shard_the_right_terms():
	n = LLAMA2_7B.param_count()
	base = dict(param_dtype="bf16", master_weights=True)
	none = m.static_terms(LLAMA2_7B, m.TrainingSetup(**base))
	z1 = m.static_terms(LLAMA2_7B, m.TrainingSetup(**base, parallel=m.Parallel("zero1", 8)))
	z2 = m.static_terms(LLAMA2_7B, m.TrainingSetup(**base, parallel=m.Parallel("zero2", 8)))
	z3 = m.static_terms(LLAMA2_7B, m.TrainingSetup(**base, parallel=m.Parallel("fsdp", 8)))
	assert z1["optimizer"] == pytest.approx(none["optimizer"] / 8)
	assert z1["gradients"] == pytest.approx(none["gradients"])
	assert z2["gradients"] == pytest.approx(none["gradients"] / 8)
	assert z2["weights"] == pytest.approx(none["weights"])
	assert z3["weights"] == pytest.approx(2 * n / 8)
	assert z3["comm_buffers"] > 0  # gathered layer params


def test_ddp_adds_a_gradient_bucket_copy():
	st = m.static_terms(GPT2_SMALL, m.TrainingSetup(parallel=m.Parallel("ddp", 4)))
	assert st["comm_buffers"] == pytest.approx(st["gradients"])
	view = m.static_terms(GPT2_SMALL, m.TrainingSetup(parallel=m.Parallel("ddp", 4, ddp_bucket_view=True)))
	assert view["comm_buffers"] == 0


def test_tensor_parallel_splits_weights_and_most_activations():
	one = m.TrainingSetup()
	tp = m.TrainingSetup(parallel=m.Parallel(tensor=4))
	assert m.static_terms(LLAMA2_7B, tp)["weights"] == pytest.approx(
		m.static_terms(LLAMA2_7B, one)["weights"] / 4
	)
	a1 = m.activation_bytes_per_layer(LLAMA2_7B, one, 1, 2048)
	a4 = m.activation_bytes_per_layer(LLAMA2_7B, tp, 1, 2048)
	assert a1 / 4 < a4 < a1  # norms and the qkv input are not split


def test_pipeline_parallel_first_stage_keeps_activations():
	one = m.TrainingSetup()
	pp = m.TrainingSetup(parallel=m.Parallel(pipeline=4))
	assert m.activation_bytes(LLAMA2_7B, pp, 1, 1024) == pytest.approx(
		m.activation_bytes(LLAMA2_7B, one, 1, 1024)
	)
	assert m.static_terms(LLAMA2_7B, pp)["weights"] == pytest.approx(
		m.static_terms(LLAMA2_7B, one)["weights"] / 4
	)


def test_optimizer_offload_frees_states_and_master():
	st = m.static_terms(
		LLAMA2_7B,
		m.TrainingSetup(master_weights=True, parallel=m.Parallel("zero2", 2, offload_optimizer=True)),
	)
	assert st["optimizer"] == 0 and st["master_weights"] == 0


# ---------------------------------------------------------------- optimizers and quantisation


@pytest.mark.parametrize(
	"name, bytes_per_param",
	[
		("adamw_torch", 8),
		("adamw_torch_fused", 8),
		("adamw_8bit", 2),
		("paged_adamw_8bit", 2),
		("lion_32bit", 4),
		("sgd", 0),
		("sgd_momentum", 4),
	],
)
def test_optimizer_state_bytes(name, bytes_per_param):
	setup = m.TrainingSetup(param_dtype="fp32", compute_dtype="fp32", optimizer=name)
	st = m.static_terms(GPT2_SMALL, setup)
	assert st["optimizer"] == pytest.approx(bytes_per_param * GPT2_SMALL.param_count())


def test_adafactor_is_factored():
	st = m.static_terms(LLAMA2_7B, m.TrainingSetup(param_dtype="fp32", optimizer="adafactor"))
	assert 0 < st["optimizer"] < 0.01 * LLAMA2_7B.param_count() * 4


def test_unknown_optimizer_falls_back_to_adamw():
	assert m.optimizer_key("shampoo") == ("adamw", False)
	assert m.optimizer_key("adamw_torch_fused") == ("adamw_torch_fused", True)


@pytest.mark.parametrize(
	"q, lo, hi", [("4bit-nf4", 3.5e9, 4.5e9), ("8bit", 6.5e9, 7.5e9), ("gptq", 3.4e9, 4.4e9)]
)
def test_quantised_base_weights(q, lo, hi):
	st = m.static_terms(LLAMA2_7B, m.TrainingSetup(method="lora", lora_rank=8, quantization=q))
	assert lo < st["weights"] < hi


def test_unknown_quantisation_is_an_error():
	with pytest.raises(ValueError):
		m.quant_key("3bit-mystery")


def test_lora_all_linear_targets_every_projection():
	setup = m.TrainingSetup(method="lora", lora_rank=8, lora_targets=("all-linear",))
	per = sum(8 * (a + b) for a, b in LLAMA2_7B.linear_shapes().values())
	assert m.adapter_params(LLAMA2_7B, setup) == 32 * per


# ---------------------------------------------------------------- moments: eval, generation, RL


def test_eval_spike_without_no_grad_becomes_the_peak():
	setup = m.TrainingSetup(eval_batch=64, eval_no_grad=False)
	bd = m.breakdown(GPT2_SMALL, setup, batch=8, seq=1024)
	assert bd.peak_moment == "eval"
	assert bd.terms["eval_spike"] > 0


def test_kv_cache_formula():
	arch = m.TransformerArch(n_layers=28, d_model=1536, n_heads=12, n_kv_heads=2, vocab_size=151936)
	got = m.kv_cache_bytes(arch, sequences=64, tokens_per_seq=1024, kv_dtype="bf16")
	assert got == pytest.approx(2 * 28 * 2 * 128 * 2 * 64 * 1024)


def test_rl_generation_moment_and_reference_model():
	gen = m.Generation(sequences=256, prompt_len=512, new_tokens=1024)
	full = m.TrainingSetup(generation=gen, reference_model=True)
	lora = m.TrainingSetup(generation=gen, reference_model=True, method="lora", lora_rank=16)
	st_full = m.static_terms(LLAMA2_7B, full)
	st_lora = m.static_terms(LLAMA2_7B, lora)
	assert st_full["frozen_models"] == pytest.approx(st_full["weights"])  # a frozen copy
	assert st_lora["frozen_models"] == 0  # adapters off = reference
	bd = m.breakdown(LLAMA2_7B, full, batch=8, seq=1536)
	assert "generate" in bd.moments and bd.terms["kv_cache"] > 0


def test_colocated_vllm_reserves_its_fraction():
	gen = m.Generation(sequences=64, prompt_len=256, new_tokens=512, engine="vllm", colocated_fraction=0.3)
	setup = m.TrainingSetup(method="lora", lora_rank=16, generation=gen)
	cap = 80 * 2**30
	bd = m.breakdown(LLAMA2_7B, setup, batch=4, seq=768, capacity_bytes=cap)
	assert bd.moments["generate"] >= 0.3 * cap


def test_ppo_value_model_adds_weights_and_optimizer():
	st = m.static_terms(GPT2_SMALL, m.TrainingSetup(value_model=True))
	assert st["frozen_models"] > 2 * GPT2_SMALL.param_count()


def test_grad_accum_keeps_grads_alive_in_forward():
	one = m.breakdown(LLAMA2_7B, m.TrainingSetup(), 1, 512)
	acc = m.breakdown(LLAMA2_7B, m.TrainingSetup(grad_accum=8), 1, 512)
	assert acc.moments["train"] > one.moments["train"]  # the peak itself is the optimizer step at batch 1


def test_moe_counts_all_experts_but_activates_k():
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
	dense = m.TransformerArch(n_layers=2, d_model=512, n_heads=8, vocab_size=1000, d_ff=1024, gated_mlp=True)
	assert moe.param_count() > 7 * (dense.param_breakdown()["linear"] // 2)
	assert moe.active_linear_params() < moe.param_breakdown()["linear"]


def test_stack_sums_to_total():
	setup = m.TrainingSetup(eval_batch=16, generation=m.Generation(32, 256, 256))
	bd = m.breakdown(LLAMA2_7B, setup, batch=4, seq=1024)
	assert sum(bd.stack.values()) == pytest.approx(bd.total)


def test_same_work_units_with_other_knobs_is_not_measured():
	setup = m.TrainingSetup()
	probes = _three_probes(GPT2_SMALL, setup, 8, 1024)
	sgd = m.TrainingSetup(optimizer="sgd")
	est = m.predict_peak(GPT2_SMALL, sgd, 8, 1024, probes=probes, gpu="A100-SXM-80GB")
	assert est.status == "predicted"


def test_capacity_defaults_to_usable_memory():
	setup = m.TrainingSetup()
	a = m.predict_peak(GPT2_SMALL, setup, 4, 1024, gpu="L4")
	b = m.predict_peak(GPT2_SMALL, setup, 4, 1024, capacity_bytes=24 * 2**30)
	assert a.p_oom >= b.p_oom


# ---------------------------------------------------------------- Review, Oct 27: memory model refinements


def test_optimizer_step_moment_can_be_the_peak():
	bd = m.breakdown(LLAMA2_7B, m.TrainingSetup(), 1, 512)
	assert bd.peak_moment == "optimizer_step"
	st = m.static_terms(LLAMA2_7B, m.TrainingSetup())
	scratch = st["optimizer"] / 2  # foreach AdamW: one temporary the size of exp_avg_sq
	expected = st["weights"] + st["optimizer"] + st["gradients"] + scratch
	assert bd.moments["optimizer_step"] == pytest.approx(expected)
	fused = m.breakdown(LLAMA2_7B, m.TrainingSetup(optimizer_impl="fused"), 1, 512)
	assert fused.moments["optimizer_step"] == pytest.approx(expected - scratch)
	assert "optimizer_step" not in m.breakdown(LLAMA2_7B, m.TrainingSetup(method="frozen"), 1, 512).moments


def test_lora_frozen_linears_save_fewer_activations():
	full = m.activation_bytes_per_layer(LLAMA2_7B, m.TrainingSetup(), 4, 1024)
	qv = m.activation_bytes_per_layer(
		LLAMA2_7B, m.TrainingSetup(method="lora", lora_rank=16, lora_targets=("q", "v")), 4, 1024
	)
	every = m.activation_bytes_per_layer(
		LLAMA2_7B, m.TrainingSetup(method="lora", lora_rank=16, lora_targets=("all-linear",)), 4, 1024
	)
	assert qv < every and every < full * 1.05  # all-linear saves what full does, plus rank-16 intermediates
	h, ff, sb = LLAMA2_7B.d_model, LLAMA2_7B.ff, 4 * 1024
	# q,v-only skips the MLP input and the down input
	assert full - qv == pytest.approx(2 * (sb * h + sb * ff) - 2 * 2 * sb * 16, rel=1e-6)


def test_flash_attention_output_is_counted_once():
	flash = m.activation_bytes_per_layer(LLAMA2_7B, m.TrainingSetup(flash_attention=True), 1, 1024)
	sbh = 1024 * LLAMA2_7B.d_model
	ffr = LLAMA2_7B.ff / LLAMA2_7B.d_model
	norms, qkv_in, qkv, out_in, mlp_in, mlp = (
		2 * 2 * sbh,
		2 * sbh,
		2 * sbh * 3,
		2 * sbh,
		2 * sbh,
		2 * sbh * 4 * ffr,
	)
	lse = 4 * LLAMA2_7B.n_heads * 1024
	assert flash == pytest.approx(norms + qkv_in + qkv + out_in + lse + mlp_in + mlp)


def test_autocast_keeps_half_precision_weight_copies():
	amp = m.TrainingSetup(param_dtype="fp32", compute_dtype="bf16")
	cache = m.autocast_cache_bytes(LLAMA2_7B, amp)
	assert cache == pytest.approx(LLAMA2_7B.param_breakdown()["linear"] * 2)
	assert m.autocast_cache_bytes(LLAMA2_7B, m.TrainingSetup()) == 0  # bf16 weights: nothing to cast
	assert (
		m.autocast_cache_bytes(
			LLAMA2_7B, m.TrainingSetup(param_dtype="fp32", compute_dtype="bf16", method="lora")
		)
		== 0
	)
	bd = m.breakdown(GPT2_SMALL, amp, 8, 1024)
	assert bd.stack["weights"] > m.static_terms(GPT2_SMALL, amp)["weights"]


def test_fsdp_root_unit_stays_gathered():
	par = m.Parallel(strategy="fsdp", world_size=8)
	st = m.static_terms(LLAMA2_7B, m.TrainingSetup(parallel=par))
	pb = LLAMA2_7B.param_breakdown()
	layer = 2 * LLAMA2_7B.largest_layer_params() * 2
	root = (pb["embedding"] + pb["head"] + LLAMA2_7B.d_model) * 2 * (1 - 1 / 8)
	assert st["comm_buffers"] == pytest.approx(layer + root)


def test_hf_eval_logit_accumulation_grows_the_eval_peak():
	base = m.TrainingSetup(eval_batch=8, eval_no_grad=True)
	acc = m.TrainingSetup(eval_batch=8, eval_no_grad=True, eval_logits_accumulated=500)
	a = m.breakdown(LLAMA2_7B, base, 1, 512).moments["eval"]
	b = m.breakdown(LLAMA2_7B, acc, 1, 512).moments["eval"]
	assert b - a == pytest.approx(500 * 512 * LLAMA2_7B.vocab_size * 2)


def test_below_the_smallest_probe_the_range_still_reaches_that_probe():
	# a steep line through the probes drops fast below them; the top of the range never goes under what the line
	# gives at the smallest probe (memory doesn't shrink faster than the work)
	setup = m.TrainingSetup(param_dtype="bf16")
	probes = []
	for b, extra in ((4, 2e9), (8, 6e9)):  # residual grows steeply: a fixed floor the line can't see
		an = m.breakdown(GPT2_SMALL, setup, batch=b, seq=1024).allocator_peak
		probes.append(m.MemoryProbe(tokens=b * 1024, measured_bytes=an + extra, analytical_bytes=an))
	low = m.predict_peak(GPT2_SMALL, setup, batch=2, seq=1024, probes=probes, gpu="A100-SXM-80GB")
	cal = m.Calibration.fit(probes)
	an_q = m.breakdown(GPT2_SMALL, setup, batch=2, seq=1024).allocator_peak
	floor = cal.correct(cal.min_tokens, an_q) + m.DEFAULT_CONTEXT_BYTES
	assert low.status == "extrapolated" and low.hi >= floor > low.point * math.exp(
		m.HALF_WIDTH_CALIBRATED * 1.25
	)
