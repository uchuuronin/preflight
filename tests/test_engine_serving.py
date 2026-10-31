"""Serving model (engine/serving.py): KV cache, rooflines, the continuous-batching simulation, and its targets."""

from pathlib import Path

import numpy as np
import pytest

from preflight.engine import api, gpus, serving
from preflight.engine.memory import TransformerArch
from preflight.schema import FlightRecord, load_record
from preflight.schema.record import GenerationStats

EX = Path(__file__).resolve().parents[1] / "schema" / "examples"
LLAMA8B = TransformerArch(
	n_layers=32, d_model=4096, n_heads=32, vocab_size=128256, d_ff=14336, n_kv_heads=8, gated_mlp=True,
	tie_embeddings=False,
)  # fmt: skip


def _setup(gpu="A100-SXM-80GB", **kw) -> serving.ServeSetup:
	return serving.ServeSetup(arch=LLAMA8B, gpu=gpu, weight_bytes=LLAMA8B.param_count() * 2, **kw)


def test_kv_bytes_per_token_is_the_textbook_formula():
	# Llama-3-8B: 2 x 32 layers x 8 KV heads x 128 dims x 2 bytes = 128 KiB per token
	assert serving.kv_bytes_per_token(LLAMA8B) == 2 * 32 * 8 * 128 * 2
	assert serving.kv_bytes_per_token(LLAMA8B, "fp8") == 2 * 32 * 8 * 128
	assert serving.kv_bytes_per_token(LLAMA8B, tensor=2) == 2 * 32 * 8 * 128


def test_kv_capacity_is_what_is_left_after_the_weights():
	s = _setup()
	cap = gpus.get_gpu("A100-SXM-80GB").usable_capacity_gib() * 2**30
	expected = (cap * 0.9 - s.weight_bytes - serving.WORKSPACE_BYTES) / serving.kv_bytes_per_token(LLAMA8B)
	assert serving.kv_capacity_tokens(s) == int(expected)
	assert serving.kv_capacity_tokens(_setup(mem_util=0.5)) < serving.kv_capacity_tokens(s)
	assert (
		serving.kv_capacity_tokens(_setup("L4")) < serving.kv_capacity_tokens(s) / 10
	)  # 24 GB, 16 GB of weights


def test_decode_is_memory_bound_and_prefill_compute_bound():
	s = _setup()
	spec = gpus.get_gpu(s.gpu)
	one = serving.decode_seconds(s, 1, 512)
	# one token: read every weight once, at the guessed share of the bandwidth
	assert one == pytest.approx(s.weight_bytes / (spec.bandwidth_bytes() * s.eff_bw) + s.overhead_s, rel=0.02)
	assert serving.decode_seconds(s, 32, 512) < 2 * one  # batching decode is nearly free while memory-bound
	pre = serving.prefill_seconds(s, 4096)
	flops = 2 * LLAMA8B.active_linear_params() * 4096
	assert pre >= flops / (
		spec.peak_flops("bf16") * s.eff_flops
	)  # at least the dense matmuls at the peak share
	assert serving.prefill_seconds(_setup("H100-SXM-80GB"), 4096) < pre / 2  # about 3x the FLOPs


def test_one_request_is_a_prefill_then_its_decodes():
	s = _setup()
	r = serving.simulate(s, [serving.Request(0.0, 1000, 100)])
	assert r.ttft[0] == pytest.approx(serving.iteration_seconds(s, 1000, 500, 0, 0))
	assert r.latency[0] == pytest.approx(r.ttft[0] + 99 * serving.decode_seconds(s, 1, 1050), rel=0.05)
	assert r.output_tokens == 100 and r.peak_running == 1


def test_batching_raises_throughput_and_the_kv_cache_caps_concurrency():
	reqs = serving.workload(256, 512, 256, seed=1)
	solo = serving.simulate(_setup(max_num_seqs=1), reqs)
	many = serving.simulate(_setup(max_num_seqs=256), reqs)
	assert many.throughput > 10 * solo.throughput
	tight = serving.simulate(_setup("L4"), reqs)
	cap = tight.kv_capacity_tokens
	needs = sorted(r.prompt + r.output for r in reqs)
	assert sum(needs[: tight.peak_running]) <= cap  # never more admitted at once than the cache holds
	assert tight.peak_running < many.peak_running


def test_light_traffic_waits_for_nothing():
	s = _setup()
	light = serving.forecast(s, serving.workload(200, 512, 128, rate_per_s=0.5, seed=2))
	heavy = serving.forecast(s, serving.workload(200, 512, 128, rate_per_s=200.0, seed=2))
	assert light.ttft_p95.point < 0.5 and heavy.ttft_p95.point > light.ttft_p95.point
	assert light.tpot_p50.point < heavy.tpot_p50.point


def test_requests_longer_than_the_cache_are_reported_not_hung():
	s = _setup("L4")
	r = serving.simulate(s, [serving.Request(0.0, 10**6, 10)])
	assert r.rejected == 1 and r.output_tokens == 0 and "never run" in r.notes[0]


def test_one_probe_calibrates_the_scale():
	s = _setup()
	reqs = serving.workload(64, 512, 200, seed=3)
	sim = serving.simulate(s, reqs).seconds
	cal = serving.calibrate(s, reqs, 1.5 * sim)
	assert cal.scale == pytest.approx(1.5) and cal.sigma == serving.SIGMA_CALIBRATED
	f = serving.forecast(s, reqs, cal)
	assert f.throughput.point == pytest.approx(serving.simulate(s, reqs).throughput / 1.5)
	assert f.throughput.lo < f.throughput.point < f.throughput.hi


# ---------------------------------------------------------------- on the serving lens


def _serve_record(backend="vllm") -> FlightRecord:
	base = load_record((EX / "finetune.json").read_text())
	probes = [
		p.model_copy(
			update={
				"generation": GenerationStats(
					calls=1,
					seconds_total=12.0,
					sequences=64,
					new_tokens_mean=180,
					new_tokens_p95=400,
					new_tokens_max=512,
					max_new_tokens=512,
					backend=backend,
				)
			}
		)
		for p in base.probes
	]
	return base.model_copy(update={"job": base.job.model_copy(update={"lens": "serving"}), "probes": probes})


def test_serving_targets_stay_off_until_the_flag(monkeypatch):
	monkeypatch.setitem(api.FEATURES, "serving", False)
	assert "serving.throughput" not in api.predict(_serve_record()).root


def test_serving_lens_reports_calibrated_throughput_latency_and_kv(monkeypatch):
	monkeypatch.setitem(api.FEATURES, "serving", True)
	rec = _serve_record()
	pred = api.predict(rec).root
	assert set(api.SERVING_TARGETS) <= set(pred) and set(api.SERVING_TARGETS) <= {
		t.id for t in api.targets_for(rec)
	}
	tp = pred["serving.throughput"]
	assert tp.status != "abstain" and tp.lo < tp.point < tp.hi and "calibrated on the probe" in tp.reason
	# calibrated on this workload: about the probe's own tokens over its own seconds
	assert tp.point == pytest.approx(64 * 180 / 12.0, rel=0.35)
	assert pred["serving.kv_tokens"].point > 0
	cards = {m.id: m for m in api.fill_predictions(rec).models}
	assert "not yet checked against a full vLLM run" in cards[serving.MODEL_ID].notes


def test_an_hf_probe_doesnt_calibrate_a_vllm_simulation(monkeypatch):
	monkeypatch.setitem(api.FEATURES, "serving", True)
	tp = api.predict(_serve_record("hf")).root["serving.throughput"]
	assert "uncalibrated" in tp.reason and tp.status == "extrapolated"
	assert np.log(tp.hi / tp.point) > 0.4  # the uncalibrated band is wide
