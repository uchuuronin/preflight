import math

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.torch

from preflight.corpus import configs as c  # noqa: E402
from preflight.corpus import worker as w  # noqa: E402

TINY = {
	"gpt": {
		"n_layers": 2,
		"d_model": 64,
		"n_heads": 4,
		"n_kv_heads": 2,
		"d_ff": 192,
		"gated_mlp": True,
		"vocab_size": 512,
		"seq": 32,
		"batch": 2,
		"dtype": "fp32",
		"attention": "sdpa",
		"grad_checkpointing": "full",
		"causal": True,
	},
	"encoder": {
		"n_layers": 2,
		"d_model": 64,
		"n_heads": 4,
		"n_kv_heads": 4,
		"d_ff": 256,
		"gated_mlp": False,
		"vocab_size": 512,
		"seq": 32,
		"batch": 2,
		"dtype": "bf16",
		"attention": "math",
		"grad_checkpointing": "none",
		"causal": False,
	},
	"lora": {
		"n_layers": 2,
		"d_model": 64,
		"n_heads": 4,
		"n_kv_heads": 4,
		"d_ff": 256,
		"gated_mlp": False,
		"vocab_size": 512,
		"seq": 32,
		"batch": 2,
		"dtype": "bf16",
		"attention": "sdpa",
		"grad_checkpointing": "none",
		"causal": True,
		"lora_rank": 4,
		"lora_targets": ["q", "v"],
	},
	"cnn": {
		"depth": 10,
		"width": 8,
		"image": 32,
		"batch": 4,
		"channels_last": True,
		"dtype": "fp32",
		"n_classes": 10,
	},
	"mlp": {"depth": 3, "width": 64, "batch": 16, "in_features": 32, "dtype": "fp32"},
	"gnn": {
		"conv": "sage",
		"nodes": 300,
		"avg_degree": 5,
		"hidden": 32,
		"layers": 2,
		"in_features": 16,
		"n_classes": 5,
		"dtype": "fp32",
	},
}


@pytest.mark.parametrize("family", sorted(TINY))
def test_kernel_run_produces_complete_record(family):
	cfg = c.CorpusConfig("kernels", family, TINY[family])
	rec = w.run_config(cfg, device="cpu")
	assert rec["status"] == "ok", rec.get("error")
	assert rec["n_params"] > 0
	assert rec["n_trainable"] > 0
	assert rec["timing"]["step_s"] > 0
	assert set(rec["timing"]["phases"]) == {"forward", "backward", "optimizer"}
	assert rec["timing"]["n_steps"] >= 3
	assert rec["optimizer_steps_before_measure"] >= 2
	assert rec["ops"], "profiler op table is empty"
	assert all({"name", "count", "device_s", "flops", "bytes"} <= set(op) for op in rec["ops"])
	assert rec["graph"]["nodes"] and rec["graph"]["edges"]
	assert rec["gpu_seconds"] > 0
	assert rec["work_units"] > 0


def test_lora_trains_only_adapters():
	rec = w.run_config(c.CorpusConfig("kernels", "lora", TINY["lora"]), device="cpu")
	assert rec["n_trainable"] == 2 * 2 * 4 * (64 + 64)


def test_param_count_matches_engine_formula_for_gpt():
	from preflight.engine import memory as m

	p = TINY["gpt"]
	arch = m.TransformerArch(
		n_layers=p["n_layers"],
		d_model=p["d_model"],
		n_heads=p["n_heads"],
		n_kv_heads=p["n_kv_heads"],
		vocab_size=p["vocab_size"],
		d_ff=p["d_ff"],
		gated_mlp=True,
		tie_embeddings=True,
		bias=False,
		norm="rmsnorm",
	)
	rec = w.run_config(c.CorpusConfig("kernels", "gpt", p), device="cpu")
	assert rec["n_params"] == arch.param_count()


def test_oom_errors_are_classified_on_cuda_and_rocm():
	assert w.classify_exception(RuntimeError("CUDA out of memory. Tried to allocate 2 GiB")) == "oom"
	assert w.classify_exception(RuntimeError("HIP out of memory. Tried to allocate 2 GiB")) == "oom"
	assert w.classify_exception(ValueError("bad shape")) == "error"


SMALL_DIV = {
	"n_layers": 2,
	"d_model": 64,
	"n_heads": 4,
	"seq": 32,
	"batch": 8,
	"steps": 60,
	"warmup": 0,
	"init_std": 0.02,
	"precision": "fp32",
	"qk_norm": False,
	"z_loss": 0.0,
	"beta2": 0.95,
	"weight_decay": 0.1,
	"grad_clip": None,
	"seed": 1,
}


def test_divergence_run_logs_stability_signals():
	cfg = c.CorpusConfig("divergence", "gpt-small", {**SMALL_DIV, "lr": 1e-3})
	rec = w.run_config(cfg, device="cpu")
	assert rec["status"] == "ok", rec.get("error")
	s = rec["series"]
	for k in ("loss", "grad_norm", "update_ratio", "max_attn_logit", "output_logit_norm"):
		assert len(s[k]) == 60, k
	assert len(s["update_ratio"][0]) == 2  # one per layer
	assert rec["label"]["diverged"] is False
	assert s["loss"][-1] < s["loss"][0]


def test_huge_learning_rate_is_labelled_diverged():
	cfg = c.CorpusConfig("divergence", "gpt-small", {**SMALL_DIV, "lr": 3.0, "init_std": 0.2})
	rec = w.run_config(cfg, device="cpu")
	assert rec["label"]["diverged"] is True
	assert rec["label"]["diverged_step"] is not None


def test_label_rules():
	flat = [5.0] * 100
	assert w.label_divergence(flat)["diverged"] is False
	nan = [5.0] * 50 + [math.nan] * 50
	lab = w.label_divergence(nan)
	assert lab["diverged"] is True and lab["diverged_step"] == 50
	down_then_up = [5.0 - 0.03 * i for i in range(100)] + [2.0 + 0.1 * i for i in range(100)]
	assert w.label_divergence(down_then_up)["diverged"] is True


def test_label_ignores_recovered_spikes_and_plateaus():
	base = [5.0 * (i + 1) ** -0.3 + 1.0 for i in range(400)]
	spike = list(base)
	for i in range(150, 180):
		spike[i] *= 3.0
	assert w.label_divergence(spike)["diverged"] is False
	plateau = [6.0 + 0.05 * ((i * 7919) % 13 - 6) / 6 for i in range(400)]
	assert w.label_divergence(plateau)["diverged"] is False


def test_label_catches_slow_drift_and_gives_onset():
	drift = [3.0 * (i + 1) ** -0.2 + 1.0 for i in range(200)] + [1.6 + 0.01 * k for k in range(200)]
	lab = w.label_divergence(drift)
	assert lab["diverged"] is True and lab["rule"] == "D9"
	assert 200 <= lab["diverged_step"] <= 320


def test_early_collapse_to_the_baseline_is_diverged():
	collapse = [6.9, 40.0, 270.0, 120.0, 30.0] + [5.55 + 0.001 * (i % 3) for i in range(200)]
	lab = w.label_divergence(collapse, vocab_size=256)
	assert lab["diverged"] is True and lab["diverged_step"] == 2
	assert w.label_divergence(collapse)["diverged"] is False  # without the vocab, 5.55 < 0.9 x 6.9


def test_classified_failures_are_recorded_with_the_shared_code(monkeypatch):
	def oom(*_a, **_k):
		raise RuntimeError("HIP out of memory. Tried to allocate 2.00 GiB")

	monkeypatch.setattr(w, "_run_kernels", oom)
	rec = w.run_config(c.CorpusConfig("kernels", "mlp", TINY["mlp"]), device="cpu")
	assert (rec["status"], rec["error_code"]) == ("oom", "oom")
	assert rec["gpu_seconds"] >= 0


def test_bugs_in_corpus_code_are_raised_not_recorded():
	bad = c.CorpusConfig("kernels", "gnn", {**TINY["gnn"], "in_features": -1})
	with pytest.raises(RuntimeError):
		w.run_config(bad, device="cpu")


def test_op_rows_carry_a_kernel_class():
	rec = w.run_config(c.CorpusConfig("kernels", "mlp", TINY["mlp"]), device="cpu")
	assert any(op["class"] == "gemm" for op in rec["ops"])


# ---------------------------------------------------------------- isolation


def test_isolated_run_returns_the_workers_row():
	from preflight.corpus.isolate import run_isolated

	rec = run_isolated(c.CorpusConfig("kernels", "mlp", TINY["mlp"]), device="cpu", timeout_s=300)
	assert rec["status"] == "ok" and rec["isolated"] and rec["gpu_seconds"] > 0


def test_hung_config_becomes_a_timeout_row():
	from preflight.corpus.isolate import run_isolated

	big = {**TINY["gnn"], "nodes": 3_000_000, "avg_degree": 50, "hidden": 512, "layers": 4}
	rec = run_isolated(c.CorpusConfig("kernels", "gnn", big), device="cpu", timeout_s=3)
	assert (rec["status"], rec["error_code"]) == ("timeout", "timeout")
	assert rec["gpu_seconds"] >= 3


def test_corpus_bug_in_the_child_is_recorded_as_user_error():
	from preflight.corpus.isolate import run_isolated

	bad = c.CorpusConfig("kernels", "gnn", {**TINY["gnn"], "in_features": -1})
	rec = run_isolated(bad, device="cpu", timeout_s=300)
	assert (rec["status"], rec["error_code"]) == ("error", "user_error")
