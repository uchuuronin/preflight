"""Recorder signals beyond the basics: eval pass, checkpoints, AMP, attention logits, generation,
trainer logs, divergence and classified failures. CPU, through the real subprocess path."""

import importlib.util
import itertools
from pathlib import Path

import pytest

pytest.importorskip("torch")

from preflight.harness.static import map_static
from preflight.probes.local import ProbeSpec, run_probe
from preflight.recorder.overrides import apply_overrides
from preflight.recorder.runtime import PHASE_STEPS

FIX = Path(__file__).resolve().parent / "fixtures"
# scripts that import transformers; CI's torch job installs it with accelerate
needs_transformers = pytest.mark.skipif(
	importlib.util.find_spec("transformers") is None, reason="transformers not installed"
)
pytestmark = pytest.mark.torch


def probe(script, overrides=None, **kw):
	files = {script: (FIX / script).read_text()}
	if overrides:
		files = apply_overrides(files, map_static(files, entrypoint=script), overrides)
	kw.setdefault("timeout_s", 180)
	return run_probe(ProbeSpec(probe_id="p1", files=files, entrypoint=script, **kw)).probe


def test_eval_is_pulled_in_capped_and_shows_grads_were_kept():
	p = probe("eval_nograd.py", {"eval_every": 3}, warmup=2, timed=3, eval_batches=2)
	assert p.status == "ok", p.error
	assert p.eval is not None
	assert p.eval.batches == 2  # 100k-sample eval loader cut to two batches
	assert p.eval.batch_size == 16
	assert p.eval.grad_enabled is True
	assert p.eval.seconds_per_batch > 0
	assert p.eval.loader_batches == 100_000 // 16  # the full pass, so its time and memory can be extrapolated


def test_checkpoint_saves_are_timed_and_kept_out_of_step_time():
	p = probe("ckpt.py", warmup=2, timed=4)
	assert p.status == "ok", p.error
	assert p.checkpoint.saves >= 1
	assert p.checkpoint.bytes_mean > 30_000_000
	assert p.step_time.p95_s < p.checkpoint.seconds_mean


def test_loss_scale_and_max_attention_logit_series():
	p = probe("amp_attn.py", warmup=1, timed=3)
	assert p.status == "ok", p.error
	s = p.stability.series
	assert len(s["loss_scale"]) == 4 + PHASE_STEPS and s["loss_scale"][0] == 1024.0
	assert all(v == 1024.0 for v in s["loss_scale"])  # timed steps read the scale after the probe
	# the attention check is skipped on timed steps (it would add GPU work to the timing), kept elsewhere
	kinds = [x.kind for x in p.step_log]
	assert all(
		v is not None and v > 0 for v, k in zip(s["max_attn_logit"], kinds, strict=True) if k != "timed"
	)


@needs_transformers
def test_generation_stats_catch_truncated_rollouts():
	p = probe("rollout_hf.py", warmup=1, timed=2)
	assert p.status == "ok", p.error
	g = p.generation
	assert g.calls == 3 + PHASE_STEPS and g.backend == "hf"
	assert g.max_new_tokens == 8
	assert g.new_tokens_max <= 8
	assert g.truncation_rate > 0.5  # a random model rarely emits EOS within 8 tokens
	assert "rollout" in p.step_time.by_phase


@needs_transformers
def test_trainer_logs_become_rl_signals_and_logging_is_forced_every_step():
	p = probe("trl_logs.py", warmup=1, timed=2)
	assert p.status == "ok", p.error
	assert p.rl.truncation_rate == pytest.approx(0.9)
	assert p.rl.zero_spread_fraction == pytest.approx(0.4)
	assert len(p.rl.kl) == 3 + PHASE_STEPS and len(p.rl.entropy) == 3 + PHASE_STEPS


def test_nan_loss_is_counted():
	p = probe("nan_loss.py", warmup=1, timed=2)
	assert p.stability.nan_count >= 1


def test_failures_carry_codes():
	assert probe("missing_pkg.py").error_code == "missing_package"
	assert probe("oom.py").error_code == "oom"
	assert probe("crash_at_step.py").error_code == "user_error"
	assert probe("hang.py", timeout_s=8).error_code == "timeout"


def test_host_memory_is_sampled():
	p = probe("tiny_loop.py", warmup=1, timed=2)
	assert p.extras["host_rss_peak_bytes"] > 50_000_000


def test_a_model_left_in_eval_mode_is_not_an_eval_pass():
	# from_pretrained calls .eval(); a loop that never calls .train() must keep its whole data loader
	p = probe("eval_mode_loop.py", warmup=2, timed=4)
	assert p.status == "ok", p.error
	assert p.steps >= 6 and p.eval is None


@needs_transformers
def test_from_pretrained_eval_mode_does_not_cut_the_training_loader():
	p = probe("eval_mode_hf.py", warmup=2, timed=4)
	assert p.status == "ok", p.error
	assert p.steps >= 6 and p.eval is None


def test_a_validation_pass_before_training_is_cut_like_an_eval():
	from preflight.probes.local import ProbeSpec, run_probe
	from preflight.recorder.runtime import PRETRAIN_EVAL_CAP

	files = {"sanity_eval_first.py": (FIX / "sanity_eval_first.py").read_text()}
	r = run_probe(
		ProbeSpec(
			probe_id="p1", files=files, entrypoint="sanity_eval_first.py", warmup=1, timed=2, timeout_s=120
		)
	)
	assert r.probe.status == "ok", r.probe.error
	assert f"VAL BATCHES {PRETRAIN_EVAL_CAP}" in r.log_tail  # not 12,500
	assert r.probe.steps == 3 + PHASE_STEPS


def test_timed_steps_keep_each_steps_own_loss_scale():
	p = probe("growing_scale.py", warmup=1, timed=3)
	assert p.status == "ok", p.error
	s = p.stability.series["loss_scale"]
	assert all(b > a for a, b in itertools.pairwise(s)), s  # doubling every step, timed ones included
	assert all(v is not None and v > 0 for v in p.stability.series["loss"])


def test_power_is_read_through_amdsmi_on_rocm(monkeypatch):
	import sys
	import types

	import torch

	from preflight.recorder import runtime

	fake = types.ModuleType("amdsmi")
	fake.amdsmi_init = lambda: None
	fake.amdsmi_get_processor_handles = lambda: ["gpu0", "gpu1"]
	fake.amdsmi_get_power_info = lambda h: {
		"current_socket_power": "N/A",
		"average_socket_power": 512 if h == "gpu1" else 0,
	}
	monkeypatch.setitem(sys.modules, "amdsmi", fake)
	monkeypatch.setattr(torch.version, "hip", "6.2.0", raising=False)
	monkeypatch.setenv("HIP_VISIBLE_DEVICES", "1")
	read = runtime.gpu_power_reader()
	assert read is not None and read() == 512.0
	fake.amdsmi_get_power_info = lambda h: (_ for _ in ()).throw(RuntimeError("no smi"))
	monkeypatch.setitem(sys.modules, "amdsmi", fake)
	with pytest.raises(RuntimeError):
		read()  # the sampler suppresses this; the reader itself doesn't hide it
