"""Recorder signals beyond the basics: eval pass, checkpoints, AMP, attention logits, generation,
trainer logs, divergence and classified failures. CPU, through the real subprocess path."""

from pathlib import Path

import pytest

pytest.importorskip("torch")

from preflight.harness.static import map_static
from preflight.probes.local import ProbeSpec, run_probe
from preflight.recorder.overrides import apply_overrides

FIX = Path(__file__).resolve().parent / "fixtures"
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
	assert len(s["loss_scale"]) == 4 and s["loss_scale"][0] == 1024.0
	assert all(v is not None and v > 0 for v in s["max_attn_logit"])


def test_generation_stats_catch_truncated_rollouts():
	p = probe("rollout_hf.py", warmup=1, timed=2)
	assert p.status == "ok", p.error
	g = p.generation
	assert g.calls == 3 and g.backend == "hf"
	assert g.max_new_tokens == 8
	assert g.new_tokens_max <= 8
	assert g.truncation_rate > 0.5  # a random model rarely emits EOS within 8 tokens
	assert "rollout" in p.step_time.by_phase


def test_trainer_logs_become_rl_signals_and_logging_is_forced_every_step():
	p = probe("trl_logs.py", warmup=1, timed=2)
	assert p.status == "ok", p.error
	assert p.rl.truncation_rate == pytest.approx(0.9)
	assert p.rl.zero_spread_fraction == pytest.approx(0.4)
	assert len(p.rl.kl) == 3 and len(p.rl.entropy) == 3


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
