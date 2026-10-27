"""The probe recorder, end to end through a subprocess (the path Modal uses), on CPU."""

from pathlib import Path

import pytest

pytest.importorskip("torch")

from preflight.probes.local import ProbeSpec, run_probe

FIX = Path(__file__).resolve().parent / "fixtures"
pytestmark = pytest.mark.torch


def spec(script: str, **kw) -> ProbeSpec:
	files = {script: (FIX / script).read_text()}
	return ProbeSpec(probe_id=kw.pop("probe_id", "p1"), files=files, entrypoint=script, **kw)


@pytest.fixture(scope="module")
def loop_result():
	return run_probe(spec("tiny_loop.py", warmup=2, timed=3, knobs={"batch_size": 4}, timeout_s=120))


def test_plain_loop_probe_is_ok_and_stops_at_target(loop_result):
	p = loop_result.probe
	assert p.status == "ok", p.error
	assert p.steps == 5
	assert p.step_time.n_steps == 3
	assert p.knobs == {"batch_size": 4}


def test_timings_come_from_unprofiled_steps_and_phases_add_up(loop_result):
	st = loop_result.probe.step_time
	assert st.unprofiled
	assert 0 < st.p50_s <= st.p95_s
	assert st.first_step_s >= st.p50_s
	assert set(st.by_phase) >= {"data", "forward", "backward", "optimizer", "other"}
	assert sum(st.by_phase.values()) == pytest.approx(1.0, abs=0.02)


def test_memory_terms_are_exact_for_weights_and_adam_state(loop_result):
	m = loop_result.probe.memory
	n_params = 512 * 64 + (64 * 128 + 128) + (128 * 64 + 64) + 64 * 512
	assert m.by_term["weights"] == n_params * 4
	assert m.by_term["gradients"] == n_params * 4
	# AdamW: exp_avg + exp_avg_sq per param, plus a scalar step per param tensor
	assert m.by_term["optimizer"] >= 2 * n_params * 4
	assert m.by_term["logits"] == 4 * 32 * 512 * 4


def test_stability_series_cover_every_step(loop_result):
	s = loop_result.probe.stability
	for key in ("loss", "grad_norm", "grad_norm_ma", "update_to_weight", "output_logit_norm", "lr"):
		assert len(s.series[key]) == 5, key
	assert all(v is not None and v > 0 for v in s.series["loss"])
	assert s.nan_count == 0


def test_data_and_throughput(loop_result):
	p = loop_result.probe
	assert p.data.num_workers == 0
	assert 0 <= p.data.loader_wait_fraction <= 1
	assert p.data.tokens_per_sample_mean == 32
	assert p.throughput.tokens_per_s > 0
	assert p.throughput.samples_per_s == pytest.approx(p.throughput.tokens_per_s / 32, rel=1e-6)
	assert p.throughput.flops_per_step > 0


def test_env_is_detected(loop_result):
	env = loop_result.env
	assert env.backend == "cpu"
	assert "torch" in env.libraries


def test_profiled_steps_export_a_trace(tmp_path):
	r = run_probe(spec("tiny_loop.py", warmup=2, timed=2, profile=True, out_dir=tmp_path, timeout_s=120))
	assert r.probe.status == "ok"
	assert r.trace_file is not None and r.trace_file.exists()
	assert r.probe.step_time.n_steps == 2  # profiled steps are never timed


def test_hf_trainer_probe_measures_padding(tmp_path):
	r = run_probe(spec("tiny_hf.py", warmup=2, timed=2, out_dir=tmp_path, timeout_s=180))
	p = r.probe
	assert p.status == "ok", p.error
	assert p.steps == 4
	assert 0.05 < p.data.padding_fraction < 0.8
	# grad accumulation 2: two backward passes per optimizer step
	assert p.extras["micro_steps_per_step"] == 2
	assert len(p.stability.series["loss"]) == 4


def test_user_error_names_the_step_and_the_line():
	r = run_probe(spec("crash_at_step.py", warmup=2, timed=3, timeout_s=60))
	p = r.probe
	assert p.status == "error"
	assert "labels and logits have different lengths" in p.error
	assert "crash_at_step.py:11" in p.error
	assert "after step 3" in p.error
	assert p.steps == 3


def test_oom_is_its_own_status():
	r = run_probe(spec("oom.py", warmup=2, timed=3, timeout_s=60))
	assert r.probe.status == "oom"
	assert r.probe.steps == 2


def test_scripts_that_swallow_exceptions_still_stop():
	r = run_probe(spec("swallow.py", warmup=1, timed=2, timeout_s=60))
	assert r.probe.status == "ok"
	assert r.probe.steps == 3


def test_hard_wall_clock_cap_names_the_last_stage():
	r = run_probe(spec("hang.py", warmup=1, timed=1, timeout_s=8))
	p = r.probe
	assert p.status == "timeout"
	assert "before the first optimizer step" in p.error
