"""The probe recorder, end to end through a subprocess (the path Modal uses), on CPU."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("torch")

from preflight.probes.local import ProbeSpec, run_probe
from preflight.recorder.runtime import PHASE_STEPS

FIX = Path(__file__).resolve().parent / "fixtures"
# scripts that import transformers; CI's torch job installs it with accelerate
needs_transformers = pytest.mark.skipif(
	importlib.util.find_spec("transformers") is None, reason="transformers not installed"
)
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
	assert p.steps == 5 + PHASE_STEPS  # warmup, timed, then synchronised steps for the phase split
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
		assert len(s.series[key]) == 5 + PHASE_STEPS, key
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


@needs_transformers
def test_hf_trainer_probe_measures_padding(tmp_path):
	r = run_probe(spec("tiny_hf.py", warmup=2, timed=2, out_dir=tmp_path, timeout_s=180))
	p = r.probe
	assert p.status == "ok", p.error
	assert p.steps == 4 + PHASE_STEPS
	assert 0.05 < p.data.padding_fraction < 0.8
	# grad accumulation 2: two backward passes per optimizer step
	assert p.extras["micro_steps_per_step"] == 2
	# the real run's length: the Trainer plans args.max_steps optimizer steps; no epoch count then
	assert p.extras["planned_steps"] == 500 and "epochs" not in p.extras
	assert p.extras["train_loader_batches"] > 0  # the Varlen set over per-device batches
	# the training set's rows as the script built them: 8..47 tokens each (B, Oct 30)
	assert 40 <= p.extras["dataset_max_tokens"] <= 47 and p.extras["dataset_rows_total"] == 10_000
	assert 0 < p.extras["dataset_rows_scanned"] <= 10_000
	assert len(p.stability.series["loss"]) == 4 + PHASE_STEPS


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
	assert r.probe.steps == 3 + PHASE_STEPS


def test_hard_wall_clock_cap_names_the_last_stage():
	r = run_probe(spec("hang.py", warmup=1, timed=1, timeout_s=8))
	p = r.probe
	assert p.status == "timeout"
	assert "before the first optimizer step" in p.error


def test_extras_carry_the_counts_b_reads_by_name(loop_result):
	n_params = 512 * 64 + (64 * 128 + 128) + (128 * 64 + 64) + 64 * 512
	x = loop_result.probe.extras
	assert x["params_total"] == n_params
	assert x["params_trainable"] == n_params
	assert x["tokens_per_step"] == 4 * 32
	assert x["train_loader_batches"] == 100_000 // 4  # the training loader's length, one epoch
	assert "planned_steps" not in x and "epochs" not in x  # a plain loop says neither
	assert x["dataset_max_tokens"] == 32 == x["dataset_p999_tokens"]  # fixed blocks of seq_len
	assert 0 < x["dataset_rows_scanned"] <= 100_000 == x["dataset_rows_total"]
	assert x["samples_per_step"] == 4
	assert x["micro_steps_per_step"] == 1


def rows_of(result):
	return [r.model_dump() for r in result.kernels.rows]


@pytest.fixture(scope="module")
def profiled(tmp_path_factory):
	out = tmp_path_factory.mktemp("prof")
	return run_probe(spec("tiny_loop.py", warmup=2, timed=2, profile=True, out_dir=out, timeout_s=180))


def test_profiled_probe_returns_a_kernel_table(profiled):
	from preflight.recorder.runtime import KERNEL_ROWS

	k = profiled.kernels.model_dump()
	assert k["steps"] == 3 and k["backend"] == "cpu"
	assert profiled.probe.kernels == profiled.kernels  # rides on the Probe in the record
	rows = k["rows"]
	assert 0 < len(rows) <= KERNEL_ROWS + 17
	for r in rows:
		assert set(r) >= {"name", "kernel_class", "calls", "device_s", "flops", "bytes", "dtype"}
		assert r["device_s"] > 0 and r["calls"] > 0
	# sorted by time, top rows first, the rolled-up tail last
	top = [r for r in rows if not r["name"].startswith("(tail)")]
	assert [r["device_s"] for r in top] == sorted((r["device_s"] for r in top), reverse=True)


def test_matmuls_carry_flops_and_bytes(profiled):
	mm = [
		r
		for r in rows_of(profiled)
		if r["name"] in ("aten::mm", "aten::addmm", "aten::linear", "aten::matmul")
	]
	assert mm and all(r["flops"] and r["flops"] > 0 for r in mm if r["name"] in ("aten::mm", "aten::addmm"))
	assert all(r["bytes"] and r["bytes"] > 0 for r in mm)


def test_model_shape_rides_with_the_probe(loop_result):
	# script-defined model, no blocks: what can be read and nothing more
	assert loop_result.arch == {
		"d_model": 64,
		"vocab_size": 512,
		"tie_embeddings": False,
		"source": "modules",
	}


@needs_transformers
def test_hf_model_shape_comes_from_its_config(profiled_hf):
	a = profiled_hf.arch
	assert a["source"] == "hf_config" and a["model_type"] == "llama" and a["norm"] == "rmsnorm"
	assert a["gated_mlp"] is True and a["n_layers"] >= 1 and a["head_dim"] >= 1


def test_host_bookkeeping_is_left_to_the_host_bucket(profiled):
	# on CPU every op has self time; autograd nodes and view ops are host work on any device
	names = [r["name"] for r in rows_of(profiled)]
	assert not [
		n for n in names if n.endswith(("Backward0", "Backward1")) or n.startswith("torch::autograd::")
	]
	assert not {"aten::view", "aten::t", "aten::as_strided", "aten::reshape", "aten::item"} & set(names)
	assert any(n in ("aten::mm", "aten::addmm") for n in names)  # real compute stays


def test_kernel_time_is_per_step_and_bounded_by_the_step(profiled):
	per_step = sum(r["device_s"] for r in rows_of(profiled))
	# CPU self time of ops in one profiled step can't exceed a few steps' wall time
	assert per_step < 20 * profiled.probe.step_time.p50_s + 0.05


@pytest.fixture(scope="module")
def profiled_hf(tmp_path_factory):
	out = tmp_path_factory.mktemp("prof_hf")
	return run_probe(spec("tiny_hf.py", warmup=2, timed=2, profile=True, out_dir=out, timeout_s=240))


@needs_transformers
def test_profiler_ranges_are_not_counted_as_kernels(profiled_hf):
	names = [r["name"] for r in rows_of(profiled_hf)]
	assert not any(n.startswith(("ProfilerStep", "Optimizer.", "enumerate(DataLoader)")) for n in names), (
		names[:10]
	)


@needs_transformers
def test_attention_rows_carry_estimated_flops(profiled_hf):
	attn = [r for r in rows_of(profiled_hf) if "scaled_dot_product" in r["name"]]
	assert attn and all(r["flops"] and r["flops"] > 0 for r in attn)


def test_every_step_is_logged_with_its_shape_time_and_kind(loop_result):
	log = loop_result.probe.step_log
	assert [x.kind for x in log] == ["warmup"] * 2 + ["timed"] * 3 + ["phase"] * PHASE_STEPS
	assert [x.index for x in log] == list(range(1, len(log) + 1))
	for x in log:
		assert (
			x.wall_s > 0
			and x.tokens == 4 * 32
			and x.samples == 4
			and x.padded_len == 32
			and x.micro_steps == 1
		)
	# the summary's timed steps are exactly the logged timed ones
	timed = sorted(x.wall_s for x in log if x.kind == "timed")
	assert loop_result.probe.step_time.p50_s == pytest.approx(timed[1])


def test_timed_steps_lose_no_signal_to_deferred_reads(loop_result):
	# losses, grad norms and logit norms are read after the probe on timed steps, not dropped
	s = loop_result.probe.stability.series
	# (the output head is found at the first optimizer step, so its norm starts at step 2)
	assert all(v is not None and v > 0 for v in s["loss"] + s["grad_norm"] + s["output_logit_norm"][1:])


@needs_transformers
@pytest.mark.parametrize("variant", ["ref_model", "peft", "vllm"])
def test_a_grpo_run_splits_rollout_reference_and_reward_and_records_pass_rates(tmp_path, variant):
	"""A CPU stand-in shaped as TRL 0.14 is (tests/fixtures/fake_grpo): generate, the reference
	log-probs in a closure under inference_mode (through ref_model, or the policy with PEFT),
	reward functions, then the update, two accumulation micro-steps fetched up front."""
	root = FIX / "fake_grpo"
	files = {p.relative_to(root).as_posix(): p.read_text() for p in root.rglob("*.py")}
	flag = {"peft": "FAKE_GRPO_PEFT", "vllm": "FAKE_GRPO_VLLM"}.get(variant)
	if flag:
		files["train.py"] = f"import os\nos.environ['{flag}'] = '1'\n" + files["train.py"]
	r = run_probe(
		ProbeSpec(probe_id="p1", files=files, entrypoint="train.py", warmup=2, timed=3, timeout_s=240)
	)
	p = r.probe
	assert p.status == "ok", p.error
	phases = p.step_time.by_phase
	assert phases.get("rollout", 0) > 0 and phases.get("reference", 0) > 0 and phases.get("reward", 0) > 0
	# every micro-step's policy forward counts, not only the first one's (review, Oct 31)
	assert phases.get("forward", 0) > 0.5 * phases["reference"]
	assert (phases.get("weight_sync", 0) > 0) == (variant == "vllm")
	assert sum(phases.values()) == pytest.approx(1.0, abs=0.02)
	rl = p.rl
	assert rl.group_size == 4
	assert rl.pass_rate_per_prompt and all(0 <= x <= 1 for x in rl.pass_rate_per_prompt)
	assert rl.zero_spread_fraction is not None


def test_a_closure_loop_keeps_its_forward_out_of_the_optimizer():
	"""opt.step(closure): the pre-step hook runs before the closure's forward (review, Oct 31)."""
	r = run_probe(ProbeSpec(probe_id="p1", files={"train.py": (FIX / "closure_loop.py").read_text()},
		entrypoint="train.py", warmup=2, timed=3, timeout_s=120))  # fmt: skip
	assert r.probe.status == "ok", r.probe.error
	ph = r.probe.step_time.by_phase
	assert ph["forward"] > ph["optimizer"]


def test_an_image_batch_records_its_size_and_layout():
	r = run_probe(spec("vision_loop.py", warmup=1, timed=2))
	assert r.probe.status == "ok", r.probe.error
	ex = r.probe.extras
	assert ex["images_per_batch"] == 16 and ex["image_height"] == 32 and ex["image_width"] == 32
	assert ex["channels_last"] == 1.0


def test_a_sampled_graph_batch_records_nodes_edges_and_hops():
	r = run_probe(spec("graph_loop.py", warmup=1, timed=2))
	assert r.probe.status == "ok", r.probe.error
	ex = r.probe.extras
	assert ex["graph_seed_nodes"] == 32 and ex["graph_nodes_hop1"] == 320 and ex["graph_nodes_hop2"] == 1600
	assert ex["graph_nodes_per_batch"] == ex["graph_nodes_max"] == 32 + 320 + 1600
	assert ex["graph_edges_per_batch"] == 320 + 1600


def test_retrieval_stages_record_items_and_throughput():
	"""A stand-in with sentence-transformers' and faiss's seams (tests/fixtures/fake_retrieval)."""
	root = FIX / "fake_retrieval"
	files = {p.relative_to(root).as_posix(): p.read_text() for p in root.rglob("*.py")}
	r = run_probe(
		ProbeSpec(probe_id="p1", files=files, entrypoint="train.py", warmup=1, timed=2, timeout_s=120)
	)
	assert r.probe.status == "ok", r.probe.error
	ex = r.probe.extras
	assert ex["encode_items"] == 2100 and ex["encode_items_per_s"] > 0
	assert ex["index_add_items"] == 2000 and ex["search_items"] == 100
	assert "index_train_items" not in ex  # the script never trained the index
