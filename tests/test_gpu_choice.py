"""Which GPU a probe runs on, and the gpu_type knob (B's request). Policy from scripts/mc_kernels_gpu.py (G)."""

from pathlib import Path

import pytest

from preflight.harness.skeleton import skeleton_record
from preflight.harness.static import map_static
from preflight.probes.gpu_choice import (
	LADDER,
	MARGIN,
	choose_probe_gpu,
	estimate_need_gb,
	next_gpu,
	params_from_name,
)
from preflight.schema import Constraints, Source

ROOT = Path(__file__).resolve().parents[1]


def sm_of(path):
	return map_static({"train.py": (ROOT / path).read_text()}, entrypoint="train.py")


def test_policy_constants_follow_the_simulation():
	assert MARGIN == 1.3
	assert LADDER == ["L4", "L40S", "A100-SXM-80GB", "H200-SXM-141GB"]


@pytest.mark.parametrize(
	("name", "params"),
	[
		("Qwen/Qwen2.5-1.5B", 1.5e9),
		("meta-llama/Llama-3.1-8B-Instruct", 8e9),
		("gpt2-125M", 125e6),
		("mistralai/Mixtral-8x7B-v0.1", 46.7e9),
		("bert-base-uncased", None),
	],
)
def test_params_from_model_name(name, params):
	got = params_from_name(name)
	assert got == pytest.approx(params, rel=0.05) if params else got is None


def test_small_lora_fits_the_cheapest_gpu():
	sm = sm_of("zoo/finetune_lora_hf/train.py")  # Qwen2.5-1.5B, LoRA
	need = estimate_need_gb(sm)
	assert 4 < need < 15
	assert choose_probe_gpu(need) == "L4"


def test_full_finetune_of_an_8b_model_needs_a_big_gpu():
	src = 'from transformers import AutoModelForCausalLM, Trainer\nm = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")\nTrainer(model=m).train()\n'
	need = estimate_need_gb(map_static({"t.py": src}, entrypoint="t.py"))
	assert need > 100
	# nothing on the ladder holds it: the largest single GPU; the probe reports the OOM
	# and multi-GPU probes come later
	assert choose_probe_gpu(need) == "H200-SXM-141GB"
	# an 80 GB GPU is the cheapest that holds a 50 GB need with the margin
	assert choose_probe_gpu(50.0) == "A100-SXM-80GB"


def test_unknown_size_starts_cheap():
	src = "import torch\nm = torch.nn.Linear(4, 4)\nopt = torch.optim.SGD(m.parameters(), 0.1)\nopt.step()\n"
	assert choose_probe_gpu(estimate_need_gb(map_static({"t.py": src}, entrypoint="t.py"))) == "L4"


def test_visitor_allowed_list_is_respected():
	assert choose_probe_gpu(30.0, allowed=["A100-SXM-80GB", "H100-SXM-80GB"]) == "A100-SXM-80GB"
	# nothing allowed is big enough: the largest allowed, and the probe says so if it OOMs
	assert choose_probe_gpu(500.0, allowed=["L4", "L40S"]) == "L40S"


def test_escalation_ladder():
	assert next_gpu("L4") == "L40S"
	assert next_gpu("A100-SXM-80GB") == "H200-SXM-141GB"
	assert next_gpu("H200-SXM-141GB") is None
	assert next_gpu("L4", allowed=["L4", "A100-SXM-80GB"]) == "A100-SXM-80GB"


def test_gpu_type_knob_is_declared_with_the_chosen_gpu():
	rec = skeleton_record(sm_of("zoo/finetune_lora_hf/train.py"), Source(kind="sample"))
	k = rec.knob("gpu_type")
	assert k.kind == "categorical" and k.current == "L4"
	assert {"L4", "A100-SXM-80GB", "H100-SXM-80GB"} <= set(k.domain.values)


def test_gpu_type_follows_the_visitors_request_and_constraints():
	sm = sm_of("zoo/finetune_lora_hf/train.py")
	rec = skeleton_record(sm, Source(kind="sample"), gpu="H100-SXM-80GB")
	assert rec.knob("gpu_type").current == "H100-SXM-80GB"
	rec = skeleton_record(
		sm, Source(kind="sample"), constraints=Constraints(gpu_types=["A100-SXM-80GB", "L40S"])
	)
	k = rec.knob("gpu_type")
	assert set(k.domain.values) == {"A100-SXM-80GB", "L40S"} and k.current == "L40S"


def test_agents_have_no_gpu_type():
	src = "from openai import OpenAI\nc = OpenAI()\nc.chat.completions.create(model='muse-spark-1.3', messages=[])\n"
	rec = skeleton_record(map_static({"r.py": src}, entrypoint="r.py"), Source(kind="upload"))
	with pytest.raises(KeyError):
		rec.knob("gpu_type")


class _R:
	def __init__(self, status, code=None):
		self.probe = type("P", (), {"status": status, "error_code": code})()


def test_one_step_up_after_a_gpu_oom_only():
	from preflight.probes.gpu_choice import probe_on_ladder

	seen = []

	def run(gpu):
		seen.append(gpu)
		return _R("oom", "oom") if gpu == "L4" else _R("ok")

	res, tried, err = probe_on_ladder(run, "L4")
	assert tried == ["L4", "L40S"] and res.probe.status == "ok" and err is None
	# host memory: a bigger GPU won't help
	assert probe_on_ladder(lambda g: _R("oom", "oom.host"), "L4").tried == ["L4"]
	# an eval OOM is about the eval batch: no bigger GPU; the pipeline re-runs with eval off instead
	out = probe_on_ladder(lambda g: _R("oom", "oom.eval") if g == "L4" else _R("ok"), "L4")
	assert out.tried == ["L4"]
	# still OOM after one step: stop there, the record says so
	assert probe_on_ladder(lambda g: _R("oom", "oom"), "L4").tried == ["L4", "L40S"]
	assert probe_on_ladder(lambda g: _R("oom", "oom"), "H200-SXM-141GB").tried == ["H200-SXM-141GB"]


def test_step_up_respects_the_caller_and_survives_a_failed_start():
	from preflight.probes.gpu_choice import probe_on_ladder

	oom = _R("oom", "oom")
	out = probe_on_ladder(lambda g: oom, "L4", may_step_up=lambda g: False)
	assert out.tried == ["L4"] and out.result is oom

	def run(gpu):
		if gpu == "L40S":
			raise RuntimeError("no L40S capacity")
		return oom

	out = probe_on_ladder(run, "L4")
	assert out.result is oom and out.tried == ["L4", "L40S"] and "not available" in out.step_up_error
	assert "capacity" not in out.step_up_error  # the dispatcher's text stays in the server log


def test_ties_on_memory_go_to_the_cheaper_gpu():
	# A100-80GB and H100 have the same usable memory; the cheaper one wins, in any order given
	assert choose_probe_gpu(40.0, allowed=["H100-SXM-80GB", "A100-SXM-80GB"]) == "A100-SXM-80GB"
	assert next_gpu("L40S", allowed=["H100-SXM-80GB", "A100-SXM-80GB", "L40S"]) == "A100-SXM-80GB"
	assert next_gpu("A100-SXM-80GB", allowed=["A100-SXM-80GB", "H100-SXM-80GB"]) is None  # H100 holds no more


def test_unknown_or_unlisted_gpus_never_step_down():
	assert next_gpu("Mystery Accelerator") is None
	assert next_gpu("Mystery Accelerator", allowed=["Mystery Accelerator", "L4"]) is None
	assert next_gpu("L4", allowed=["L4"]) is None


def test_adversarial_names_are_cheap():
	import time

	from preflight.probes.gpu_choice import canonical

	t = time.perf_counter()
	assert params_from_name("org/" + "1" * 2_000_000) is None
	assert params_from_name("x" * 100_000 + "-7B") is None  # past the 128-char cut
	assert len(canonical("H" * 100_000)) <= 64
	assert time.perf_counter() - t < 0.5


def test_unknown_size_walks_the_ladder():
	from preflight.probes.gpu_choice import max_steps_up, probe_on_ladder

	assert max_steps_up(12.0) == 1 and max_steps_up(0.0) == len(LADDER) - 1

	def run(gpu):  # a script-built model that needs an 80 GB GPU
		return _R("ok") if gpu in ("A100-SXM-80GB", "H200-SXM-141GB") else _R("oom", "oom")

	assert probe_on_ladder(run, "L4", steps=1).tried == ["L4", "L40S"]
	out = probe_on_ladder(run, "L4", steps=max_steps_up(0.0))
	assert out.tried == ["L4", "L40S", "A100-SXM-80GB"] and out.result.probe.status == "ok"
	# a budget check can stop the walk part-way
	out = probe_on_ladder(run, "L4", steps=3, may_step_up=lambda g: g != "A100-SXM-80GB")
	assert out.tried == ["L4", "L40S"] and out.result.probe.status == "oom"


def test_nothing_fits_takes_the_cheaper_of_near_equal_largest():
	assert choose_probe_gpu(500.0, allowed=["H100-SXM-80GB", "A100-SXM-80GB", "L4"]) == "A100-SXM-80GB"
