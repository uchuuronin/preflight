"""Probe planning. Defaults come from the Oct 27 Monte Carlo (docs/decisions/probe-plan.md)."""

from pathlib import Path

from preflight.harness.skeleton import skeleton_record
from preflight.harness.static import map_static
from preflight.probes.plan import TIMED_STEPS, WARMUP_STEPS, plan_probes
from preflight.schema import Source

ROOT = Path(__file__).resolve().parents[1]


def rec(path):
	sm = map_static({"train.py": (ROOT / path).read_text()}, entrypoint="train.py")
	return skeleton_record(sm, Source(kind="sample"))


def test_step_defaults_follow_the_monte_carlo():
	assert (WARMUP_STEPS, TIMED_STEPS) == (2, 5)


def test_memory_lens_gets_three_probes_current_half_batch_half_seq():
	plan = plan_probes(rec("zoo/pretrain_nanogpt/train.py"))
	knobs = [p.knobs for p in plan]
	assert knobs[0] == {"batch_size": 12, "seq_len": 1024}
	assert {"batch_size": 6, "seq_len": 1024} in knobs
	assert {"batch_size": 12, "seq_len": 512} in knobs
	assert len(plan) == 3
	assert plan[0].role == "current"


def test_batch_of_one_varies_sequence_twice():
	r = rec("zoo/pretrain_nanogpt/train.py")
	r.knob("batch_size").current = 1
	knobs = [p.knobs for p in plan_probes(r)]
	assert {"batch_size": 1, "seq_len": 512} in knobs
	assert {"batch_size": 1, "seq_len": 256} in knobs


def test_eval_and_checkpoint_happen_inside_the_current_probe():
	plan = plan_probes(rec("zoo/finetune_lora_hf/train.py"))
	cur = plan[0]
	# EVAL_EVERY is pulled in so the first eval (where the zoo script OOMs) runs during the probe
	assert cur.overrides["eval_every"] == WARMUP_STEPS + 1
	assert cur.eval_batches == 2
	# the measured probes keep the script's own value
	assert all("eval_every" not in p.overrides for p in plan[1:])


def test_checkpoint_probe_saves_once_at_the_last_step():
	plan = plan_probes(rec("zoo/pretrain_nanogpt/train.py"))
	# two steps before the end: lands inside the probe whether the script counts from 0 or 1
	assert plan[0].overrides["checkpoint_every"] == WARMUP_STEPS + TIMED_STEPS - 2


def test_without_seq_len_knob_falls_back_to_batch_points():
	src = "import torch\nbatch_size = 8\nmodel = torch.nn.Linear(4, 4)\nopt = torch.optim.SGD(model.parameters(), 0.1)\nopt.step()\n"
	sm = map_static({"t.py": src}, entrypoint="t.py")
	plan = plan_probes(skeleton_record(sm, Source(kind="upload")))
	assert [p.knobs for p in plan] == [{"batch_size": 8}, {"batch_size": 4}, {"batch_size": 2}]


def test_agent_lens_plans_one_sample_run():
	src = "from openai import OpenAI\nc = OpenAI()\nMAX_TURNS = 40\nc.chat.completions.create(model='muse-spark-1.3', messages=[])\n"
	sm = map_static({"r.py": src}, entrypoint="r.py")
	plan = plan_probes(skeleton_record(sm, Source(kind="upload")))
	assert len(plan) == 1 and plan[0].role == "agent_sample"
	assert plan[0].agent_tasks >= 30
