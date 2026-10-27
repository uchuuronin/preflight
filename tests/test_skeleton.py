"""Script map -> skeleton record: job-map graph, declared knobs, read-time findings."""

from pathlib import Path

import pytest

from preflight.harness.skeleton import skeleton_record
from preflight.harness.static import map_static
from preflight.schema import FlightRecord, Source

ROOT = Path(__file__).resolve().parents[1]


def mapped(path):
	return map_static({"train.py": (ROOT / path).read_text()}, entrypoint="train.py")


@pytest.fixture(scope="module")
def lora_rec():
	return skeleton_record(
		mapped("zoo/finetune_lora_hf/train.py"), Source(kind="sample", uri="sample://zoo/lora-hf")
	)


@pytest.fixture(scope="module")
def nano_rec():
	return skeleton_record(mapped("zoo/pretrain_nanogpt/train.py"), Source(kind="sample"))


def test_skeleton_is_a_valid_record(lora_rec, nano_rec):
	for rec in (lora_rec, nano_rec):
		FlightRecord.model_validate_json(rec.model_dump_json())
		assert rec.graph.skeleton is True
		assert rec.job.script_map is not None


def test_repeated_blocks_are_collapsed(lora_rec, nano_rec):
	block = next(n for n in nano_rec.graph.nodes if n.id == "block")
	assert block.repeat == 12
	assert any(n.parent == "block" for n in nano_rec.graph.nodes)


def test_graph_has_the_training_path(nano_rec):
	kinds = {n.kind for n in nano_rec.graph.nodes}
	assert {"data", "loss", "optimizer"} <= kinds
	assert any(e.kind == "grad" for e in nano_rec.graph.edges)


def test_eval_and_checkpoint_nodes_appear_when_the_script_has_them(lora_rec, nano_rec):
	assert "eval" in {n.id for n in lora_rec.graph.nodes}
	assert "ckpt" in {n.id for n in nano_rec.graph.nodes}


def test_precision_knob_is_normalised_and_categorical(nano_rec, lora_rec):
	p = nano_rec.knob("precision")
	assert p.kind == "categorical"
	assert p.current == "fp32"
	assert set(p.domain.values) >= {"fp32", "bf16"}
	assert lora_rec.knob("precision").current == "bf16"


def test_numeric_knobs_get_domains_that_contain_the_current_value(lora_rec):
	bs = lora_rec.knob("batch_size")
	assert bs.kind == "int" and bs.domain.log
	assert bs.domain.min <= bs.current <= bs.domain.max
	assert bs.label_key == "knob.batch_size"
	assert bs.group == "memory"
	lr = lora_rec.knob("lr")
	assert lr.kind == "float" and lr.domain.log


def test_code_fixes_become_needs_code_knobs(lora_rec):
	ng = lora_rec.knob("eval_no_grad")
	assert ng.needs_code and ng.current is False and ng.patch_template == "eval.wrap_no_grad"
	pk = lora_rec.knob("packing")
	assert pk.needs_code and pk.current is False


def test_read_time_findings_have_codes_and_nodes(lora_rec, nano_rec):
	codes = {f.code for f in lora_rec.findings}
	assert "eval.no_grad" in codes
	assert "eval.batch_large" in codes
	assert "data.no_packing" in codes
	ncodes = {f.code for f in nano_rec.findings}
	assert {"data.num_workers_zero", "precision.fp32"} <= ncodes
	for f in lora_rec.findings + nano_rec.findings:
		assert f.source == "rule"
		assert f.node_id in {n.id for n in (lora_rec if f in lora_rec.findings else nano_rec).graph.nodes}


def test_agent_skeleton():
	src = """
from openai import OpenAI
client = OpenAI()
MAX_TURNS = 40
TOOLS = [{"type": "function", "function": {"name": "search_code"}}, {"type": "function", "function": {"name": "run_tests"}}]
def run(task):
	for _ in range(MAX_TURNS):
		client.chat.completions.create(model="muse-spark-1.3", messages=[], tools=TOOLS)
"""
	sm = map_static({"run.py": src}, entrypoint="run.py")
	rec = skeleton_record(sm, Source(kind="upload"))
	ids = {n.id for n in rec.graph.nodes}
	assert {"tasks", "agent", "llm", "context"} <= ids
	assert rec.knob("max_turns").current == 40
	assert rec.env.backend == "api"


# ---------------------------------------------------------------- every lens gets its own map


import importlib.util  # noqa: E402

_LENSES = importlib.util.spec_from_file_location(
	"lens_src", Path(__file__).with_name("test_static_lenses.py")
)
LENS_SRC = importlib.util.module_from_spec(_LENSES)
_LENSES.loader.exec_module(LENS_SRC)


@pytest.mark.parametrize(
	("src", "lens", "must_have"),
	[
		("GRPO", "rl", {"rollout", "reward", "reference", "optimizer"}),
		("VLLM", "serving", {"requests", "scheduler", "kv_cache"}),
		("EVAL", "eval", {"prompts", "generate", "grader", "activations"}),
		("AGENT", "agent", {"tasks", "llm", "context", "tools"}),
		("RETRIEVAL", "retrieval", {"pairs", "encoder", "negatives", "corpus", "index"}),
		("GNN", "gnn", {"graph", "sampler", "gnn", "loss"}),
		("VISION", "vision", {"images", "augment", "backbone", "loss"}),
	],
)
def test_each_lens_has_its_own_job_map(src, lens, must_have):
	sm = map_static({"s.py": getattr(LENS_SRC, src)}, entrypoint="s.py")
	rec = skeleton_record(sm, Source(kind="upload"))
	assert rec.job.lens == lens
	ids = {n.id for n in rec.graph.nodes}
	assert must_have <= ids, ids
	FlightRecord.model_validate_json(rec.model_dump_json())


def test_every_declared_knob_has_a_domain_and_a_group():
	for name in ("GRPO", "VLLM", "EVAL", "AGENT", "RETRIEVAL", "GNN", "VISION", "FSDP"):
		sm = map_static({"s.py": getattr(LENS_SRC, name)}, entrypoint="s.py")
		rec = skeleton_record(sm, Source(kind="upload"))
		for k in rec.knobs:
			assert k.group, (name, k.id)
			if k.kind in ("int", "float"):
				assert k.domain.min is not None and k.domain.max is not None, (name, k.id)
				assert k.domain.min <= k.current <= k.domain.max


def test_memory_share_is_bounded_by_one():
	sm = map_static({"s.py": LENS_SRC.VLLM}, entrypoint="s.py")
	k = skeleton_record(sm, Source(kind="upload")).knob("gpu_memory_utilization")
	assert k.domain.max == 1.0 and k.domain.min > 0
