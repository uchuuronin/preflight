"""The static reader across every lens in the plan, and its hardening against hostile input."""

import time

import pytest

from preflight.harness.static import ScriptParseError, map_static
from preflight.recorder.overrides import apply_overrides


def m(src, name="s.py"):
	return map_static({name: src}, entrypoint=name)


def knobs(sm):
	return {k.id: k.current for k in sm.knobs}


# ---------------------------------------------------------------- RL post-training (TRL)

GRPO = """
from trl import GRPOConfig, GRPOTrainer
cfg = GRPOConfig(
	output_dir="o",
	num_generations=8,
	max_completion_length=512,
	max_prompt_length=256,
	per_device_train_batch_size=8,
	gradient_accumulation_steps=2,
	learning_rate=1e-6,
	beta=0.04,
	temperature=0.9,
	epsilon=0.2,
	epsilon_high=0.28,
	loss_type="dr_grpo",
	use_vllm=True,
	vllm_gpu_memory_utilization=0.3,
	num_iterations=1,
	logging_steps=10,
	bf16=True,
)
trainer = GRPOTrainer(model="Qwen/Qwen2.5-0.5B-Instruct", reward_funcs=[correct], args=cfg, train_dataset=ds)
trainer.train()
"""


def test_grpo_knobs_cover_the_rl_lens():
	sm = m(GRPO)
	k = knobs(sm)
	assert k["group_size"] == 8 and k["max_completion_len"] == 512 and k["max_prompt_len"] == 256
	assert k["kl_beta"] == 0.04 and k["temperature"] == 0.9
	assert k["clip_low"] == 0.2 and k["clip_high"] == 0.28
	assert k["loss_type"] == "dr_grpo" and k["use_vllm"] is True
	assert k["gpu_memory_utilization"] == 0.3
	assert sm.rl.algorithm == "grpo" and sm.rl.generation_backend == "vllm"
	assert sm.rl.reward_functions == ["correct"]


# ---------------------------------------------------------------- self-hosted serving and eval sweeps

VLLM = """
from vllm import LLM, SamplingParams
llm = LLM(model="meta-llama/Llama-3.1-8B-Instruct", max_model_len=32768, max_num_seqs=256,
	gpu_memory_utilization=0.92, tensor_parallel_size=2, enable_prefix_caching=True, kv_cache_dtype="fp8")
params = SamplingParams(max_tokens=1024, temperature=0.0)
outs = llm.generate(prompts, params)
"""


def test_vllm_serving_knobs():
	sm = m(VLLM)
	k = knobs(sm)
	assert sm.lens == "serving" and sm.framework == "vllm"
	assert k["max_model_len"] == 32768 and k["max_num_seqs"] == 256
	assert k["gpu_memory_utilization"] == 0.92 and k["tensor_parallel"] == 2
	assert k["prefix_caching"] is True and k["kv_cache_dtype"] == "fp8"
	assert k["max_new_tokens"] == 1024
	assert sm.model.hf_id == "meta-llama/Llama-3.1-8B-Instruct"


EVAL = """
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
MODEL = "Qwen/Qwen3-8B"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)
EVAL_BATCH_SIZE = 16
for batch in batches(EVAL_BATCH_SIZE):
	with torch.no_grad():
		out = model.generate(**batch, max_new_tokens=4096, output_hidden_states=True, return_dict_in_generate=True)
	torch.save(out.hidden_states, f"acts/{i}.pt")
"""


def test_eval_sweep_knobs_and_activation_dump_risk():
	sm = m(EVAL)
	k = knobs(sm)
	assert sm.lens == "eval"
	assert k["max_new_tokens"] == 4096 and k["eval_batch_size"] == 16
	assert k["dump_hidden_states"] is True
	assert any("hidden states" in r for r in sm.risks)


# ---------------------------------------------------------------- agents

AGENT = """
import asyncio
from openai import AsyncOpenAI
client = AsyncOpenAI(base_url="https://api.meta.ai/v1")
MODEL = "muse-spark-1.3"
MAX_TURNS = 40
CONCURRENCY = 8
N_TASKS = 500
TOOL_RETRIES = 1000
async def solve(task):
	for turn in range(MAX_TURNS):
		r = await client.chat.completions.create(model=MODEL, messages=msgs, tools=TOOLS, temperature=0.2,
			max_tokens=2048, reasoning_effort="medium")
		while True:
			try:
				run_tool(r)
				break
			except ToolError:
				continue
"""


def test_agent_knobs_and_runaway_retry_risk():
	sm = m(AGENT)
	k = knobs(sm)
	assert sm.lens == "agent"
	assert k["max_turns"] == 40 and k["concurrency"] == 8 and k["n_tasks"] == 500
	assert k["temperature"] == 0.2 and k["max_new_tokens"] == 2048 and k["reasoning_effort"] == "medium"
	assert k["tool_retry_cap"] == 1000
	assert sm.agent.model == "muse-spark-1.3" and sm.agent.client == "openai"
	assert any("retries" in r for r in sm.risks)


# ---------------------------------------------------------------- retrieval, GNN, vision

RETRIEVAL = """
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader
model = SentenceTransformer("BAAI/bge-small-en-v1.5")
loader = DataLoader(pairs, batch_size=16, shuffle=True)
loss = losses.MultipleNegativesRankingLoss(model)
model.fit(train_objectives=[(loader, loss)], epochs=1, warmup_steps=100)
emb = model.encode(corpus, batch_size=256)
import faiss
index = faiss.IndexHNSWFlat(384, 32)
"""


def test_retrieval_knobs_and_small_contrastive_batch_risk():
	sm = m(RETRIEVAL)
	k = knobs(sm)
	assert sm.lens == "retrieval"
	assert k["batch_size"] == 16 and k["encode_batch_size"] == 256 and k["index_m"] == 32
	assert any("in-batch negatives" in r for r in sm.risks)


GNN = """
import torch
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import GraphSAGE
loader = NeighborLoader(data, num_neighbors=[25, 10, 10], batch_size=1024, num_workers=4)
model = GraphSAGE(in_channels=128, hidden_channels=256, num_layers=3, out_channels=47)
opt = torch.optim.Adam(model.parameters(), lr=3e-3)
for batch in loader:
	loss = model(batch.x, batch.edge_index).sum()
	loss.backward()
	opt.step()
"""


def test_gnn_fanout_becomes_one_knob_per_hop():
	sm = m(GNN)
	k = knobs(sm)
	assert sm.lens == "gnn" and sm.framework == "pyg"
	assert (k["fanout.hop1"], k["fanout.hop2"], k["fanout.hop3"]) == (25, 10, 10)
	assert k["hidden_channels"] == 256 and k["gnn_layers"] == 3 and k["batch_size"] == 1024
	out = apply_overrides({"s.py": GNN}, sm, {"fanout.hop1": 15})
	assert "num_neighbors=[15, 10, 10]" in out["s.py"]


VISION = """
import timm
import torch
from torchvision import transforms
IMAGE_SIZE = 224
model = timm.create_model("resnet50", pretrained=True, num_classes=10)
tf = transforms.Compose([transforms.RandomResizedCrop(IMAGE_SIZE), transforms.ToTensor()])
loader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=2)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
for x, y in loader:
	loss = torch.nn.functional.cross_entropy(model(x), y)
	loss.backward()
	opt.step()
"""


def test_vision_knobs_and_cpu_augmentation_risk():
	sm = m(VISION)
	k = knobs(sm)
	assert sm.lens == "vision"
	assert k["image_size"] == 224 and k["batch_size"] == 128 and k["num_workers"] == 2
	assert sm.model.name == "resnet50"
	assert any("channels_last" in r for r in sm.risks)
	assert any("augmentation" in r for r in sm.risks)


# ---------------------------------------------------------------- multi-GPU and data repetition

FSDP = """
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
args = TrainingArguments(output_dir="o", per_device_train_batch_size=2, fsdp="full_shard auto_wrap",
	gradient_checkpointing=True, bf16=True, num_train_epochs=4, dataloader_num_workers=4)
Trainer(model=model, args=args, train_dataset=ds).train()
"""


def test_fsdp_and_epochs():
	sm = m(FSDP)
	k = knobs(sm)
	assert sm.training.distributed == "fsdp"
	assert k["epochs"] == 4 and k["gradient_checkpointing"] is True
	assert k["fsdp_mode"] == "full_shard auto_wrap"


def test_torchrun_launch_reads_gpu_count():
	sm = map_static(
		{"train.py": "import torch\n", "launch.sh": "torchrun --nproc_per_node=4 train.py --lr 3e-4\n"},
		entrypoint="train.py",
	)
	assert knobs(sm)["gpus"] == 4
	assert sm.training is None or sm.training.distributed in ("ddp", "fsdp", "none")


def test_repeated_epochs_flag_data_repetition():
	sm = m(FSDP)
	assert any("epochs" in r and "repeat" in r for r in sm.risks)


# ---------------------------------------------------------------- hardening


def test_oversized_submission_is_refused_quickly():
	# an entry point over the budget alone: refused, and not called a syntax error (D, Nov 2)
	with pytest.raises(ScriptParseError, match="more than Preflight maps") as exc:
		m("x = 1\n" * 600_000)
	assert getattr(exc.value, "code", None) == "submission.too_large"


def test_deeply_nested_expression_is_a_parse_error_not_a_crash():
	with pytest.raises(ScriptParseError):
		m("X = " + "+".join(["1"] * 20_000) + "\n")


def test_many_calls_stay_fast():
	src = "\n".join(f"f{i}(batch_size={i}, num_workers=2, lr=0.1)" for i in range(3000))
	t0 = time.perf_counter()
	m(src)
	assert time.perf_counter() - t0 < 2.0


def test_override_strings_cannot_inject_code():
	src = 'dtype = "float32"\n'
	sm = m(src)
	out = apply_overrides({"s.py": src}, sm, {"precision": 'bf16" + __import__("os").getenv("HOME") + "'})
	ns = {}
	exec(compile(out["s.py"], "s.py", "exec"), ns)
	assert "__import__" in ns["dtype"]  # it is data, not code


def test_override_keeps_torch_dtype_attributes():
	src = "import torch\nDTYPE = torch.float32\n"
	sm = m(src)
	out = apply_overrides({"s.py": src}, sm, {"precision": "bf16"})
	assert "DTYPE = torch.bfloat16" in out["s.py"]


def test_override_with_form_feed_in_file():
	src = "# section\x0c\nbatch_size = 4\n"
	sm = m(src)
	out = apply_overrides({"s.py": src}, sm, {"batch_size": 8})
	assert "batch_size = 8" in out["s.py"]
