"""What each knob acts on in the job map, and whether moving it changes what the model learns.

`affects` lets the guide light the map nodes a knob moves; `changes_results`
tells it which knobs to offer only to clear a failure, never as a saving (C's
request, docs/requests/from-c.md). Both are written on every declared knob at
skeleton time.

AFFECTS lists candidate node ids across every lens's graph, in the order the
guide should light them; a knob keeps the ones its record's graph has. The ids
are the skeleton's own (harness/skeleton.py, one graph per lens).
"""

import re

from preflight.schema import Graph, Knob

_DATA = ["data", "pairs", "images", "sampler", "requests", "prompts", "tasks"]
_MODEL = ["model", "encoder", "backbone", "gnn", "body"]
_BLOCK = ["block", "attn", "mlp", "layer", "body"]

# derived: from what each knob changes in the code the static reader recognises
AFFECTS: dict[str, list[str]] = {
	# memory and batch shape
	"batch_size": [*_DATA, *_MODEL, "generate"],
	"seq_len": ["data", "attn", "lm_head", "body", "pairs", "encoder"],
	"eval_batch_size": ["eval"],
	"eval_no_grad": ["eval"],
	"eval_every": ["eval"],
	"gradient_checkpointing": [*_BLOCK, "backbone", "encoder"],
	"grad_accum": ["data", "optimizer"],
	"precision": [*_MODEL],
	"quantization": ["model", "attn", "mlp"],
	"attn": ["attn"],
	"image_size": ["images", "augment", "backbone"],
	"encode_batch_size": ["corpus"],
	"dump_hidden_states": ["activations"],
	# the model and adapters
	"lora_r": ["attn", "mlp", "optimizer"],
	"lora_alpha": ["attn", "mlp"],
	"lora_dropout": ["attn", "mlp"],
	"n_layer": ["block", "model"],
	"d_model": ["block", "embed", "lm_head"],
	"n_head": ["attn"],
	"hidden_channels": ["layer"],
	"gnn_layers": ["layer", "gnn"],
	"index_m": ["index"],
	"index_nlist": ["index"],
	# optimisation
	"optimizer": ["optimizer"],
	"lr": ["optimizer"],
	"lr_schedule": ["optimizer"],
	"warmup_steps": ["optimizer"],
	"weight_decay": ["optimizer"],
	"grad_clip": ["optimizer"],
	"epochs": [*_DATA],
	"max_steps": [*_DATA, "optimizer"],
	# data pipeline
	"num_workers": [*_DATA],
	"packing": ["data"],
	"group_by_length": ["data"],
	# hardware and parallelism
	"gpu_type": [*_MODEL, "scheduler"],
	"gpus": [*_MODEL, "scheduler"],
	"fsdp_mode": ["model", "optimizer"],
	"tensor_parallel": ["model", "scheduler"],
	"checkpoint_every": ["ckpt"],
	# RL
	"group_size": ["rollout", "reward"],
	"max_prompt_len": ["rollout", "data"],
	"max_completion_len": ["rollout"],
	"use_vllm": ["rollout", "weight_sync"],
	"kl_beta": ["reference", "loss"],
	"clip_low": ["loss"],
	"clip_high": ["loss"],
	"loss_type": ["loss"],
	# generation, serving and eval
	"max_new_tokens": ["rollout", "decode", "generate", "llm"],
	"temperature": ["rollout", "generate", "llm"],
	"max_model_len": ["kv_cache", "prefill"],
	"max_num_seqs": ["scheduler", "kv_cache"],
	"gpu_memory_utilization": ["kv_cache", "rollout"],
	"kv_cache_dtype": ["kv_cache"],
	"prefix_caching": ["kv_cache", "prefill"],
	"concurrency": ["scheduler", "requests", "agent", "llm"],
	# agents
	"n_tasks": ["tasks"],
	"max_turns": ["agent", "context"],
	"reasoning_effort": ["llm"],
	"context_limit": ["context"],
	"tool_retry_cap": ["tools"],
}
_FANOUT = re.compile(r"fanout\.hop\d+")

# cited: C's rule-based tour (web/src/guide/ruleTour.ts CHANGES_RESULTS, Oct 27), the definition
# the guide uses, so the record and the tour agree. Everything else A declares changes only speed,
# memory or cost.
CHANGES_RESULTS = frozenset(
	{
		# what the model sees
		"seq_len",
		"packing",
		"group_by_length",
		"max_prompt_len",
		"max_completion_len",
		"max_new_tokens",
		"context_limit",
		"image_size",
		# what it can learn
		"lora_r",
		"lora_alpha",
		"lora_dropout",
		"quantization",
		"precision",
		"n_layer",
		"d_model",
		"n_head",
		"hidden_channels",
		"gnn_layers",
		"model",
		# how it learns
		"lr",
		"lr_schedule",
		"warmup_steps",
		"weight_decay",
		"optimizer",
		"grad_clip",
		"epochs",
		"max_steps",
		# RL and agents
		"group_size",
		"temperature",
		"reasoning_effort",
		"kl_beta",
		"clip_low",
		"clip_high",
		"loss_type",
		"max_turns",
	}
)
# Knobs whose effect on results depends on what else moves with them (a smaller batch with more
# accumulation learns the same; alone it doesn't). The guide treats null as "check first".
DEPENDS = frozenset({"batch_size", "grad_accum", "gpus", "fsdp_mode", "tensor_parallel"})


def affects(knob_id: str, node_ids: set[str]) -> list[str]:
	cands = AFFECTS.get(knob_id) or (["sampler"] if _FANOUT.fullmatch(knob_id) else [])
	return list(dict.fromkeys(n for n in cands if n in node_ids))


def changes_results(knob_id: str) -> bool | None:
	if knob_id in DEPENDS:
		return None
	return knob_id in CHANGES_RESULTS


def annotate(knobs: list[Knob], graph: Graph) -> list[Knob]:
	"""Each knob with `affects` and `changes_results` filled; values a knob already carries stay."""
	node_ids = {n.id for n in graph.nodes}
	out = []
	for k in knobs:
		update = {}
		if not k.affects:
			update["affects"] = affects(k.id, node_ids)
		if k.changes_results is None:
			update["changes_results"] = changes_results(k.id)
		out.append(k.model_copy(update=update) if update else k)
	return out
