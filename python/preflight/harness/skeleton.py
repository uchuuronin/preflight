"""Script map -> skeleton record: the map a visitor sees within 60 s, before any probe.

Builds, for every lens, the job-map graph (hierarchy collapsed, repeated
blocks shown once with ×N, metric references for what animates), the declared
knobs with domains and groups, and the findings that can be read off the code
alone. Probes later fill measured numbers in; B fills probed ranges.
"""

import itertools

from preflight.probes.gpu_choice import canonical, choose_probe_gpu, estimate_need_gb, offered_gpus
from preflight.schema import (
	Constraints,
	Domain,
	Env,
	Evidence,
	Finding,
	FlightRecord,
	Graph,
	GraphEdge,
	GraphNode,
	Job,
	Knob,
	Source,
)
from preflight.schema.script_map import ScriptKnob, ScriptMap

from .knob_meta import annotate
from .static import PRECISION, PREQUANTISED, QUANT_FORMATS

TRAINING_LENSES = ("finetune", "pretrain", "rl", "vision", "gnn", "retrieval")
TEXT_LENSES = ("finetune", "pretrain", "rl")  # the lenses whose data is token sequences and documents
NO_PADDING = ("", "none", "false", "no", "do_not_pad")  # how a map says a batch isn't padded

GROUPS = {
	**dict.fromkeys(
		[
			"batch_size",
			"eval_batch_size",
			"seq_len",
			"gradient_checkpointing",
			"eval_no_grad",
			"encode_batch_size",
			"image_size",
			"max_model_len",
			"max_num_seqs",
			"max_batched_tokens",
			"gpu_memory_utilization",
			"kv_cache_dtype",
			"dump_hidden_states",
			"dump_attentions",
			"quantization",
		],
		"memory",
	),
	**dict.fromkeys(
		["num_workers", "packing", "pin_memory", "prefetch_factor", "epochs", "group_by_length"], "data"
	),
	**dict.fromkeys(
		[
			"lr",
			"warmup_steps",
			"grad_clip",
			"n_layer",
			"d_model",
			"n_head",
			"lora_r",
			"lora_alpha",
			"lora_dropout",
			"weight_decay",
			"optimizer",
			"lr_schedule",
			"hidden_channels",
			"gnn_layers",
			"index_m",
			"index_nlist",
		],
		"model",
	),
	**dict.fromkeys(
		[
			"grad_accum",
			"precision",
			"compile",
			"tf32",
			"tensor_parallel",
			"pipeline_parallel",
			"prefix_caching",
			"fsdp_mode",
			"deepspeed_config",
			"gpus",
			"attn",
		],
		"speed",
	),
	**dict.fromkeys(["max_steps", "checkpoint_every", "eval_every", "log_every", "gpu_type"], "cluster"),
	**dict.fromkeys(
		[
			"max_turns",
			"concurrency",
			"model",
			"prompt_caching",
			"n_tasks",
			"tool_retry_cap",
			"reasoning_effort",
			"context_limit",
		],
		"agent",
	),
	**dict.fromkeys(
		[
			"group_size",
			"max_completion_len",
			"max_prompt_len",
			"kl_beta",
			"clip_low",
			"clip_high",
			"loss_type",
			"use_vllm",
			"rl_iterations",
		],
		"rl",
	),
	**dict.fromkeys(["max_new_tokens", "temperature"], "generation"),
}

# id: (min, floor of max, log, multiplier of current for max)
INT_DOMAINS: dict[str, tuple[float, float, bool, float]] = {
	"batch_size": (1, 1024, True, 16),
	"eval_batch_size": (1, 512, True, 8),
	"encode_batch_size": (1, 4096, True, 8),
	"seq_len": (64, 32_768, True, 8),
	"num_workers": (0, 32, False, 1),
	"prefetch_factor": (1, 16, False, 1),
	"grad_accum": (1, 256, True, 1),
	"max_steps": (1, 1_000_000, True, 10),
	"checkpoint_every": (1, 1_000_000, True, 10),
	"eval_every": (1, 100_000, True, 10),
	"log_every": (1, 10_000, True, 10),
	"warmup_steps": (0, 100_000, False, 10),
	"n_layer": (1, 128, False, 4),
	"d_model": (64, 16_384, True, 4),
	"n_head": (1, 256, False, 4),
	"max_turns": (1, 200, False, 4),
	"concurrency": (1, 256, True, 1),
	"n_tasks": (1, 100_000, True, 10),
	"tool_retry_cap": (0, 1_000, False, 1),
	"group_size": (1, 64, True, 4),
	"max_completion_len": (16, 32_768, True, 16),
	"max_prompt_len": (16, 32_768, True, 16),
	"max_new_tokens": (1, 131_072, True, 16),
	"max_model_len": (256, 1_048_576, True, 8),
	"max_num_seqs": (1, 4_096, True, 8),
	"max_batched_tokens": (256, 1_048_576, True, 8),
	"tensor_parallel": (1, 8, True, 1),
	"pipeline_parallel": (1, 8, True, 1),
	"rl_iterations": (1, 8, False, 1),
	"lora_r": (1, 512, True, 16),
	"lora_alpha": (1, 1024, True, 16),
	"context_limit": (1_024, 2_097_152, True, 4),
	"index_m": (4, 128, True, 4),
	"index_nlist": (16, 1_048_576, True, 16),
	"hidden_channels": (8, 4_096, True, 4),
	"gnn_layers": (1, 16, False, 2),
	"image_size": (32, 1_024, True, 4),
	"gpus": (1, 64, True, 8),
}
# id: (min, max, log) for floats with a natural range
FLOAT_DOMAINS: dict[str, tuple[float, float, bool]] = {
	"gpu_memory_utilization": (0.05, 1.0, False),
	"temperature": (0.0, 2.0, False),
	"kl_beta": (0.0, 1.0, False),
	"clip_low": (0.0, 1.0, False),
	"clip_high": (0.0, 1.0, False),
	"weight_decay": (0.0, 1.0, False),
	"epochs": (0.1, 100.0, True),
	"lora_dropout": (0.0, 0.5, False),
	"lora_alpha": (1.0, 1024.0, True),
}
CATEGORIES: dict[str, list] = {
	"precision": ["fp32", "bf16", "fp16"],
	"loss_type": ["grpo", "bnpo", "dr_grpo", "dapo"],
	"reasoning_effort": ["minimal", "low", "medium", "high", "xhigh"],
	"kv_cache_dtype": ["auto", "fp8", "fp8_e4m3", "fp8_e5m2"],
	"optimizer": ["adamw_torch", "adamw_torch_fused", "adamw_8bit", "adafactor", "sgd", "lion_32bit"],
	"lr_schedule": ["linear", "cosine", "constant", "constant_with_warmup", "cosine_with_restarts"],
	"attn": ["eager", "sdpa", "flash_attention_2", "flash_attention_3", "flex_attention"],
	"quantization": list(QUANT_FORMATS),
}
AGENT_MODELS = ["muse-spark-1.3", "muse-spark-open", "muse-glimmer"]


def _group(kid: str) -> str:
	if kid in GROUPS:
		return GROUPS[kid]
	if kid.startswith("fanout."):
		return "memory"
	return "model"


def _domain(sk: ScriptKnob) -> Domain:
	v = sk.current
	if sk.kind == "int":
		if sk.id in INT_DOMAINS:
			lo, hi, log, mult = INT_DOMAINS[sk.id]
			return Domain(min=min(lo, v), max=max(hi, v * mult, v), log=log, step=1)
		if sk.id.startswith("fanout."):
			return Domain(min=1, max=max(100, 4 * v), step=1)
		return Domain(min=min(0, v), max=max(10 * v, 1), step=1)
	if sk.kind == "float" and isinstance(v, int | float):
		if sk.id in FLOAT_DOMAINS:
			lo, hi, log = FLOAT_DOMAINS[sk.id]
			return Domain(min=min(lo, v), max=max(hi, v), log=log)
		if sk.id in ("lr", "grad_clip") and v > 0:
			return Domain(min=v / 100, max=v * 100, log=True)
		return Domain(min=min(0.0, v), max=max(10 * v, 1.0))
	if sk.kind == "categorical":
		values = list(CATEGORIES.get(sk.id, []))
		if v not in values:
			values.append(v)
		return Domain(values=values)
	return Domain()


def _knob(sk: ScriptKnob) -> Knob:
	current = sk.current
	if sk.id == "quantization":
		# read from a BitsAndBytesConfig: same knob, same patch as the one declared when the code has none
		return _choice_knob("quantization", current, list(QUANT_FORMATS), "model.quantize_weights")
	if sk.id == "precision" and isinstance(current, str):
		current = PRECISION.get(current.lower(), current)
		order = CATEGORIES["precision"]
		values = sorted(set(order) | {current}, key=lambda p: order.index(p) if p in order else 9)
		return Knob(
			id="precision",
			label_key="knob.precision",
			kind="categorical",
			domain=Domain(values=values),
			current=current,
			group="speed",
		)
	return Knob(
		id=sk.id,
		label_key=f"knob.{sk.id}",
		unit=sk.unit,
		kind=sk.kind,
		domain=_domain(sk),
		current=current,
		needs_code=sk.needs_code,
		group=_group(sk.id),
	)


def _code_knob(kid: str, current: bool, template: str, *, needs_code: bool = True) -> Knob:
	return Knob(
		id=kid,
		label_key=f"knob.{kid}",
		kind="bool",
		current=current,
		needs_code=needs_code,
		patch_template=template,
		group=_group(kid),
	)


def _choice_knob(kid: str, current: str, values: list[str], template: str) -> Knob:
	return Knob(
		id=kid,
		label_key=f"knob.{kid}",
		kind="categorical",
		domain=Domain(values=values if current in values else [current, *values]),
		current=current,
		needs_code=True,
		patch_template=template,
		group=_group(kid),
	)


def declared_knobs(sm: ScriptMap) -> list[Knob]:
	knobs = [_knob(sk) for sk in sm.knobs if sk.current is not None]
	have = {k.id for k in knobs}
	tr, data = sm.training, sm.data
	if sm.lens in TRAINING_LENSES:
		if tr is not None and tr.eval_under_no_grad is False and "eval_no_grad" not in have:
			knobs.append(_code_knob("eval_no_grad", False, "eval.wrap_no_grad"))
		if data is not None and data.packing is False and "packing" not in have:
			knobs.append(_code_knob("packing", False, "data.enable_packing"))
		if "gradient_checkpointing" not in have and sm.lens in ("finetune", "pretrain", "rl"):
			knobs.append(
				_code_knob(
					"gradient_checkpointing",
					False,
					"memory.gradient_checkpointing",
					needs_code=sm.framework not in ("hf_trainer", "trl"),
				)
			)
		m = sm.model
		hf_decoder = m is not None and m.hf_id is not None and m.family == "decoder"
		prequantised = m is not None and m.quantization in PREQUANTISED  # GPTQ/AWQ: fixed by the checkpoint
		if (
			hf_decoder
			and m.lora is not None
			and "quantization" not in have
			and sm.lens == "finetune"
			and not prequantised
		):
			# QLoRA: the largest memory lever a LoRA fine-tune has; B models it, a patch adds the config
			knobs.append(_choice_knob("quantization", m.quantization or "none", list(QUANT_FORMATS),
				"model.quantize_weights"))  # fmt: skip
		if hf_decoder and "attn" not in have:
			# transformers loads with SDPA unless told otherwise
			knobs.append(_choice_knob("attn", "sdpa", CATEGORIES["attn"], "model.attn_implementation"))
		if sm.lens == "vision" and "channels_last" not in have:
			knobs.append(_code_knob("channels_last", False, "vision.channels_last"))
	if sm.lens == "agent":
		model = sm.agent.model if sm.agent and sm.agent.model else AGENT_MODELS[0]
		if "model" not in have:
			knobs.append(
				Knob(
					id="model",
					label_key="knob.model",
					kind="categorical",
					domain=Domain(values=AGENT_MODELS if model in AGENT_MODELS else [model, *AGENT_MODELS]),
					current=model,
					group="agent",
				)
			)
		if "prompt_caching" not in have:
			knobs.append(_code_knob("prompt_caching", False, "agent.enable_prompt_caching"))
		if "concurrency" not in have:
			knobs.append(
				Knob(
					id="concurrency",
					label_key="knob.concurrency",
					unit="requests",
					kind="int",
					domain=Domain(min=1, max=256, log=True, step=1),
					current=1,
					group="agent",
				)
			)
	return sorted(knobs, key=lambda k: (k.group or "z", k.id))


# ---------------------------------------------------------------- graphs, one per lens


def _N(id, label, kind="module", **kw):
	return GraphNode(id=id, label=label, kind=kind, **kw)


def _E(a, b, kind="data", metric=None):
	return GraphEdge(source=a, target=b, kind=kind, metric=metric)


def _knob_values(sm: ScriptMap) -> dict:
	return {k.id: k.current for k in sm.knobs}


def _model_chain(sm: ScriptMap, nodes: list, edges: list) -> list[str]:
	m = sm.model
	lora = " + LoRA" if m and m.lora else ""
	nodes.append(_N("model", m.name if m else "Model", "group", params=m.param_count if m else None))
	if m and m.family in ("decoder", "encoder", "encoder_decoder"):
		kind = "Decoder block" if m.family == "decoder" else "Encoder block"
		nodes += [
			_N("embed", "Embedding", parent="model"),
			_N("block", kind, "group", parent="model", repeat=max(1, m.n_layers or 1)),
			_N("attn", f"Self-attention{lora}", parent="block", metrics=["probe.stability.max_attn_logit"]),
			_N("mlp", f"MLP{lora}", parent="block"),
			_N(
				"lm_head",
				"LM head" if m.family == "decoder" else "Pooler",
				parent="model",
				metrics=["memory.by_term.logits"],
			),
		]
		chain = ["embed", "attn", "mlp", "lm_head"]
	else:
		nodes.append(_N("body", (m.family if m else "network").upper(), parent="model"))
		chain = ["body"]
	edges += [_E(a, b) for a, b in itertools.pairwise(chain)]
	return chain


def _training_tail(sm: ScriptMap, nodes, edges, first, last) -> None:
	tr, m = sm.training, sm.model
	knob = _knob_values(sm)
	nodes.append(_N("loss", "Loss", "loss", metrics=["probe.stability.loss"]))
	opt_label = (tr.optimizer if tr and tr.optimizer else "Optimizer") + (
		" (LoRA params)" if m and m.lora else ""
	)
	nodes.append(_N("optimizer", opt_label, "optimizer", metrics=["probe.stability.grad_norm"]))
	edges += [_E(last, "loss"), _E("loss", "optimizer", "grad"), _E("optimizer", first, "weights")]
	if "eval_batch_size" in knob or "eval_every" in knob or (tr and tr.eval_under_no_grad is False):
		label = "Eval loop" + (f" · batch {knob['eval_batch_size']}" if "eval_batch_size" in knob else "")
		nodes.append(_N("eval", label, "phase", metrics=["risk.oom", "probe.eval.peak_allocated_bytes"]))
		edges.append(_E("data", "eval", "control"))
	if "checkpoint_every" in knob:
		nodes.append(
			_N(
				"ckpt",
				f"Checkpoint every {knob['checkpoint_every']:,} steps",
				"io",
				metrics=["risk.preemption_loss", "probe.checkpoint.seconds_mean"],
			)
		)
		edges.append(_E("optimizer", "ckpt", "control"))


def _training_graph(sm: ScriptMap) -> Graph:
	d = sm.data
	nodes: list[GraphNode] = []
	edges: list[GraphEdge] = []
	dlabel = (d.dataset if d and d.dataset else "Training data").split("/")[-1]
	if d and d.num_workers is not None:
		dlabel += f" · num_workers={d.num_workers}"
	if d and d.packing is False:
		dlabel += f" · padded ({d.padding or 'per batch'})"
	nodes.append(
		_N("data", dlabel, "data", metrics=["probe.data.loader_wait_fraction", "probe.data.padding_fraction"])
	)
	chain = _model_chain(sm, nodes, edges)
	edges.append(_E("data", chain[0], "tokens", "probe.throughput.tokens_per_s"))
	if sm.lens == "rl":
		rl = sm.rl
		g = f" · G={rl.group_size}" if rl and rl.group_size else ""
		cap = f" · cap {rl.max_completion_len}" if rl and rl.max_completion_len else ""
		backend = f" ({rl.generation_backend})" if rl and rl.generation_backend else ""
		rewards = f" ({', '.join(rl.reward_functions)})" if rl and rl.reward_functions else ""
		nodes += [
			_N(
				"rollout",
				f"Rollouts{backend}{g}{cap}",
				"phase",
				metrics=["probe.rl.truncation_rate", "probe.generation.new_tokens_p95"],
			),
			_N("reward", f"Reward{rewards}", "phase", metrics=["probe.rl.zero_spread_fraction"]),
			_N("reference", "Reference model", "phase", metrics=["probe.rl.kl"]),
			_N("weight_sync", "Weight sync", "phase"),
		]
		edges += [
			_E("data", "rollout", "control"),
			_E("rollout", "reward", "tokens"),
			_E("rollout", "reference", "tokens"),
			_E("reward", "loss"),
			_E("optimizer", "weight_sync", "weights"),
			_E("weight_sync", "rollout", "weights"),
		]
	_training_tail(sm, nodes, edges, chain[0], chain[-1])
	return Graph(nodes=nodes, edges=edges, skeleton=True)


def _retrieval_graph(sm: ScriptMap) -> Graph:
	knob = _knob_values(sm)
	m = sm.model
	bs = knob.get("batch_size")
	nodes = [
		_N(
			"pairs",
			"Query-passage pairs" + (f" · batch {bs}" if bs else ""),
			"data",
			metrics=["probe.data.loader_wait_fraction"],
		),
		_N("encoder", m.name if m else "Encoder", metrics=["probe.throughput.samples_per_s"]),
		_N(
			"negatives",
			f"In-batch negatives ({bs - 1} per query)" if isinstance(bs, int) else "In-batch negatives",
			"phase",
		),
		_N("loss", "Contrastive loss", "loss", metrics=["probe.stability.loss"]),
		_N("optimizer", "Optimizer", "optimizer"),
		_N(
			"corpus",
			"Corpus encoding"
			+ (f" · batch {knob['encode_batch_size']}" if "encode_batch_size" in knob else ""),
			"phase",
			metrics=["time.corpus_encode"],
		),
		_N(
			"index",
			"Vector index" + (f" · M={knob['index_m']}" if "index_m" in knob else ""),
			"store",
			metrics=["time.index_build", "memory.index"],
		),
	]
	edges = [
		_E("pairs", "encoder", "tokens"),
		_E("encoder", "negatives"),
		_E("negatives", "loss"),
		_E("loss", "optimizer", "grad"),
		_E("optimizer", "encoder", "weights"),
		_E("encoder", "corpus", "control"),
		_E("corpus", "index"),
	]
	return Graph(nodes=nodes, edges=edges, skeleton=True)


def _gnn_graph(sm: ScriptMap) -> Graph:
	knob = _knob_values(sm)
	hops = sorted((k for k in knob if k.startswith("fanout.hop")), key=lambda k: int(k.rsplit("hop", 1)[1]))
	fan = " × ".join(str(knob[h]) for h in hops)
	layers = knob.get("gnn_layers") or len(hops) or 1
	nodes = [
		_N("graph", "Graph store", "store"),
		_N(
			"sampler",
			f"Neighbour sampler (CPU){' · ' + fan if fan else ''}",
			"phase",
			metrics=["probe.data.loader_wait_fraction"],
		),
		_N("gnn", sm.model.name if sm.model else "GNN", "group"),
		_N("layer", "Message-passing layer", parent="gnn", repeat=max(1, int(layers))),
		_N("loss", "Loss", "loss", metrics=["probe.stability.loss"]),
		_N("optimizer", "Optimizer", "optimizer"),
	]
	edges = [
		_E("graph", "sampler"),
		_E("sampler", "layer", "data", "probe.throughput.samples_per_s"),
		_E("layer", "loss"),
		_E("loss", "optimizer", "grad"),
		_E("optimizer", "layer", "weights"),
	]
	return Graph(nodes=nodes, edges=edges, skeleton=True)


def _vision_graph(sm: ScriptMap) -> Graph:
	knob = _knob_values(sm)
	nw = knob.get("num_workers")
	size = knob.get("image_size")
	nodes = [
		_N("images", "Images" + (f" · {size}px" if size else ""), "data"),
		_N(
			"augment",
			f"Decode + augment (CPU{f', {nw} workers' if nw is not None else ''})",
			"phase",
			metrics=["probe.data.loader_wait_fraction"],
		),
		_N("backbone", sm.model.name if sm.model else "Backbone", metrics=["probe.throughput.samples_per_s"]),
		_N("head", "Classifier head"),
		_N("loss", "Loss", "loss", metrics=["probe.stability.loss"]),
		_N("optimizer", "Optimizer", "optimizer"),
	]
	edges = [
		_E("images", "augment"),
		_E("augment", "backbone"),
		_E("backbone", "head"),
		_E("head", "loss"),
		_E("loss", "optimizer", "grad"),
		_E("optimizer", "backbone", "weights"),
	]
	if "checkpoint_every" in knob:
		nodes.append(_N("ckpt", f"Checkpoint every {knob['checkpoint_every']:,} steps", "io"))
		edges.append(_E("optimizer", "ckpt", "control"))
	return Graph(nodes=nodes, edges=edges, skeleton=True)


def _agent_graph(sm: ScriptMap) -> Graph:
	a = sm.agent
	knob = _knob_values(sm)
	model = a.model if a and a.model else "LLM"
	n = knob.get("n_tasks")
	conc = knob.get("concurrency")
	nodes = [
		_N("tasks", f"{n:,} tasks" if isinstance(n, int) else "Tasks", "data"),
		_N(
			"agent",
			"Agent loop" + (f" · {conc} at once" if conc else ""),
			"group",
			metrics=["agent.turns_per_task"],
		),
		_N("llm", model, "model_call", parent="agent", metrics=["agent.tokens_per_turn", "cost.total"]),
		_N("context", "Context window", "store", parent="agent", metrics=["agent.context_fill"]),
		_N(
			"tools",
			"Tools" + (f" ({len(a.tools)})" if a and a.tools else ""),
			"tool",
			parent="agent",
			metrics=["agent.tool_errors"],
		),
		_N("grader", "Grader", "phase"),
	]
	edges = [
		_E("tasks", "llm", "control"),
		_E("llm", "tools", "control"),
		_E("tools", "context", "tokens"),
		_E("context", "llm", "tokens", "agent.tokens_per_turn"),
		_E("llm", "grader", "control"),
	]
	return Graph(nodes=nodes, edges=edges, skeleton=True)


def _serving_graph(sm: ScriptMap) -> Graph:
	knob = _knob_values(sm)
	m = sm.model
	tp = knob.get("tensor_parallel")
	nodes = [
		_N("requests", "Requests", "data"),
		_N(
			"scheduler",
			"Scheduler" + (f" · {knob['max_num_seqs']} in flight" if "max_num_seqs" in knob else ""),
			"queue",
		),
		_N(
			"model",
			(m.name if m else "Model") + (f" · TP={tp}" if tp else ""),
			metrics=["serving.tokens_per_s"],
		),
		_N("prefill", "Prefill", "phase", parent=None),
		_N("decode", "Decode", "phase"),
		_N(
			"kv_cache",
			"KV cache" + (f" ({knob['kv_cache_dtype']})" if "kv_cache_dtype" in knob else ""),
			"store",
			metrics=["memory.kv_cache"],
		),
		_N("outputs", "Outputs", "io"),
	]
	edges = [
		_E("requests", "scheduler", "control"),
		_E("scheduler", "prefill", "tokens"),
		_E("prefill", "model"),
		_E("model", "decode"),
		_E("decode", "kv_cache"),
		_E("kv_cache", "decode"),
		_E("decode", "outputs", "tokens"),
	]
	return Graph(nodes=nodes, edges=edges, skeleton=True)


def _eval_graph(sm: ScriptMap) -> Graph:
	knob = _knob_values(sm)
	m = sm.model
	cap = knob.get("max_new_tokens")
	nodes = [
		_N(
			"prompts",
			"Prompts" + (f" · batch {knob['eval_batch_size']}" if "eval_batch_size" in knob else ""),
			"data",
		),
		_N(
			"generate",
			f"Generate ({m.name if m else 'model'})"
			+ (f" · cap {cap:,} tokens" if isinstance(cap, int) else ""),
			"phase",
			metrics=["probe.generation.truncation_rate", "probe.generation.new_tokens_p95"],
		),
		_N("grader", "Grader", "phase"),
		_N("results", "Results", "io"),
	]
	edges = [_E("prompts", "generate", "tokens"), _E("generate", "grader", "tokens"), _E("grader", "results")]
	if knob.get("dump_hidden_states") or knob.get("dump_attentions"):
		nodes.append(_N("activations", "Activation dumps", "store", metrics=["storage.activations"]))
		edges.append(_E("generate", "activations"))
	else:
		nodes.append(_N("activations", "Activations (not stored)", "store"))
	return Graph(nodes=nodes, edges=edges, skeleton=True)


def skeleton_graph(sm: ScriptMap) -> Graph:
	return {
		"agent": _agent_graph,
		"serving": _serving_graph,
		"eval": _eval_graph,
		"retrieval": _retrieval_graph,
		"gnn": _gnn_graph,
		"vision": _vision_graph,
	}.get(sm.lens, _training_graph)(sm)


# ---------------------------------------------------------------- findings from the code alone


def read_time_findings(sm: ScriptMap, graph: Graph) -> list[Finding]:
	nodes = {n.id for n in graph.nodes}
	out: list[Finding] = []

	def add(code, severity, title, node, knob=None, fix=None, evidence=None, detail=None):
		out.append(
			Finding(
				id=f"read.{code}",
				severity=severity,
				code=code,
				title=title,
				detail=detail,
				node_id=node if node in nodes else None,
				evidence=evidence or [],
				source="rule",
				fix_template=fix,
				knob_id=knob,
			)
		)

	def risk(*needles):
		return next((r for r in sm.risks if all(n in r for n in needles)), None)

	tr, d = sm.training, sm.data
	if sm.lens in TRAINING_LENSES:
		if tr is not None and tr.eval_under_no_grad is False:
			add(
				"eval.no_grad",
				"error",
				"Eval runs with gradients kept",
				"eval",
				"eval_no_grad",
				"eval.wrap_no_grad",
				[Evidence(metric="script.eval_under_no_grad", value="false")],
				detail=risk("no_grad"),
			)
		if d and d.batch_size and d.eval_batch_size and d.eval_batch_size >= 4 * d.batch_size:
			add(
				"eval.batch_large",
				"warn",
				f"Eval batch is {d.eval_batch_size // d.batch_size}x the train batch",
				"eval",
				"eval_batch_size",
				"eval.reduce_batch",
				[Evidence(metric="eval_batch_size", value=d.eval_batch_size, threshold=4 * d.batch_size)],
			)
		# B, Nov 2 (airbench, CIFAR-10): packing is about token sequences, and only a padded batch wastes anything
		if (
			d
			and d.packing is False
			and sm.lens in TEXT_LENSES
			and str(d.padding or "").lower() not in NO_PADDING
		):
			add(
				"data.no_packing",
				"warn",
				"No sequence packing; batches are padded",
				"data",
				"packing",
				"data.enable_packing",
			)
		if d and d.num_workers == 0:
			add(
				"data.num_workers_zero",
				"warn",
				"Data loader runs in the training process (num_workers=0)",
				"data" if "data" in nodes else "sampler",
				"num_workers",
				"data.num_workers",
				[Evidence(metric="num_workers", value=0)],
			)
		if tr and tr.precision == "fp32":
			add("precision.fp32", "warn", "Trains in fp32", "model", "precision", "precision.bf16_autocast")
		# repeating documents is a text-corpus cost (Muennighoff et al.); several epochs over images or a graph is the
		# usual recipe, not a warning (B, Nov 2: "Every document is seen 8 times" on CIFAR-10)
		if tr and tr.epochs and tr.epochs > 1 and sm.lens in TEXT_LENSES:
			add(
				"data.repetition",
				"info",
				f"Every document is seen {tr.epochs:g} times",
				"data",
				"epochs",
				None,
				[Evidence(metric="epochs", value=tr.epochs)],
				detail=risk("epochs", "repeat"),
			)
	if sm.lens == "retrieval" and (r := risk("in-batch negatives")):
		add(
			"retrieval.small_batch",
			"warn",
			"Few in-batch negatives per query",
			"negatives",
			"batch_size",
			None,
			detail=r,
		)
	if sm.lens == "vision":
		if r := risk("channels_last"):
			add(
				"vision.no_channels_last",
				"info",
				"No channels_last memory format",
				"backbone",
				"channels_last",
				"vision.channels_last",
				detail=r,
			)
		if r := risk("augmentation"):
			add(
				"vision.cpu_augment",
				"warn",
				"Augmentation runs on the CPU",
				"augment",
				"num_workers",
				None,
				detail=r,
			)
	if r := risk("hidden states"):
		add(
			"eval.activation_dump",
			"warn",
			"Hidden states dumped for every layer",
			"activations",
			"dump_hidden_states",
			None,
			detail=r,
		)
	if sm.lens == "agent":
		for r in [x for x in sm.risks if "retries" in x]:
			add(
				"agent.tool_retry_loop",
				"error",
				"A tool call can retry without a useful cap",
				"tools",
				"tool_retry_cap",
				"agent.cap_tool_retries",
				detail=r,
			)
			break
	return out


def gpu_type_knob(
	sm: ScriptMap, *, gpu: str | None = None, constraints: Constraints | None = None
) -> Knob | None:
	"""B's gpu_type knob: the GPU the probe runs on, and the ones the visitor may move to.

	Values are the visitor's allowed GPUs when they set any, else Modal's list.
	Current is the GPU they asked for, else the size-aware choice the probe
	will make (probes/gpu_choice.py); the probe overwrites it with env.gpu.name.
	"""
	if sm.lens == "agent":
		return None
	allowed = [canonical(g) for g in constraints.gpu_types] if constraints and constraints.gpu_types else None
	values = list(dict.fromkeys(allowed or offered_gpus()))
	current = canonical(gpu) if gpu else choose_probe_gpu(estimate_need_gb(sm), allowed)
	if current not in values:
		values.append(current)
	return Knob(
		id="gpu_type",
		label_key="knob.gpu_type",
		kind="categorical",
		domain=Domain(values=values),
		current=current,
		group="cluster",
	)


def skeleton_record(
	sm: ScriptMap,
	source: Source,
	*,
	gpu: str | None = None,
	constraints: Constraints | None = None,
	**job_fields,
) -> FlightRecord:
	"""gpu: the GPU the visitor asked for (JobRequest.gpu); constraints: their limits, kept on the job."""
	graph = skeleton_graph(sm)
	env = Env(backend="api", provider="api") if sm.lens == "agent" else None
	knobs = declared_knobs(sm)
	gk = gpu_type_knob(sm, gpu=gpu, constraints=constraints)
	if gk is not None and all(k.id != "gpu_type" for k in knobs):
		knobs = sorted([*knobs, gk], key=lambda k: (k.group or "z", k.id))
	knobs = annotate(knobs, graph)
	if constraints is not None:
		job_fields["constraints"] = constraints
	return FlightRecord(
		job=Job(lens=sm.lens, source=source, framework=sm.framework, script_map=sm, **job_fields),
		env=env,
		graph=graph,
		knobs=knobs,
		findings=read_time_findings(sm, graph),
	)
