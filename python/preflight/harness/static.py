"""Static script mapping: read a script's AST and produce a ScriptMap, no model call.

This is the adapter path the contract promises for when Muse Spark is down
(HF Trainer, TRL, Lightning, torchtitan, vLLM, sentence-transformers, PyG,
timm and plain loops), and the cross-check that keeps Spark's map honest:
every knob it reports must sit at a location this reader can find.

Coverage, by lens (the vocabularies below are the single list to extend):
  fine-tune, pretrain   batch, sequence, accumulation, lr, steps, epochs, precision, checkpointing,
                        eval cadence and batch, packing, LoRA rank, workers, FSDP/DeepSpeed, GPUs
  RL post-training      group size, completion and prompt caps, KL beta, clip bounds, temperature,
                        loss type, vLLM rollouts and its memory share, reward functions
  agent and API runs    model, turns, concurrency, tasks, temperature, max tokens, reasoning
                        effort, tool-retry caps (and uncapped retry loops)
  eval sweeps           max new tokens (thinking cap), eval batch, hidden-state dumps
  serving (vLLM)        max model length, sequences in flight, memory share, tensor parallel,
                        prefix caching, KV-cache dtype
  retrieval             contrastive batch (in-batch negatives), encode batch, index parameters
  GNN                   fan-out per hop, hidden width, layers, batch of seed nodes
  vision                image size, batch, workers, channels_last, CPU augmentation
  launchers             torchrun / accelerate / Slurm GPU counts in .sh, .sbatch, .yaml files

It only reads code. It never imports or runs the user's script, and it is
bounded: over MAX_SOURCE_BYTES of Python, only the entry point and what it imports is read
(mapped_files), an entry point over it alone is refused, and pathological ASTs
(deep nesting) become a ScriptParseError rather than a crash.
"""

import ast
import itertools
import re
from dataclasses import dataclass, field

from preflight.schema.script_map import (
	AgentInfo,
	CodeLocation,
	DataInfo,
	LoopInfo,
	LoraInfo,
	ModelInfo,
	RLInfo,
	ScriptKnob,
	ScriptMap,
	TrainInfo,
)

MAX_SOURCE_BYTES = 2 * 1024 * 1024  # Python source read by the static reader, all files together
TEXT_SUFFIXES = (".sh", ".sbatch", ".slurm", ".yaml", ".yml", ".toml", ".cfg", "Makefile", ".bash")


class ScriptParseError(ValueError):
	pass


class SourceTooLarge(ScriptParseError):
	"""The entry point alone is more Python than the static reader reads: not a syntax error."""

	code = "map.too_large"


def mapped_files(
	files: dict[str, str], entrypoint: str, *, closure: bool | None = None
) -> tuple[dict[str, str], dict | None]:
	"""The files the static reader maps: all of them when their Python fits MAX_SOURCE_BYTES, else
	the entry point and the Python files it imports, nearest first, up to that budget, beside the
	other text files (harness/closure.py; D, Nov 2: torchtune's repository is over 2 MB of Python).
	The second value says what was left out, or None when nothing was. The closure needs
	runner.read_closure on (`closure`, default: the flag); off, over the budget is refused, saying
	how big it is (a SourceTooLarge, code map.too_large, never a syntax error)."""
	py = {p: s for p, s in files.items() if p.endswith(".py")}
	sizes = {p: len(s.encode("utf-8", "surrogatepass")) for p, s in py.items()}
	total = sum(sizes.values())
	if total <= MAX_SOURCE_BYTES:
		if len(py) > 1 and entrypoint in py and _map_closure():
			return _closure_only(files, py, sizes, entrypoint), None
		return files, None
	limit = f"more than Preflight maps ({MAX_SOURCE_BYTES // 2**20} MB)"
	if sizes.get(entrypoint, 0) > MAX_SOURCE_BYTES:
		raise SourceTooLarge(f"{entrypoint} alone is {_mb(sizes[entrypoint])} of Python, {limit}")
	if closure is None:
		from preflight.runner.flags import flag

		closure = flag("runner.read_closure")
	if not closure:
		raise SourceTooLarge(f"the source is {_mb(total)} of Python in {len(py):,} files, {limit}")
	if entrypoint not in py:
		raise SourceTooLarge(
			f"the source is {_mb(total)} of Python in {len(py):,} files, {limit}, and {entrypoint} isn't a "
			"Python file whose imports Preflight can follow; name the Python script it runs as the entry point"
		)
	from .closure import import_closure

	got, info = import_closure(py, lambda paths: {p: py[p] for p in paths}, entrypoint,
		max_files=len(py), max_bytes=MAX_SOURCE_BYTES, sizes=sizes)  # fmt: skip
	info["python_files"] = len(py)
	info["python_bytes"] = total
	return {**{p: s for p, s in files.items() if not p.endswith(".py")}, **got}, info


def _map_closure() -> bool:
	from preflight.runner.flags import flag

	return flag("harness.map_closure")


def _closure_only(
	files: dict[str, str], py: dict[str, str], sizes: dict[str, int], entrypoint: str
) -> dict[str, str]:
	"""harness.map_closure (off; the re-validation, Nov 3): a repo that fits is mapped from the entry point and the
	Python files it imports, not every file, so a model, a loop or a knob in code the job never runs isn't read as
	the job's (the framework and lens already follow the entry point's imports, entry_imports). The other text
	files stay. Not a "partial" read: nothing was left out for size, so the map says nothing about it."""
	from .closure import import_closure

	got, _ = import_closure(py, lambda paths: {p: py[p] for p in paths}, entrypoint,
		max_files=len(py), max_bytes=MAX_SOURCE_BYTES, sizes=sizes)  # fmt: skip
	return {
		**{p: s for p, s in files.items() if not p.endswith(".py")},
		**{p: s for p, s in got.items() if p in py},
	}


def _mb(n: int) -> str:
	return f"{n / 2**20:.2f} MB"


# ---------------------------------------------------------------- vocabularies


def _ids(names, kid):
	return dict.fromkeys(names, kid)


# module-level constant names (lowercased) -> canonical knob id
CONST_KNOBS: dict[str, str] = {
	**_ids(
		["batch_size", "train_batch_size", "micro_batch_size", "per_device_batch_size", "bs"], "batch_size"
	),
	**_ids(["eval_batch_size", "val_batch_size", "test_batch_size"], "eval_batch_size"),
	**_ids(
		[
			"block_size",
			"seq_len",
			"max_len",
			"max_length",
			"max_seq_len",
			"max_seq_length",
			"context_length",
			"ctx_len",
			"seq_length",
		],
		"seq_len",
	),
	"num_workers": "num_workers",
	**_ids(["learning_rate", "lr"], "lr"),
	**_ids(
		["gradient_accumulation_steps", "grad_accum", "grad_accum_steps", "accumulation_steps"], "grad_accum"
	),
	**_ids(["max_iters", "max_steps", "num_steps", "train_steps", "total_steps"], "max_steps"),
	**_ids(["epochs", "num_epochs", "num_train_epochs", "n_epochs"], "epochs"),
	**_ids(
		[
			"checkpoint_every",
			"save_every",
			"save_steps",
			"save_interval",
			"ckpt_interval",
			"checkpoint_interval",
		],
		"checkpoint_every",
	),
	**_ids(["eval_every", "eval_steps", "eval_interval", "val_every"], "eval_every"),
	**_ids(["warmup_iters", "warmup_steps"], "warmup_steps"),
	**_ids(["dtype", "precision", "ptdtype"], "precision"),
	**_ids(["n_layer", "n_layers", "num_layers", "num_hidden_layers"], "n_layer"),
	**_ids(["n_embd", "d_model", "hidden_size", "n_embed", "dim"], "d_model"),
	**_ids(["n_head", "n_heads", "num_heads", "num_attention_heads"], "n_head"),
	"max_turns": "max_turns",
	**_ids(
		["concurrency", "max_concurrency", "n_concurrent", "num_concurrent", "parallel_requests"],
		"concurrency",
	),
	**_ids(["num_generations", "group_size", "n_samples_per_prompt"], "group_size"),
	**_ids(["max_completion_length", "max_completion_len", "max_response_length"], "max_completion_len"),
	**_ids(["max_prompt_length", "max_prompt_len"], "max_prompt_len"),
	**_ids(["max_new_tokens", "max_tokens", "thinking_budget", "max_gen_len"], "max_new_tokens"),
	**_ids(["grad_clip", "max_grad_norm", "clip_grad"], "grad_clip"),
	"gradient_checkpointing": "gradient_checkpointing",
	**_ids(["n_tasks", "num_tasks", "num_samples", "n_problems"], "n_tasks"),
	**_ids(["tool_retries", "max_tool_retries", "tool_retry_cap", "max_retries"], "tool_retry_cap"),
	"temperature": "temperature",
	"reasoning_effort": "reasoning_effort",
	**_ids(["image_size", "img_size", "resolution", "crop_size", "input_size"], "image_size"),
	**_ids(["kl_coef", "beta", "kl_beta", "init_kl_coef"], "kl_beta"),
	"lora_r": "lora_r",
	"lora_alpha": "lora_alpha",
	"lora_dropout": "lora_dropout",
	"weight_decay": "weight_decay",
	**_ids(
		["context_limit", "max_context_tokens", "context_window", "max_context", "max_input_tokens"],
		"context_limit",
	),
	**_ids(["attn_implementation", "attn_impl"], "attn"),
}

# kwargs of *Arguments / *Config calls (TrainingArguments, SFTConfig, GRPOConfig, DPOConfig, ...)
ARGS_KNOBS: dict[str, str] = {
	"lora_alpha": "lora_alpha",
	"lora_dropout": "lora_dropout",
	"attn_implementation": "attn",
	"per_device_train_batch_size": "batch_size",
	"per_device_eval_batch_size": "eval_batch_size",
	"gradient_accumulation_steps": "grad_accum",
	"learning_rate": "lr",
	"num_train_epochs": "epochs",
	"max_steps": "max_steps",
	"warmup_steps": "warmup_steps",
	"save_steps": "checkpoint_every",
	"eval_steps": "eval_every",
	"gradient_checkpointing": "gradient_checkpointing",
	"max_seq_length": "seq_len",
	"max_length": "seq_len",
	"packing": "packing",
	"num_generations": "group_size",
	"max_completion_length": "max_completion_len",
	"max_prompt_length": "max_prompt_len",
	"dataloader_num_workers": "num_workers",
	"beta": "kl_beta",
	"temperature": "temperature",
	"epsilon": "clip_low",
	"epsilon_high": "clip_high",
	"cliprange": "clip_low",
	"loss_type": "loss_type",
	"use_vllm": "use_vllm",
	"vllm_gpu_memory_utilization": "gpu_memory_utilization",
	"num_iterations": "rl_iterations",
	"fsdp": "fsdp_mode",
	"deepspeed": "deepspeed_config",
	"logging_steps": "log_every",
	"max_grad_norm": "grad_clip",
	"optim": "optimizer",
	"lr_scheduler_type": "lr_schedule",
	"weight_decay": "weight_decay",
	"torch_compile": "compile",
	"tf32": "tf32",
	"group_by_length": "group_by_length",
}

# kwargs of any call
CALL_KNOBS: dict[str, str] = {
	"batch_size": "batch_size",
	"num_workers": "num_workers",
	"max_length": "seq_len",
	"max_seq_length": "seq_len",
	"lr": "lr",
	"max_new_tokens": "max_new_tokens",
	"max_tokens": "max_new_tokens",
	"temperature": "temperature",
	"reasoning_effort": "reasoning_effort",
	"output_hidden_states": "dump_hidden_states",
	"output_attentions": "dump_attentions",
	"epochs": "epochs",
	"warmup_steps": "warmup_steps",
	"max_model_len": "max_model_len",
	"max_num_seqs": "max_num_seqs",
	"max_num_batched_tokens": "max_batched_tokens",
	"gpu_memory_utilization": "gpu_memory_utilization",
	"tensor_parallel_size": "tensor_parallel",
	"pipeline_parallel_size": "pipeline_parallel",
	"enable_prefix_caching": "prefix_caching",
	"kv_cache_dtype": "kv_cache_dtype",
	"pin_memory": "pin_memory",
	"prefetch_factor": "prefetch_factor",
	"hidden_channels": "hidden_channels",
	"max_turns": "max_turns",
	"attn_implementation": "attn",
	"max_context_tokens": "context_limit",
	"context_limit": "context_limit",
	"max_input_tokens": "context_limit",
}

# call-specific kwargs: (call name suffix, kwarg) -> knob id; these win over CALL_KNOBS
SPECIAL_KWARGS: dict[tuple[str, str], str] = {
	("encode", "batch_size"): "encode_batch_size",
	("encode_corpus", "batch_size"): "encode_batch_size",
	("encode_queries", "batch_size"): "encode_batch_size",
}
GNN_CLASSES = {
	"GraphSAGE",
	"GCN",
	"GAT",
	"GIN",
	"EdgeCNN",
	"PNA",
	"GraphUNet",
	"SAGEConv",
	"GCNConv",
	"GATConv",
}
FAISS_POSITIONAL = {
	"IndexHNSWFlat": (1, "index_m"),
	"IndexIVFFlat": (2, "index_nlist"),
	"IndexIVFPQ": (2, "index_nlist"),
}

LABELS = {
	"batch_size": "Batch size",
	"eval_batch_size": "Eval batch size",
	"seq_len": "Sequence length",
	"num_workers": "Data loader workers",
	"lr": "Learning rate",
	"grad_accum": "Gradient accumulation",
	"max_steps": "Training steps",
	"epochs": "Epochs",
	"checkpoint_every": "Checkpoint interval",
	"eval_every": "Eval interval",
	"warmup_steps": "Warm-up steps",
	"precision": "Precision",
	"n_layer": "Layers",
	"d_model": "Model width",
	"n_head": "Attention heads",
	"max_turns": "Max turns",
	"concurrency": "Concurrency",
	"group_size": "Group size",
	"max_completion_len": "Max completion length",
	"max_prompt_len": "Max prompt length",
	"grad_clip": "Gradient clip",
	"gradient_checkpointing": "Gradient checkpointing",
	"packing": "Sequence packing",
	"lora_r": "LoRA rank",
	"n_tasks": "Tasks",
	"max_new_tokens": "Max new tokens",
	"temperature": "Temperature",
	"reasoning_effort": "Reasoning effort",
	"tool_retry_cap": "Tool retry cap",
	"kl_beta": "KL coefficient",
	"clip_low": "Clip range (low)",
	"clip_high": "Clip range (high)",
	"loss_type": "Loss type",
	"use_vllm": "vLLM rollouts",
	"gpu_memory_utilization": "GPU memory share",
	"max_model_len": "Max model length",
	"max_num_seqs": "Sequences in flight",
	"tensor_parallel": "Tensor parallel",
	"prefix_caching": "Prefix caching",
	"kv_cache_dtype": "KV-cache dtype",
	"dump_hidden_states": "Dump hidden states",
	"encode_batch_size": "Encode batch",
	"index_m": "HNSW links (M)",
	"index_nlist": "IVF lists",
	"hidden_channels": "Hidden width",
	"gnn_layers": "GNN layers",
	"image_size": "Image size",
	"fsdp_mode": "FSDP mode",
	"gpus": "GPUs",
	"log_every": "Log interval",
	"weight_decay": "Weight decay",
	"lora_alpha": "LoRA alpha",
	"lora_dropout": "LoRA dropout",
	"attn": "Attention kernel",
	"quantization": "Weight quantization",
	"context_limit": "Context limit",
	"gpu_type": "GPU type",
}

UNITS = {
	"batch_size": "samples",
	"eval_batch_size": "samples",
	"encode_batch_size": "samples",
	"seq_len": "tokens",
	"max_steps": "steps",
	"checkpoint_every": "steps",
	"eval_every": "steps",
	"warmup_steps": "steps",
	"max_turns": "turns",
	"max_completion_len": "tokens",
	"max_prompt_len": "tokens",
	"max_new_tokens": "tokens",
	"max_model_len": "tokens",
	"n_tasks": "tasks",
	"image_size": "px",
	"gpus": "GPUs",
	"concurrency": "requests",
	"context_limit": "tokens",
}

PRECISION = {
	"float32": "fp32",
	"fp32": "fp32",
	"float": "fp32",
	"bfloat16": "bf16",
	"bf16": "bf16",
	"float16": "fp16",
	"fp16": "fp16",
	"half": "fp16",
	"float8": "fp8",
	"fp8": "fp8",
}

AGENT_CALLS = ("chat.completions.create", "responses.create", "messages.create", "completion", "acompletion")
RL_TRAINERS = {
	"GRPOTrainer": "grpo",
	"PPOTrainer": "ppo",
	"RLOOTrainer": "rloo",
	"OnlineDPOTrainer": "online_dpo",
	"DPOTrainer": "dpo",
	"GKDTrainer": "gkd",
	"KTOTrainer": "kto",
}
LAUNCH_GPUS = [
	re.compile(r"--nproc[_-]per[_-]node[= ](\d+)"),
	re.compile(r"--num[_-]processes[= ](\d+)"),
	re.compile(r"#SBATCH\s+--gres=gpu(?::[\w-]+)?:(\d+)"),
	re.compile(r"#SBATCH\s+--gpus(?:-per-node)?=(?:[\w-]+:)?(\d+)"),
]


# ---------------------------------------------------------------- AST helpers


def dotted(node: ast.AST) -> str:
	if isinstance(node, ast.Name):
		return node.id
	if isinstance(node, ast.Attribute):
		base = dotted(node.value)
		return f"{base}.{node.attr}" if base else node.attr
	if isinstance(node, ast.Call):
		return dotted(node.func) + "()"
	if isinstance(node, ast.Await):
		return dotted(node.value)
	return ""


def literal(node: ast.AST):
	"""A literal value, or raise ValueError. Handles -x and torch.bfloat16-style dtypes."""
	if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "torch":
		return node.attr
	return ast.literal_eval(node)


class Source:
	"""One file's text with line offsets computed once (ast.get_source_segment re-splits every call)."""

	def __init__(self, text: str):
		self.text = text
		self.lines = text.split("\n")

	def segment(self, node: ast.AST, limit: int = 300) -> str | None:
		try:
			l0, l1, c0, c1 = node.lineno - 1, node.end_lineno - 1, node.col_offset, node.end_col_offset
		except AttributeError:
			return None
		if l0 == l1:
			seg = self.lines[l0].encode("utf-8")[c0:c1].decode("utf-8", errors="replace")
		else:
			first = self.lines[l0].encode("utf-8")[c0:].decode("utf-8", errors="replace")
			middle = self.lines[l0 + 1 : l1]
			last = self.lines[l1].encode("utf-8")[:c1].decode("utf-8", errors="replace")
			seg = "\n".join([first, *middle, last])
		return seg[:limit]


@dataclass
class Candidate:
	knob_id: str
	value: object
	file: str
	line: int
	expr: str
	priority: int


@dataclass
class Facts:
	imports: set[str] = field(default_factory=set)
	names: set[str] = field(default_factory=set)
	consts: dict[str, tuple[object, str, int, str]] = field(default_factory=dict)
	calls: list[tuple[str, ast.Call, str]] = field(default_factory=list)
	candidates: list[Candidate] = field(default_factory=list)
	trainer_vars: dict[str, str] = field(default_factory=dict)
	optimizer_vars: set[str] = field(default_factory=set)
	module_classes: list[ast.ClassDef] = field(default_factory=list)
	callback_classes: list[tuple[ast.ClassDef, str]] = field(default_factory=list)
	sources: dict[str, Source] = field(default_factory=dict)
	texts: dict[str, str] = field(default_factory=dict)
	trees: dict[str, ast.Module] = field(default_factory=dict)
	retry_loops: list[str] = field(default_factory=list)


def _parse(files: dict[str, str]) -> Facts:
	f = Facts()
	py = {p: s for p, s in files.items() if p.endswith(".py")}
	if sum(len(s.encode("utf-8", "surrogatepass")) for s in py.values()) > MAX_SOURCE_BYTES:
		# callers pass mapped_files(); a direct caller over the budget gets the same accurate refusal
		raise SourceTooLarge(
			f"the Python source is more than Preflight maps ({MAX_SOURCE_BYTES // 2**20} MB)"
		)
	f.texts = {p: s for p, s in files.items() if p.endswith(TEXT_SUFFIXES) and len(s) < 256 * 1024}
	for path, src in py.items():
		try:
			tree = ast.parse(src, filename=path)
		except SyntaxError as exc:
			raise ScriptParseError(f"{path}:{exc.lineno}: {exc.msg}") from exc
		except (RecursionError, MemoryError, ValueError) as exc:
			raise ScriptParseError(f"{path}: too deeply nested to read ({type(exc).__name__})") from exc
		f.trees[path] = tree
		f.sources[path] = Source(src)
	try:
		for path, tree in f.trees.items():
			_scan(f, path, tree)
	except (RecursionError, MemoryError) as exc:
		raise ScriptParseError(f"too deeply nested to read ({type(exc).__name__})") from exc
	return f


def _const_target(node: ast.AST) -> tuple[str, ast.AST] | None:
	"""(name, value) for `x = 1` and `x: int = 1` with a plain name on the left."""
	if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
		return node.targets[0].id, node.value
	if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value is not None:
		return node.target.id, node.value
	return None


def _scan(f: Facts, path: str, tree: ast.Module) -> None:
	src = f.sources[path]
	# module constants first; then fields of config classes (dataclasses, HfArgumentParser
	# dataclasses, nanoGPT-style GPTConfig), which a module constant of the same name outranks
	class_fields = [n for c in tree.body if isinstance(c, ast.ClassDef) for n in c.body]
	for node, in_class in [(n, False) for n in tree.body] + [(n, True) for n in class_fields]:
		target = _const_target(node)
		if target is None:
			continue
		name, value = target
		if in_class and name in f.consts:
			continue
		try:
			val = literal(value)
		except (ValueError, SyntaxError, TypeError, RecursionError):
			continue
		f.consts[name] = (val, path, node.lineno, src.segment(node) or name)
	for node in ast.walk(tree):
		if isinstance(node, ast.Import):
			for a in node.names:
				f.imports.add(a.name)
		elif isinstance(node, ast.ImportFrom) and node.module:
			f.imports.add(node.module)
			for a in node.names:
				f.names.add(a.name)
		elif isinstance(node, ast.Call):
			f.calls.append((dotted(node.func), node, path))
		elif isinstance(node, ast.ClassDef):
			bases = [dotted(b) for b in node.bases]
			if any(b.endswith("Module") for b in bases):
				f.module_classes.append(node)
			if any(b.endswith("Callback") for b in bases):
				f.callback_classes.append((node, path))
		elif isinstance(node, ast.While) and isinstance(node.test, ast.Constant) and node.test.value is True:
			for inner in ast.walk(node):
				if isinstance(inner, ast.ExceptHandler) and any(
					isinstance(b, ast.Continue | ast.Pass) for b in inner.body
				):
					f.retry_loops.append(f"{path}:{node.lineno}")
					break
		if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call) and len(node.targets) == 1:
			tgt = dotted(node.targets[0])
			fn = dotted(node.value.func)
			short = fn.rsplit(".", 1)[-1]
			if short.endswith("Trainer"):
				f.trainer_vars[tgt] = short
			if "optim." in fn or short in ("AdamW", "Adam", "SGD", "Adafactor", "Lion", "Muon", "AdamW8bit"):
				f.optimizer_vars.add(tgt)


def _resolve(f: Facts, node: ast.AST, path: str) -> tuple[object, str, int, str] | None:
	"""Value and location for an argument: a literal where it is, or a module constant where it is defined."""
	if isinstance(node, ast.Name) and node.id in f.consts:
		return f.consts[node.id]
	try:
		return literal(node), path, node.lineno, ""
	except (ValueError, SyntaxError, TypeError, RecursionError):
		return None


def _kwargs(call: ast.Call) -> dict[str, ast.AST]:
	return {k.arg: k.value for k in call.keywords if k.arg}


def _first_arg(f: Facts, call: ast.Call, path: str, kw: str | None = None):
	node = call.args[0] if call.args else (_kwargs(call).get(kw) if kw else None)
	if node is None:
		return None
	r = _resolve(f, node, path)
	return r[0] if r else None


# ---------------------------------------------------------------- extraction

SCALAR = int | float | str | bool


def _collect_knobs(f: Facts) -> None:
	for name, (val, path, line, expr) in f.consts.items():
		kid = CONST_KNOBS.get(name.lower())
		if kid and isinstance(val, SCALAR):
			f.candidates.append(Candidate(kid, val, path, line, expr, priority=2))
	for fn, call, path in f.calls:
		short = fn.rsplit(".", 1)[-1]
		src = f.sources[path]
		is_args = short.endswith("Arguments") or short.endswith("Config")
		table = ARGS_KNOBS if is_args else CALL_KNOBS
		for kw in call.keywords:
			kid = SPECIAL_KWARGS.get((short, kw.arg or "")) or table.get(kw.arg or "")
			if short in GNN_CLASSES and kw.arg == "num_layers":
				kid = "gnn_layers"
			if kw.arg == "num_neighbors" and isinstance(kw.value, ast.List):
				# one knob per hop: num_neighbors=[25, 10] -> fanout.hop1, fanout.hop2
				seg = src.segment(kw) or "num_neighbors"
				for i, elt in enumerate(kw.value.elts, 1):
					r = _resolve(f, elt, path)
					if r and isinstance(r[0], int):
						f.candidates.append(Candidate(f"fanout.hop{i}", r[0], path, kw.value.lineno, seg, 3))
				continue
			if not kid:
				continue
			r = _resolve(f, kw.value, path)
			if r is None or not isinstance(r[0], SCALAR):
				continue
			val, rpath, rline, rexpr = r
			if not rexpr:
				rexpr = src.segment(kw) or kw.arg
			# a kwarg that just forwards a constant is the constant's knob
			p = (
				2
				if isinstance(kw.value, ast.Name)
				else (3 if is_args or (short, kw.arg) in SPECIAL_KWARGS else 1)
			)
			f.candidates.append(Candidate(kid, val, rpath, rline, rexpr, p))
		if is_args:
			kws = _kwargs(call)
			for flag, prec in (("bf16", "bf16"), ("fp16", "fp16")):
				if flag in kws:
					r = _resolve(f, kws[flag], path)
					if r and r[0] is True:
						seg = src.segment(next(k for k in call.keywords if k.arg == flag))
						f.candidates.append(
							Candidate("precision", prec, path, kws[flag].lineno, seg or flag, 3)
						)
		if short == "LoraConfig":
			kws = _kwargs(call)
			if "r" in kws:
				r = _resolve(f, kws["r"], path)
				if r:
					seg = src.segment(next(k for k in call.keywords if k.arg == "r"))
					f.candidates.append(Candidate("lora_r", r[0], r[1], r[2], r[3] or seg or "r", 3))
		if short == "add_argument":
			a = _argparse_candidate(f, call, path, src)
			if a is not None:
				f.candidates.append(a)
		if short in ("BitsAndBytesConfig", "from_pretrained"):
			q = _quant_candidate(f, call, path, src)
			if q is not None:
				f.candidates.append(q)
		if short in FAISS_POSITIONAL:
			idx, kid = FAISS_POSITIONAL[short]
			if len(call.args) > idx:
				r = _resolve(f, call.args[idx], path)
				if r and isinstance(r[0], int):
					f.candidates.append(
						Candidate(kid, r[0], r[1], r[2], r[3] or src.segment(call) or short, 3)
					)
	for path, text in f.texts.items():
		for i, line in enumerate(text.split("\n"), 1):
			for pat in LAUNCH_GPUS:
				mt = pat.search(line)
				if mt:
					f.candidates.append(Candidate("gpus", int(mt.group(1)), path, i, mt.group(0), 3))


# weight formats, named as B's memory model keys them (engine/memory.py QUANT_BYTES)
QUANT_FORMATS = ("none", "int8", "nf4", "nf4_dq", "fp4", "fp4_dq")
FOUR_BIT = frozenset({"nf4", "nf4_dq", "fp4", "fp4_dq", "gptq_4bit", "awq_4bit", "hqq_4bit"})
PREQUANTISED = frozenset({"gptq_4bit", "gptq_8bit", "awq_4bit"})


def _prequantised(hf_id: str) -> str | None:
	"""Checkpoints quantised before loading, named in the model id (Llama-2-7B-GPTQ, Qwen2.5-7B-Instruct-GPTQ-Int8)."""
	low = hf_id.lower()
	if "gptq" in low:
		return "gptq_8bit" if re.search(r"int8|8bit|8-bit|w8a", low) else "gptq_4bit"
	if "awq" in low:
		return "awq_4bit"
	return None


def argparse_flags(call: ast.Call) -> tuple[list[str], str | None]:
	"""(option strings, dest) of a parser.add_argument call; dest as argparse would derive it."""
	flags = [a.value for a in call.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
	opts = [x for x in flags if x.startswith("-")]
	dest_kw = _kwargs(call).get("dest")
	if isinstance(dest_kw, ast.Constant) and isinstance(dest_kw.value, str):
		return opts, dest_kw.value
	longs = [x for x in opts if x.startswith("--")]
	name = (longs or opts or flags or [None])[0]
	return opts, name.lstrip("-").replace("-", "_") if name else None


def _argparse_candidate(f: Facts, call: ast.Call, path: str, src) -> Candidate | None:
	"""parser.add_argument("--batch_size", type=int, default=16): a knob at its default."""
	_opts, dest = argparse_flags(call)
	kid = CONST_KNOBS.get((dest or "").lower())
	default = next((k for k in call.keywords if k.arg == "default"), None)
	if kid is None or default is None:
		return None
	try:
		val = literal(default.value)
	except (ValueError, SyntaxError, TypeError, RecursionError):
		return None
	if not isinstance(val, SCALAR) or val is None:
		return None
	return Candidate(kid, val, path, default.value.lineno, src.segment(default) or "default", 2)


def _quant_label(kws: dict, f: Facts, path: str) -> str | None:
	"""nf4, nf4_dq, fp4, fp4_dq or int8 from BitsAndBytesConfig / from_pretrained flags."""

	def val(k):
		r = _resolve(f, kws[k], path) if k in kws else None
		return r[0] if r else None

	if val("load_in_4bit") is True:
		qt = val("bnb_4bit_quant_type")
		base = qt if qt in ("nf4", "fp4") else "fp4"  # bitsandbytes' default is fp4
		return f"{base}_dq" if val("bnb_4bit_use_double_quant") is True else base
	if val("load_in_8bit") is True:
		return "int8"
	return None


def _quant_candidate(f: Facts, call: ast.Call, path: str, src) -> Candidate | None:
	kws = _kwargs(call)
	label = _quant_label(kws, f, path)
	if label is None:
		return None
	flag = "load_in_8bit" if label == "int8" else "load_in_4bit"
	kw = next(k for k in call.keywords if k.arg == flag)
	return Candidate("quantization", label, path, kw.value.lineno, src.segment(kw) or flag, 3)


# changing these rewrites more than one value (quantization swaps a config object's flags)
NEEDS_CODE = frozenset({"quantization"})


def _knobs(f: Facts) -> list[ScriptKnob]:
	best: dict[str, Candidate] = {}
	for c in f.candidates:
		cur = best.get(c.knob_id)
		if cur is None or c.priority > cur.priority:
			best[c.knob_id] = c
	out = []
	for kid, c in best.items():
		v = c.value
		kind = (
			"bool"
			if isinstance(v, bool)
			else "int"
			if isinstance(v, int)
			else "float"
			if isinstance(v, float)
			else "categorical"
		)
		out.append(
			ScriptKnob(
				id=kid,
				label=LABELS.get(kid, kid.replace("_", " ").replace(".", " ").capitalize()),
				kind=kind,
				current=v,
				unit=UNITS.get(kid) or ("nodes" if kid.startswith("fanout.") else None),
				location=CodeLocation(file=c.file, line=c.line, expr=c.expr),
				needs_code=kid in NEEDS_CODE,
			)
		)
	return sorted(out, key=lambda k: k.id)


def _has(f: Facts, *mods: str) -> bool:
	return any(m == i or i.startswith(m + ".") for m in mods for i in f.imports)


def _short_calls(f: Facts) -> set[str]:
	return {fn.rsplit(".", 1)[-1] for fn, _, _ in f.calls}


# an API client's method passed, not called (smolagents' models.py: `self.retryer(self.client.chat.completions.create,
# **kwargs)`): seen only from the job's own files (harness.map_closure), where another example's call no longer answers
_API_REFS = ("chat.completions.create", "responses.create", "messages.create")


def _api_ref(f: Facts) -> tuple[str, int, str] | None:
	"""(file, line, name) of the first API client method named in the job's code, called or not: an attribute in the
	AST, so a comment or docstring that mentions one doesn't count (review, Nov 3). Only with harness.map_closure."""
	if not _map_closure():
		return None
	for path, tree in f.trees.items():
		for node in ast.walk(tree):
			if isinstance(node, ast.Attribute) and node.attr == "create":
				name = dotted(node)
				if name.endswith(_API_REFS):
					return path, node.lineno, name
	return None


def _api_refs(f: Facts) -> bool:
	return _api_ref(f) is not None


def _framework(f: Facts) -> str:
	trainers = set(f.trainer_vars.values()) | {s for s in _short_calls(f) if s.endswith("Trainer")}
	if _has(f, "trl") or trainers & (set(RL_TRAINERS) | {"SFTTrainer", "RewardTrainer"}):
		return "trl"
	if _has(f, "vllm"):
		return "vllm"
	if _has(f, "sentence_transformers"):
		return "sentence_transformers"
	if _has(f, "torch_geometric"):
		return "pyg"
	if _has(f, "dgl"):
		return "dgl"
	if _has(f, "torchtitan"):
		return "torchtitan"
	if _has(f, "lightning", "pytorch_lightning"):
		return "lightning"
	if _has(f, "transformers") and ("Trainer" in trainers or "Trainer" in f.names):
		return "hf_trainer"
	if (
		_has(f, "openai", "anthropic", "litellm")
		or any(fn.endswith(AGENT_CALLS) for fn, _, _ in f.calls)
		or _api_refs(f)
	):
		return "openai_client"
	if _has(f, "timm"):
		return "timm"
	if (
		any(fn.endswith(".generate") for fn, _, _ in f.calls)
		and not f.optimizer_vars
		and _has(f, "transformers")
	):
		return "hf_generate"
	if _has(f, "torch"):
		return "plain_pytorch"
	return "other"


def _pretrained(f: Facts) -> tuple[str | None, ast.Call | None, str | None]:
	for fn, call, path in f.calls:
		if fn.endswith("from_pretrained") and "Tokenizer" not in fn and "Processor" not in fn:
			v = _first_arg(f, call, path, "pretrained_model_name_or_path")
			if isinstance(v, str):
				return v, call, path
	for fn, call, path in f.calls:
		short = fn.rsplit(".", 1)[-1]
		if short in RL_TRAINERS or short == "SFTTrainer" or (short == "LLM" and _has(f, "vllm")):
			v = _kwargs(call).get("model") or (call.args[0] if call.args and short == "LLM" else None)
			if v is not None:
				r = _resolve(f, v, path)
				if r and isinstance(r[0], str):
					return r[0], call, path
		if short == "SentenceTransformer":
			v = _first_arg(f, call, path, "model_name_or_path")
			if isinstance(v, str):
				return v, call, path
	return None, None, None


def _lens(f: Facts, framework: str, hf_id: str | None) -> str:
	shorts = _short_calls(f)
	if shorts & set(RL_TRAINERS):
		return "rl"
	if framework == "trl":
		return "finetune"
	if framework == "openai_client":
		return "agent"
	if framework == "vllm":
		return "eval" if _has(f, "lm_eval", "evaluate") else "serving"
	if framework == "sentence_transformers" or _has(f, "faiss"):
		return "retrieval"
	if framework in ("pyg", "dgl"):
		return "gnn"
	# B, Nov 2 (the unseen runs): airbench imports torchvision and its transforms, not torchvision.models, and builds
	# its own conv net, so it read as pretraining; a torchvision import with a conv layer is a vision job
	if (
		framework == "timm"
		or _has(f, "torchvision.models", "torchvision.datasets", "torchvision.transforms")
		or (_has(f, "torchvision") and any("Conv2d" in s.text for s in f.sources.values()))
	):
		return "vision"
	if framework == "hf_generate":
		return "eval"
	return "finetune" if hf_id else "pretrain"


# harness.lens_by_entry (off; the PM, Nov 3): the unseen runs' three misreads the import rules leave. LitGPT's
# litgpt/pretrain.py loads a tokenizer from the Hub, so it read as fine-tuning; torchtune's
# recipes/lora_finetune_single_device.py loads its model through its own config, so it read as pretraining;
# lm-evaluation-harness's lm_eval/__main__.py has no training loop and read as pretraining
# whole words of the file name (split on anything not a letter or digit): "pretrained" is not "pretrain"
ENTRY_PRETRAIN = frozenset({"pretrain", "pretraining"})
ENTRY_FINETUNE = frozenset({"finetune", "finetuning", "lora", "qlora", "sft"})


def _lens_by_entry(lens: str, reach: Facts, f: Facts, entrypoint: str) -> str:
	"""A training lens corrected by the entry point's own name, by whole words (pretrain; finetune, lora, sft,
	fine_tune; fine-tuning words win when both appear), and an entry point inside lm_eval's or lighteval's own
	package read as an eval; every other lens as the import rules read it. Not a job that imports lm_eval:
	a fine-tune can run benchmarks in its loop, and the reader can't always see its optimizer (review, Nov 3)."""
	from preflight.runner.flags import flag

	if not flag("harness.lens_by_entry") or lens not in ("finetune", "pretrain"):
		return lens
	# lm-evaluation-harness's own entry point imports its package as local code, which the job's imports leave out
	evals = ("lm_eval", "lighteval")
	if entrypoint.removeprefix("src/").split("/", 1)[0] in evals:  # lighteval keeps its package under src/
		return "eval"
	words = re.split(r"[^a-z0-9]+", entrypoint.rsplit("/", 1)[-1].lower().removesuffix(".py"))
	if set(words) & ENTRY_FINETUNE or any(
		a == "fine" and b in ("tune", "tuning") for a, b in itertools.pairwise(words)
	):
		return "finetune"
	if set(words) & ENTRY_PRETRAIN:
		return "pretrain"
	return lens


def _const(f: Facts, *names):
	for n in names:
		for k, (v, *_rest) in f.consts.items():
			if k.lower() == n:
				return v
	return None


def _model(f: Facts, hf_id, call, path) -> ModelInfo | None:
	lora = quant = None
	for fn, c, p in f.calls:
		short = fn.rsplit(".", 1)[-1]
		if short == "LoraConfig":
			vals = {k: (_resolve(f, v, p) or (None,))[0] for k, v in _kwargs(c).items()}
			r = vals.get("r")
			if isinstance(r, int):
				tm = vals.get("target_modules")
				lora = LoraInfo(
					r=r,
					alpha=vals.get("lora_alpha"),
					target_modules=list(tm) if isinstance(tm, list | tuple) else [],
					dropout=vals.get("lora_dropout"),
				)
		if short in ("BitsAndBytesConfig", "from_pretrained"):
			# first one wins, as for the knob: a reward or reference model loaded later doesn't override it
			quant = quant or _quant_label(_kwargs(c), f, p)
		if short == "create_model" and "timm" in fn:
			name = _first_arg(f, c, p, "model_name")
			if isinstance(name, str):
				fam = "vit" if re.search(r"vit|deit|swin|eva", name) else "cnn"
				return ModelInfo(name=name, family=fam)
	if hf_id:
		if quant is None:
			quant = _prequantised(hf_id[:200])
		dtype = None
		if call is not None:
			for k in ("torch_dtype", "dtype"):
				v = _kwargs(call).get(k)
				if v is not None:
					r = _resolve(f, v, path)
					dtype = str(r[0]) if r else None
		decoder = any("CausalLM" in fn or fn.endswith(("Trainer", ".LLM", "LLM")) for fn, _, _ in f.calls)
		family = "encoder" if _has(f, "sentence_transformers") else "decoder" if decoder else "other"
		return ModelInfo(
			name=hf_id.rsplit("/", 1)[-1],
			hf_id=hf_id,
			family=family,
			dtype=dtype,
			lora=lora,
			quantization=quant,
		)
	for fn, c, p in f.calls:
		short = fn.rsplit(".", 1)[-1]
		if short in GNN_CLASSES:
			kw = {k: (_resolve(f, v, p) or (None,))[0] for k, v in _kwargs(c).items()}
			return ModelInfo(
				name=short,
				family="gnn",
				n_layers=kw.get("num_layers") if isinstance(kw.get("num_layers"), int) else None,
				d_model=kw.get("hidden_channels") if isinstance(kw.get("hidden_channels"), int) else None,
			)
	if f.module_classes:
		n_layer = _const(f, "n_layer", "n_layers", "num_layers", "num_hidden_layers")
		d = _const(f, "n_embd", "d_model", "hidden_size", "n_embed", "dim")
		n_head = _const(f, "n_head", "n_heads", "num_heads", "num_attention_heads")
		vocab = _const(f, "vocab_size")
		text = " ".join(s.text for s in f.sources.values())
		attn = "attention" in text or "attn" in text
		family = "decoder" if attn and vocab else "cnn" if "Conv2d" in text else "mlp"
		params = (
			12 * n_layer * d * d + (vocab or 0) * d
			if isinstance(n_layer, int) and isinstance(d, int)
			else None
		)
		return ModelInfo(
			name=f.module_classes[-1].name,
			family=family,
			n_layers=n_layer if isinstance(n_layer, int) else None,
			d_model=d if isinstance(d, int) else None,
			n_heads=n_head if isinstance(n_head, int) else None,
			vocab_size=vocab if isinstance(vocab, int) else None,
			param_count=params,
			lora=lora,
		)
	return None


def _no_grad_ctx(with_node: ast.With) -> bool:
	return any(
		dotted(item.context_expr).endswith(("no_grad()", "inference_mode()")) for item in with_node.items
	)


def _forward_without_no_grad(f: Facts) -> list[str]:
	"""Callback methods or eval functions that call the model outside no_grad."""
	hits = []

	def scan(fn: ast.FunctionDef, owner: str, path: str):
		if any(
			dotted(d).endswith(("no_grad", "no_grad()", "inference_mode", "inference_mode()"))
			for d in fn.decorator_list
		):
			return
		stack = [(fn, False)]
		while stack:
			node, guarded = stack.pop()
			if isinstance(node, ast.With) and _no_grad_ctx(node):
				guarded = True
			if isinstance(node, ast.Call) and not guarded:
				name = dotted(node.func)
				if name.split(".")[-1] == "model" and (node.args or node.keywords):
					hits.append(f"{owner}.{fn.name} ({path}:{node.lineno})")
			stack.extend((child, guarded) for child in ast.iter_child_nodes(node))

	for cls, path in f.callback_classes:
		for item in cls.body:
			if isinstance(item, ast.FunctionDef):
				scan(item, cls.name, path)
	for path, tree in f.trees.items():
		for node in tree.body:
			if isinstance(node, ast.FunctionDef) and re.search(r"eval|valid", node.name):
				scan(node, "<module>", path)
	return sorted(set(hits))


def _loop(f: Facts, framework: str) -> LoopInfo | None:
	for fn, call, path in f.calls:
		head, _, tail = fn.rpartition(".")
		if tail in ("train", "fit") and (head in f.trainer_vars or head.endswith("Trainer()")):
			seg = f.sources[path].segment(call)
			kind = (
				"lightning"
				if framework == "lightning"
				else "trl_trainer"
				if framework == "trl"
				else "hf_trainer"
			)
			return LoopInfo(
				kind=kind, location=CodeLocation(file=path, line=call.lineno, expr=seg), step_call=seg
			)
	for fn, call, path in f.calls:
		head, _, tail = fn.rpartition(".")
		if tail == "step" and (head in f.optimizer_vars or "optim" in head.lower() or head == "opt"):
			seg = f.sources[path].segment(call)
			return LoopInfo(
				kind="plain",
				location=CodeLocation(file=path, line=call.lineno, expr=seg),
				step_call=seg,
				optimizer_var=head,
			)
	for fn, call, path in f.calls:
		if fn.endswith("fit") and framework == "sentence_transformers":
			seg = f.sources[path].segment(call)
			return LoopInfo(
				kind="other", location=CodeLocation(file=path, line=call.lineno, expr=seg), step_call=seg
			)
	for fn, call, path in f.calls:
		if fn.endswith(AGENT_CALLS):
			return LoopInfo(
				kind="agent_loop", location=CodeLocation(file=path, line=call.lineno), step_call=fn
			)
	ref = _api_ref(f) if framework == "openai_client" else None
	if ref is not None:  # an API method passed, not called (harness.map_closure)
		return LoopInfo(kind="agent_loop", location=CodeLocation(file=ref[0], line=ref[1]), step_call=ref[2])
	for fn, call, path in f.calls:
		if fn.endswith(".generate"):
			return LoopInfo(
				kind="eval_loop", location=CodeLocation(file=path, line=call.lineno), step_call=fn
			)
	return None


def _data(f: Facts, knobs: dict[str, ScriptKnob]) -> DataInfo:
	dataset = padding = None
	packing = None
	for fn, call, path in f.calls:
		short = fn.rsplit(".", 1)[-1]
		if short == "load_dataset" and dataset is None:
			v = _first_arg(f, call, path, "path")
			if isinstance(v, str):
				dataset = v
		if short.startswith("DataCollator"):
			v = _kwargs(call).get("padding")
			r = _resolve(f, v, path) if v is not None else None
			padding = (r[0] if r else None) or (
				"longest" if "Padding" in short or "Seq2Seq" in short else padding
			)
	if dataset is None and isinstance(_const(f, "data_source", "dataset", "dataset_name"), str):
		dataset = _const(f, "data_source", "dataset", "dataset_name")
	if "packing" in knobs:
		packing = bool(knobs["packing"].current)
	elif padding:
		packing = False

	def kv(k):
		v = knobs.get(k)
		return (
			v.current
			if v is not None and isinstance(v.current, int) and not isinstance(v.current, bool)
			else None
		)

	return DataInfo(
		dataset=dataset,
		batch_size=kv("batch_size"),
		eval_batch_size=kv("eval_batch_size"),
		seq_len=kv("seq_len"),
		num_workers=kv("num_workers"),
		packing=packing,
		padding=padding if isinstance(padding, str) else None,
	)


def _precision(knobs: dict[str, ScriptKnob], model: ModelInfo | None) -> str:
	p = knobs.get("precision")
	if p is not None and isinstance(p.current, str):
		return PRECISION.get(p.current.lower(), "unknown")
	if model and model.dtype:
		return PRECISION.get(model.dtype.lower(), "unknown")
	return "unknown"


def _distributed(f: Facts, by_id: dict[str, ScriptKnob]) -> str:
	if ("fsdp_mode" in by_id and by_id["fsdp_mode"].current) or any(
		s in ("FullyShardedDataParallel", "fully_shard") for s in _short_calls(f)
	):
		return "fsdp"
	if "deepspeed_config" in by_id or _has(f, "deepspeed"):
		return "deepspeed"
	gpus = by_id.get("gpus")
	if "DistributedDataParallel" in _short_calls(f) or (
		gpus is not None and isinstance(gpus.current, int) and gpus.current > 1
	):
		return "ddp"
	return "none"


def _module_paths(mod: str, trees: dict, here: str = "") -> list[str]:
	"""The repo files a dotted module name can be: a module or a package's __init__, looked up from the entrypoint's
	own folder first (Python puts the script's folder on the path; review, Nov 2), then the repo root, then src/."""
	stem = mod.replace(".", "/")
	out = []
	for root in dict.fromkeys((f"{here}/" if here else "", "", "src/")):
		for cand in (f"{root}{stem}.py", f"{root}{stem}/__init__.py"):
			if cand in trees and cand not in out:
				out.append(cand)
	return out


def _top_level_imports(tree: ast.Module) -> list[tuple[str, int, list[str]]]:
	"""(module, level, names) for imports at module level, in if/try blocks too, not inside a def or class: a
	library's optional backend is imported inside the function that uses it."""
	out = []

	def visit(body):
		for node in body:
			if isinstance(node, ast.Import):
				out.extend((a.name, 0, []) for a in node.names)
			elif isinstance(node, ast.ImportFrom):
				out.append((node.module or "", node.level, [a.name for a in node.names]))
			elif isinstance(node, ast.If | ast.Try | ast.With):
				for part in ("body", "orelse", "finalbody", "handlers"):
					for sub in getattr(node, part, []) or []:
						visit(sub.body if isinstance(sub, ast.ExceptHandler) else [sub])

	visit(tree.body)
	return out


def _all_imports(tree: ast.Module) -> list[tuple[str, int, list[str]]]:
	out = []
	for node in ast.walk(tree):
		if isinstance(node, ast.Import):
			out.extend((a.name, 0, []) for a in node.names)
		elif isinstance(node, ast.ImportFrom):
			out.append((node.module or "", node.level, [a.name for a in node.names]))
	return out


def entry_imports(f: Facts, entrypoint: str) -> set[str]:
	"""What the job itself imports: every import in the entrypoint, then the module-level imports of the repo files
	it reaches. B, Nov 2 (the unseen runs): a library repo imports vLLM somewhere, so reading every file's imports
	sent BEIR (retrieval) and a smolagents example (an API agent) to the serving lens and an API agent to a GPU.

	A package's __init__ that re-exports its submodules (`from .vllm import VLLMEmbed`) is followed only into the
	ones whose names the job's code uses: BEIR's models package re-exports a vLLM embedder beside the SentenceBERT
	the example builds."""
	here = entrypoint.rsplit("/", 1)[0] if "/" in entrypoint else ""
	seen, todo, out = set(), [entrypoint], set()
	used: set[str] = set()
	while todo:
		path = todo.pop()
		if path in seen or path not in f.trees:
			continue
		seen.add(path)
		tree = f.trees[path]
		is_init = path.endswith("__init__.py")
		if not is_init:
			used |= set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", f.sources[path].text))
		pkg = path.rsplit("/", 1)[0].split("/") if "/" in path else []
		for mod, level, names in _all_imports(tree) if path == entrypoint else _top_level_imports(tree):
			if is_init and names and "*" not in names and not (set(names) & used):
				continue  # a re-export the job never names
			full = mod
			if level:  # relative: from the file's own package
				base = pkg[: len(pkg) - (level - 1)] if level > 1 else pkg
				full = ".".join([*base, *([mod] if mod else [])])
			local = _module_paths(full, f.trees, here) if full else []
			if full and not level and not local:
				out.add(full)
			todo.extend(local)
			for n in names:  # "from pkg import sub": sub may be a module
				if not is_init or n in used or n == "*":
					todo.extend(_module_paths(f"{full}.{n}" if full else n, f.trees, here))
	return out


def _job_framework(f: Facts, reach: Facts, entrypoint: str) -> str:
	"""The entrypoint's own imports first; then what it reaches in the repo (entry_imports)."""
	own = Facts(
		**{**f.__dict__, "imports": {m for m, lvl, _ in _all_imports(f.trees[entrypoint]) if m and not lvl}}
	)
	fw = _framework(own)
	if fw not in ("plain_pytorch", "other"):
		return fw
	return _framework(reach)


def map_static(files: dict[str, str], *, entrypoint: str) -> ScriptMap:
	if entrypoint not in files:
		raise ScriptParseError(f"entrypoint {entrypoint} is not among the submitted files")
	files, _ = mapped_files(files, entrypoint)
	f = _parse(files)
	_collect_knobs(f)
	knobs = _knobs(f)
	by_id = {k.id: k for k in knobs}
	# the job's own imports decide its framework and lens, not every file in the repo (B, Nov 2)
	reach = Facts(**{**f.__dict__, "imports": entry_imports(f, entrypoint)})
	framework = _job_framework(f, reach, entrypoint)
	hf_id, pcall, ppath = _pretrained(f)
	lens = _lens(reach, framework, hf_id)
	lens = _lens_by_entry(lens, reach, f, entrypoint)
	model = _model(f, hf_id, pcall, ppath)
	data = _data(f, by_id)
	loop = _loop(f, framework)
	risks: list[str] = []
	all_text = "\n".join(s.text for s in f.sources.values())

	def num(k):
		v = by_id.get(k)
		return (
			v.current
			if v is not None and isinstance(v.current, int | float) and not isinstance(v.current, bool)
			else None
		)

	training_lenses = ("finetune", "pretrain", "rl", "vision", "gnn", "retrieval")
	no_grad_hits = _forward_without_no_grad(f) if lens in training_lenses else []
	for h in no_grad_hits:
		risks.append(f"Eval forward in {h} runs without torch.no_grad(); activations for backward are kept")

	training = None
	if lens in training_lenses:
		precision = _precision(by_id, model)
		training = TrainInfo(
			lr=num("lr"),
			warmup_steps=num("warmup_steps"),
			grad_accum=num("grad_accum"),
			precision=precision,
			max_steps=num("max_steps"),
			epochs=num("epochs"),
			checkpoint_every=num("checkpoint_every"),
			eval_every=num("eval_every"),
			grad_checkpointing=by_id["gradient_checkpointing"].current
			if "gradient_checkpointing" in by_id
			else None,
			eval_under_no_grad=False if no_grad_hits else None,
			distributed=_distributed(f, by_id),
		)
		if data.num_workers == 0:
			risks.append(
				"num_workers=0: the data loader runs in the training process, so the GPU waits on it"
			)
		if precision == "fp32":
			risks.append("Trains in fp32; bf16 autocast is usually several times faster on tensor-core GPUs")
		if data.batch_size and data.eval_batch_size and data.eval_batch_size >= 4 * data.batch_size:
			risks.append(
				f"Eval batch {data.eval_batch_size} is {data.eval_batch_size // data.batch_size}x the train batch"
			)
		if data.packing is False and data.padding:
			risks.append(f"No sequence packing; batches are padded ({data.padding})")
		ep = num("epochs")
		if ep and ep > 1:
			risks.append(
				f"Trains for {ep:g} epochs, so every document repeats {ep:g} times; repeated low-quality data costs more "
				"than unique data (give per-document quality scores for the repetition view)"
			)
	elif lens in ("eval", "serving"):
		training = None

	if (
		lens == "retrieval"
		and "MultipleNegativesRankingLoss" in all_text
		and data.batch_size
		and data.batch_size < 64
	):
		risks.append(
			f"Contrastive loss with in-batch negatives at batch {data.batch_size}: each query sees only "
			f"{data.batch_size - 1} negatives"
		)
	if lens == "vision":
		# B, Nov 2: the zoo script's header names the finding ("vision.no_channels_last"), and any comment would do
		# the same, so only a real use of the format counts as using it
		if not re.search(r"\btorch\.channels_last\b|memory_format\s*=\s*(?:\w+\.)*channels_last", all_text):
			risks.append(
				"No channels_last memory format; convolutions usually run faster with it on tensor cores"
			)
		if re.search(r"transforms\.(Random|Color|Auto|Rand)", all_text) and (data.num_workers or 0) <= 4:
			risks.append(
				f"Image augmentation runs on the CPU with {data.num_workers or 0} workers; it can starve the GPU"
			)
	if by_id.get("dump_hidden_states") and by_id["dump_hidden_states"].current is True:
		risks.append(
			"Dumps hidden states for every layer; activation storage grows with tokens x layers x width"
		)
	if lens == "agent":
		for where in f.retry_loops:
			risks.append(f"Tool call retries without a cap ({where}: while True / except: continue)")
		if num("tool_retry_cap") and num("tool_retry_cap") >= 50:
			risks.append(
				f"Tool retries capped at {num('tool_retry_cap'):g}: a failing tool can burn turns and tokens"
			)

	rl = None
	for fn, call, _path in f.calls:
		short = fn.rsplit(".", 1)[-1]
		if short in RL_TRAINERS:
			rf = _kwargs(call).get("reward_funcs")
			names = [dotted(e) for e in rf.elts] if isinstance(rf, ast.List | ast.Tuple) else []
			vllm = by_id.get("use_vllm")
			backend = "vllm" if (vllm is not None and vllm.current is True) or _has(f, "vllm") else "hf"
			rl = RLInfo(
				algorithm=RL_TRAINERS[short],
				group_size=num("group_size"),
				max_completion_len=num("max_completion_len"),
				reward_functions=[n for n in names if n],
				generation_backend=backend,
			)
			break

	agent = None
	if lens == "agent":
		model_name = None
		for fn, call, path in f.calls:
			if fn.endswith(AGENT_CALLS):
				v = _kwargs(call).get("model")
				r = _resolve(f, v, path) if v is not None else None
				if r and isinstance(r[0], str):
					model_name = r[0]
					break
		client = "anthropic" if _has(f, "anthropic") else "litellm" if _has(f, "litellm") else "openai"
		agent = AgentInfo(
			client=client,
			model=model_name,
			max_turns=num("max_turns"),
			n_tasks=num("n_tasks") if isinstance(num("n_tasks"), int) else None,
		)

	return ScriptMap(
		lens=lens,
		framework=framework,
		entrypoint=entrypoint,
		launch=f"python {entrypoint}",
		model=model,
		data=data if lens != "agent" else None,
		training=training,
		loop=loop,
		knobs=knobs,
		agent=agent,
		rl=rl,
		risks=risks,
		produced_by="adapter",
		confidence=0.6 if loop is None else 0.85,
	)
