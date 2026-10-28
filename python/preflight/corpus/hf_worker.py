"""Real Hugging Face fine-tuning jobs for the corpus (Review, Oct 27): LoRA in bf16 and QLoRA in 4-bit.

Each config loads a public model, wraps it with PEFT LoRA (and bitsandbytes NF4 for QLoRA), and
measures what A's probes measure: peak reserved memory after two optimizer steps, and the median of
five timed steps split into forward, backward and optimizer. Inputs are random token ids, so no dataset
download is needed; the weights download from the Hub at run time (ungated models only).

Runs in corpus.isolate's child process, like every corpus config.
"""

from __future__ import annotations

import contextlib
import itertools
import statistics
import time
from typing import Any

import torch

from preflight.corpus.configs import CorpusConfig

WARMUP_STEPS = 2  # cited: A's probe-plan D1
TIMED_STEPS = 5  # cited: A's probe-plan D1
SHAPE_FIELDS = (
	"num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads", "intermediate_size",
	"vocab_size", "head_dim", "tie_word_embeddings", "max_position_embeddings",
)  # fmt: skip


def _versions() -> dict[str, str]:
	import importlib.metadata as md

	out = {"torch": torch.__version__}
	for pkg in ("transformers", "peft", "bitsandbytes", "accelerate"):
		with contextlib.suppress(md.PackageNotFoundError):
			out[pkg] = md.version(pkg)
	return out


def model_shape(config) -> dict[str, Any]:
	"""The fields the engine's ModelInfo reads, from a Hugging Face config."""
	shape = {k: getattr(config, k) for k in SHAPE_FIELDS if getattr(config, k, None) is not None}
	shape["model_type"] = getattr(config, "model_type", None)
	act = str(getattr(config, "hidden_act", "")).lower()
	shape["gated_mlp"] = (
		getattr(config, "model_type", "") in ("llama", "qwen2", "mistral", "gemma", "phi3") or "silu" in act
	)
	return shape


def run_hf(cfg: CorpusConfig, device: torch.device) -> dict[str, Any]:
	from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
	from transformers import AutoModelForCausalLM, BitsAndBytesConfig

	p = cfg.params
	if device.type == "cuda":
		torch.cuda.reset_peak_memory_stats(device)
	kw: dict[str, Any] = {"dtype": torch.bfloat16}
	if p["quantization"] == "nf4":
		kw["quantization_config"] = BitsAndBytesConfig(
			load_in_4bit=True,
			bnb_4bit_quant_type="nf4",
			bnb_4bit_compute_dtype=torch.bfloat16,
			bnb_4bit_use_double_quant=False,
		)
		kw["device_map"] = {"": device.index or 0} if device.type == "cuda" else None
	model = AutoModelForCausalLM.from_pretrained(p["model_id"], **kw)
	if p["quantization"] == "nf4":
		model = prepare_model_for_kbit_training(
			model, use_gradient_checkpointing=bool(p["grad_checkpointing"])
		)
	else:
		model = model.to(device)
		if p["grad_checkpointing"]:
			model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
	model.config.use_cache = False
	lora = LoraConfig(
		r=p["lora_rank"],
		lora_alpha=p["lora_alpha"],
		target_modules=p["lora_targets"],
		lora_dropout=0.0,
		task_type="CAUSAL_LM",
	)
	model = get_peft_model(model, lora)
	model.train()
	params = [t for t in model.parameters() if t.requires_grad]
	opt = torch.optim.AdamW(params, lr=1e-4)
	gen = torch.Generator().manual_seed(0)
	vocab = model.config.vocab_size
	ids = torch.randint(0, vocab, (p["batch"], p["seq"]), generator=gen).to(device)

	def step(marks: list | None = None) -> None:
		def mark():
			if marks is not None:
				if device.type == "cuda":
					ev = torch.cuda.Event(enable_timing=True)
					ev.record()
					marks.append(ev)
				else:
					marks.append(time.perf_counter())

		opt.zero_grad(set_to_none=True)
		mark()
		loss = model(input_ids=ids, labels=ids).loss
		mark()
		loss.backward()
		mark()
		opt.step()
		mark()

	for _ in range(WARMUP_STEPS):
		step()
	if device.type == "cuda":
		torch.cuda.synchronize(device)
	runs, times = [], []
	for _ in range(TIMED_STEPS):
		marks: list = []
		t0 = time.perf_counter()
		step(marks)
		if device.type == "cuda":
			torch.cuda.synchronize(device)
			sec = [a.elapsed_time(b) / 1e3 for a, b in itertools.pairwise(marks)]
		else:
			sec = [b - a for a, b in itertools.pairwise(marks)]
		times.append(time.perf_counter() - t0)
		runs.append(dict(zip(("forward", "backward", "optimizer"), sec, strict=True)))
	memory = None
	if device.type == "cuda":
		memory = {
			"peak_allocated": torch.cuda.max_memory_allocated(device),
			"peak_reserved": torch.cuda.max_memory_reserved(device),
		}
	return {
		"status": "ok",
		"n_params": sum(t.numel() for t in model.parameters()),
		"n_trainable": sum(t.numel() for t in params),
		"work_units": p["batch"] * p["seq"],
		"model_shape": model_shape(model.config),
		"library_versions": _versions(),
		"timing": {
			"step_s": statistics.median(times),
			"step_s_all": times,
			"phases": {
				k: statistics.median(r[k] for r in runs) for k in ("forward", "backward", "optimizer")
			},
			"n_steps": TIMED_STEPS,
			"source": "unprofiled steps",
		},
		"memory": memory,
	}
