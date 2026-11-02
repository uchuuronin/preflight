"""FakeTensor cross-check of the engine's analytical memory terms on the zoo models (A's item 4; PM, Nov 1).

Each zoo model is built under torch's FakeTensorMode, so nothing is allocated and no weights are downloaded (the
shapes come from the records' script maps). One training step runs on fake inputs: a forward with
`saved_tensors_hooks` recording every tensor autograd keeps for the backward, then the backward and one optimizer
step. Each of the engine's terms is compared with what the trace holds:

- weights: every parameter's bytes; gradients: the trainable parameters' bytes; optimizer: the optimizer's state
  after one step (AdamW's two moments, Adafactor's factored rows and columns);
- activations: the tensors saved for backward, less the parameters and the vocabulary-wide ones;
- logits: the vocabulary-wide tensors saved for the loss, plus the logits themselves.

Per term, over a grid of settings for each model, the report gives the median of engine / trace and how many
settings land within 20%, with a Wilson 95% interval. Writes python/preflight/engine/_faketensor_check.py, which the
engine reads to name a term that disagrees in its memory reason.

    uv run --with transformers==4.46.3 --with peft==0.13.2 python scripts/faketensor_memory_check.py

The trace is torch's own accounting of one eager step on CPU semantics: it sees what autograd keeps, not the CUDA
caching allocator, fused kernels or fragmentation, so it checks the terms, not the peak.
"""

from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import pprint
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

TERMS = ("weights", "gradients", "optimizer", "activations", "logits")
WITHIN = 0.20  # cited: main plan, the memory target is within 20%
OUT = ROOT / "python/preflight/engine/_faketensor_check.py"
LORA_RECORD = ROOT / "web/src/workspace/fixtures/lora-live-nov1.record.json"
GRPO_RECORD = ROOT / "zoo/grpo_trl/record_nov1_a10.json"
# Qwen/Qwen2.5-0.5B-Instruct's config.json (the GRPO record names the model but not its shape)
QWEN05 = {
	"n_layers": 24, "d_model": 896, "n_heads": 14, "n_kv_heads": 2, "d_ff": 4864, "vocab_size": 151936,
	"head_dim": 64, "gated_mlp": True, "tie_embeddings": True, "norm": "rmsnorm", "attn_impl": "sdpa",
	"dtype": "bfloat16", "max_positions": 32768, "family": "decoder", "param_count": 494032768,
}  # fmt: skip
# zoo/pretrain_nanogpt/train.py's module constants
NANOGPT = {"n_layer": 12, "n_head": 12, "n_embd": 768, "vocab_size": 50304, "block_size": 1024}


# ---------------------------------------------------------------- the trace


def _trace(
	build, batch: int, seq: int, vocab: int, optimizer: str, autocast: bool = False
) -> dict[str, float]:
	"""Bytes per term for one training step of the model `build()` makes, under FakeTensorMode."""
	import torch
	from torch._subclasses.fake_tensor import FakeTensorMode

	saved: dict[int, tuple[int, tuple]] = {}

	def pack(t):
		st = t.untyped_storage()
		saved[st._cdata] = (st.nbytes(), tuple(t.shape))
		return t

	with FakeTensorMode(allow_non_fake_inputs=True):
		model, forward = build()
		params = list(model.parameters())
		pkeys = {p.untyped_storage()._cdata for p in params}
		weights = sum(p.numel() * p.element_size() for p in params)
		trainable = [p for p in params if p.requires_grad]
		grads = sum(p.numel() * p.element_size() for p in trainable)
		ids = torch.zeros(batch, seq, dtype=torch.long)
		amp = torch.autocast("cpu", dtype=torch.bfloat16) if autocast else contextlib.nullcontext()
		with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t), amp:
			logits, loss = forward(model, ids)
		loss.backward()
		if optimizer == "adafactor":
			from transformers.optimization import Adafactor

			opt = Adafactor(trainable, lr=1e-6, relative_step=False, scale_parameter=False)
		else:
			opt = torch.optim.AdamW(trainable, lr=1e-4)
		opt.step()
		state = sum(
			v.numel() * v.element_size()
			for s in opt.state.values()
			for v in s.values()
			if isinstance(v, torch.Tensor) and v.dim() > 0
		)
	vocab_wide = sum(n for k, (n, shape) in saved.items() if k not in pkeys and shape and shape[-1] == vocab)
	act = sum(n for k, (n, shape) in saved.items() if k not in pkeys and not (shape and shape[-1] == vocab))
	logit_bytes = logits.numel() * logits.element_size()
	return {
		"weights": float(weights),
		"gradients": float(grads),
		"optimizer": float(state),
		"activations": float(act),
		"logits": float(vocab_wide + logit_bytes),
	}


def _qwen(shape: dict, lora: dict | None, checkpointing: bool):
	def build():
		import peft.tuners.tuners_utils as tu
		import torch
		from transformers import Qwen2Config, Qwen2ForCausalLM

		cfg = Qwen2Config(
			hidden_size=shape["d_model"], intermediate_size=shape["d_ff"], num_hidden_layers=shape["n_layers"],
			num_attention_heads=shape["n_heads"], num_key_value_heads=shape["n_kv_heads"],
			vocab_size=shape["vocab_size"], tie_word_embeddings=shape["tie_embeddings"],
			max_position_embeddings=shape.get("max_positions") or 32768, attn_implementation="sdpa",
			use_cache=False,
		)  # fmt: skip
		torch.set_default_dtype(torch.bfloat16)
		try:
			model = Qwen2ForCausalLM(cfg)
			if lora:
				from peft import LoraConfig, get_peft_model

				# peft moves each new adapter to its base layer's device, a swap fake parameters don't allow; they
				# are already on that (fake) device, so the moves are skipped while it wraps the model
				move, tu.BaseTunerLayer._move_adapter_to_device_of_base_layer = (
					tu.BaseTunerLayer._move_adapter_to_device_of_base_layer,
					lambda self, *a, **k: None,
				)
				to, torch.nn.Module.to = torch.nn.Module.to, lambda self, *a, **k: self
				try:
					model = get_peft_model(
						model,
						LoraConfig(
							r=lora["r"], lora_alpha=lora["alpha"], target_modules=lora["target_modules"],
							lora_dropout=lora.get("dropout", 0.0),
						),
					)  # fmt: skip
				finally:
					tu.BaseTunerLayer._move_adapter_to_device_of_base_layer = move
					torch.nn.Module.to = to
		finally:
			torch.set_default_dtype(torch.float32)
		if checkpointing:
			model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
			if lora:
				model.enable_input_require_grads()
		model.train()

		def forward(m, ids):
			out = m(input_ids=ids, labels=ids)
			return out.logits, out.loss

		return model, forward

	return build


def _nanogpt():
	def build():
		import torch
		import torch.nn.functional as F
		from torch import nn

		c = NANOGPT

		class Block(nn.Module):  # zoo/pretrain_nanogpt/train.py's Block, dropout 0
			def __init__(self):
				super().__init__()
				d = c["n_embd"]
				self.ln_1, self.ln_2 = nn.LayerNorm(d), nn.LayerNorm(d)
				self.c_attn, self.c_proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
				self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

			def forward(self, x):
				b, t, d = x.shape
				q, k, v = self.c_attn(self.ln_1(x)).split(d, dim=2)
				q, k, v = (z.view(b, t, c["n_head"], d // c["n_head"]).transpose(1, 2) for z in (q, k, v))
				y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
				x = x + self.c_proj(y.transpose(1, 2).contiguous().view(b, t, d))
				return x + self.mlp(self.ln_2(x))

		class GPT(nn.Module):
			def __init__(self):
				super().__init__()
				d = c["n_embd"]
				self.wte, self.wpe = nn.Embedding(c["vocab_size"], d), nn.Embedding(c["block_size"], d)
				self.blocks = nn.ModuleList(Block() for _ in range(c["n_layer"]))
				self.ln_f = nn.LayerNorm(d)
				self.lm_head = nn.Linear(d, c["vocab_size"], bias=False)
				self.wte.weight = self.lm_head.weight

		model = GPT()
		model.train()

		def forward(m, ids):
			pos = torch.arange(ids.size(1))
			x = m.wte(ids) + m.wpe(pos)
			for blk in m.blocks:
				x = blk(x)
			logits = m.lm_head(m.ln_f(x))
			return logits, F.cross_entropy(logits.view(-1, logits.size(-1)), ids.view(-1))

		return model, forward

	return build


# ---------------------------------------------------------------- the engine


def _engine_terms(record, values: dict, batch: int, seq: int) -> dict[str, float]:
	"""The engine's terms for this record's model and settings (every knob at its current value, as predict reads
	them), at the same batch and sequence as the trace, on the trace's definitions (review, Nov 1):

	- activations: what is saved for backward. Under full checkpointing the engine's term adds the one layer being
	recomputed in the backward, which the forward's saved tensors never hold, so that layer comes off.
	- logits: what is saved for the loss. The engine's term adds the transient fp32 upcast, alive during the loss
	forward but not saved, so that copy comes off."""
	from preflight.engine import api, knobs, memory

	arch, setup, _shape = knobs.setup_from(api._script_map(record), api._values(record, values))
	t = memory.static_terms(arch, setup)
	act = memory.activation_bytes(arch, setup, batch, seq)
	if setup.grad_checkpointing == "full":
		act -= memory.activation_bytes_per_layer(arch, setup, batch, seq)
	logits = memory.logits_bytes(arch, setup, batch, seq)
	if memory._act_bytes(setup) < 4 and setup.logits_fp32_upcast:
		logits -= batch * seq * getattr(arch, "vocab_size", 0) * 4.0 / max(setup.parallel.tensor, 1)
	return {
		"weights": t.get("weights", 0.0) + t.get("frozen_models", 0.0),
		"gradients": t.get("gradients", 0.0),
		"optimizer": t.get("optimizer", 0.0) + t.get("master_weights", 0.0),
		"activations": act,
		"logits": logits,
	}


def _record(path: Path, model_update: dict | None = None, training_update: dict | None = None):
	from preflight.schema import FlightRecord

	raw = json.loads(path.read_text())
	sm = raw["job"]["script_map"]
	if model_update:
		sm["model"] = {**sm["model"], **model_update}
	if training_update:
		sm["training"] = {**sm["training"], **training_update}
	return FlightRecord.model_validate(raw)


def cases():
	"""(case, key, record, knob values, trace builder, batch, seq, vocab, optimizer, autocast) over each model's grid.
	The key is what the engine matches a record on: the model's shape, method, checkpointing, optimizer family."""
	lora = _record(LORA_RECORD)
	m = lora.job.script_map.model
	lshape = {
		"d_model": m.d_model, "d_ff": m.d_ff, "n_layers": m.n_layers, "n_heads": m.n_heads,
		"n_kv_heads": m.n_kv_heads, "vocab_size": m.vocab_size, "tie_embeddings": m.tie_embeddings,
		"max_positions": m.max_positions,
	}  # fmt: skip
	lcfg = {
		"r": m.lora.r, "alpha": m.lora.alpha, "target_modules": list(m.lora.target_modules),
		"dropout": m.lora.dropout or 0.0,
	}  # fmt: skip
	sig = [m.n_layers, m.d_model, m.vocab_size]
	for gc in (False, True):
		# the zoo job as it runs: bf16 weights, Trainer bf16=True (autocast), the record's LoRA dropout
		name = (
			f"Qwen2.5-1.5B LoRA, bf16 autocast, {'with' if gc else 'no'} checkpointing (zoo/finetune_lora_hf)"
		)
		key = {"model": sig, "method": "lora", "checkpointing": gc, "optimizer": "adamw"}
		for b, s in itertools.product((1, 2, 4, 8), (256, 512, 1024)):
			v = {"batch_size": b, "seq_len": s, "gradient_checkpointing": gc, "eval_batch_size": 1}
			yield (name, key, lora, v, _qwen(lshape, lcfg, gc), b, s, m.vocab_size, "adamw", True)
	grpo = _record(GRPO_RECORD, model_update=QWEN05, training_update={"precision": "bf16"})
	sig05 = [QWEN05["n_layers"], QWEN05["d_model"], QWEN05["vocab_size"]]
	for gc in (False, True):
		# zoo/grpo_trl's policy shape and optimizer, with bf16 weights (the script keeps fp32 weights under autocast)
		name = (
			f"Qwen2.5-0.5B full, bf16 weights, Adafactor, {'with' if gc else 'no'} checkpointing "
			"(zoo/grpo_trl's policy shape)"
		)
		key = {"model": sig05, "method": "full", "checkpointing": gc, "optimizer": "adafactor"}
		for b, s in itertools.product((1, 2, 4, 8), (256, 768)):
			v = {"batch_size": b, "gradient_checkpointing": gc}
			yield (name, key, grpo, v, _qwen(QWEN05, None, gc), b, s, 151936, "adafactor", False)


def _nanogpt_cases():
	from preflight.schema import FlightRecord

	raw = json.loads(LORA_RECORD.read_text())
	sm = raw["job"]["script_map"]
	sm["lens"] = "pretrain"
	sm["model"] = {
		"name": "nanoGPT (zoo)", "family": "decoder", "n_layers": 12, "d_model": 768, "n_heads": 12,
		"n_kv_heads": 12, "d_ff": 3072, "vocab_size": 50304, "head_dim": 64, "max_positions": 1024,
		"gated_mlp": False, "tie_embeddings": True, "norm": "layernorm", "attn_impl": "sdpa", "dtype": "float32",
		"param_count": 124_475_904,
	}  # fmt: skip
	sm["training"] = {**sm["training"], "precision": "fp32", "eval_every": None, "optimizer": "adamw_torch"}
	sm["data"] = {**sm["data"], "eval_batch_size": None}
	raw["job"]["lens"] = "pretrain"
	# the LoRA record's own knobs (lora_r, precision) aren't the zoo nanoGPT's: only the shape knobs stay
	raw["knobs"] = [
		k
		for k in raw["knobs"]
		if k["id"] in ("batch_size", "seq_len", "gradient_checkpointing", "eval_batch_size")
	]
	rec = FlightRecord.model_validate(raw)
	for b, s in itertools.product((1, 4, 12), (256, 512, 1024)):
		v = {"batch_size": b, "seq_len": s, "gradient_checkpointing": False, "eval_batch_size": 1}
		key = {"model": [12, 768, 50304], "method": "full", "checkpointing": False, "optimizer": "adamw"}
		name = "nanoGPT 124M full, fp32 (zoo/pretrain_nanogpt)"
		yield (name, key, rec, v, _nanogpt(), b, s, 50304, "adamw", False)


# ---------------------------------------------------------------- the report


def _wilson(k: int, n: int) -> tuple[float, float]:
	from preflight.engine.conformal import wilson

	return wilson(k, n)


def run() -> dict:
	rows = []
	for case, key, rec, values, build, b, s, vocab, opt, amp in itertools.chain(cases(), _nanogpt_cases()):
		vals = {k: v for k, v in values.items() if any(kn.id == k for kn in rec.knobs)}
		try:
			eng = _engine_terms(rec, vals, b, s)
		except (ValueError, AttributeError, KeyError) as exc:
			print(f"{case} b={b} s={s}: engine has no terms ({exc})")
			continue
		tr = _trace(build, b, s, vocab, opt, autocast=amp)
		rows.append({"case": case, "key": key, "batch": b, "seq": s, "engine": eng, "trace": tr})
		print(case, b, s, {t: round(eng[t] / tr[t], 3) if tr[t] else None for t in TERMS})
	report: dict = {"within": WITHIN, "cases": {}}
	for case in dict.fromkeys(r["case"] for r in rows):
		rs = [r for r in rows if r["case"] == case]
		terms = {}
		for t in TERMS:
			ratios = [r["engine"][t] / r["trace"][t] for r in rs if r["trace"][t] > 0]
			if not ratios:
				continue
			k = sum(1 for x in ratios if abs(x - 1) <= WITHIN)
			lo, hi = _wilson(k, len(ratios))
			terms[t] = {
				"median_ratio": round(statistics.median(ratios), 4),
				"min_ratio": round(min(ratios), 4),
				"max_ratio": round(max(ratios), 4),
				"within": k,
				"n": len(ratios),
				"wilson_95": [round(lo, 4), round(hi, 4)],
				"agrees": abs(statistics.median(ratios) - 1) <= WITHIN,
			}
		report["cases"][case] = {"key": rs[0]["key"], "settings": len(rs), "terms": terms}
	return report


def write(report: dict, out: Path) -> None:
	body = pprint.pformat(report, indent=1, width=110, sort_dicts=True, compact=True)
	out.write_text(
		'"""Generated by scripts/faketensor_memory_check.py; do not edit. The engine\'s analytical memory terms '
		'against a FakeTensor trace of the zoo models."""\n\nDATA = ' + body + "\n"
	)


def main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument("--out", type=Path, default=OUT)
	ap.add_argument("--json", type=Path, help="also write the report as JSON")
	a = ap.parse_args(argv)
	report = run()
	write(report, a.out)
	if a.json:
		a.json.write_text(json.dumps(report, indent=1) + "\n")
	print(json.dumps(report, indent=1))
	return 0


if __name__ == "__main__":
	sys.exit(main())
