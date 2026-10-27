"""Write the development record the workspace is built against before B's engine lands.

	uv run python web/scripts/make_dev_record.py

The record is validated by the shared Pydantic schema (FlightRecord) and encoded with
the shared grid encoder, so it is a valid v0.1 record. Its numbers come from a rough
analytical model (weights, LoRA optimiser state, Korthikanti activations, logits, an eval
spike, roofline throughput) and are NOT measured: every model card says so, and the UI
labels records whose predictions come only from fixture models.

Why it exists beside the golden records: the golden fine-tune grids only batch and
sequence length, so the planted eval OOM cannot be fixed on screen until B grids the eval
batch. This record grids eval batch, GPU and gradient checkpointing too, plus per-phase
step time and per-term memory under the target ids C asked B for in docs/requests.md.

Planted problems match the bug-zoo fine-tune: eval batch too large and eval without
no_grad (eval keeps activations), no sequence packing.
"""

import itertools
import json
import math
from pathlib import Path

import numpy as np

from preflight.schema import FlightRecord, dump_record
from preflight.schema.encoding import choose_scale, encode_f16, encode_mask

OUT = Path(__file__).resolve().parents[1] / "src" / "workspace" / "fixtures" / "dev-finetune.record.json"
GIB = 1024**3
MODEL_ID = "dev-fixture@1"

SEQ = [256, 512, 1024, 2048, 3072, 4096]
BATCH = [1, 2, 4, 8, 16, 32]
EVAL_BATCH = [1, 8, 16, 32]
GPUS = ["A100-80GB", "L40S", "H100-SXM-80GB"]
CKPT = [False, True]

GPU_SPEC = {
	# name: (memory bytes, bf16 dense TFLOPs, $ per hour on demand)
	"A100-80GB": (80 * GIB, 312, 2.50),
	"L40S": (48 * GIB, 362, 1.95),
	"H100-SXM-80GB": (80 * GIB, 989, 3.95),
}
MFU_PEAK = {"A100-80GB": 0.46, "L40S": 0.34, "H100-SXM-80GB": 0.41}

PARAMS = 7.0e9
LAYERS, HIDDEN, VOCAB = 32, 4096, 32000
LORA_R = 16
ACT_BYTES = 24  # bytes per token per hidden unit per layer, bf16 with fused attention
CONTEXT = 0.9 * GIB
DATASET_TOKENS = 40e6
PADDING_WASTE = 0.38
EVAL_EVERY = 500
CHECKPOINT_EVERY = 1000
EVAL_TOKENS = 2e6
PROBED = {"seq_len": (512, 2048), "batch_size": (2, 8), "eval_batch_size": (1, 16)}

PHASES = ["data", "forward", "backward", "optimizer", "eval"]
TERMS = ["weights", "gradients", "optimizer", "activations", "logits", "context", "eval"]


def model(seq, batch, eval_batch, gpu, ckpt):
	mem_cap, tflops, price = GPU_SPEC[gpu]
	weights = PARAMS * 2
	lora = LAYERS * 4 * 2 * HIDDEN * LORA_R
	gradients = lora * 2
	optimizer = lora * 12  # fp32 master plus Adam m and v
	per_layer = seq * batch * HIDDEN * ACT_BYTES
	activations = per_layer * LAYERS if not ckpt else seq * batch * HIDDEN * 2 * LAYERS + per_layer
	logits = batch * seq * VOCAB * 4 * 2
	eval_full = eval_batch * seq * HIDDEN * ACT_BYTES * LAYERS + eval_batch * seq * VOCAB * 4 * 2
	train_peak = weights + gradients + optimizer + activations + logits + CONTEXT
	eval_peak = weights + gradients + optimizer + eval_full + CONTEXT
	peak = max(train_peak, eval_peak)
	eval_extra = max(0.0, eval_peak - train_peak)
	p_oom = 1 / (1 + math.exp(-(peak - 0.96 * mem_cap) / (0.025 * mem_cap)))

	tokens = seq * batch
	mfu = MFU_PEAK[gpu] * (1 - math.exp(-tokens / 3000))
	tps = tflops * 1e12 * mfu / (4 * PARAMS * (1.33 if ckpt else 1.0))
	fwd_bwd = tokens / tps
	t_forward = fwd_bwd * (0.42 if ckpt else 0.34)
	t_backward = fwd_bwd - t_forward
	t_data = 0.004 + 0.000002 * tokens
	t_optimizer = 0.012
	eval_time = EVAL_TOKENS / (tps * 3 * (1 + eval_batch / 64))
	t_eval = eval_time / EVAL_EVERY
	step = t_data + t_forward + t_backward + t_optimizer + t_eval
	steps = DATASET_TOKENS / (tokens * (1 - PADDING_WASTE))
	total = steps * step
	return {
		"memory.peak": (peak, 0.08),
		"risk.oom": (p_oom, 0.0),
		"time.total": (total, 0.15),
		"cost.total": (total / 3600 * price, 0.15),
		"risk.divergence": (min(0.6, 0.03 + 0.12 / batch), 0.5),
		"probe.mfu": mfu,
		"throughput.tokens_per_s": (tokens * (1 - PADDING_WASTE) / step, 0.12),
		"memory.term.weights": (weights, 0.02),
		"memory.term.gradients": (gradients, 0.05),
		"memory.term.optimizer": (optimizer, 0.05),
		"memory.term.activations": (activations, 0.12),
		"memory.term.logits": (logits, 0.05),
		"memory.term.context": (CONTEXT, 0.1),
		"memory.term.eval": (eval_extra, 0.15),
		"time.phase.data": (t_data, 0.2),
		"time.phase.forward": (t_forward, 0.15),
		"time.phase.backward": (t_backward, 0.15),
		"time.phase.optimizer": (t_optimizer, 0.2),
		"time.phase.eval": (t_eval, 0.2),
	}


TARGETS = [
	("memory.peak", "bytes", "memory"),
	("time.total", "s", "time"),
	("cost.total", "usd", "cost"),
	("risk.oom", "probability", "risk"),
	("risk.divergence", "probability", None),
	("throughput.tokens_per_s", "tokens/s", None),
	*[(f"memory.term.{t}", "bytes", None) for t in TERMS],
	*[(f"time.phase.{p}", "s", None) for p in PHASES],
]


def abstains(seq, batch, eval_batch):
	# Past 2x the largest probed batch x sequence length, the model doesn't trust itself.
	big = 2 * PROBED["seq_len"][1]
	return (
		seq * batch > big * PROBED["batch_size"][1] or seq * eval_batch > big * PROBED["eval_batch_size"][1]
	)


def grids():
	axes_values = [SEQ, BATCH, EVAL_BATCH, GPUS]
	axes = [
		{"knob_id": "seq_len", "values": SEQ, "scale": "log"},
		{"knob_id": "batch_size", "values": BATCH, "scale": "log"},
		{"knob_id": "eval_batch_size", "values": EVAL_BATCH, "scale": "log"},
		{"knob_id": "gpu", "values": GPUS, "scale": "categorical"},
	]
	out = []
	for ckpt in CKPT:
		cells = list(itertools.product(*axes_values))
		results = [model(s, b, e, g, ckpt) for s, b, e, g in cells]
		mask = [abstains(s, b, e) for s, b, e, _ in cells]
		for tid, unit, _ in TARGETS:
			pts = np.array([r[tid][0] for r in results])
			rel = np.array([r[tid][1] for r in results])
			lo, hi = pts * (1 - rel), pts * (1 + rel)
			if unit == "probability":
				lo, hi = np.clip(lo, 0, 1), np.clip(hi, 0, 1)
			scale = choose_scale(pts, lo, hi)
			out.append(
				{
					"id": f"g.{tid}.{'ckpt' if ckpt else 'nockpt'}",
					"target_id": tid,
					"unit": unit,
					"axes": axes,
					"point": encode_f16(pts, scale),
					"lo": encode_f16(lo, scale),
					"hi": encode_f16(hi, scale),
					"abstain": encode_mask(mask),
					"model_id": MODEL_ID,
					"scale": scale,
					"fixed": {"gradient_checkpointing": ckpt, "lora_r": LORA_R, "precision": "bf16"},
				}
			)
	return out


CURRENT = {
	"seq_len": 2048,
	"batch_size": 4,
	"eval_batch_size": 32,
	"gpu": "A100-80GB",
	"gradient_checkpointing": False,
}


def estimates():
	r = model(2048, 4, 32, "A100-80GB", False)
	out = {}
	for tid, unit, _ in TARGETS:
		p, rel = r[tid]
		lo, hi = p * (1 - rel), p * (1 + rel)
		if unit == "probability":
			lo, hi = max(0.0, lo), min(1.0, hi)
		# eval batch 32 is outside the probed range (1 to 16)
		out[tid] = {
			"point": p,
			"lo": lo,
			"hi": hi,
			"unit": unit,
			"status": "extrapolated",
			"model_id": MODEL_ID,
			"source": "grid",
			"reason": "eval batch above the probed range",
			"level": 0.9,
		}
	return out


def probe(pid, seq, batch, eval_batch, ckpt):
	r = model(seq, batch, eval_batch, "A100-80GB", ckpt)
	terms = {t: round(r[f"memory.term.{t}"][0]) for t in TERMS if t != "eval"}
	step = sum(r[f"time.phase.{p}"][0] for p in PHASES if p != "eval")
	return {
		"id": pid,
		"knobs": {
			"seq_len": seq,
			"batch_size": batch,
			"eval_batch_size": eval_batch,
			"gradient_checkpointing": ckpt,
		},
		"status": "ok",
		"steps": 7,
		"duration_s": 95.0,
		"memory": {
			"peak_allocated_bytes": round(r["memory.peak"][0]),
			"device_total_bytes": 80 * GIB,
			"by_term": terms,
			"peak_step": 2,
		},
		"step_time": {
			"n_steps": 5,
			"p50_s": step,
			"by_phase": {p: r[f"time.phase.{p}"][0] / step for p in PHASES if p != "eval"},
			"unprofiled": True,
		},
		"throughput": {"tokens_per_s": r["throughput.tokens_per_s"][0], "mfu": r["probe.mfu"]},
		"data": {
			"padding_fraction": PADDING_WASTE,
			"packing": False,
			"num_workers": 2,
			"loader_wait_fraction": 0.03,
		},
		"eval": {"batches": 2, "batch_size": eval_batch, "grad_enabled": True},
		"cost": {"gpu_seconds": 95.0, "usd": 95 / 3600 * 2.5},
	}


def graph():
	def n(nid, label, kind, parent=None, repeat=1, metrics=()):
		node = {"id": nid, "label": label, "kind": kind, "repeat": repeat, "metrics": list(metrics)}
		if parent:
			node["parent"] = parent
		return node

	nodes = [
		n(
			"data",
			"Support tickets, padded to longest",
			"data",
			metrics=["time.phase.data", "probe.data.padding_fraction"],
		),
		n("model", "Llama-style 7B", "group", metrics=["time.phase.forward", "memory.term.activations"]),
		n("embed", "Embedding", "module", parent="model", metrics=["memory.term.weights"]),
		n("block", "Decoder block", "group", parent="model", repeat=32),
		n("attn", "Self-attention + LoRA", "module", parent="block"),
		n("mlp", "MLP + LoRA", "module", parent="block"),
		n("lm_head", "LM head", "module", parent="model", metrics=["memory.term.logits"]),
		n("loss", "Cross-entropy", "loss"),
		n(
			"optimizer",
			"AdamW (LoRA params)",
			"optimizer",
			metrics=["time.phase.optimizer", "memory.term.optimizer"],
		),
		n("eval", "Eval loop", "phase", metrics=["risk.oom", "memory.term.eval", "time.phase.eval"]),
		n("ckpt", "Checkpoints", "io"),
	]
	edges = [
		{"source": "data", "target": "embed", "kind": "tokens", "metric": "throughput.tokens_per_s"},
		{"source": "embed", "target": "attn", "kind": "data"},
		{"source": "attn", "target": "mlp", "kind": "data"},
		{"source": "mlp", "target": "lm_head", "kind": "data"},
		{"source": "lm_head", "target": "loss", "kind": "data"},
		{"source": "loss", "target": "optimizer", "kind": "grad"},
		{"source": "optimizer", "target": "attn", "kind": "weights"},
		{"source": "data", "target": "eval", "kind": "control"},
		{"source": "optimizer", "target": "ckpt", "kind": "control"},
	]
	return {"nodes": nodes, "edges": edges, "skeleton": False}


def knobs():
	return [
		{
			"id": "seq_len",
			"label_key": "knob.seq_len",
			"unit": "tokens",
			"kind": "int",
			"domain": {"min": 256, "max": 4096, "log": True},
			"current": 2048,
			"probed_range": {"min": 512, "max": 2048},
			"gridded": True,
			"group": "memory",
		},
		{
			"id": "batch_size",
			"label_key": "knob.batch_size",
			"unit": "samples",
			"kind": "int",
			"domain": {"min": 1, "max": 32, "log": True},
			"current": 4,
			"probed_range": {"min": 2, "max": 8},
			"gridded": True,
			"group": "memory",
		},
		{
			"id": "eval_batch_size",
			"label_key": "knob.eval_batch_size",
			"unit": "samples",
			"kind": "int",
			"domain": {"min": 1, "max": 32, "log": True},
			"current": 32,
			"probed_range": {"min": 1, "max": 16},
			"gridded": True,
			"group": "memory",
		},
		{
			"id": "gpu",
			"label_key": "knob.gpu",
			"kind": "categorical",
			"domain": {"values": GPUS},
			"current": "A100-80GB",
			"gridded": True,
			"group": "cluster",
		},
		{
			"id": "gradient_checkpointing",
			"label_key": "knob.gradient_checkpointing",
			"kind": "bool",
			"current": False,
			"gridded": True,
			"needs_code": False,
			"group": "memory",
		},
		{
			"id": "lora_r",
			"label_key": "knob.lora_r",
			"kind": "int",
			"domain": {"min": 4, "max": 128, "log": True},
			"current": LORA_R,
			"group": "model",
		},
		{
			"id": "precision",
			"label_key": "knob.precision",
			"kind": "categorical",
			"domain": {"values": ["bf16", "fp16", "fp32"]},
			"current": "bf16",
			"group": "speed",
		},
		{
			"id": "packing",
			"label_key": "knob.packing",
			"kind": "bool",
			"current": False,
			"needs_code": True,
			"patch_template": "data.enable_packing",
			"group": "data",
		},
		{
			"id": "eval_no_grad",
			"label_key": "knob.eval_no_grad",
			"kind": "bool",
			"current": False,
			"needs_code": True,
			"patch_template": "eval.wrap_no_grad",
			"group": "memory",
		},
	]


def main():
	record = {
		"schema_version": "0.1.0",
		"record_id": "01JBDEVFINETUNE00000000000",
		"share_token": "dev-finetune-share-token",
		"created_at": "2026-10-27T03:00:00Z",
		"job": {
			"lens": "finetune",
			"source": {
				"kind": "sample",
				"uri": "sample://dev/lora-7b",
				"entrypoint": "zoo/finetune_lora_hf/train.py",
			},
			"framework": "hf_trainer",
			"title": "LoRA fine-tune of a 7B model on support tickets",
			"script_map": {
				"lens": "finetune",
				"framework": "hf_trainer",
				"entrypoint": "zoo/finetune_lora_hf/train.py",
				"model": {
					"name": "Llama-style 7B",
					"family": "decoder",
					"param_count": int(PARAMS),
					"n_layers": LAYERS,
					"d_model": HIDDEN,
					"vocab_size": VOCAB,
					"dtype": "bfloat16",
					"lora": {"r": LORA_R, "alpha": 32},
				},
				"data": {
					"dataset": "your-org/support-tickets",
					"batch_size": 4,
					"eval_batch_size": 32,
					"seq_len": 2048,
					"num_workers": 2,
					"packing": False,
					"padding": "longest",
				},
				"training": {
					"precision": "bf16",
					"max_steps": 6000,
					"eval_every": EVAL_EVERY,
					"checkpoint_every": CHECKPOINT_EVERY,
					"grad_checkpointing": False,
					"eval_under_no_grad": False,
				},
				"risks": [
					"Eval runs without torch.no_grad()",
					"Eval batch is 8x the train batch",
					"No sequence packing",
				],
				"produced_by": "manual",
				"confidence": 1.0,
			},
			"objective": {
				"kind": "min_cost",
				"description": "Fine-tune a 7B model on our support tickets, one A100, done by Friday, cheapest possible.",
			},
			"constraints": {
				"budget_usd": 60,
				"deadline": "2026-10-30T17:00:00-04:00",
				"gpu_types": GPUS,
				"max_gpus": 1,
			},
		},
		"env": {
			"backend": "cuda",
			"provider": "modal",
			"gpu": {"name": "A100-80GB", "count": 1, "memory_bytes": 80 * GIB, "vendor": "nvidia"},
			"libraries": {"torch": "2.5.1", "transformers": "4.46.2", "peft": "0.13.2"},
			"seed": 0,
		},
		"graph": graph(),
		"knobs": knobs(),
		"probes": [
			probe("p1", 2048, 4, 8, False),
			probe("p2", 2048, 2, 8, False),
			probe("p3", 1024, 4, 8, False),
		],
		"predictions": {
			"targets": [
				{"id": t, "label_key": f"target.{t}", "unit": u, **({"readout": r} if r else {})}
				for t, u, r in TARGETS
			],
			"estimates": estimates(),
			"grids": grids(),
		},
		"models": [
			{
				"id": MODEL_ID,
				"kind": "analytical",
				"version": "1",
				"training_data": "none",
				"notes": "Hand-made fixture for UI work; not a real model and not measured",
			}
		],
		"findings": [
			{
				"id": "f1",
				"severity": "critical",
				"code": "eval.no_grad",
				"title": "Eval runs with gradients kept",
				"detail": "The eval loop calls the model outside torch.no_grad(), so it keeps training-sized activations at the eval batch size.",
				"node_id": "eval",
				"evidence": [{"metric": "script.eval_under_no_grad", "value": "false"}],
				"source": "rule",
				"fix_template": "eval.wrap_no_grad",
				"knob_id": "eval_no_grad",
			},
			{
				"id": "f2",
				"severity": "error",
				"code": "eval.batch_oom",
				"title": "First eval will run out of memory",
				"node_id": "eval",
				"evidence": [
					{
						"metric": "memory.peak@eval_batch_size=32",
						"value": model(2048, 4, 32, "A100-80GB", False)["memory.peak"][0],
						"threshold": 80 * GIB,
					}
				],
				"source": "rule",
				"fix_template": "eval.reduce_batch",
				"knob_id": "eval_batch_size",
			},
			{
				"id": "f3",
				"severity": "warn",
				"code": "data.padding_waste",
				"title": "38% of tokens are padding",
				"node_id": "data",
				"evidence": [
					{
						"metric": "probe.data.padding_fraction",
						"value": PADDING_WASTE,
						"threshold": 0.2,
						"probe_id": "p1",
					}
				],
				"source": "rule",
				"fix_template": "data.enable_packing",
				"knob_id": "packing",
			},
		],
		"costs": {
			"gpu_usd": 3 * 95 / 3600 * 2.5,
			"llm_usd": 0.0,
			"breakdown": {"probes": 3 * 95 / 3600 * 2.5},
		},
	}
	rec = FlightRecord.model_validate(record)
	raw = dump_record(rec)
	cells = sum(g.cells for g in rec.predictions.grids)
	assert cells <= 16_384, f"{cells} grid cells; the record budget is 16,384"
	OUT.parent.mkdir(parents=True, exist_ok=True)
	OUT.write_text(json.dumps(json.loads(raw), indent="\t") + "\n")
	print(
		f"wrote {OUT.name}: {len(raw)} bytes raw, {len(dump_record(rec, gzipped=True))} gzipped, {cells} grid cells"
	)


if __name__ == "__main__":
	main()
