"""Generate the hand-made fine-tuning record used to develop the workspace before A's
golden records exist. Numbers come from a rough analytical model (weights, LoRA optimizer
state, Korthikanti activations, logits, eval spike, roofline throughput); they are NOT
measured and the record carries fixture=true so the UI labels it.

Planted problems, matching the bug-zoo fine-tune script: eval batch too large and eval
without no_grad (eval keeps activations), no sequence packing.

Run: python3 web/scripts/make_handmade_record.py > web/src/workspace/fixtures/handmade-finetune.record.json
"""

import base64
import itertools
import json
import math
import sys

import numpy as np

GIB = 1024**3

SEQ = [256, 512, 1024, 1536, 2048, 3072, 4096]
BATCH = [1, 2, 4, 8, 16, 32]
EVAL_BATCH = [1, 8, 16, 32]
GPU = ["A100-80GB", "L40S", "H100"]
CKPT = [False, True]

GPU_SPEC = {
	# name: (memory bytes, bf16 dense TFLOPs, HBM TB/s, $/h)
	"A100-80GB": (80 * GIB, 312, 2.0, 2.50),
	"L40S": (48 * GIB, 362, 0.86, 1.95),
	"H100": (80 * GIB, 989, 3.35, 3.95),
}

PARAMS = 7.0e9
LAYERS, HIDDEN, HEADS, VOCAB = 32, 4096, 32, 32000
LORA_RANK = 16
DATASET_TOKENS = 40e6
PADDING_WASTE = 0.38  # no packing: share of tokens that are padding
EVAL_EVERY = 500
EVAL_TOKENS = 2e6
CONTEXT = 0.9 * GIB
ACT_BYTES = 24  # bytes per token per hidden unit per layer, bf16 with fused attention

PROBED = {"seq_len": (512, 2048), "batch_size": (1, 8), "eval_batch_size": (1, 16)}


def model(seq, batch, eval_batch, gpu, ckpt):
	mem_cap, tflops, _bw, price = GPU_SPEC[gpu]
	weights = PARAMS * 2
	lora_params = LAYERS * 4 * 2 * HIDDEN * LORA_RANK
	grads_opt = lora_params * (2 + 4 + 8)  # bf16 grads, fp32 master, Adam m and v
	per_layer_act = seq * batch * HIDDEN * ACT_BYTES  # flash attention removes the s^2 term
	act = per_layer_act * LAYERS
	if ckpt:
		act = seq * batch * HIDDEN * 2 * LAYERS + per_layer_act
	logits = batch * seq * VOCAB * 4 * 2  # fp32 logits plus softmax copy
	# Planted bug: eval runs without no_grad, so it keeps full activations at eval batch.
	eval_spike = eval_batch * seq * HIDDEN * ACT_BYTES * LAYERS + eval_batch * seq * VOCAB * 4 * 2
	train_peak = weights + grads_opt + act + logits + CONTEXT
	eval_peak = weights + grads_opt + eval_spike + CONTEXT
	peak = max(train_peak, eval_peak)
	eval_extra = max(0.0, eval_peak - train_peak)
	p_oom = 1 / (1 + math.exp(-(peak - 0.96 * mem_cap) / (0.025 * mem_cap)))

	tokens_step = seq * batch
	mfu_peak = {"A100-80GB": 0.46, "L40S": 0.34, "H100": 0.41}[gpu]
	mfu = mfu_peak * (1 - math.exp(-tokens_step / 3000))
	flops_token = 4 * PARAMS * (1.33 if ckpt else 1.0)
	tps = tflops * 1e12 * mfu / flops_token
	t_fwd_bwd = tokens_step / tps
	t_fwd = t_fwd_bwd * (0.42 if ckpt else 0.34)
	t_bwd = t_fwd_bwd - t_fwd
	t_data = 0.004 + 0.000002 * tokens_step
	steps = DATASET_TOKENS / (tokens_step * (1 - PADDING_WASTE))
	eval_time = EVAL_TOKENS / (tps * 3 * (1 + eval_batch / 64))
	t_eval = eval_time / EVAL_EVERY
	step_time = t_data + t_fwd + t_bwd + t_eval
	time_to_finish = steps * step_time
	cost = time_to_finish / 3600 * price
	effective_tps = tokens_step * (1 - PADDING_WASTE) / step_time
	risk = min(0.6, 0.03 + 0.12 / batch)
	return {
		"peak_memory": (peak, 0.08),
		"p_oom": (p_oom, 0.0),
		"time_to_finish": (time_to_finish, 0.15),
		"total_cost": (cost, 0.15),
		"divergence_risk": (risk, 0.5),
		"tokens_per_sec": (effective_tps, 0.12),
		"mfu": (mfu, 0.12),
		"mem_weights": (weights, 0.02),
		"mem_grads_optimizer": (grads_opt, 0.05),
		"mem_activations": (act, 0.12),
		"mem_logits": (logits, 0.05),
		"mem_eval_spike": (eval_extra, 0.15),
		"t_data": (t_data, 0.2),
		"t_forward": (t_fwd, 0.15),
		"t_backward": (t_bwd, 0.15),
		"t_eval": (t_eval, 0.2),
	}


TARGETS = [
	("peak_memory", "bytes", "lower"),
	("p_oom", "prob", "lower"),
	("time_to_finish", "s", "lower"),
	("total_cost", "usd", "lower"),
	("divergence_risk", "prob", "lower"),
	("tokens_per_sec", "tokens/s", "higher"),
	("mfu", "ratio", "higher"),
	("mem_weights", "bytes", "lower"),
	("mem_grads_optimizer", "bytes", "lower"),
	("mem_activations", "bytes", "lower"),
	("mem_logits", "bytes", "lower"),
	("mem_eval_spike", "bytes", "lower"),
	("t_data", "s", "lower"),
	("t_forward", "s", "lower"),
	("t_backward", "s", "lower"),
	("t_eval", "s", "lower"),
]


def f16(arr):
	return base64.b64encode(np.asarray(arr, dtype="<f2").tobytes()).decode()


def main():
	axes = [SEQ, BATCH, EVAL_BATCH, GPU, CKPT]
	names = ["seq_len", "batch_size", "eval_batch_size", "gpu", "grad_ckpt"]
	cells = list(itertools.product(*axes))
	results = [model(*c) for c in cells]
	grids = []
	for tid, unit, _better in TARGETS:
		pts = np.array([r[tid][0] for r in results], dtype=np.float64)
		rel = np.array([r[tid][1] for r in results])
		lo = pts * (1 - rel)
		hi = pts * (1 + rel)
		if unit == "prob":
			lo, hi = np.clip(lo, 0, 1), np.clip(hi, 0, 1)
		abstain = np.zeros(len(cells), dtype=np.uint8)
		for i, (s, b, eb, _g, _c) in enumerate(cells):
			# Abstain beyond 2x the largest probed batch x sequence length.
			if s * b > 2 * PROBED["seq_len"][1] * PROBED["batch_size"][1]:
				abstain[i] = 1
			if s * eb > 2 * PROBED["seq_len"][1] * PROBED["eval_batch_size"][1]:
				abstain[i] = 1
		# float16 tops out at 65504 with ~3 significant digits, so large units are stored
		# scaled (GiB, hours) and the grid's "scale" multiplies them back on read.
		div = GIB if unit == "bytes" else 3600.0 if tid == "time_to_finish" else 1.0
		grids.append(
			{
				"id": f"g_{tid}",
				"target_id": tid,
				"axes": [{"knob_id": n, "values": v} for n, v in zip(names, axes)],
				"point": f16(pts / div),
				"lo": f16(lo / div),
				"hi": f16(hi / div),
				"abstain": base64.b64encode(abstain.tobytes()).decode(),
				"model_id": "analytical-handmade@0",
				"fixed": {"lora_rank": LORA_RANK, "precision": "bf16"},
				"scale": div,
			}
		)
	cell_count = len(cells) * len(TARGETS)
	assert cell_count <= 16384, cell_count

	record = {
		"schema_version": "0.1.0",
		"record_id": "01HANDMADEFINETUNE0000000",
		"parent_id": None,
		"created_at": "2026-10-27T00:00:00Z",
		"fixture": True,
		"job": {
			"lens": "finetune",
			"title": "LoRA fine-tune, 7B on support tickets",
			"source": {"kind": "template", "ref": "zoo/finetune_lora_hf"},
			"framework": "hf-trainer",
			"objective": "cost",
			"constraints": {"budget_usd": 60, "deadline": "2026-10-30T17:00:00-04:00", "gpus": GPU},
			"total_steps": 6000,
			"eval_every": EVAL_EVERY,
			"checkpoint_every": 1000,
		},
		"env": {
			"backend": "cuda",
			"provider": "modal",
			"gpu": {"name": "A100-80GB", "count": 1, "memory_bytes": 80 * GIB},
			"libs": {"torch": "2.9.0", "transformers": "4.58.0", "peft": "0.17.0"},
			"seed": 0,
		},
		"graph": graph(),
		"knobs": knobs(),
		"probes": [
			{"id": "p1", "knobs": {"seq_len": 512, "batch_size": 4, "eval_batch_size": 8, "gpu": "A100-80GB", "grad_ckpt": False}, "steps": 6, "status": "ok"},
			{"id": "p2", "knobs": {"seq_len": 2048, "batch_size": 4, "eval_batch_size": 8, "gpu": "A100-80GB", "grad_ckpt": False}, "steps": 6, "status": "ok"},
			{"id": "p3", "knobs": {"seq_len": 2048, "batch_size": 8, "eval_batch_size": 16, "gpu": "A100-80GB", "grad_ckpt": True}, "steps": 6, "status": "ok"},
		],
		"predictions": {
			"targets": [{"id": t, "unit": u, "label_key": f"target.{t}", "better": b} for t, u, b in TARGETS],
			"grids": grids,
		},
		"models": [
			{"id": "analytical-handmade@0", "kind": "analytical", "version": "0", "trained_on": "none (hand-made fixture)"}
		],
		"findings": [
			{
				"id": "f1",
				"severity": "critical",
				"node": "eval",
				"title": "finding.eval_no_grad.title",
				"evidence": "finding.eval_no_grad.evidence",
				"source": "rule",
				"fix_template": "eval_no_grad",
				"knob": "eval_batch_size",
			},
			{
				"id": "f2",
				"severity": "warn",
				"node": "data",
				"title": "finding.padding_waste.title",
				"evidence": "finding.padding_waste.evidence",
				"source": "rule",
				"fix_template": "sequence_packing",
			},
		],
		"patches": [],
		"provenance": [],
		"costs": {"gpu_usd": 0.0, "llm_usd": 0.0},
	}
	json.dump(record, sys.stdout, indent="\t")
	sys.stdout.write("\n")


def knobs():
	return [
		{"id": "seq_len", "label_key": "knob.seq_len", "unit": "tokens", "kind": "int",
		 "domain": {"min": 256, "max": 4096, "values": SEQ}, "value": 2048,
		 "probed_range": {"min": 512, "max": 2048}, "gridded": True, "needs_code": False},
		{"id": "batch_size", "label_key": "knob.batch_size", "unit": "count", "kind": "int",
		 "domain": {"min": 1, "max": 32, "values": BATCH}, "value": 4,
		 "probed_range": {"min": 1, "max": 8}, "gridded": True, "needs_code": False},
		{"id": "eval_batch_size", "label_key": "knob.eval_batch_size", "unit": "count", "kind": "int",
		 "domain": {"min": 1, "max": 32, "values": EVAL_BATCH}, "value": 32,
		 "probed_range": {"min": 1, "max": 16}, "gridded": True, "needs_code": False},
		{"id": "grad_ckpt", "label_key": "knob.grad_ckpt", "unit": "none", "kind": "bool",
		 "domain": {"values": CKPT}, "value": False, "gridded": True, "needs_code": True,
		 "patch_template": "gradient_checkpointing"},
		{"id": "gpu", "label_key": "knob.gpu", "unit": "none", "kind": "enum",
		 "domain": {"values": GPU}, "value": "A100-80GB",
		 "probed_range": {"values": ["A100-80GB"]}, "gridded": True, "needs_code": False},
		{"id": "lora_rank", "label_key": "knob.lora_rank", "unit": "count", "kind": "int",
		 "domain": {"min": 4, "max": 128, "values": [4, 8, 16, 32, 64, 128]}, "value": LORA_RANK,
		 "probed_range": {"min": 16, "max": 16}, "gridded": False, "needs_code": False},
		{"id": "precision", "label_key": "knob.precision", "unit": "none", "kind": "enum",
		 "domain": {"values": ["bf16", "fp16", "fp32"]}, "value": "bf16", "gridded": False, "needs_code": True,
		 "patch_template": "bf16"},
	]


def graph():
	n = lambda id, kind, label, col, row, **kw: {"id": id, "kind": kind, "label": label, "label_key": f"node.{id}", "pos": {"col": col, "row": row}, **kw}
	nodes = [
		n("data", "data", "Data", 0, 0, metrics={"time": "t_data", "throughput": "tokens_per_sec"}),
		n("embed", "embed", "Embeddings", 1, 0, metrics={"memory": "mem_weights"}),
		n("blocks", "blocks", "Transformer blocks", 2, 0, repeat=32, metrics={"time": "t_forward", "memory": "mem_activations"}),
		n("head", "head", "LM head", 3, 0, metrics={"memory": "mem_logits"}),
		n("loss", "loss", "Loss", 4, 0),
		n("backward", "backward", "Backward", 3, 1, metrics={"time": "t_backward"}),
		n("optimizer", "optimizer", "Optimizer (LoRA)", 1, 1, metrics={"memory": "mem_grads_optimizer"}),
		n("eval", "eval", "Eval pass", 4, 1, metrics={"time": "t_eval", "memory": "mem_eval_spike"}),
		n("checkpoint", "checkpoint", "Checkpoint", 4, 2),
	]
	e = lambda s, t, kind="flow", **kw: {"id": f"{s}-{t}", "source": s, "target": t, "kind": kind, **kw}
	edges = [
		e("data", "embed", throughput="tokens_per_sec"),
		e("embed", "blocks", throughput="tokens_per_sec"),
		e("blocks", "head", throughput="tokens_per_sec"),
		e("head", "loss", throughput="tokens_per_sec"),
		e("loss", "backward", throughput="tokens_per_sec"),
		e("backward", "optimizer", throughput="tokens_per_sec"),
		e("optimizer", "data", "loop"),
		e("loss", "eval", "branch", label="every 500 steps"),
		e("eval", "checkpoint", "branch", label="every 1,000 steps"),
	]
	return {"nodes": nodes, "edges": edges}


if __name__ == "__main__":
	main()
