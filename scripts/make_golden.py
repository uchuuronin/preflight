"""Write the golden records in schema/examples/.

	uv run python scripts/make_golden.py

Fixed ids, tokens and timestamps keep the output byte-stable, so CI can
round-trip it. The numbers are hand-made but physically plausible (they follow
the bug-zoo scripts) so C can build the workspace against them before real
probes land. Predictions here are fixtures with model_id "golden-fixture@0";
B's engine replaces them.
"""

import json
import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from preflight.schema import FlightRecord
from preflight.schema.encoding import choose_scale, encode_f16, encode_mask

OUT = Path(__file__).resolve().parents[1] / "schema" / "examples"
T0 = datetime(2026, 10, 27, 14, 0, 0, tzinfo=UTC)
GIB = 1024**3


def iso(t: datetime) -> str:
	return t.isoformat().replace("+00:00", "Z")


def fixture_grid(gid, target, unit, axes, fn, fixed, rel=0.1, abstain_fn=None):
	shape = [len(a["values"]) for a in axes]
	pts, lo, hi, ab = [], [], [], []
	for idx in np.ndindex(*shape):
		vals = [axes[i]["values"][j] for i, j in enumerate(idx)]
		p = fn(*vals)
		pts.append(p)
		lo.append(p * (1 - rel))
		hi.append(p * (1 + rel))
		ab.append(bool(abstain_fn and abstain_fn(*vals)))
	scale = choose_scale(pts, lo, hi)
	return {
		"id": gid,
		"target_id": target,
		"unit": unit,
		"axes": axes,
		"point": encode_f16(pts, scale),
		"lo": encode_f16(lo, scale),
		"hi": encode_f16(hi, scale),
		"abstain": encode_mask(ab),
		"model_id": "golden-fixture@0",
		"scale": scale,
		"fixed": fixed,
	}


def est(point, unit, status="predicted", rel=0.1, source="grid", reason=None):
	e = {
		"point": point,
		"lo": point * (1 - rel),
		"hi": point * (1 + rel),
		"unit": unit,
		"status": status,
		"model_id": "golden-fixture@0",
		"source": source,
		"level": 0.9,
	}
	if reason:
		e["reason"] = reason
	return e


def kernel_table(rows, backend="cuda", steps=3, allow_tf32=False):
	"""Probe.kernels: (name, class, calls, device_s, flops, bytes) per step; dtype bf16 unless given."""
	out = []
	for name, cls, calls, sec, flops, nbytes, *dt in rows:
		out.append({"name": name, "kernel_class": cls, "calls": calls, "device_s": sec, "flops": flops,
			"bytes": nbytes, "dtype": dt[0] if dt else "bf16"})  # fmt: skip
	return {
		"rows": out,
		"steps": steps,
		"backend": backend,
		"allow_tf32": allow_tf32,
		"classified_by": "engine.steptime",
	}


def targets():
	return [
		{"id": "memory.peak", "label_key": "target.memory.peak", "unit": "bytes", "readout": "memory"},
		{"id": "time.total", "label_key": "target.time.total", "unit": "s", "readout": "time"},
		{"id": "cost.total", "label_key": "target.cost.total", "unit": "usd", "readout": "cost"},
		{"id": "risk.oom", "label_key": "target.risk.oom", "unit": "probability", "readout": "risk"},
	]


# ------------------------------------------------------------------ fine-tune


ACCUM = 4  # the fine-tune's grad_accum
FT_PARAMS = 1_562_179_072  # Qwen2.5-1.5B with r=16 LoRA on all seven projections
FT_TRAINABLE = 18_464_768
L4_BF16 = 121e12


def ft_step(bs):
	"""Seconds per optimizer step (four micro-steps) on the L4. The fixture's time.total (C's tests
	read it) is built on this, so it stays as first written: an optimistic 58-63% MFU."""
	return round(0.31 * bs + 0.12, 3)


def finetune() -> dict:
	# Qwen2.5-1.5B + LoRA r=16 on one L4 (24 GB), HF Trainer. Bug zoo planted problems:
	# eval batch too large, eval without no_grad, no packing.
	vocab, layers, d = 151_936, 28, 1536

	def peak(bs, seq):
		weights = 3.09 * GIB
		lora_state = 0.22 * GIB
		act = bs * seq * d * layers * 34 / 2  # bytes, bf16, rough per-layer factor
		logits = bs * seq * vocab * 4  # fp32 upcast for the loss
		return weights + lora_state + act + logits + 0.8 * GIB

	knobs = [
		{
			"id": "batch_size",
			"label_key": "knob.batch_size",
			"unit": "samples",
			"kind": "int",
			"domain": {"min": 1, "max": 64, "log": True},
			"current": 4,
			"probed_range": {"min": 2, "max": 4},
			"gridded": True,
			"group": "memory",
		},
		{
			"id": "seq_len",
			"label_key": "knob.seq_len",
			"unit": "tokens",
			"kind": "int",
			"domain": {"min": 128, "max": 8192, "log": True},
			"current": 1024,
			"probed_range": {"min": 1024, "max": 1024},
			"gridded": True,
			"group": "memory",
		},
		{
			"id": "grad_accum",
			"label_key": "knob.grad_accum",
			"unit": "steps",
			"kind": "int",
			"domain": {"min": 1, "max": 64},
			"current": 4,
			"gridded": False,
			"group": "speed",
		},
		{
			"id": "eval_batch_size",
			"label_key": "knob.eval_batch_size",
			"unit": "samples",
			"kind": "int",
			"domain": {"min": 1, "max": 128, "log": True},
			"current": 32,
			"gridded": False,
			"group": "memory",
		},
		{
			"id": "lora_r",
			"label_key": "knob.lora_r",
			"kind": "int",
			"domain": {"min": 4, "max": 128, "log": True},
			"current": 16,
			"group": "model",
		},
		{
			"id": "gradient_checkpointing",
			"label_key": "knob.gradient_checkpointing",
			"kind": "bool",
			"current": False,
			"group": "memory",
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
	nodes = [
		{
			"id": "data",
			"label": "alpaca-cleaned · padded to longest",
			"kind": "data",
			"metrics": ["probe.data.padding_fraction"],
		},
		{"id": "model", "label": "Qwen2.5-1.5B", "kind": "group", "params": 1_543_714_304},
		{"id": "embed", "label": "Embedding", "kind": "module", "parent": "model", "params": vocab * d},
		{"id": "block", "label": "Decoder block", "kind": "group", "parent": "model", "repeat": layers},
		{
			"id": "attn",
			"label": "Self-attention + LoRA",
			"kind": "module",
			"parent": "block",
			"metrics": ["probe.step_time.forward"],
		},
		{"id": "mlp", "label": "MLP + LoRA", "kind": "module", "parent": "block"},
		{
			"id": "lm_head",
			"label": "LM head (tied)",
			"kind": "module",
			"parent": "model",
			"metrics": ["memory.by_term.logits"],
		},
		{"id": "loss", "label": "Cross-entropy", "kind": "loss"},
		{"id": "optimizer", "label": "AdamW (LoRA params)", "kind": "optimizer", "params": 18_464_768},
		{"id": "eval", "label": "Eval loop", "kind": "phase", "metrics": ["risk.oom"]},
		{"id": "ckpt", "label": "Checkpoints", "kind": "io"},
	]
	edges = [
		{"source": "data", "target": "embed", "kind": "tokens", "metric": "probe.throughput.tokens_per_s"},
		{"source": "embed", "target": "attn"},
		{"source": "attn", "target": "mlp"},
		{"source": "mlp", "target": "lm_head"},
		{"source": "lm_head", "target": "loss"},
		{"source": "loss", "target": "optimizer", "kind": "grad"},
		{"source": "optimizer", "target": "attn", "kind": "weights"},
		{"source": "data", "target": "eval", "kind": "control"},
		{"source": "optimizer", "target": "ckpt", "kind": "control"},
	]

	def ft_kernels(step, tokens):
		# Qwen2.5-1.5B LoRA on an L4, four micro-steps a step: GEMMs (about 80% of peak) and flash
		# attention dominate; RMSNorm and SiLU run as elementwise ops; the rest rolls up per class.
		# FLOPs add up to the recorder's 6N-style count for the step, and no row beats the peak.
		f = tokens * (4 * FT_PARAMS + 2 * FT_TRAINABLE)
		return kernel_table([
			("aten::mm", "gemm", 4704.0, 0.62 * step, 0.85 * f, 1.16e11),
			("aten::_flash_attention_backward", "attention", 112.0, 0.10 * step, 0.10 * f, 4.4e9),
			("aten::_flash_attention_forward", "attention", 112.0, 0.04 * step, 0.05 * f, 3.0e9),
			("aten::mul", "elementwise", 7648.0, 0.05 * step, None, 8.8e10),
			("aten::silu_backward", "elementwise", 112.0, 0.015 * step, None, 1.9e10),
			("aten::_log_softmax_backward_data", "softmax", 8.0, 0.015 * step, None, 1.0e10),
			("aten::_foreach_addcdiv_", "optimizer", 1.0, 0.004 * step, None, 2.2e8),
			("(tail) elementwise (41 ops)", "elementwise", 12880.0, 0.04 * step, None, 7.6e10),
			("(tail) copy (6 ops)", "copy", 1640.0, 0.015 * step, None, 2.4e10),
			("(tail) reduction (9 ops)", "reduction", 932.0, 0.01 * step, None, 8.0e9),
			("(unattributed device time)", "other", 0.0, 0.008 * step, None, None),
		])  # fmt: skip

	def probe(pid, bs, i):
		pk = peak(bs, 1024)
		step = ft_step(bs)
		tokens = ACCUM * bs * 1024  # padded batches, all four micro-steps: what tokens_per_s counts
		loss = [2.41, 2.37, 2.30, 2.26, 2.21]
		return {
			"id": pid,
			"knobs": {"batch_size": bs, "seq_len": 1024},
			"status": "ok",
			"steps": 5,
			"started_at": iso(T0 + timedelta(minutes=1 + 2 * i)),
			"duration_s": 71.0 + 9 * i,
			"memory": {
				"peak_allocated_bytes": int(pk),
				"peak_reserved_bytes": int(pk * 1.07),
				"device_total_bytes": 23_034_920_960,
				"by_term": {
					"weights": int(3.09 * GIB),
					"gradients": int(0.07 * GIB),
					"optimizer": int(0.15 * GIB),
					"activations": int(bs * 1024 * d * layers * 17),
					"logits": int(bs * 1024 * vocab * 4),
					"context": int(0.8 * GIB),
				},
				"peak_step": 2,
			},
			"step_time": {
				"n_steps": 3,
				"mean_s": round(step, 3),
				"p50_s": round(step, 3),
				"p95_s": round(step * 1.06, 3),
				"first_step_s": round(step * 4.2, 3),
				"by_phase": {
					"data": 0.04,
					"forward": 0.31,
					"backward": 0.55,
					"optimizer": 0.07,
					"idle": 0.03,
				},
				"unprofiled": True,
			},
			"throughput": {
				"tokens_per_s": round(tokens / step, 1),
				"samples_per_s": round(ACCUM * bs / step, 2),
				# the recorder's rule: forward 2N + activation grads 2N + weight grads 2N_trainable per token
				"mfu": round(tokens * (4 * FT_PARAMS + 2 * FT_TRAINABLE) / step / L4_BF16, 3),
			},
			"stability": {
				"series": {
					"loss": loss,
					"grad_norm": [0.81, 0.77, 0.74, 0.79, 0.72],
					"lr": [2e-05, 4e-05, 6e-05, 8e-05, 1e-04],
				},
			},
			"data": {
				"padding_fraction": 0.46,
				"packing": False,
				"tokens_per_sample_mean": 553.0,
				"tokens_per_sample_p95": 1024.0,
				"loader_wait_fraction": 0.04,
				"num_workers": 2,
			},
			"power": {"mean_w": 61.0, "max_w": 72.0, "samples": 70},
			"extras": {
				"params_total": float(FT_PARAMS),
				"params_trainable": float(FT_TRAINABLE),
				"tokens_per_step": float(tokens),
				"samples_per_step": float(ACCUM * bs),
				"micro_steps_per_step": float(ACCUM),
				"host_rss_peak_bytes": 6_442_450_944.0,
			},
			**({"kernels": ft_kernels(step, tokens)} if i == 0 else {}),
			"cost": {"gpu_seconds": 71.0 + 9 * i, "usd": round((71.0 + 9 * i) * 0.80 / 3600, 4)},
			"cache_key": f"sha256:{'a' * 60}{i:04d}",
			"trace": f"blob://traces/01JBGOLDENFT/{pid}.json.gz",
			"log": f"blob://logs/01JBGOLDENFT/{pid}.txt",
		}

	axes = [
		{"knob_id": "batch_size", "values": [1, 2, 4, 8, 16, 32], "scale": "log"},
		{"knob_id": "seq_len", "values": [256, 512, 1024, 2048, 4096], "scale": "log"},
	]
	grid = fixture_grid(
		"g.memory.bs_seq",
		"memory.peak",
		"bytes",
		axes,
		peak,
		fixed={"grad_accum": 4, "lora_r": 16, "gradient_checkpointing": False, "precision": "bf16"},
		abstain_fn=lambda bs, seq: bs * seq > 2 * 4 * 1024,
	)
	risk = fixture_grid(
		"g.risk.bs_seq",
		"risk.oom",
		"probability",
		axes,
		lambda bs, seq: 1 / (1 + math.exp(-(peak(bs, seq) - 22.0 * GIB) / (0.6 * GIB))),
		fixed=grid["fixed"],
		rel=0.0,
		abstain_fn=lambda bs, seq: bs * seq > 2 * 4 * 1024,
	)
	return {
		"schema_version": "0.1.0",
		"record_id": "01JBGOLDENFT0000000000000A",
		"share_token": "golden-finetune-share-tok",
		"created_at": iso(T0),
		"job": {
			"lens": "finetune",
			"source": {
				"kind": "sample",
				"uri": "sample://zoo/lora-hf",
				"entrypoint": "zoo/finetune_lora_hf/train.py",
				"content_hash": "sha256:" + "f" * 64,
			},
			"framework": "hf_trainer",
			"title": "LoRA fine-tune of Qwen2.5-1.5B on Alpaca",
			"script_map": {
				"lens": "finetune",
				"framework": "hf_trainer",
				"entrypoint": "zoo/finetune_lora_hf/train.py",
				"launch": "python train.py",
				"model": {
					"name": "Qwen2.5-1.5B",
					"hf_id": "Qwen/Qwen2.5-1.5B",
					"family": "decoder",
					"param_count": 1_543_714_304,
					"n_layers": layers,
					"d_model": d,
					"n_heads": 12,
					"n_kv_heads": 2,
					"d_ff": 8960,
					"vocab_size": vocab,
					"head_dim": 128,
					"max_positions": 131_072,
					"gated_mlp": True,
					"tie_embeddings": True,
					"norm": "rmsnorm",
					"attn_impl": "sdpa",
					"dtype": "bfloat16",
					"lora": {
						"r": 16,
						"alpha": 32.0,
						"target_modules": [
							"q_proj",
							"k_proj",
							"v_proj",
							"o_proj",
							"gate_proj",
							"up_proj",
							"down_proj",
						],
						"dropout": 0.05,
					},
				},
				"data": {
					"dataset": "yahma/alpaca-cleaned",
					"batch_size": 4,
					"eval_batch_size": 32,
					"seq_len": 1024,
					"num_workers": 2,
					"packing": False,
					"padding": "longest",
					"n_examples": 51_760,
				},
				"training": {
					"optimizer": "adamw_torch",
					"lr": 1e-4,
					"warmup_steps": 50,
					"grad_accum": 4,
					"precision": "bf16",
					"epochs": 1.0,
					"eval_every": 200,
					"checkpoint_every": 500,
					"grad_checkpointing": False,
					"eval_under_no_grad": False,
				},
				"loop": {
					"kind": "hf_trainer",
					"location": {"file": "train.py", "line": 88, "expr": "trainer.train()"},
					"step_call": "trainer.train()",
					"model_var": "model",
				},
				"knobs": [
					{
						"id": "batch_size",
						"label": "per_device_train_batch_size",
						"kind": "int",
						"current": 4,
						"location": {"file": "train.py", "line": 61, "expr": "per_device_train_batch_size=4"},
					},
					{
						"id": "eval_batch_size",
						"label": "per_device_eval_batch_size",
						"kind": "int",
						"current": 32,
						"location": {"file": "train.py", "line": 62, "expr": "per_device_eval_batch_size=32"},
					},
				],
				"risks": [
					"Custom eval callback runs the model without torch.no_grad()",
					"Eval batch is 8x the train batch",
					"No sequence packing; padding to the longest in batch",
				],
				"produced_by": "adapter",
				"confidence": 0.9,
			},
			"objective": {"kind": "min_time", "description": "Fine-tune before Friday on my one lab GPU"},
			"constraints": {
				"budget_usd": 20.0,
				"gpu_types": ["L4"],
				"max_gpus": 1,
				"wall_time_cap_s": 86_400.0,
				"preemption": {"kind": "none"},
			},
		},
		"env": {
			"backend": "cuda",
			"provider": "modal",
			"gpu": {
				"name": "L4",
				"count": 1,
				"memory_bytes": 23_034_920_960,
				"vendor": "nvidia",
				"compute_capability": "8.9",
				"driver": "550.90.07",
			},
			"nodes": 1,
			"libraries": {
				"torch": "2.5.1",
				"transformers": "4.46.2",
				"peft": "0.13.2",
				"accelerate": "1.1.1",
			},
			"python": "3.11.10",
			"seed": 42,
			"region": "us-east",
			"image": "sha256:" + "1" * 64,
		},
		"graph": {"nodes": nodes, "edges": edges, "skeleton": False},
		"knobs": knobs,
		"probes": [probe("p1", 2, 0), probe("p2", 4, 1)],
		"predictions": {
			"targets": targets(),
			"estimates": {
				"memory.peak": est(peak(4, 1024), "bytes"),
				"time.total": est(51_760 / (4 * ACCUM) * ft_step(4), "s", rel=0.15),  # 3,235 steps at bs 4
				"cost.total": est(51_760 / (4 * ACCUM) * ft_step(4) * 0.80 / 3600, "usd", rel=0.15),
				"risk.oom": {
					"point": 0.97,
					"lo": 0.9,
					"hi": 1.0,
					"unit": "probability",
					"status": "predicted",
					"model_id": "golden-fixture@0",
					"source": "exact",
					"level": 0.9,
					"reason": "first eval at step 200 runs batch 32 with gradients kept",
				},
			},
			"grids": [grid, risk],
		},
		"models": [
			{
				"id": "golden-fixture@0",
				"kind": "rule",
				"version": "0",
				"notes": "Hand-made fixture for UI work; not a real model",
			}
		],
		"findings": [
			{
				"id": "f1",
				"severity": "critical",
				"code": "eval.no_grad",
				"title": "Eval runs with gradients kept",
				"detail": "The eval callback calls model(**batch) outside torch.no_grad(), so activations for backward are kept at eval batch 32.",
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
					{"metric": "memory.peak@eval_batch_size=32", "value": 38.2e9, "threshold": 23_034_920_960}
				],
				"source": "rule",
				"fix_template": "eval.reduce_batch",
				"knob_id": "eval_batch_size",
			},
			{
				"id": "f3",
				"severity": "warn",
				"code": "data.padding_waste",
				"title": "46% of tokens are padding",
				"node_id": "data",
				"evidence": [
					{
						"metric": "probe.data.padding_fraction",
						"value": 0.46,
						"threshold": 0.2,
						"probe_id": "p2",
					}
				],
				"source": "rule",
				"fix_template": "data.enable_packing",
				"knob_id": "packing",
			},
		],
		"provenance": [
			{
				"provider": "baseten",
				"model": "meta-models/muse-spark-open-1026",
				"purpose": "triage",
				"input_tokens": 6120,
				"output_tokens": 412,
				"cost_usd": 0.0021,
				"request_id": "req_golden_1",
				"at": iso(T0 + timedelta(minutes=5)),
				"latency_s": 3.4,
				"ok": True,
			},
		],
		"costs": {"gpu_usd": 0.0336, "llm_usd": 0.0021, "breakdown": {"probes": 0.0336, "triage": 0.0021}},
	}


# ------------------------------------------------------------------ pretrain


def pretrain() -> dict:
	# nanoGPT-style 124M on one A100-40GB; planted: num_workers=0, fp32, sparse checkpoints.
	def step_s(bs, seq, workers, prec):
		compute = bs * seq * 124e6 * 6 / (312e12 * (0.42 if prec == "bf16" else 0.11))
		loader = 0.0 if workers > 0 else bs * seq * 2.2e-7
		return compute + loader

	knobs = [
		{
			"id": "batch_size",
			"label_key": "knob.batch_size",
			"unit": "samples",
			"kind": "int",
			"domain": {"min": 1, "max": 128, "log": True},
			"current": 12,
			"probed_range": {"min": 8, "max": 12},
			"gridded": True,
			"group": "memory",
		},
		{
			"id": "seq_len",
			"label_key": "knob.seq_len",
			"unit": "tokens",
			"kind": "int",
			"domain": {"min": 128, "max": 4096, "log": True},
			"current": 1024,
			"probed_range": {"min": 1024, "max": 1024},
			"gridded": True,
			"group": "memory",
		},
		{
			"id": "num_workers",
			"label_key": "knob.num_workers",
			"kind": "int",
			"domain": {"min": 0, "max": 16},
			"current": 0,
			"group": "data",
		},
		{
			"id": "precision",
			"label_key": "knob.precision",
			"kind": "categorical",
			"domain": {"values": ["bf16", "fp32"]},
			"current": "fp32",
			"group": "speed",
		},
		{
			"id": "lr",
			"label_key": "knob.lr",
			"kind": "float",
			"domain": {"min": 1e-5, "max": 1e-2, "log": True},
			"current": 6e-4,
			"group": "model",
		},
		{
			"id": "checkpoint_every",
			"label_key": "knob.checkpoint_every",
			"unit": "steps",
			"kind": "int",
			"domain": {"min": 50, "max": 100_000, "log": True},
			"current": 20_000,
			"group": "cluster",
		},
		{
			"id": "max_steps",
			"label_key": "knob.max_steps",
			"unit": "steps",
			"kind": "int",
			"domain": {"min": 100, "max": 1_000_000, "log": True},
			"current": 60_000,
			"group": "cluster",
		},
	]
	nodes = [
		{
			"id": "data",
			"label": "OpenWebText shards · num_workers=0",
			"kind": "data",
			"metrics": ["probe.data.loader_wait_fraction"],
		},
		{"id": "model", "label": "GPT-2 124M", "kind": "group", "params": 124_439_808},
		{"id": "wte", "label": "Token + position embedding", "kind": "module", "parent": "model"},
		{"id": "block", "label": "Transformer block", "kind": "group", "parent": "model", "repeat": 12},
		{
			"id": "attn",
			"label": "Causal self-attention",
			"kind": "module",
			"parent": "block",
			"metrics": ["probe.stability.max_attn_logit"],
		},
		{"id": "mlp", "label": "MLP", "kind": "module", "parent": "block"},
		{"id": "lm_head", "label": "LM head", "kind": "module", "parent": "model"},
		{"id": "loss", "label": "Cross-entropy", "kind": "loss", "metrics": ["probe.stability.loss"]},
		{"id": "optimizer", "label": "AdamW", "kind": "optimizer", "params": 124_439_808},
		{
			"id": "ckpt",
			"label": "Checkpoint every 20k steps",
			"kind": "io",
			"metrics": ["risk.preemption_loss"],
		},
	]
	edges = [
		{"source": "data", "target": "wte", "kind": "tokens", "metric": "probe.throughput.tokens_per_s"},
		{"source": "wte", "target": "attn"},
		{"source": "attn", "target": "mlp"},
		{"source": "mlp", "target": "lm_head"},
		{"source": "lm_head", "target": "loss"},
		{"source": "loss", "target": "optimizer", "kind": "grad"},
		{"source": "optimizer", "target": "wte", "kind": "weights"},
		{"source": "optimizer", "target": "ckpt", "kind": "control"},
	]

	def pt_kernels(s, prec):
		# GPT-2 124M in fp32 with TF32 matmuls (nanoGPT turns it on) on an A100: the data loader leaves
		# the GPU idle 37% of the step, so the kernels sum to well under it; the rest is B's host bucket
		dt = "fp32" if prec == "fp32" else "bf16"
		return kernel_table([
			("aten::mm", "gemm", 444.0, 0.30 * s, 6.1e12, 9.8e9, dt),
			("aten::_efficient_attention_backward", "attention", 12.0, 0.08 * s, 7.7e11, 1.2e9, dt),
			("aten::_efficient_attention_forward", "attention", 12.0, 0.04 * s, 3.1e11, 6.0e8, dt),
			("aten::native_layer_norm_backward", "norm", 25.0, 0.025 * s, None, 3.1e9, dt),
			("aten::gelu_backward", "elementwise", 12.0, 0.02 * s, None, 2.4e9, dt),
			("aten::_foreach_lerp_", "optimizer", 1.0, 0.012 * s, None, 1.5e9, dt),
			("(tail) elementwise (28 ops)", "elementwise", 980.0, 0.05 * s, None, 7.0e9, dt),
			("(tail) copy (5 ops)", "copy", 120.0, 0.012 * s, None, 1.4e9, dt),
		], allow_tf32=prec == "fp32")  # fmt: skip

	def probe(pid, bs, i, prec="fp32"):
		s = step_s(bs, 1024, 0, prec)
		gn = [1.9, 1.4, 1.2, 1.1, 1.05, 0.98, 0.95, 0.93]
		return {
			"id": pid,
			"knobs": {"batch_size": bs, "seq_len": 1024, "num_workers": 0, "precision": prec},
			"status": "ok",
			"steps": 8,
			"started_at": iso(T0 + timedelta(minutes=1 + 2 * i)),
			"duration_s": 64.0 + 5 * i,
			"memory": {
				"peak_allocated_bytes": int((1.99 + 0.84 * bs) * GIB),
				"peak_reserved_bytes": int((2.2 + 0.86 * bs) * GIB),
				"device_total_bytes": 42_298_834_944,
				"by_term": {
					"weights": 497_759_232,
					"gradients": 497_759_232,
					"optimizer": 995_518_464,
					"activations": int(0.66 * bs * GIB),
					"logits": bs * 1024 * 50_304 * 4,
					"context": int(0.6 * GIB),
				},
				"peak_step": 2,
			},
			"step_time": {
				"n_steps": 5,
				"mean_s": round(s, 3),
				"p50_s": round(s, 3),
				"p95_s": round(s * 1.04, 3),
				"first_step_s": round(s + 21.0, 3),
				"by_phase": {"data": 0.37, "forward": 0.2, "backward": 0.36, "optimizer": 0.05, "idle": 0.02},
				"unprofiled": True,
			},
			"throughput": {
				"tokens_per_s": round(bs * 1024 / s, 1),
				"samples_per_s": round(bs / s, 2),
				"flops_per_step": bs * 1024 * 124e6 * 6,
				# against the TF32 peak (156 TFLOP/s) in fp32, bf16's (312) otherwise
				"mfu": round(bs * 1024 * 124e6 * 6 / s / (156e12 if prec == "fp32" else 312e12), 3),
			},
			"stability": {
				"series": {
					"loss": [10.98, 10.21, 9.62, 9.11, 8.70, 8.33, 8.02, 7.79],
					"grad_norm": gn,
					"grad_norm_ma": [round(sum(gn[: k + 1]) / (k + 1), 3) for k in range(len(gn))],
					"update_to_weight": [0.0021, 0.0018, 0.0016, 0.0015, 0.0015, 0.0014, 0.0014, 0.0013],
					"max_attn_logit": [11.2, 12.0, 12.6, 13.1, 13.5, 13.8, 14.0, 14.3],
					"output_logit_norm": [41.0, 39.2, 38.7, 38.1, 37.9, 37.6, 37.4, 37.2],
					"lr": [6e-6, 1.2e-5, 1.8e-5, 2.4e-5, 3e-5, 3.6e-5, 4.2e-5, 4.8e-5],
				},
				"nan_count": 0,
				"inf_count": 0,
				"loss_scale_events": 0,
			},
			"data": {"loader_wait_fraction": 0.37, "num_workers": 0, "tokens_per_sample_mean": 1024.0},
			"power": {"mean_w": 248.0, "max_w": 301.0, "samples": 64},
			"extras": {
				"lr_range_divergence_lr": 4.8e-3,
				"resume_exact_match": 1.0,
				"params_total": 124_475_904.0,
				"params_trainable": 124_475_904.0,
				"tokens_per_step": float(bs * 1024),
				"samples_per_step": float(bs),
				"micro_steps_per_step": 1.0,
			},
			**({"kernels": pt_kernels(s, prec)} if i == 0 else {}),
			"cost": {"gpu_seconds": 64.0 + 5 * i, "usd": round((64.0 + 5 * i) * 2.10 / 3600, 4)},
			"trace": f"blob://traces/01JBGOLDENPT/{pid}.json.gz",
		}

	axes = [
		{"knob_id": "batch_size", "values": [4, 8, 12, 16, 24, 32, 48], "scale": "log"},
		{"knob_id": "precision", "values": ["fp32", "bf16"], "scale": "categorical"},
	]
	fixed = {"seq_len": 1024, "num_workers": 0}
	time_grid = fixture_grid(
		"g.time.bs_prec",
		"time.total",
		"s",
		axes,
		lambda bs, p: 60_000 * step_s(bs, 1024, 0, p) * 12 / bs,
		fixed,
		rel=0.15,
		abstain_fn=lambda bs, p: bs > 24,
	)
	mem_grid = fixture_grid(
		"g.memory.bs_prec",
		"memory.peak",
		"bytes",
		axes,
		lambda bs, p: (1.99 + 0.84 * bs * (0.55 if p == "bf16" else 1.0)) * GIB,
		fixed,
		abstain_fn=lambda bs, p: bs > 24,
	)
	total = 60_000 * step_s(12, 1024, 0, "fp32")
	return {
		"schema_version": "0.1.0",
		"record_id": "01JBGOLDENPT0000000000000B",
		"share_token": "golden-pretrain-share-tok",
		"created_at": iso(T0),
		"job": {
			"lens": "pretrain",
			"source": {
				"kind": "sample",
				"uri": "sample://zoo/nanogpt-pretrain",
				"entrypoint": "zoo/pretrain_nanogpt/train.py",
				"content_hash": "sha256:" + "e" * 64,
			},
			"framework": "plain_pytorch",
			"title": "GPT-2 124M pretrain on OpenWebText",
			"objective": {
				"kind": "min_cost",
				"description": "Small LM baseline for a class project, cheapest finish",
			},
			"constraints": {
				"budget_usd": 60.0,
				"max_gpus": 1,
				"wall_time_cap_s": 172_800.0,
				"preemption": {
					"kind": "policy",
					"policy": "Unity preempt partition: jobs may be requeued after 2 h",
					"grace_s": 120.0,
				},
			},
		},
		"env": {
			"backend": "cuda",
			"provider": "modal",
			"gpu": {
				"name": "A100-SXM4-40GB",
				"count": 1,
				"memory_bytes": 42_298_834_944,
				"vendor": "nvidia",
				"compute_capability": "8.0",
			},
			"libraries": {"torch": "2.5.1"},
			"python": "3.11.10",
			"seed": 1337,
		},
		"graph": {"nodes": nodes, "edges": edges, "skeleton": False},
		"knobs": knobs,
		"probes": [probe("p1", 8, 0), probe("p2", 12, 1)],
		"predictions": {
			"targets": [
				*targets(),
				{
					"id": "risk.preemption_loss",
					"label_key": "target.risk.preemption_loss",
					"unit": "s",
					"readout": "risk",
				},
			],
			"estimates": {
				"memory.peak": est(
					(1.99 + 0.84 * 12) * GIB, "bytes", status="measured", rel=0.02, source="exact"
				),
				"time.total": est(total, "s", rel=0.15),
				"cost.total": est(total * 2.10 / 3600, "usd", rel=0.15),
				"risk.oom": {
					"point": 0.0,
					"lo": 0.0,
					"hi": 0.01,
					"unit": "probability",
					"status": "predicted",
					"model_id": "golden-fixture@0",
					"source": "exact",
				},
				"risk.preemption_loss": est(3_600.0, "s", rel=0.5, source="exact"),
			},
			"grids": [time_grid, mem_grid],
		},
		"models": [
			{
				"id": "golden-fixture@0",
				"kind": "rule",
				"version": "0",
				"notes": "Hand-made fixture for UI work; not a real model",
			}
		],
		"findings": [
			{
				"id": "f1",
				"severity": "error",
				"code": "data.num_workers_zero",
				"title": "GPU waits on the data loader 37% of each step",
				"node_id": "data",
				"evidence": [
					{
						"metric": "probe.data.loader_wait_fraction",
						"value": 0.37,
						"threshold": 0.1,
						"probe_id": "p2",
					}
				],
				"source": "rule",
				"fix_template": "data.num_workers",
				"knob_id": "num_workers",
			},
			{
				"id": "f2",
				"severity": "warn",
				"code": "precision.fp32",
				"title": "Training in fp32 on a GPU with bf16 tensor cores",
				"node_id": "model",
				"evidence": [{"metric": "probe.throughput.mfu", "value": 0.07, "threshold": 0.3}],
				"source": "rule",
				"fix_template": "precision.bf16_autocast",
				"knob_id": "precision",
			},
			{
				"id": "f3",
				"severity": "warn",
				"code": "ckpt.sparse",
				"title": "Checkpoints every 20k steps; a requeue loses up to 7 h",
				"node_id": "ckpt",
				"evidence": [{"metric": "checkpoint_every", "value": 20_000}],
				"source": "rule",
				"fix_template": "ckpt.interval",
				"knob_id": "checkpoint_every",
			},
			{
				"id": "f4",
				"severity": "info",
				"code": "lr.headroom",
				"title": "Learning rate is 8x below where the probe diverged",
				"node_id": "optimizer",
				"evidence": [{"metric": "probe.extras.lr_range_divergence_lr", "value": 4.8e-3}],
				"source": "rule",
				"knob_id": "lr",
			},
		],
		"costs": {"gpu_usd": 0.0782, "llm_usd": 0.0, "breakdown": {"probes": 0.0782}},
	}


# ------------------------------------------------------------------ agent


def agent() -> dict:
	# Tool-using agent eval on Muse Spark; planted: no prompt caching, context overflow at
	# step 12, a tool that retries forever.
	knobs = [
		{
			"id": "model",
			"label_key": "knob.model",
			"kind": "categorical",
			"domain": {"values": ["muse-spark", "muse-spark-open", "muse-glimmer"]},
			"current": "muse-spark",
			"group": "agent",
		},
		{
			"id": "prompt_caching",
			"label_key": "knob.prompt_caching",
			"kind": "bool",
			"current": False,
			"group": "agent",
		},
		{
			"id": "max_turns",
			"label_key": "knob.max_turns",
			"unit": "turns",
			"kind": "int",
			"domain": {"min": 1, "max": 100},
			"current": 40,
			"group": "agent",
		},
		{
			"id": "context_limit",
			"label_key": "knob.context_limit",
			"unit": "tokens",
			"kind": "int",
			"domain": {"min": 8_192, "max": 262_144, "log": True},
			"current": 32_768,
			"group": "agent",
		},
		{
			"id": "concurrency",
			"label_key": "knob.concurrency",
			"unit": "requests",
			"kind": "int",
			"domain": {"min": 1, "max": 64},
			"current": 4,
			"group": "agent",
		},
		{
			"id": "n_tasks",
			"label_key": "knob.n_tasks",
			"unit": "tasks",
			"kind": "int",
			"domain": {"min": 1, "max": 5000},
			"current": 500,
			"group": "agent",
		},
		{
			"id": "tool_retry_cap",
			"label_key": "knob.tool_retry_cap",
			"kind": "int",
			"domain": {"min": 0, "max": 1000},
			"current": 1000,
			"needs_code": True,
			"patch_template": "agent.cap_tool_retries",
			"group": "agent",
		},
	]
	nodes = [
		{"id": "tasks", "label": "500 repo tasks", "kind": "data"},
		{"id": "agent", "label": "Agent loop", "kind": "group", "metrics": ["agent.turns_per_task"]},
		{
			"id": "llm",
			"label": "Muse Spark",
			"kind": "model_call",
			"parent": "agent",
			"metrics": ["agent.tokens_per_turn"],
		},
		{
			"id": "context",
			"label": "Context window 32k",
			"kind": "store",
			"parent": "agent",
			"metrics": ["agent.context_fill"],
		},
		{"id": "tool_search", "label": "search_code", "kind": "tool", "parent": "agent"},
		{"id": "tool_run", "label": "run_tests (retries forever)", "kind": "tool", "parent": "agent"},
		{"id": "grader", "label": "Grader", "kind": "phase"},
	]
	edges = [
		{"source": "tasks", "target": "llm", "kind": "control"},
		{"source": "llm", "target": "tool_search", "kind": "control"},
		{"source": "llm", "target": "tool_run", "kind": "control"},
		{"source": "tool_search", "target": "context", "kind": "tokens"},
		{"source": "tool_run", "target": "context", "kind": "tokens"},
		{"source": "context", "target": "llm", "kind": "tokens", "metric": "agent.tokens_per_turn"},
		{"source": "llm", "target": "grader", "kind": "control"},
	]
	rows = [
		(
			"t017",
			9,
			61_200,
			3_900,
			1,
			0,
			48.1,
			True,
			"done",
			[3100, 6800, 10400, 14900, 18800, 23500, 27100, 30200, 31900],
		),
		("t042", 6, 33_800, 2_700, 2, 0, 31.4, True, "done", [2900, 5600, 8400, 11000, 13700, 16400]),
		(
			"t088",
			12,
			213_000,
			6_100,
			3,
			1,
			96.0,
			False,
			"context_overflow",
			[3300, 7100, 10900, 14800, 18100, 21000, 23900, 26200, 28400, 30300, 31800, 32768],
		),
		("t103", 40, 1_102_000, 9_800, 0, 38, 412.0, False, "loop", [3000 + 780 * k for k in range(40)]),
		("t151", 7, 41_900, 3_100, 1, 0, 36.2, True, "done", [3200, 6200, 9100, 12300, 15200, 18000, 20600]),
		("t230", 10, 98_400, 5_200, 2, 0, 63.9, False, "done", [3100 + 3300 * k for k in range(10)]),
	]
	tasks = []
	for tid, turns, inp, out, reasoning, errs, lat, ok, stop, ctx in rows:
		ctx = [min(c, 32_768) for c in ctx]
		tasks.append(
			{
				"task_id": tid,
				"model": "muse-spark",
				"turns": turns,
				"input_tokens": inp,
				"output_tokens": out,
				"reasoning_tokens": reasoning * 800,
				"cached_input_tokens": 0,
				"cache_creation_tokens": 0,
				"tool_calls": turns - 1 + errs,
				"errors": errs,
				"latency_s": lat,
				"success": ok,
				"context_tokens_by_turn": ctx,
				"cost_usd": round(inp * 0.6e-6 + out * 2.4e-6, 5),
				"stop_reason": stop,
				"trace": f"blob://traces/01JBGOLDENAG/{tid}.otel.json",
			}
		)
	per_task = sum(t["cost_usd"] for t in tasks) / len(tasks)
	return {
		"schema_version": "0.1.0",
		"record_id": "01JBGOLDENAG0000000000000C",
		"share_token": "golden-agent-share-tokenn",
		"created_at": iso(T0),
		"job": {
			"lens": "agent",
			"source": {
				"kind": "sample",
				"uri": "sample://zoo/agent-eval-spark",
				"entrypoint": "zoo/agent_eval_spark/run.py",
				"content_hash": "sha256:" + "d" * 64,
			},
			"framework": "openai_client",
			"title": "Coding-agent eval on Muse Spark, 500 tasks",
			"objective": {
				"kind": "max_quality_per_usd",
				"description": "Pick the model for our agent eval within $150",
			},
			"constraints": {
				"budget_usd": 150.0,
				"deadline": iso(T0 + timedelta(days=2)),
				"quota": {"api_usd": 150.0},
			},
		},
		"env": {
			"backend": "api",
			"provider": "api",
			"libraries": {"openai": "1.54.3"},
			"python": "3.11.10",
			"seed": 0,
		},
		"graph": {"nodes": nodes, "edges": edges, "skeleton": False},
		"knobs": knobs,
		"probes": [
			{
				"id": "p1",
				"knobs": {"model": "muse-spark", "prompt_caching": False},
				"status": "ok",
				"steps": 6,
				"started_at": iso(T0 + timedelta(minutes=1)),
				"duration_s": 412.0,
				"extras": {"tasks_sampled": 6, "otel_spans": 84},
				"cost": {"gpu_seconds": 0, "usd": round(sum(t["cost_usd"] for t in tasks), 4)},
			}
		],
		"agent_sample": {
			"tasks": tasks,
			"population": 500,
			"selection": "6 tasks from the 30-70% historical pass-rate band",
			"rate_limit_rpm": 600.0,
			"rate_limit_tpm": 2_000_000.0,
		},
		"predictions": {
			"targets": [
				{"id": "cost.total", "label_key": "target.cost.total", "unit": "usd", "readout": "cost"},
				{"id": "time.total", "label_key": "target.time.total", "unit": "s", "readout": "time"},
				{
					"id": "memory.context_fill",
					"label_key": "target.memory.context_fill",
					"unit": "fraction",
					"readout": "memory",
				},
				{
					"id": "risk.runaway",
					"label_key": "target.risk.runaway",
					"unit": "probability",
					"readout": "risk",
				},
			],
			"estimates": {
				"cost.total": {
					"point": round(per_task * 500, 2),
					"lo": round(per_task * 500 * 0.55, 2),
					"hi": round(per_task * 500 * 1.9, 2),
					"unit": "usd",
					"status": "predicted",
					"model_id": "golden-fixture@0",
					"source": "exact",
					"level": 0.9,
					"reason": "6 tasks may under-sample rare runaway tasks; upper bound uses the 95th percentile",
				},
				"time.total": est(500 * 114.6 / 4, "s", rel=0.4, source="exact"),
				"memory.context_fill": est(0.84, "fraction", rel=0.15, source="exact"),
				"risk.runaway": est(0.17, "probability", rel=0.6, source="exact"),
			},
		},
		"models": [
			{
				"id": "golden-fixture@0",
				"kind": "rule",
				"version": "0",
				"notes": "Hand-made fixture for UI work; not a real model",
			}
		],
		"findings": [
			{
				"id": "f1",
				"severity": "error",
				"code": "agent.tool_retry_loop",
				"title": "run_tests retries forever on a failing build",
				"detail": "Task t103 called run_tests 38 times and stopped only at max_turns; it alone cost more than the other five tasks.",
				"node_id": "tool_run",
				"evidence": [{"metric": "agent_sample.t103.errors", "value": 38}],
				"source": "rule",
				"fix_template": "agent.cap_tool_retries",
				"knob_id": "tool_retry_cap",
			},
			{
				"id": "f2",
				"severity": "error",
				"code": "agent.context_overflow",
				"title": "Context fills at turn 12",
				"node_id": "context",
				"evidence": [
					{
						"metric": "agent_sample.t088.context_tokens_by_turn",
						"value": 32_768,
						"threshold": 32_768,
						"step": 12,
					}
				],
				"source": "rule",
				"knob_id": "context_limit",
			},
			{
				"id": "f3",
				"severity": "warn",
				"code": "agent.no_prompt_caching",
				"title": "No prompt caching; 0% of input tokens cached",
				"node_id": "llm",
				"evidence": [{"metric": "agent_sample.cached_input_fraction", "value": 0.0}],
				"source": "rule",
				"knob_id": "prompt_caching",
			},
		],
		"provenance": [
			{
				"provider": "meta",
				"model": "muse-spark",
				"purpose": "probe",
				"input_tokens": sum(t["input_tokens"] for t in tasks),
				"output_tokens": sum(t["output_tokens"] for t in tasks),
				"cost_usd": round(sum(t["cost_usd"] for t in tasks), 4),
				"at": iso(T0 + timedelta(minutes=8)),
				"ok": True,
			},
		],
		"costs": {
			"gpu_usd": 0.0,
			"llm_usd": round(sum(t["cost_usd"] for t in tasks), 4),
			"breakdown": {"agent_sample": round(sum(t["cost_usd"] for t in tasks), 4)},
		},
	}


def main() -> None:
	OUT.mkdir(parents=True, exist_ok=True)
	for name, fn in [("finetune", finetune), ("pretrain", pretrain), ("agent", agent)]:
		raw = fn()
		rec = FlightRecord.model_validate(raw)
		# write the canonical serialisation so the round-trip test is exact
		canonical = json.loads(rec.model_dump_json(exclude_none=True))
		(OUT / f"{name}.json").write_text(json.dumps(canonical, indent="\t", ensure_ascii=False) + "\n")
		print(f"schema/examples/{name}.json")


if __name__ == "__main__":
	main()
