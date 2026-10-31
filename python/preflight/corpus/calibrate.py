"""Build the engine's conformal table from grouped corpus rows (Review, Oct 27).

Each group is one architecture measured at five batch x sequence points (configs.grouped_configs,
configs.hf_configs): three probes laid out like A's probe plan and two held-out queries. For every
group the engine is fitted on the probes exactly as it is for a user's job, predicts the queries,
and each query becomes a conformal score |log(actual / predicted)| for memory.peak and time.step.
Scores are grouped by architecture, so the table's coverage check never calibrates and tests on the
same architecture.

	python -m preflight.corpus.calibrate --store ./corpus --out python/preflight/engine/_calibration_table.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

from preflight.engine import conformal, knobs, memory, oom, steptime
from preflight.engine.memory import TrainingSetup, TransformerArch

SYNTHETIC_LENSES = frozenset({"finetune_synthetic"})
LENS = {
	"gpt": "pretrain",
	"lora": "finetune_synthetic",  # the corpus's own LoRA models: kept apart, users' fine-tunes are real HF models
	"encoder": "retrieval",
	"cnn": "vision",
	"mlp": "vision",
	"gnn": "gnn",
	"hf_lora": "finetune",
}


def _transformer(p: dict[str, Any], family: str) -> tuple[TransformerArch, TrainingSetup, int, int]:
	"""The corpus's synthetic transformer (corpus/models.py) as the engine sees it."""
	arch = TransformerArch(
		n_layers=p["n_layers"], d_model=p["d_model"], n_heads=p["n_heads"], n_kv_heads=p.get("n_kv_heads"),
		d_ff=p["d_ff"], vocab_size=p["vocab_size"], gated_mlp=p.get("gated_mlp", False),
		norm="rmsnorm" if p.get("causal", True) else "layernorm", bias=False, tie_embeddings=True,
	)  # fmt: skip
	amp = p.get("dtype", "fp32") != "fp32"
	setup = TrainingSetup(
		param_dtype="fp32",  # the corpus trains under autocast with fp32 weights
		compute_dtype=p.get("dtype", "fp32") if amp else "fp32",
		method="lora" if family == "lora" else "full",
		lora_rank=p.get("lora_rank", 0),
		lora_targets=tuple(p.get("lora_targets", ("q", "v"))),
		optimizer="adamw",
		grad_checkpointing="full" if p.get("grad_checkpointing") == "full" else "none",
		flash_attention=p.get("attention", "sdpa") == "sdpa",
	)
	return arch, setup, p["batch"], p["seq"]


def _hf(p: dict[str, Any], shape: dict[str, Any]) -> tuple[Any, TrainingSetup, int, int]:
	from preflight.schema.script_map import LoraInfo, ModelInfo

	info = ModelInfo(
		name=p["model_id"], hf_id=p["model_id"], family="decoder",
		n_layers=shape.get("num_hidden_layers"), d_model=shape.get("hidden_size"),
		n_heads=shape.get("num_attention_heads"), n_kv_heads=shape.get("num_key_value_heads"),
		d_ff=shape.get("intermediate_size"), vocab_size=shape.get("vocab_size"), head_dim=shape.get("head_dim"),
		gated_mlp=shape.get("gated_mlp"), tie_embeddings=shape.get("tie_word_embeddings"),
		lora=LoraInfo(r=p["lora_rank"], alpha=p["lora_alpha"], target_modules=["all-linear"]),
	)  # fmt: skip
	arch = knobs.arch_from_model_info(info)
	quant = None if p["quantization"] == "none" else p["quantization"]
	setup = TrainingSetup(
		param_dtype="bf16", compute_dtype="bf16", quantization=quant, method="qlora" if quant else "lora",
		lora_rank=p["lora_rank"], lora_targets=("all-linear",), optimizer="adamw",
		grad_checkpointing="full" if p.get("grad_checkpointing") else "none",
	)  # fmt: skip
	return arch, setup, p["batch"], p["seq"]


def _analytical(row: dict[str, Any]) -> float | None:
	"""The engine's allocator peak for this row's config, or None when it has no analytical model for the family."""
	cfg = row["config"]
	fam, p = cfg["family"], cfg["params"]
	if fam in ("gpt", "encoder", "lora"):
		arch, setup, b, s = _transformer(p, fam)
	elif fam == "hf_lora" and row.get("model_shape"):
		arch, setup, b, s = _hf(p, row["model_shape"])
	else:
		return None  # CNN, MLP and GNN: the probes-only line, as the API uses without a model shape
	return memory.breakdown(arch, setup, b, s, context_bytes=0).allocator_peak


# What makes two configs the same architecture: everything that shapes the model, nothing about how it is run
# (dtype, attention kernel, checkpointing, LoRA rank). Scores are grouped by it, so leave-one-architecture-out
# never calibrates on a near-duplicate of the architecture it tests (Chiya, Oct 30).
_TRANSFORMER = ("n_layers", "d_model", "n_heads", "n_kv_heads", "d_ff", "gated_mlp", "vocab_size", "causal")
ARCH_FIELDS = {
	"gpt": _TRANSFORMER,
	"encoder": _TRANSFORMER,
	"lora": _TRANSFORMER,
	"cnn": ("depth", "width"),
	"mlp": ("depth", "width", "in_features"),
	"gnn": ("layers", "hidden", "conv"),
	"hf_lora": ("model_id",),
}


def arch_key(config: dict[str, Any]) -> str:
	fam, p = config.get("family", ""), config.get("params", {})
	fields = ARCH_FIELDS.get(fam)
	if fields is None:
		return f"{fam}:{p.get('group', '')}"
	return f"{fam}:" + json.dumps({k: p.get(k) for k in fields}, sort_keys=True, default=str)


def _ok(row: dict[str, Any]) -> bool:
	return row.get("status") == "ok" and bool((row.get("memory") or {}).get("peak_reserved"))


def scores_from_rows(rows: Iterable[dict[str, Any]]) -> list[conformal.Score]:
	# one fit per architecture per GPU: a group id names the architecture, and the same groups run on every GPU
	groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
	for r in rows:
		g = r.get("config", {}).get("params", {}).get("group")
		if g and _ok(r):
			groups[(g, r.get("gpu", ""))].append(r)
	out: list[conformal.Score] = []
	for (_gid, gpu), rs in sorted(groups.items()):
		probes = [r for r in rs if r["config"]["params"].get("role") == "probe"]
		queries = [r for r in rs if r["config"]["params"].get("role") == "query"]
		if len(probes) < 2 or not queries:
			continue
		fam = rs[0]["config"]["family"]
		lens = LENS.get(fam, "pretrain")
		arch = arch_key(rs[0]["config"])
		an = {id(r): _analytical(r) for r in rs}
		mp = [
			memory.MemoryProbe(float(r["work_units"]), float(r["memory"]["peak_reserved"]), an[id(r)] or 0.0)
			for r in probes
		]
		cal = memory.Calibration.fit(mp)
		sp = [
			steptime.StepProbe(float(r["work_units"]), _phases(r), r["key"] if "key" in r else str(i))
			for i, r in enumerate(probes)
		]
		model = steptime.SameGpuModel.fit(sp)
		for q in queries:
			x = float(q["work_units"])
			pred = cal.correct(x, an[id(q)] or 0.0)
			if pred > 0:
				out.append(
					conformal.Score("memory.peak", gpu, lens, arch, pred, float(q["memory"]["peak_reserved"]))
				)
			est = model.predict(x)
			if est.status != "abstain" and est.point and q.get("timing", {}).get("step_s"):
				out.append(
					conformal.Score("time.step", gpu, lens, arch, est.point, float(q["timing"]["step_s"]))
				)
	return out


def _capacity(row: dict[str, Any]) -> float | None:
	env = row.get("env") or {}
	mem = row.get("memory") or {}
	return env.get("total_memory") or mem.get("total")


def oom_pairs(rows: list[dict[str, Any]], scores: list[conformal.Score]) -> list[tuple[float, bool]]:
	"""(predicted P(OOM), did it run out of memory) for every held-out query, fitted and OOM alike.

	Each query's P(OOM) comes the way the engine computes it: from the residuals of the table cell its GPU and job
	type look up, in a table built without the query's own architecture, so the check never grades a group on itself.
	"""
	mem = [s for s in scores if s.target == "memory.peak"]
	groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
	for r in rows:
		g = r.get("config", {}).get("params", {}).get("group")
		if g:
			groups[(g, r.get("gpu", ""))].append(r)
	tables: dict[str, conformal.Table] = {}
	out = []
	for (_gid, gpu), rs in sorted(groups.items()):
		probes = [r for r in rs if r["config"]["params"].get("role") == "probe" and _ok(r)]
		if len(probes) < 2:
			continue
		arch = arch_key(rs[0]["config"])
		if arch not in tables:
			tables[arch] = conformal.calibrate([s for s in mem if s.group != arch])
		lens = LENS.get(rs[0]["config"]["family"], "pretrain")
		cell = tables[arch].lookup("memory.peak", gpu, lens)
		if cell is None or not cell.residuals:
			continue
		mp = [
			memory.MemoryProbe(
				float(r["work_units"]), float(r["memory"]["peak_reserved"]), _analytical(r) or 0.0
			)
			for r in probes
		]
		cal = memory.Calibration.fit(mp)
		for q in rs:
			if q["config"]["params"].get("role") != "query" or q.get("status") not in ("ok", "oom"):
				continue
			cap = _capacity(q)
			p = q["config"]["params"]
			x = q.get("work_units") or p.get("batch", 1) * p.get("seq", 1)
			pred = cal.correct(float(x), _analytical(q) or 0.0)
			if cap and pred > 0:
				out.append((oom.empirical_p_oom(cell.residuals, pred, float(cap)), q.get("status") == "oom"))
	return out


def _phases(row: dict[str, Any]) -> dict[str, float]:
	t = row.get("timing") or {}
	ph = {k: v for k, v in (t.get("phases") or {}).items() if v and v > 0}
	rest = (t.get("step_s") or 0.0) - sum(ph.values())
	if rest > 0:
		ph["other"] = rest  # host time between the timed marks
	return ph or {"step": t.get("step_s") or 0.0}


def load_rows(store: Path) -> list[dict[str, Any]]:
	"""Rows from a result store (one .json per config) or from modal_app's export (.jsonl.gz per GPU)."""
	import gzip

	rows = []
	for f in sorted(store.rglob("*.json")):
		if f.name.endswith(".meta.json"):
			continue
		try:
			rows.append(json.loads(f.read_text()))
		except (json.JSONDecodeError, OSError):
			continue
	for f in sorted(store.rglob("*.jsonl.gz")):
		try:
			with gzip.open(f, "rt") as fh:
				rows += [json.loads(line) for line in fh if line.strip()]
		except (json.JSONDecodeError, OSError, EOFError):
			continue
	return rows


def write_table(table: conformal.Table, out: Path) -> None:
	"""A plain module the engine (and Pyodide) imports: the table as a Python literal, not JSON text, whose
	true, false and null aren't Python."""
	import ast
	import math
	import pprint

	data = table.to_dict()

	def finite(v):
		if isinstance(v, float) and not math.isfinite(v):
			raise ValueError(f"the table holds a non-finite number ({v}); a band can't be built from it")
		if isinstance(v, dict):
			for x in v.values():
				finite(x)
		elif isinstance(v, (list, tuple)):
			for x in v:
				finite(x)

	finite(data)
	body = pprint.pformat(data, indent=1, width=110, sort_dicts=True)
	ast.literal_eval(body)  # the module must be a plain literal the engine and Pyodide import
	out.write_text(
		'"""Generated by python -m preflight.corpus.calibrate; do not edit. Held-out conformal half-widths."""\n\n'
		f"DATA = {body}\n"
	)


# PM, Oct 27: the cross-GPU holdout; never tuned on, reported once. Every A100 board: Modal's A100-80GB lands on
# SXM or PCIe (sweep.LANDS_ON), and no A100 row may reach calibration by default
HOLDOUT_GPUS = ("A100-SXM-80GB", "A100-PCIe-80GB", "A100-SXM-40GB", "A100-PCIe-40GB")


def held_out(gpu: str | None, holdout_gpus: tuple[str, ...] = HOLDOUT_GPUS) -> bool:
	"""A row on a holdout board, or on a MIG slice of one ("A100-SXM-80GB-MIG-3g.40gb")."""
	return bool(gpu) and any(gpu == h or gpu.startswith(f"{h}-MIG") for h in holdout_gpus)


MIN_PER_BIN = 10  # guess: fewer points than this and a bin's Wilson interval is too wide to read; not drawn


def build(
	scores: list[conformal.Score], holdout_gpus: tuple[str, ...] = HOLDOUT_GPUS, built_from: str = ""
) -> tuple[conformal.Table, str]:
	"""The table from every GPU but the holdout, its leave-one-architecture-out coverage, and the holdout scored
	once against it. Returns the table and a report to print."""
	cal = [s for s in scores if not held_out(s.gpu, holdout_gpus)]
	held = [s for s in scores if held_out(s.gpu, holdout_gpus)]
	table = conformal.calibrate(cal, built_from=built_from)
	lines, coverage = [], {}
	for target, cov in conformal.evaluate(cal).items():
		if not cov.n:
			lines.append(
				f"{target}: no held-out run found a cell with at least {conformal.MIN_CELL} runs; no coverage"
			)
			continue
		coverage[target] = (cov.hits, cov.n)
		lines.append(
			f"{target}: leave-one-architecture-out coverage {cov.hits} of {cov.n} ({cov.rate:.1%}, "
			f"Wilson 95% {cov.lo:.1%} to {cov.hi:.1%}), median half-width {cov.median_half_width:.3f}"
		)
	holdout: dict[str, dict[str, tuple[int, int]]] = {}
	# the holdout GPU has no cell of its own, so it is scored on the merged cells, as a user's GPU is, and on a table
	# without its own architecture, as a user's new model is
	by_group: dict[str, conformal.Table] = {}
	for s in held:
		if s.group not in by_group:
			by_group[s.group] = conformal.calibrate([c for c in cal if c.group != s.group])
		cell = by_group[s.group].lookup(s.target, s.gpu, s.lens)
		if cell is None:
			continue
		h, n = holdout.setdefault(s.target, {}).get(s.gpu, (0, 0))
		holdout[s.target][s.gpu] = (h + (s.value <= cell.half_width), n + 1)
	for target, gs in sorted(holdout.items()):
		for gpu, (h, n) in sorted(gs.items()):
			lo, hi = conformal.wilson(h, n)
			lines.append(
				f"{target}: holdout {gpu}, {h} of {n} inside the band (Wilson 95% {lo:.1%} to {hi:.1%}); reported once"
			)
	table = replace(table, coverage=coverage, holdout=holdout)
	return table, "\n".join(lines)


def reliability_lines(pairs: list[tuple[float, bool]], min_per_bin: int = MIN_PER_BIN) -> list[str]:
	"""The reliability diagram as text: predicted against observed per bin, with Wilson intervals; thin bins
	are named but not drawn."""
	out = ["P(OOM) reliability (predicted vs observed, Wilson 95% on each bin):"]
	for b in oom.reliability(pairs):
		if b.n == 0:
			continue
		if b.n < min_per_bin:
			out.append(f"  [{b.lo:.2f}, {b.hi:.2f}): {b.n} points, too few to draw")
			continue
		out.append(
			f"  [{b.lo:.2f}, {b.hi:.2f}) n={b.n} predicted {b.mean_predicted:.2f} observed {b.observed:.2f} "
			f"(Wilson 95% {b.wilson_lo:.2f} to {b.wilson_hi:.2f}){'' if b.consistent else ' INCONSISTENT'}"
		)
	return out


def parse_lenses(spec: str) -> dict[str, frozenset[str]]:
	"""'time.step=pretrain+retrieval+vision;memory.peak=vision' -> {target: lenses}; empty means no limit."""
	out: dict[str, frozenset[str]] = {}
	for part in (p.strip() for p in spec.split(";")):
		if not part:
			continue
		target, _, lenses = part.partition("=")
		names = frozenset(x.strip() for x in lenses.split("+") if x.strip())
		if not target.strip() or not names:
			raise SystemExit(f"--lenses wants target=lens+lens, got {part!r}")
		out[target.strip()] = names
	return out


def main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser(prog="preflight.corpus.calibrate")
	ap.add_argument("--store", type=Path, required=True)
	ap.add_argument("--out", type=Path, help="write the engine's _calibration_table.py here")
	ap.add_argument("--holdout-gpus", default=",".join(HOLDOUT_GPUS), help="never calibrated on; scored once")
	ap.add_argument(
		"--built-from", help="where the rows came from, for the model cards (default: the store path)"
	)
	ap.add_argument(
		"--targets",
		default="memory.peak,time.step",
		help="targets whose cells go into the table; the others keep their simulated bands and say so",
	)
	ap.add_argument(
		"--keep-holdout",
		type=Path,
		help="carry the holdout scores forward from this table; the store must then hold no holdout rows",
	)
	ap.add_argument(
		"--score-holdout",
		action="store_true",
		help="score the holdout rows in the store: once only, refused when --out already holds holdout scores",
	)
	ap.add_argument(
		"--cross",
		action="store_true",
		help="also build the cross-GPU band (time.step.cross) from configs two GPUs both ran",
	)
	ap.add_argument(
		"--lenses",
		default="",
		help="'target=lens+lens;...': for these targets, only these lenses get cells, and there is no pooled "
		"all-lens cell, so every other lens keeps its simulated band",
	)
	args = ap.parse_args(argv)
	only = parse_lenses(args.lenses)
	holdout = tuple(g.strip() for g in args.holdout_gpus.split(",") if g.strip())
	rows = load_rows(args.store)
	held_rows = [r for r in rows if held_out(r.get("gpu"), holdout)]
	kept = None
	if args.keep_holdout:
		import ast

		if held_rows:
			raise SystemExit(
				f"--keep-holdout: the store still holds {len(held_rows)} holdout rows; remove them so they aren't scored again"
			)
		kept = ast.literal_eval(args.keep_holdout.read_text().split("=", 1)[1]).get("holdout") or {}
	elif held_rows and not args.score_holdout:
		raise SystemExit(
			f"the store holds {len(held_rows)} holdout rows: pass --score-holdout to score them (once), or remove them "
			"and pass --keep-holdout <table> to carry the earlier scores forward"
		)
	elif held_rows and args.out and args.out.exists() and "'holdout': {}" not in args.out.read_text():
		raise SystemExit(f"{args.out} already holds holdout scores: the holdout is scored once")
	scores = scores_from_rows(rows)  # grouped rows -> conformal scores
	# the corpus's synthetic LoRA models aren't what users send (real HF models are): reported, never calibrated on
	synthetic = [s for s in scores if s.lens in SYNTHETIC_LENSES]
	scores = [s for s in scores if s.lens not in SYNTHETIC_LENSES]
	if synthetic:
		print(
			f"left out of the table: {len(synthetic)} scores from synthetic lenses {sorted({s.lens for s in synthetic})}"
		)
	scores = [s for s in scores if s.target not in only or s.lens in only[s.target]]
	if not scores:
		print("no grouped rows with probes and queries; nothing to calibrate")
		return 1
	cross_cases = []
	if args.cross:
		from preflight.corpus import crossgpu

		cross_cases = crossgpu.cases(rows)  # held-out boards never enter (calibrate.held_out)
		scores = scores + crossgpu.scores(cross_cases)
	table, report = build(
		scores,
		holdout,
		built_from=f"{args.built_from or args.store} ({len(scores)} scores, holdout {', '.join(holdout) or 'none'})",
	)
	print(report)
	cal_rows = [r for r in rows if not held_out(r.get("gpu"), holdout)]
	pairs = oom_pairs(cal_rows, [s for s in scores if not held_out(s.gpu, holdout)])
	print("\n".join(reliability_lines(pairs)))
	ok = oom.reliable(oom.reliability(pairs), min_per_bin=MIN_PER_BIN)
	print(f"  reliable enough to show as a probability: {ok}")
	middle = oom.middle_verified(oom.reliability(pairs, edges=oom.SHOWN_EDGES))
	print(f"  shown as a number between 5% and 95%: {middle} (each middle bin needs {oom.MIN_SHOWN} points)")
	keep = {t.strip() for t in args.targets.split(",") if t.strip()}
	if cross_cases:
		from preflight.corpus import crossgpu

		keep.add(crossgpu.TARGET)
		h, n = crossgpu.gpu_out_coverage(cross_cases)
		lo, hi = conformal.wilson(h, n)
		print(
			f"{crossgpu.TARGET}: leave-one-GPU-out coverage {h} of {n} ({h / n:.1%}, Wilson 95% {lo:.1%} to {hi:.1%}; "
			"pairs share architectures, so the honest interval is wider: see the decision doc)"
		)
		table = replace(table, coverage={**table.coverage, crossgpu.TARGET: (h, n)})
	table = replace(
		table,
		oom_reliable=ok,
		oom_bins=tuple(
			(b.lo, b.hi, b.n, b.consistent, round(b.observed * b.n) if b.n else 0)
			for b in oom.reliability(pairs, edges=oom.SHOWN_EDGES)
		),
		# a lens-limited target has no all-lens cell: an unlisted lens must not borrow the listed lenses' band
		cells={k: c for k, c in table.cells.items() if k[0] in keep and not (k[0] in only and not k[2])},
		coverage={t: v for t, v in table.coverage.items() if t in keep},
		holdout={t: v for t, v in table.holdout.items() if t in keep},
	)
	if kept is not None:
		table = replace(
			table, holdout={t: {g: (int(v[0]), int(v[1])) for g, v in gs.items()} for t, gs in kept.items()}
		)
		print(f"holdout scores carried forward from {args.keep_holdout}, not re-scored")
	# only memory's residuals are read (P(OOM)); the rest would only bloat the module the browser loads
	table = replace(
		table,
		cells={k: (c if k[0] == "memory.peak" else replace(c, residuals=())) for k, c in table.cells.items()},
	)
	# the table's version is a hash of its bands, so a record can tell when the bands behind its numbers changed
	import hashlib

	body = json.dumps({"cells": table.to_dict()["cells"], "oom": table.to_dict()["oom_bins"]}, sort_keys=True)
	table = replace(table, version=hashlib.sha256(body.encode()).hexdigest()[:10])
	if args.out:
		write_table(table, args.out)
		print(f"wrote {args.out} with {len(table.cells)} cells")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
