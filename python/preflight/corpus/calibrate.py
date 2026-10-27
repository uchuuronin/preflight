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
from pathlib import Path
from typing import Any

from preflight.engine import conformal, knobs, memory, steptime
from preflight.engine.memory import TrainingSetup, TransformerArch

LENS = {
	"gpt": "pretrain",
	"lora": "finetune",
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


def _ok(row: dict[str, Any]) -> bool:
	return row.get("status") == "ok" and bool((row.get("memory") or {}).get("peak_reserved"))


def scores_from_rows(rows: Iterable[dict[str, Any]]) -> list[conformal.Score]:
	groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
	for r in rows:
		g = r.get("config", {}).get("params", {}).get("group")
		if g and _ok(r):
			groups[g].append(r)
	out: list[conformal.Score] = []
	for gid, rs in sorted(groups.items()):
		probes = [r for r in rs if r["config"]["params"].get("role") == "probe"]
		queries = [r for r in rs if r["config"]["params"].get("role") == "query"]
		if len(probes) < 2 or not queries:
			continue
		fam = rs[0]["config"]["family"]
		lens = LENS.get(fam, "pretrain")
		gpu = rs[0]["gpu"]
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
					conformal.Score("memory.peak", gpu, lens, gid, pred, float(q["memory"]["peak_reserved"]))
				)
			est = model.predict(x)
			if est.status != "abstain" and est.point and q.get("timing", {}).get("step_s"):
				out.append(
					conformal.Score("time.step", gpu, lens, gid, est.point, float(q["timing"]["step_s"]))
				)
	return out


def _phases(row: dict[str, Any]) -> dict[str, float]:
	t = row.get("timing") or {}
	ph = {k: v for k, v in (t.get("phases") or {}).items() if v and v > 0}
	rest = (t.get("step_s") or 0.0) - sum(ph.values())
	if rest > 0:
		ph["other"] = rest  # host time between the timed marks
	return ph or {"step": t.get("step_s") or 0.0}


def load_rows(store: Path) -> list[dict[str, Any]]:
	rows = []
	for f in sorted(store.rglob("*.json")):
		if f.name.endswith(".meta.json"):
			continue
		try:
			rows.append(json.loads(f.read_text()))
		except (json.JSONDecodeError, OSError):
			continue
	return rows


def write_table(table: conformal.Table, out: Path) -> None:
	body = json.dumps(table.to_dict(), indent="\t", sort_keys=True)
	out.write_text(
		'"""Generated by python -m preflight.corpus.calibrate; do not edit. Held-out conformal half-widths."""\n\n'
		f"DATA = {body}\n"
	)


def main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser(prog="preflight.corpus.calibrate")
	ap.add_argument("--store", type=Path, required=True)
	ap.add_argument("--out", type=Path, help="write the engine's _calibration_table.py here")
	args = ap.parse_args(argv)
	scores = scores_from_rows(load_rows(args.store))
	if not scores:
		print("no grouped rows with probes and queries; nothing to calibrate")
		return 1
	for target, cov in conformal.evaluate(scores).items():
		print(
			f"{target}: leave-one-architecture-out coverage {cov.rate:.1%} "
			f"(Wilson 95% {cov.lo:.1%}-{cov.hi:.1%}, n={cov.n}), median half-width {cov.median_half_width:.3f}"
		)
	table = conformal.calibrate(scores, built_from=f"{args.store} ({len(scores)} scores)")
	if args.out:
		write_table(table, args.out)
		print(f"wrote {args.out} with {len(table.cells)} cells")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
