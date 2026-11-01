"""Spark Open labels the corpus's failures from each row's numbers (contract item; Chiya, Nov 1, up to $1).

For every corpus row that ran out of memory and every divergence run that diverged, Muse Spark Open on Baseten
(`meta-models/muse-spark-open-1026`, effort low, structured output) names the failure mode from the row's own
numbers. B's rules label the same rows independently: an out-of-memory row by the engine's largest memory term at
the row's settings (engine/memory.breakdown), a diverged run by the D9 rule's reason (corpus/labels.py). The rule
labels are never in the prompt. The report is the agreement rate with a Wilson interval; every call, failed or not,
is logged with its cost in the corpus provenance.

    python scripts/spark_label_corpus.py prepare --store <corpus store> --divergence <dir of divergence rows>
        writes python/preflight/corpus/data/failures_nov1.jsonl (inputs and rule labels; no calls)
    python scripts/spark_label_corpus.py label        # in remote-run.yml, which holds BASETEN_API_KEY
        writes out/spark_labels.jsonl (one line per row: the label and the call) and out/spark_labels_calls.jsonl
    python scripts/spark_label_corpus.py score --labels out/spark_labels.jsonl
        prints the agreement and writes python/preflight/corpus/reports/spark_labels_nov1.json
"""

from __future__ import annotations

import argparse
import glob
import gzip
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / "python/preflight/corpus/data/failures_nov1.jsonl"
REPORT = ROOT / "python/preflight/corpus/reports/spark_labels_nov1.json"
PROVENANCE = ROOT / "python/preflight/corpus/provenance/spark_open_labels_nov1.jsonl"
BUDGET_USD = 1.00  # Chiya, Nov 1: pre-approved up to $1
MODEL = "meta-models/muse-spark-open-1026"

OOM_LABELS = ("model_state", "activations", "logits", "other")
DIV_LABELS = ("non_finite", "blow_up", "collapse", "other")
OOM_DEFS = (
	"model_state: weights, gradients and optimizer state fill the GPU, whatever the batch; "
	"activations: per-layer activations (and attention scores) for the batch and sequence fill it; "
	"logits: the output layer's batch x sequence x vocabulary logits (and their softmax) fill it; "
	"other: none of these, or you can't tell."
)
DIV_DEFS = (
	"non_finite: the loss becomes NaN or infinite; "
	"blow_up: the loss stays finite but ends well above the best it reached; "
	"collapse: the loss spikes far above its starting value and never learns below about its starting level; "
	"other: none of these, or you can't tell."
)


# ---------------------------------------------------------------- prepare (local, no calls)


def _rows(pattern: str) -> list[dict]:
	return [json.loads(line) for f in sorted(glob.glob(pattern)) for line in gzip.open(f, "rt")]


def _oom_rule(row: dict) -> str | None:
	from preflight.corpus import calibrate as cal
	from preflight.engine import memory

	cfg = row["config"]
	fam, p = cfg["family"], cfg["params"]
	if fam in ("gpt", "encoder", "lora"):
		arch, setup, b, s = cal._transformer(p, fam)
	elif fam == "hf_lora" and row.get("model_shape"):
		arch, setup, b, s = cal._hf(p, row["model_shape"])
	else:
		return None  # no analytical model for this family: Spark labels it, the agreement leaves it out
	st = memory.breakdown(arch, setup, b, s, context_bytes=0).stack
	groups = {
		"model_state": sum(
			st.get(t, 0.0) for t in ("weights", "master_weights", "gradients", "optimizer", "frozen_models")
		),
		"activations": st.get("activations", 0.0),
		"logits": st.get("logits", 0.0),
	}
	return max(groups, key=groups.get)


def _div_rule(row: dict) -> str:
	reason = row["label"]["reason"]
	if "non-finite" in reason:
		return "non_finite"
	if "well above its best" in reason:
		return "blow_up"
	return "collapse"


def _finite(xs: list) -> list[float]:
	return [x for x in xs if isinstance(x, (int, float)) and math.isfinite(x)]


def _div_evidence(row: dict) -> dict:
	s = row["series"]
	loss = s.get("loss") or []
	fin = _finite(loss)
	first_bad = next(
		(i for i, x in enumerate(loss) if not (isinstance(x, (int, float)) and math.isfinite(x))), None
	)
	best_i = (
		min(
			range(len(loss)),
			key=lambda i: (
				loss[i] if i < len(loss) and loss[i] is not None and math.isfinite(loss[i]) else math.inf
			),
		)
		if fin
		else None
	)

	def ends(name: str) -> dict | None:
		v = _finite(s.get(name) or [])
		return {"first": round(v[0], 4), "max": round(max(v), 4), "last": round(v[-1], 4)} if v else None

	gn = _finite(s.get("grad_norm") or [])
	scale = [x for x in (s.get("loss_scale") or []) if x is not None]
	drops = sum(1 for a, b in itertools.pairwise(scale) if b < a)
	return {
		"settings": row["config"]["params"],
		"steps_run": row.get("steps_run"),
		"stopped_early": row.get("stopped_early"),
		"loss": {
			"first": round(fin[0], 4) if fin else None,
			"best": round(min(fin), 4) if fin else None,
			"best_step": best_i,
			"last_finite": round(fin[-1], 4) if fin else None,
			"median_last_50": round(sorted(fin[-50:])[len(fin[-50:]) // 2], 4) if fin else None,
			"first_non_finite_step": first_bad,
		},
		"grad_norm": {"median": round(sorted(gn)[len(gn) // 2], 3), "max": round(max(gn), 3)} if gn else None,
		"max_attn_logit": ends("max_attn_logit"),
		"output_logit_norm": ends("output_logit_norm"),
		"nonfinite_grad_steps": sum(1 for x in (s.get("nonfinite_grads") or []) if x),
		"loss_scale_drops": drops,
	}


def prepare(store: str, divergence: str) -> None:
	items = []
	for kind in ("kernels", "hf", "grouped"):
		for r in _rows(f"{store}/{kind}/*.jsonl.gz"):
			if r.get("status") != "oom":
				continue
			evidence = {
				"family": r["config"]["family"],
				"settings": r["config"]["params"],
				"model_shape": r.get("model_shape"),
				"gpu": r.get("gpu"),
				"gpu_total_bytes": (r.get("memory") or {}).get("total"),
				"peak_allocated_before_failing_bytes": (r.get("memory") or {}).get("peak_allocated"),
				"n_params": r.get("n_params"),
			}
			# review, Nov 1: the config key repeats across GPUs, so the row is the GPU and the key
			items.append(
				{
					"row_id": f"{r.get('gpu')}/{r['key']}",
					"task": "oom",
					"evidence": evidence,
					"rule": _oom_rule(r),
				}
			)
	for r in _rows(f"{divergence}/*.jsonl.gz"):
		if (r.get("label") or {}).get("diverged"):
			items.append(
				{
					"row_id": f"{r.get('gpu')}/{r['key']}",
					"task": "divergence",
					"evidence": _div_evidence(r),
					"rule": _div_rule(r),
				}
			)
	INPUTS.parent.mkdir(parents=True, exist_ok=True)
	INPUTS.write_text("".join(json.dumps(i, sort_keys=True) + "\n" for i in items))
	counts = {t: sum(1 for i in items if i["task"] == t) for t in ("oom", "divergence")}
	print(
		f"wrote {INPUTS.relative_to(ROOT)}: {counts}; rows without a rule label: {sum(1 for i in items if i['rule'] is None)}"
	)


# ---------------------------------------------------------------- label (Baseten; remote-run.yml)


def _schema(labels: tuple[str, ...]):
	from pydantic import BaseModel

	class Label(BaseModel):
		label: Literal[labels]  # type: ignore[valid-type]
		confidence: Literal["low", "medium", "high"]
		why: str  # review, Nov 1: no length cap (strict schemas drop it, and a long answer would fail); cut at 300 below

	return Label


def _messages(item: dict) -> list[dict]:
	if item["task"] == "oom":
		ask = (
			"A training run on one GPU ran out of memory. From its settings and numbers below, which memory use filled "
			f"the GPU? Labels: {OOM_DEFS}"
		)
	else:
		ask = f"A small GPT training run diverged. From its settings and loss numbers below, how did it fail? Labels: {DIV_DEFS}"
	return [
		{
			"role": "system",
			"content": "You label failed ML training runs for a dataset. Answer only from the numbers given. "
			"Return JSON matching the schema: one label, your confidence, and one sentence of why.",
		},
		{"role": "user", "content": f"{ask}\n\nRun:\n{json.dumps(item['evidence'], sort_keys=True)}"},
	]


RELABEL_BUDGET_USD = 0.48  # Chiya, Nov 1: relabel the unanswered rows only if it stays under $0.50
RELABEL_MAX_TOKENS = 16_384  # the first run's 8,192 cut 8 answers off before their JSON closed
RELABEL_TIMEOUT_S = 90.0


def relabel(first: str, out_dir: str) -> None:
	"""The rows the first run left unanswered, again, with room to finish: a higher output cap and a longer
	timeout. A row starts only if its worst case (two calls at the full cap) still fits under the budget."""
	from preflight.harness.spark import ModelClient

	done = {r["row_id"]: r for r in (json.loads(x) for x in Path(first).read_text().splitlines())}
	items = [
		i
		for i in (json.loads(x) for x in INPUTS.read_text().splitlines())
		if not done[i["row_id"]].get("label")
	]
	client = ModelClient.baseten_spark_open(max_output_tokens=RELABEL_MAX_TOKENS)
	client.timeout_s, client.deadline_s = RELABEL_TIMEOUT_S, 2.5 * RELABEL_TIMEOUT_S
	if client.model != MODEL:
		raise SystemExit(f"expected {MODEL}, the client would call {client.model} (SPARK_OPEN_MODEL)")
	rate_in, _cached, rate_out = client.prices.get(MODEL, (1.25, 1.25, 4.25))
	worst_row = 2 * (RELABEL_MAX_TOKENS * rate_out + 2_000 * rate_in) / 1e6
	out = Path(out_dir)
	out.mkdir(parents=True, exist_ok=True)
	spent = 0.0
	with open(out / "spark_relabels.jsonl", "w") as fh:
		for item in items:
			if spent + worst_row > RELABEL_BUDGET_USD:
				print(
					f"stopping before {item['row_id']}: ${spent:.3f} spent, a row can cost ${worst_row:.3f}"
				)
				break
			schema = _schema(OOM_LABELS if item["task"] == "oom" else DIV_LABELS)
			before = len(client.calls)
			try:
				ans, _ = client.structured(_messages(item), schema, purpose="label", reasoning_effort="low")
				got = {"label": ans.label, "confidence": ans.confidence, "why": ans.why[:300]}
			except Exception as exc:
				got = {"label": None, "error": f"{type(exc).__name__}: {exc}"[:300]}
			calls = [c.model_dump(mode="json") for c in client.calls[before:]]
			# review, Nov 1: a call that timed out records $0, though the provider may have run and billed it; the
			# guard counts it at half a row's worst case, so the $0.50 holds whatever happened
			spent += sum(c["cost_usd"] if c["ok"] or c["output_tokens"] else worst_row / 2 for c in calls)
			row = {"row_id": item["row_id"], "task": item["task"], **got, "calls": calls, "relabel": True}
			fh.write(json.dumps(row) + "\n")
	tried = len((out / "spark_relabels.jsonl").read_text().splitlines())
	print(f"tried {tried} of {len(items)} rows; {len(client.calls)} calls; ${spent:.4f} counted")


def merge(first: str, second: str, out: str) -> None:
	"""The first run's rows with the relabelled ones in place; every call of both runs is kept for provenance."""
	rows = [json.loads(x) for x in Path(first).read_text().splitlines()]
	again = {r["row_id"]: r for r in (json.loads(x) for x in Path(second).read_text().splitlines())}
	merged = []
	for r in rows:
		if r["row_id"] in again:
			new = again[r["row_id"]]
			merged.append({**new, "calls": r["calls"] + new["calls"]})
		else:
			merged.append(r)
	Path(out).write_text("".join(json.dumps(r) + "\n" for r in merged))


def label(out_dir: str) -> None:
	from preflight.harness.spark import ModelClient

	items = [json.loads(line) for line in INPUTS.read_text().splitlines()]
	client = ModelClient.baseten_spark_open()
	if client.model != MODEL:
		raise SystemExit(f"expected {MODEL}, the client would call {client.model} (SPARK_OPEN_MODEL)")
	out = Path(out_dir)
	out.mkdir(parents=True, exist_ok=True)
	spent, done = 0.0, []
	with open(out / "spark_labels.jsonl", "w") as fh:
		for n, item in enumerate(items):
			if (
				spent >= BUDGET_USD * 0.9
			):  # stop with room to spare: the next call can't take the total past $1
				print(f"stopping at {n} of {len(items)}: ${spent:.3f} spent of ${BUDGET_USD:.2f}")
				break
			schema = _schema(OOM_LABELS if item["task"] == "oom" else DIV_LABELS)
			before = len(client.calls)
			try:
				ans, _ = client.structured(_messages(item), schema, purpose="label", reasoning_effort="low")
				got = {"label": ans.label, "confidence": ans.confidence, "why": ans.why[:300]}
			except Exception as exc:  # a refusal or a schema miss: logged, and the row counts as unlabelled
				got = {"label": None, "error": f"{type(exc).__name__}: {exc}"[:300]}
			calls = [c.model_dump(mode="json") for c in client.calls[before:]]
			spent += sum(c["cost_usd"] for c in calls)
			row = {"row_id": item["row_id"], "task": item["task"], **got, "calls": calls}
			fh.write(json.dumps(row) + "\n")
			done.append(row)
	with open(out / "spark_labels_calls.jsonl", "w") as fh:
		for c in client.calls:
			fh.write(json.dumps(c.model_dump(mode="json")) + "\n")
	print(
		f"labelled {sum(1 for r in done if r.get('label'))} of {len(items)}; {len(client.calls)} calls; ${spent:.4f}"
	)


# ---------------------------------------------------------------- score (local)


def _wilson(k: int, n: int) -> tuple[float, float]:
	from preflight.engine.conformal import wilson

	return wilson(k, n)


def score(labels_path: str, run: str) -> None:
	items = {i["row_id"]: i for i in (json.loads(x) for x in INPUTS.read_text().splitlines())}
	got = [json.loads(x) for x in Path(labels_path).read_text().splitlines()]
	calls = [c for r in got for c in r["calls"]]
	rep: dict = {
		"model": MODEL,
		"effort": "low",
		"run": run,
		"rows": len(items),
		"answered": sum(1 for r in got if r.get("label")),
	}
	for task in ("oom", "divergence"):
		pairs = [(items[r["row_id"]]["rule"], r.get("label")) for r in got if r["task"] == task]
		comparable = [(a, b) for a, b in pairs if a is not None and b is not None]
		k, n = sum(1 for a, b in comparable if a == b), len(comparable)
		lo, hi = _wilson(k, n) if n else (None, None)
		confusion: dict[str, dict[str, int]] = {}
		for a, b in comparable:
			confusion.setdefault(a, {}).setdefault(b, 0)
			confusion[a][b] += 1
		rep[task] = {
			"rows": len(pairs),
			"compared": n,
			"agree": k,
			"rate": k / n if n else None,
			"wilson_95": [lo, hi],
			"rule_vs_spark": confusion,
			"no_rule_label": sum(1 for a, _ in pairs if a is None),
			"no_answer": sum(1 for _, b in pairs if b is None),
		}
	all_pairs = [(items[r["row_id"]]["rule"], r.get("label")) for r in got]
	comp = [(a, b) for a, b in all_pairs if a is not None and b is not None]
	k, n = sum(1 for a, b in comp if a == b), len(comp)
	rep["overall"] = {
		"compared": n,
		"agree": k,
		"rate": k / n if n else None,
		"wilson_95": list(_wilson(k, n)) if n else None,
	}
	rep["calls"] = len(calls)
	rep["cost_usd"] = round(sum(c["cost_usd"] for c in calls), 4)
	rep["tokens"] = {
		"input": sum(c["input_tokens"] for c in calls),
		"output": sum(c["output_tokens"] for c in calls),
	}
	REPORT.write_text(json.dumps(rep, indent=1) + "\n")
	PROVENANCE.parent.mkdir(parents=True, exist_ok=True)
	PROVENANCE.write_text(
		"".join(
			json.dumps({"row_id": r["row_id"], "task": r["task"], **c}) + "\n"
			for r in got
			for c in r["calls"]
		)
	)
	print(json.dumps(rep, indent=1))


def main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser()
	sub = ap.add_subparsers(dest="cmd", required=True)
	p = sub.add_parser("prepare")
	p.add_argument("--store", required=True)
	p.add_argument("--divergence", required=True)
	lab = sub.add_parser("label")
	lab.add_argument("--out", default="out")
	rl = sub.add_parser("relabel")
	rl.add_argument(
		"--first", default=str(ROOT / "python/preflight/corpus/reports/spark_labels_nov1_rows.jsonl")
	)
	rl.add_argument("--out", default="out")
	mg = sub.add_parser("merge")
	mg.add_argument("--first", required=True)
	mg.add_argument("--second", required=True)
	mg.add_argument("--out", required=True)
	s = sub.add_parser("score")
	s.add_argument("--labels", required=True)
	s.add_argument("--run", default="")
	a = ap.parse_args(argv)
	if a.cmd == "prepare":
		prepare(a.store, a.divergence)
	elif a.cmd == "label":
		label(a.out)
	elif a.cmd == "relabel":
		relabel(a.first, a.out)
	elif a.cmd == "merge":
		merge(a.first, a.second, a.out)
	else:
		score(a.labels, a.run)
	return 0


if __name__ == "__main__":
	sys.exit(main())
