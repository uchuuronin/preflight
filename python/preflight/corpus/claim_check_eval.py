"""Planted claims for the claim check (harness/claim_check.py; PM, Nov 2, item 3): true numbers must pass, altered
ones must be caught.

	uv run --extra probe python -m preflight.corpus.claim_check_eval [--out report.json]   # code only, no model
	modal run --env main python/preflight/corpus/claim_check_app.py::evaluate              # with Spark Open picking

Each claim is one sentence stating one raw probe number of a real record (the samples, the zoo records, a live
fixture), written the way a finding writes it: GB, seconds, a percentage, dollars, a rate. The true claim is the
number as measured; its altered twin moves it by 15%, 20% or 50%, which a reader can't tell apart by eye. A true
claim passes when it reads "supported"; an altered one is caught when it reads "unsupported". Wilson 95% intervals.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

# the repo root, or where the Modal image puts the records (corpus/claim_check_app.py)
ROOT = Path(os.environ.get("PREFLIGHT_CLAIMS_ROOT", Path(__file__).resolve().parents[3]))
RECORDS = (
	"schema/examples/finetune.json",
	"schema/examples/pretrain.json",
	"schema/examples/agent.json",
	"tests/fixtures/b-a-smoke-l4-main-4.record.json",
	"zoo/grpo_trl/record_nov1_a10.json",
	"zoo/gnn_pyg/record_nov1_l4.json",
	"zoo/retrieval_st/record_nov1_l4.json",
	"zoo/vision_timm/record_nov1_l4.json",
	"zoo/eval_sweep_reasoning/record_nov2_a10.json",
	"zoo/agent_eval_spark/record_oct31.json",
)
PER_RECORD = 12  # guess: enough fields per record to cover each kind it has
SHIFTS = (0.15, -0.20, 0.50)  # guess: errors a careless model makes, too small to see by eye
KINDS = ("bytes", "seconds", "fraction", "usd", "rate")


def render(kind: str, v: float) -> str | None:
	"""The number as a finding would write it."""
	if kind == "bytes":
		return f"{v / 1e9:.1f} GB" if v >= 1e8 else f"{v / 1e6:.0f} MB" if v >= 1e6 else None
	if kind == "seconds":
		return f"{v:.2f} s" if v < 10 else f"{v:.0f} s"
	if kind == "fraction":
		return f"{v * 100:.1f}%" if 0 < v <= 1 else None
	if kind == "usd":
		return f"${v:.3f}" if v >= 0.001 else None
	if kind == "rate":
		return f"{v:,.0f} tokens/s" if v >= 10 else None
	return None


def claims(seed: int = 0) -> list[dict]:
	from preflight.harness.claim_check import _kind_of, raw_numbers
	from preflight.schema import load_record

	rng = random.Random(seed)
	out = []
	for path in RECORDS:
		rec = load_record((ROOT / path).read_text())
		raw = raw_numbers(rec)
		fields = [(f, v, _kind_of(f)) for f, v in sorted(raw.items()) if _kind_of(f) in KINDS and v > 0]
		rng.shuffle(fields)
		picked, per_kind = [], {}
		for f, v, k in fields:  # spread across kinds first
			if per_kind.get(k, 0) < PER_RECORD // 3 and render(k, v):
				picked.append((f, v, k))
				per_kind[k] = per_kind.get(k, 0) + 1
			if len(picked) >= PER_RECORD:
				break
		for i, (f, v, k) in enumerate(picked):
			shift = SHIFTS[i % len(SHIFTS)]
			for planted, value in (("true", v), ("altered", v * (1 + shift))):
				num = render(k, value)
				# an altered number that rounds to the same text isn't a different claim ("$0.003" both ways)
				if num is None or (planted == "altered" and num == render(k, v)):
					continue
				out.append(
					{
						"record": path,
						"field": f,
						"kind": k,
						"planted": planted,
						"shift": 0.0 if planted == "true" else shift,
						"text": f"The probe measured {num} for this job.",
						"number": num,
					}
				)
	return out


def run(items: list[dict], client=None) -> list[dict]:
	from preflight.harness.claim_check import check_text
	from preflight.schema import load_record

	cache = {}
	for it in items:
		rec = cache.setdefault(it["record"], load_record((ROOT / it["record"]).read_text()))
		checks, _ = check_text(it["text"], rec, client=client)
		# the check's number is the figure as parsed ("2,877" of "2,877 tokens/s")
		c = next((c for c in checks if it["number"].startswith(c.number)), None)
		it["verdict"] = c.verdict if c else "not_found"
		it["picked"] = c.field if c else None
	return items


def summary(items: list[dict]) -> dict:
	from preflight.engine.conformal import wilson

	def tally(rows, ok):
		k, n = sum(ok(r) for r in rows), len(rows)
		lo, hi = wilson(k, n) if n else (None, None)
		return {"k": k, "n": n, "rate": k / n if n else None, "wilson95": [lo, hi]}

	true = [r for r in items if r["planted"] == "true"]
	alt = [r for r in items if r["planted"] == "altered"]
	verdicts = {p: {} for p in ("true", "altered")}
	for r in items:
		verdicts[r["planted"]][r["verdict"]] = verdicts[r["planted"]].get(r["verdict"], 0) + 1
	return {
		"true_pass": tally(true, lambda r: r["verdict"] == "supported"),
		"true_pass_right_field": tally(
			true, lambda r: r["verdict"] == "supported" and r["picked"] == r["field"]
		),
		"altered_caught": tally(alt, lambda r: r["verdict"] == "unsupported"),
		"verdicts": verdicts,
		"by_kind": {
			k: {
				"true_pass": tally(
					[r for r in true if r["kind"] == k], lambda r: r["verdict"] == "supported"
				),
				"altered_caught": tally(
					[r for r in alt if r["kind"] == k], lambda r: r["verdict"] == "unsupported"
				),
			}
			for k in KINDS
		},
	}


def main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument("--out", type=Path)
	args = ap.parse_args(argv)
	items = run(claims())
	rep = {"model": None, "summary": summary(items), "items": items}
	if args.out:
		args.out.write_text(json.dumps(rep, indent=1) + "\n")
	print(json.dumps(rep["summary"], indent=1))
	return 0


if __name__ == "__main__":
	sys.exit(main())
