"""Data-quality exposure on a real corpus (PM, Nov 2, item 5): FineWeb-Edu's sample-10BT, CPU only.

	modal run --env main python/preflight/corpus/quality_check_app.py::check

Streams the first DOCS documents of HuggingFaceFW/fineweb-edu (config sample-10BT) with the `datasets` library, bins
them by the dataset's own `int_score` and counts tokens with its own `token_count` column (engine/quality.py), scales
the bins to the sample's 10B tokens, and prints the exposure table at three budgets: one pass, three passes, and
three passes with the mixture weighted 4:1 towards the documents scoring 4 and 5. The same for
fineweb-edu-score-2 (the authors' looser set, which keeps int_score 2), so the low-quality warning has something to
read. Writes out/quality_check.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

DOCS = 200_000  # guess: enough documents that the smallest bin has thousands
SETS = (
	# cited: the dataset card, "sample-10BT": 10B GPT-2 tokens
	("HuggingFaceFW/fineweb-edu", "sample-10BT", 10e9),
	# the looser set has no sample config (Nov 2 run: "BuilderConfig 'sample-10BT' not found"): one Common Crawl dump,
	# scaled to the documents read, since the dump's own token total isn't on its card
	("HuggingFaceFW/fineweb-edu-score-2", "CC-MAIN-2025-05", None),
)
BUDGETS = ((1, None), (3, None), (3, {"int_score 4": 4.0, "int_score 5": 4.0}))

app = modal.App("preflight-quality-check")
image = (
	modal.Image.debian_slim(python_version="3.11")
	.pip_install("pydantic>=2.8,<3", "numpy>=1.26", "httpx>=0.27", "datasets==3.1.0")
	.add_local_python_source("preflight")
)


@app.function(image=image, timeout=40 * 60, cpu=2, memory=4096)
def check_remote(docs: int = DOCS) -> dict:
	from datasets import load_dataset

	from preflight.engine import quality as Q

	out = {"docs": docs, "sets": []}
	for name, config, total in SETS:
		row: dict = {"dataset": name, "config": config, "dataset_tokens": total}
		try:
			ds = load_dataset(name, config, split="train", streaming=True)
			# review, Nov 2: a streamed dataset's features can be None; the first row's keys are the columns
			cols = list(ds.features) if ds.features else list(next(iter(ds)).keys())
			col = Q.column_for(name, cols)
			row["column"] = col and col.name
			if col is None:
				row["error"] = f"no known quality column in {cols}"
				out["sets"].append(row)
				continue
			t0 = time.monotonic()
			p = Q.profile(ds.take(docs), col)
			row["read_s"] = round(time.monotonic() - t0, 1)
			row["sample"] = {"docs": p.docs, "tokens": p.tokens, "missing": p.missing, "low": sorted(p.low)}
			row["tables"] = []
			unique = total or sum(p.tokens.values())
			row["budget_base_tokens"] = (
				unique  # the dataset's total, or the documents read when its total isn't known
			)
			for passes, weights in BUDGETS:
				rows = Q.table(p, passes * unique, total, weights)
				row["tables"].append(
					{
						"budget_tokens": passes * unique,
						"weights": weights,
						"rows": [r.__dict__ for r in rows],
					}
				)
		except Exception as exc:  # one dataset's failure is its own row
			row["error"] = f"{type(exc).__name__}: {exc}"[:400]
		out["sets"].append(row)
	return out


@app.local_entrypoint()
def check(docs: int = DOCS):
	t0 = time.monotonic()
	got = check_remote.remote(docs)
	got["wall_s"] = round(time.monotonic() - t0, 1)
	Path("out").mkdir(exist_ok=True)
	Path("out/quality_check.json").write_text(json.dumps(got, indent=1))
	for s in got["sets"]:
		print(s["dataset"], s.get("column"), s.get("sample", {}).get("docs"), s.get("error", ""))
		for t in s.get("tables", []):
			print("  budget", t["budget_tokens"], t["weights"])
			for r in t["rows"]:
				print(
					f"    {r['name']:<14} low={r['low_quality']!s:<5} unique {r['unique_tokens'] / 1e9:6.2f}B  per epoch {r['seen_per_epoch'] / 1e9:6.2f}B  epochs {r['epochs']:5.2f}  {r['flag'] or ''}"
				)
