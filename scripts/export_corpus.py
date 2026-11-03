"""Export the corpus as one file for publication (gap 16): every row, tagged with its kind and split.

    python scripts/export_corpus.py --store <store with kernels/ and hf/> --divergence <dir of divergence rows> \
        [--divergence-real <dir of the Nov 2 realistic sweep's rows>] --out <dir>

Writes <out>/preflight-corpus-<version>.jsonl.gz (one JSON row per line, keys sorted), <out>/SHA256SUMS and
<out>/README.md (a dataset card for a Hugging Face dataset or a release asset, from DATASHEET.md's facts). The
rows are exactly what the corpus worker wrote, plus three fields: `kind` (kernels, hf, divergence or
divergence_real), `split` (calibration, cross_gpu_holdout, divergence or not_in_calibration) and `source_file`.
"""

from __future__ import annotations

import argparse
import collections
import glob
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

VERSION = (
	"0.4.0"  # the published file's version: 0.3.0's rows plus the Nov 2 realistic sweep and the MI300X rows
)
# cited: the A100 rows are the cross-GPU holdout, never calibrated on (calibration-l4-phase1.md); the MI300X rows
# likewise (Chiya, Nov 2: scored once for NVIDIA to MI300X, no refit on them)
HOLDOUT = ("A100", "MI300X")


def _split(kind: str, row: dict) -> str:
	if kind in ("divergence", "divergence_real"):
		return "divergence"
	if str(row.get("gpu", "")).upper().startswith(HOLDOUT):
		return "cross_gpu_holdout"
	if (row.get("config") or {}).get("family") == "lora":
		return "not_in_calibration"  # the synthetic LoRA family: scored, never calibrated on (DATASHEET.md)
	return "calibration"


def _rows(pattern: str):
	for f in sorted(glob.glob(pattern)):
		with gzip.open(f, "rt") as fh:
			for line in fh:
				yield os.path.basename(f), json.loads(line)


def export(store: str, divergence: str, out: Path, divergence_real: str | None = None) -> dict:
	out.mkdir(parents=True, exist_ok=True)
	path = out / f"preflight-corpus-{VERSION}.jsonl.gz"
	counts: collections.Counter = collections.Counter()
	skipped: collections.Counter = collections.Counter()
	seen = set()
	# a fixed mtime keeps the gzip bytes, and so the checksum, the same for the same rows
	with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
		for kind, pattern in (
			("kernels", f"{store}/kernels/*.jsonl.gz"),
			("hf", f"{store}/hf/*.jsonl.gz"),
			("divergence", f"{divergence}/*.jsonl.gz"),
			*((("divergence_real", f"{divergence_real}/*.jsonl.gz"),) if divergence_real else ()),
		):
			for fname, row in _rows(pattern):
				own = (row.get("config") or {}).get("kind")
				if (
					own and own != kind
				):  # review, Nov 2: the realistic sweep's rows never go out labelled as the Nov 1 sweep
					skipped[own] += 1  # reported apart, never counted as a published row
					continue
				ident = (kind, row.get("gpu"), row.get("key"))
				if ident in seen:  # each config is written once per GPU; a repeat would double-count
					continue
				seen.add(ident)
				split = _split(kind, row)
				row = {**row, "kind": kind, "split": split, "source_file": f"{kind}/{fname}"}
				gz.write((json.dumps(row, sort_keys=True) + "\n").encode())
				counts[(kind, row.get("gpu"), row.get("status"), split)] += 1
	digest = hashlib.sha256(path.read_bytes()).hexdigest()
	(out / "SHA256SUMS").write_text(f"{digest}  {path.name}\n")
	summary = {
		"file": path.name,
		"bytes": path.stat().st_size,
		"sha256": digest,
		"rows": sum(counts.values()),
		"skipped_by_kind": dict(skipped),
		"by": [
			{"kind": k, "gpu": g, "status": s, "split": sp, "rows": n}
			for (k, g, s, sp), n in sorted(counts.items(), key=str)
		],
	}
	(out / "README.md").write_text(_card(summary))
	return summary


def _card(s: dict) -> str:
	by_kind = collections.Counter()
	for r in s["by"]:
		by_kind[r["kind"]] += r["rows"]
	lines = "\n".join(
		f"| {r['kind']} | {r['gpu']} | {r['status']} | {r['split']} | {r['rows']} |" for r in s["by"]
	)
	return f"""---
license: apache-2.0
pretty_name: Preflight corpus
tags: [gpu, training, memory, benchmarks, calibration]
---

# Preflight corpus {VERSION}

Measured training jobs on real GPUs (Modal, Oct 28 to Nov 2, 2026, and one rented AMD MI300X on Nov 2): memory,
step time and per-op profiles for {by_kind["kernels"]} small-model configs and {by_kind["hf"]} Hugging Face LoRA
and QLoRA jobs, and the loss, grad-norm and logit series of {by_kind["divergence"] + by_kind["divergence_real"]}
short GPT runs with a hindsight divergence label ({by_kind["divergence"]} from Nov 1's sweep and
{by_kind["divergence_real"]} from Nov 2's sweep at usual learning rates). Preflight (github.com/uchuuronin/preflight)
uses it to calibrate and check its prediction intervals. The full datasheet is `python/preflight/corpus/DATASHEET.md`
in the repository.

- **File:** `{s["file"]}`, {s["rows"]} rows, {s["bytes"]:,} bytes, sha256 `{s["sha256"]}`.
- **Row:** one JSON object per line; `kind`, `split` and `source_file` added to what the corpus worker wrote.
- **GPUs:** NVIDIA L4, A10, L40S, T4, H100-SXM-80GB, A100-PCIe-80GB and A100-SXM-80GB (torch 2.8.0, CUDA 12.8), and
  AMD MI300X (torch 2.5.1, ROCm 6.2). A config's `key` includes the torch version, so pair rows across vendors on
  `config.family`, `config.kind` and `config.params`.
- **Licence:** Apache 2.0, the repository's licence, for the measurements. No model weights and no dataset text
  are included.
- **Sources:** the configs are generated by `python/preflight/corpus`; the Hugging Face jobs time
  Qwen/Qwen2.5-0.5B, Qwen/Qwen2.5-1.5B, HuggingFaceTB/SmolLM2-360M, HuggingFaceTB/SmolLM2-1.7B and
  TinyLlama/TinyLlama-1.1B-Chat-v1.0 (all Apache 2.0); the divergence runs train on Tiny Shakespeare
  (karpathy/char-rnn, public-domain text) and store only training curves, not text.
- **Splits:** `calibration` (calibrated on, leave-one-architecture-out), `cross_gpu_holdout` (every A100 and MI300X
  row, scored once, never calibrated on), `not_in_calibration` (the synthetic LoRA family), `divergence` (both
  divergence kinds; they score the divergence classifier only).
- **Limits:** small models (at most 1.7B parameters, single GPU), five timed steps per config, random inputs for
  the timing kinds, one AMD GPU on another torch version than the NVIDIA rows. Not for ranking GPUs for purchase or
  for estimating how fast a particular production model trains.

| kind | gpu | status | split | rows |
| --- | --- | --- | --- | --- |
{lines}
"""


def main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument("--store", required=True)
	ap.add_argument("--divergence", required=True)
	ap.add_argument("--divergence-real", default=None)
	ap.add_argument("--out", required=True, type=Path)
	a = ap.parse_args(argv)
	s = export(a.store, a.divergence, a.out, a.divergence_real)
	print(json.dumps({k: v for k, v in s.items() if k != "by"}, indent=1))
	return 0


if __name__ == "__main__":
	sys.exit(main())
