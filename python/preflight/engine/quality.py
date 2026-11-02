"""Per-document quality columns, read into the repetition model's bins (PM, Nov 2, item 5).

Some pretraining corpora ship a quality score per document. When the dataset has one, its documents are binned by
that score, each bin's tokens counted (from the dataset's own token column where it has one), and the bins handed to
engine/repetition.py: how many tokens of each bin a run sees in each epoch, and a warning when low-quality documents
are seen more than once or twice. A new readout beside the others; nothing in the step, memory or cost models reads it.

Known columns (each cited to its dataset card; the low-quality cut-off is the dataset authors' own where they give one):
- FineWeb-Edu (HuggingFaceFW/fineweb-edu): `int_score`, 0 to 5, the rounded educational score of their classifier
  (`score` is the unrounded value); the authors keep documents scoring 3 and up for the released edu set, so under 3
  is low quality here (their looser fineweb-edu-score-2 keeps 2s, which read as low). Rows carry `token_count`.
- DCLM (mlfoundations/dclm-baseline-1.0): `fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train_prob`, the
  fastText classifier's probability, binned in deciles. The set is already that classifier's top slice of its pool,
  so no decile is called low quality: the deciles are shown, and only repetition is flagged.
- RedPajama-V2 (togethercomputer/RedPajama-Data-V2): `quality_signals`, a JSON column; its `ccnet_bucket` (head,
  middle, tail by the CCNet perplexity model), tail as low quality, the authors' own reading of the buckets.

Pure Python: it runs in Pyodide.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

from preflight.engine.repetition import Bin, Exposure, exposure

MODEL_ID = "quality-exposure@0.1.0"


@dataclass(frozen=True)
class Column:
	dataset: str  # the hub id the column belongs to
	name: str  # the column read
	kind: str  # "int_score", "probability", "ccnet_bucket"
	low_below: float | None = None  # a document scoring under this is low quality (numeric columns)
	token_column: str | None = None  # the dataset's own token count per document, when it has one
	source: str = ""


KNOWN = (
	Column(
		"HuggingFaceFW/fineweb-edu",
		"int_score",
		"int_score",
		3,
		"token_count",
		"cited: FineWeb-Edu dataset card, the edu set keeps int_score >= 3",
	),
	# the authors' looser set keeps int_score 2 as well; 3 stays the edu cut-off, so its 2s read as low quality
	Column(
		"HuggingFaceFW/fineweb-edu-score-2",
		"int_score",
		"int_score",
		3,
		"token_count",
		"cited: FineWeb-Edu dataset card (score-2 keeps int_score >= 2; edu's cut-off is 3)",
	),
	# review, Nov 2: DCLM-baseline is already its classifier's top slice, so no decile of it is called low quality;
	# the deciles are shown, nothing is flagged for quality (repetition past 4 epochs still is)
	Column(
		"mlfoundations/dclm-baseline-1.0",
		"fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train_prob",
		"probability",
		None,
		None,
		"cited: DCLM paper, the baseline set is the classifier's top ~10% of its pool",
	),
	Column(
		"togethercomputer/RedPajama-Data-V2",
		"quality_signals",
		"ccnet_bucket",
		None,
		None,
		"cited: RedPajama-V2 card, CCNet perplexity buckets head/middle/tail; tail as low quality",
	),
)


def column_for(dataset: str, columns: list[str]) -> Column | None:
	"""The quality column of a known dataset, if this dataset is one and has it (a config name after the id is fine)."""
	base = dataset.split(":", 1)[0].strip().lower()
	for c in KNOWN:  # the exact dataset (review, Nov 2: a prefix caught other datasets' names)
		if base == c.dataset.lower() and c.name in columns:
			return c
	return None


def _bin_of(col: Column, value) -> tuple[str, bool] | None:
	"""(bin name, low quality) for one document's value, or None when it has none."""
	if col.kind == "int_score":
		if value is None:
			return None
		v = int(value)
		return f"{col.name} {v}", v < (col.low_below or 0)
	if col.kind == "probability":
		if value is None or not math.isfinite(float(value)):
			return None
		d = min(9, max(0, int(float(value) * 10)))
		return f"decile {d + 1} (p {d / 10:.1f} to {(d + 1) / 10:.1f})", False
	if col.kind == "ccnet_bucket":
		sig = json.loads(value) if isinstance(value, str) else (value or {})
		b = sig.get("ccnet_bucket")
		if isinstance(b, list):  # RedPajama-V2 stores each signal as [[start, end, value], ...]
			b = b[0][2] if b and isinstance(b[0], list) and len(b[0]) == 3 else (b[0] if b else None)
		if b is None:
			return None
		return f"ccnet {b}", str(b) == "tail"
	return None


@dataclass
class Profile:
	"""Tokens per quality bin in a sample, and how far it scales to the whole dataset."""

	column: Column
	tokens: dict[str, float] = field(default_factory=dict)  # in the sample
	docs: dict[str, int] = field(default_factory=dict)
	low: set[str] = field(default_factory=set)
	missing: int = 0  # documents with no value in the column

	def bins(self, dataset_tokens: float | None = None) -> list[Bin]:
		"""repetition.Bin per quality bin, scaled to `dataset_tokens` (the sample's own size when None)."""
		total = sum(self.tokens.values())
		scale = (dataset_tokens / total) if dataset_tokens and total else 1.0
		return [Bin(name, t * scale, 1.0, name in self.low) for name, t in sorted(self.tokens.items())]


def profile(rows, col: Column, tokens_of=None) -> Profile:
	"""Bin documents (dicts with the column) by quality, counting tokens with the dataset's own token column, or
	`tokens_of(row)` when it has none (review, Nov 2: without either, there is nothing to count)."""
	if not col.token_column and tokens_of is None:
		raise ValueError(f"{col.dataset} has no token column; pass tokens_of to count each document's tokens")
	p = Profile(col)
	for r in rows:
		got = _bin_of(col, r.get(col.name))
		if got is None:
			p.missing += 1
			continue
		name, low = got
		n = r.get(col.token_column) if col.token_column else None
		n = float(n) if n is not None else float(tokens_of(r) if tokens_of else 0)
		p.tokens[name] = p.tokens.get(name, 0.0) + n
		p.docs[name] = p.docs.get(name, 0) + 1
		if low:
			p.low.add(name)
	return p


@dataclass(frozen=True)
class EpochRow:
	name: str
	low_quality: bool
	unique_tokens: float
	seen_per_epoch: float  # tokens of this bin seen in one pass of the run's budget split by epochs
	epochs: float  # times each document in the bin is seen over the whole budget
	flag: str | None


def table(
	p: Profile,
	budget_tokens: float,
	dataset_tokens: float | None = None,
	weights: dict[str, float] | None = None,
) -> list[EpochRow]:
	"""For a run of `budget_tokens` over the dataset (mixture weights per bin, 1 by default): per quality bin, its
	unique tokens, the tokens of it seen per epoch of the dataset, the times each of its documents is seen, and
	repetition.py's warning (low-quality documents seen more than FLAG_LOW_QUALITY times, or any bin past 4 epochs)."""
	bins = [
		Bin(b.name, b.tokens, (weights or {}).get(b.name, 1.0), b.low_quality) for b in p.bins(dataset_tokens)
	]
	ex: dict[str, Exposure] = {e.name: e for e in exposure(bins, budget_tokens)}
	unique = sum(b.tokens for b in bins)
	passes = budget_tokens / unique if unique else 0.0  # the run's epochs over the whole dataset
	return [
		EpochRow(
			b.name,
			b.low_quality,
			b.tokens,
			ex[b.name].epochs * b.tokens / passes if passes else 0.0,
			ex[b.name].epochs,
			ex[b.name].flag,
		)
		for b in bins
	]
