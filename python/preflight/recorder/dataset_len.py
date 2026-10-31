"""The training set's row lengths in tokens, so B can size memory by the real longest row (B, Oct 30).

A probe only sees the batches it draws, and the longest row in the data is rarely among them, so
B's memory rule had to assume every batch is `seq_len` long. This reads the rows the script itself
built: the training loader's dataset, row by row as the script's own `__getitem__` (or HF `map`)
tokenised it, so the length is the script's tokenisation and truncation, not a guess at them.

	dataset_max_tokens     the longest row, in tokens
	dataset_p999_tokens    the 99.9th percentile, so one outlier row shows as one
	dataset_rows_scanned   how many rows these are over ("measured over N rows")
	dataset_rows_total     how many rows the dataset has, when it knows

Very large datasets: the first MAX_ROWS rows only, inside BUDGET_S; rows_scanned < rows_total
says so. Datasets without a length (streaming) or rows it can't read as tokens give nothing.
Pure Python (duck-typed: anything with `.shape`, a sequence, or an HF Dataset's Arrow column), so
it is tested without torch.
"""

import math
import time
from collections.abc import Mapping

MAX_ROWS = 200_000  # B's request: past this, the first 200,000 rows, and the record says so
BUDGET_S = 10.0  # guess: Alpaca (52k rows) is under a second from Arrow; a slow __getitem__ stops here, about $0.002 of L4 a probe
TOKEN_KEYS = ("input_ids", "tokens", "ids", "token_ids")


def _length(row) -> int | None:
	"""Tokens in one row as a dataset hands it out: a dict's input_ids, a tuple's first tensor
	(x, y), or a tensor or list of ids itself."""
	if isinstance(row, Mapping):
		row = next((row[k] for k in TOKEN_KEYS if k in row), None)
	elif isinstance(row, tuple | list) and row and not isinstance(row[0], int | float):
		row = row[0]
	if row is None or isinstance(row, str | bytes):
		return None
	shape = getattr(row, "shape", None)
	if shape is not None:
		dims = [int(d) for d in shape]
		if not dims:
			return None
		if len(dims) > 1 and any(d != 1 for d in dims[:-1]):
			return None  # several sequences in one row: not a single row's length
		return dims[-1]
	if isinstance(row, list | tuple) and all(isinstance(t, int) for t in row[:8]):
		return len(row)
	return None


def _arrow_lengths(ds, n: int) -> list[int] | None:
	"""An HF datasets.Dataset's token column lengths straight from Arrow (no row decoding), when the
	table is the dataset as indexed (no select/shuffle mapping on top)."""
	try:
		if getattr(ds, "_indices", None) is not None:
			return None
		names = getattr(ds, "column_names", None) or []
		col = next((k for k in TOKEN_KEYS if k in names), None)
		if col is None:
			return None
		import pyarrow.compute as pc

		lens = pc.list_value_length(ds.data.column(col).slice(0, n))
		return [int(x) for x in lens.to_pylist() if x is not None]
	except Exception:
		return None


def percentile(sorted_vals: list[int], q: float) -> int:
	"""The nearest-rank percentile of sorted values (q in 0..1)."""
	k = max(0, min(len(sorted_vals) - 1, math.ceil(q * len(sorted_vals)) - 1))
	return sorted_vals[k]


def measure(ds, *, max_rows: int = MAX_ROWS, budget_s: float = BUDGET_S, clock=time.monotonic) -> dict:
	"""The extras for one dataset, or {} when its rows can't be read as token sequences."""
	try:
		total = len(ds)
	except Exception:
		return {}  # streaming: no length, nothing to scan in a bounded way
	if not isinstance(total, int) or total <= 0:
		return {}
	n = min(total, max_rows)
	lens = _arrow_lengths(ds, n)
	if lens is None:
		lens = []
		deadline = clock() + budget_s
		for i in range(n):
			if i % 64 == 0 and clock() > deadline:
				break
			try:
				k = _length(ds[i])
			except Exception:
				return {}  # the script's own __getitem__ failed outside training: say nothing
			if k is None:
				return {} if not lens else _summary(lens, total)
			lens.append(k)
	return _summary(lens, total) if lens else {}


def _summary(lens: list[int], total: int) -> dict:
	s = sorted(lens)
	return {
		"dataset_max_tokens": float(s[-1]),
		"dataset_p999_tokens": float(percentile(s, 0.999)),
		"dataset_rows_scanned": float(len(s)),
		"dataset_rows_total": float(total),
	}
