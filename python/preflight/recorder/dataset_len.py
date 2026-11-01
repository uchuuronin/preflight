"""The training set's row lengths in tokens, so B can size memory by the real longest row (B, Oct 30).

A probe only sees the batches it draws, and the longest row in the data is rarely among them, so
B's memory rule had to assume every batch is `seq_len` long. This reads the rows the script itself
built: the training loader's dataset, row by row as the script's own `__getitem__` (or HF `map`)
tokenised it, so the length is the script's tokenisation and truncation, not a guess at them.

	dataset_max_tokens     the longest row, in tokens
	dataset_p999_tokens    the 99.9th percentile, so one outlier row shows as one
	dataset_p50_tokens     the median row, and
	dataset_mean_tokens    the mean row: step and total time follow these, memory the max (B, Nov 1)
	dataset_rows_scanned   how many rows these are over ("measured over N rows")
	dataset_rows_total     how many rows the dataset has, when it knows

Very large datasets: MAX_ROWS rows spread evenly over the set (so a set sorted by length can't
hide its longest rows), read in order inside BUDGET_S; rows_scanned < rows_total says so. The
99.9th percentile tells an outlier from the bulk only over more than 1,000 rows (at 1,000 or
fewer it is the max). Datasets without a length (streaming), rows that aren't integer token
sequences (text, images, audio, features) and a stuck row give nothing.
Pure Python (duck-typed: anything with `.shape`, a sequence, or an HF Dataset's Arrow column), so
it is tested without torch.
"""

import math
import numbers
import time
from collections.abc import Mapping

MAX_ROWS = 200_000  # B's request: past this, the first 200,000 rows, and the record says so
BUDGET_S = 10.0  # guess: Alpaca (52k rows) is under a second from Arrow; a slow __getitem__ stops here, about $0.002 of L4 a probe
TOKEN_KEYS = ("input_ids", "tokens", "ids", "token_ids")


def _is_float(x) -> bool:
	dt = getattr(x, "dtype", None)
	return dt is not None and (bool(getattr(dt, "is_floating_point", False)) or bool(getattr(dt, "is_complex", False))
		or any(w in str(dt) for w in ("float", "complex", "bfloat", "bool")))  # fmt: skip


def _length(row) -> int | None:
	"""Tokens in one row as a dataset hands it out: a dict's input_ids, a tuple's first tensor
	(x, y), or a tensor or list of ids itself. Floating-point values (audio, images, features)
	are never tokens."""
	if isinstance(row, Mapping):
		row = next((row[k] for k in TOKEN_KEYS if k in row), None)
	elif isinstance(row, tuple | list) and row and not isinstance(row[0], int | float):
		row = row[0]
	if row is None or isinstance(row, str | bytes) or _is_float(row):
		return None
	shape = getattr(row, "shape", None)
	if shape is not None:
		dims = [int(d) for d in shape]
		if not dims:
			return None
		if len(dims) > 1 and any(d != 1 for d in dims[:-1]):
			return None  # several sequences in one row: not a single row's length
		return dims[-1]
	if (
		isinstance(row, list | tuple)
		and row
		and all(isinstance(t, numbers.Integral) and not isinstance(t, bool) for t in row[:8])
	):  # Python or numpy integers, never bools
		return len(row)
	return None


def _picks(total: int, n: int) -> list[int]:
	"""n row indices spread evenly over the whole set, so a set sorted by length can't hide its
	longest rows past the cap."""
	if n >= total:
		return list(range(total))
	step = (total - 1) / (n - 1) if n > 1 else 0
	return sorted({round(i * step) for i in range(n)})  # first and last row included, n at most


def _arrow_lengths(ds, picks: list[int]) -> list[int] | None:
	"""An HF datasets.Dataset's token column lengths straight from Arrow (no row decoding): only a
	flat list-of-integers column, with no row mapping (shuffle, split) or custom transform on top."""
	try:
		if getattr(ds, "_indices", None) is not None:
			return None
		fmt = getattr(ds, "format", None)
		if isinstance(fmt, dict) and fmt.get("type") == "custom":
			return None  # set_transform: the rows the script sees aren't the stored ones
		names = getattr(ds, "column_names", None) or []
		col = next((k for k in TOKEN_KEYS if k in names), None)
		if col is None:
			return None
		import pyarrow as pa
		import pyarrow.compute as pc

		column = ds.data.column(col)
		t = column.type
		if not (pa.types.is_list(t) or pa.types.is_large_list(t)) or not pa.types.is_integer(t.value_type):
			return None  # nested lists (return_tensors per row) or not ids: read row by row instead
		lens = pc.list_value_length(column.take(pa.array(picks)))
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
	picks = _picks(total, min(total, max_rows))
	lens = _arrow_lengths(ds, picks)
	if lens is None:
		lens = []
		deadline = clock() + budget_s
		for i in picks:
			if clock() > deadline:
				break
			try:
				k = _length(ds[i])
			except Exception:
				return {}  # the script's own __getitem__ failed outside training: say nothing
			if k is None:
				return {}  # a row that isn't a token sequence: the set isn't one to measure
			lens.append(k)
	return _summary(lens, total) if lens else {}


def measure_bounded(ds, *, budget_s: float = BUDGET_S, grace_s: float = 2.0) -> dict:
	"""measure() in a daemon thread, given up on after budget_s + grace_s: one row that never
	returns (a stuck read) costs the probe a few seconds, never its result."""
	import threading

	box: dict = {}

	def run():
		try:
			box["out"] = measure(ds, budget_s=budget_s)
		except Exception:
			box["out"] = {}

	t = threading.Thread(target=run, name="dataset-lengths", daemon=True)
	t.start()
	t.join(budget_s + grace_s)
	return box.get("out", {})


def _summary(lens: list[int], total: int) -> dict:
	s = sorted(lens)
	return {
		"dataset_max_tokens": float(s[-1]),
		"dataset_p999_tokens": float(percentile(s, 0.999)),
		"dataset_p50_tokens": float(percentile(s, 0.5)),
		"dataset_mean_tokens": sum(s) / len(s),
		"dataset_rows_scanned": float(len(s)),
		"dataset_rows_total": float(total),
	}
