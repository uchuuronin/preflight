"""The training set's row lengths for B's memory rule (B, Oct 30), on fake datasets, no torch."""

import pytest

from preflight.recorder.dataset_len import measure, percentile


class Shape:
	def __init__(self, *dims, dtype="torch.int64"):
		self.shape = dims
		self.dtype = dtype


class Rows:
	def __init__(self, rows):
		self.rows = rows

	def __len__(self):
		return len(self.rows)

	def __getitem__(self, i):
		return self.rows[i]


def test_tokenised_rows_give_the_longest_and_the_999th_percentile():
	lens = [100] * 9999 + [4000]  # one outlier row in 10,000
	got = measure(Rows([{"input_ids": list(range(n)), "labels": [0] * n} for n in lens]))
	assert got["dataset_max_tokens"] == 4000
	assert got["dataset_p999_tokens"] == 100  # the outlier shows as one row, not as the bulk
	assert got["dataset_rows_scanned"] == got["dataset_rows_total"] == 10_000


def test_tensor_pairs_and_fixed_blocks_are_read_by_their_shape():
	got = measure(Rows([(Shape(1024), Shape(1024))] * 50))
	assert got["dataset_max_tokens"] == 1024 and got["dataset_rows_scanned"] == 50
	assert measure(Rows([Shape(1, 77)] * 3))["dataset_max_tokens"] == 77


def test_a_very_large_set_is_scanned_up_to_its_cap_and_says_so():
	class Big(Rows):
		def __len__(self):
			return 5_000_000

		def __getitem__(self, i):
			return {"input_ids": [1] * (10 + i % 7)}

	got = measure(Big([]), max_rows=1000)
	assert got["dataset_rows_scanned"] == 1000 and got["dataset_rows_total"] == 5_000_000


def test_a_slow_getitem_stops_at_the_time_budget():
	t = [0.0]

	class Slow(Rows):
		def __getitem__(self, i):
			t[0] += 0.01
			return {"input_ids": [1, 2, 3]}

	got = measure(Slow([None] * 100_000), budget_s=1.0, clock=lambda: t[0])
	assert 90 <= got["dataset_rows_scanned"] <= 102 and got["dataset_rows_total"] == 100_000


def test_a_set_sorted_by_length_doesnt_hide_its_longest_rows_past_the_cap():
	class Sorted(Rows):
		def __len__(self):
			return 1_000_000

		def __getitem__(self, i):
			return {"input_ids": [1] * (1 + i // 1000)}  # longest rows last

	got = measure(Sorted([]), max_rows=2000)
	assert got["dataset_max_tokens"] == 1000 and got["dataset_rows_scanned"] <= 2000


def test_a_stuck_row_costs_seconds_not_the_probe():
	import threading
	import time

	from preflight.recorder.dataset_len import measure_bounded

	never = threading.Event()

	class Stuck(Rows):
		def __getitem__(self, i):
			if i > 10:
				never.wait()
			return {"input_ids": [1, 2]}

	t0 = time.monotonic()
	assert measure_bounded(Stuck([None] * 1000), budget_s=0.2, grace_s=0.3) == {}
	assert time.monotonic() - t0 < 2.0
	never.set()


@pytest.mark.parametrize(
	"ds",
	[
		Rows([{"text": "raw text, not tokens"}] * 3),
		Rows([]),
		Rows([Shape(4, 128)] * 2),
		Rows([(Shape(1, 16000, dtype="torch.float32"), 16000)] * 3),  # audio: a waveform, not tokens
		Rows([(Shape(784, dtype="torch.float32"), 3)] * 3),  # a flattened image
		Rows([{"input_ids": [1, 2]}, {"text": "x"}]),  # stops at a row that isn't tokens: nothing
	],
)
def test_rows_that_arent_token_sequences_give_nothing(ds):
	assert measure(ds) == {}


def test_a_dataset_without_a_length_or_that_fails_gives_nothing():
	class Stream:
		def __iter__(self):
			yield {"input_ids": [1]}

	class Broken(Rows):
		def __getitem__(self, i):
			raise RuntimeError("needs the GPU")

	assert measure(Stream()) == {} and measure(Broken([1, 2])) == {}


def test_an_hf_dataset_is_read_from_its_arrow_column():
	pa = pytest.importorskip("pyarrow")

	class HF:
		_indices = None
		column_names = ("input_ids", "labels")

		def __init__(self, lens):
			self.data = pa.table({"input_ids": [list(range(n)) for n in lens], "labels": [[0]] * len(lens)})
			self.n = len(lens)

		def __len__(self):
			return self.n

		def __getitem__(self, i):
			raise AssertionError("rows are read from Arrow, not one by one")

	got = measure(HF([5, 9, 300, 7]))
	assert got["dataset_max_tokens"] == 300 and got["dataset_rows_scanned"] == 4


def test_percentile_is_nearest_rank():
	assert percentile([1, 2, 3, 4], 0.5) == 2 and percentile([7], 0.999) == 7


def test_nested_arrow_lists_are_read_row_by_row_not_as_length_one():
	pa = pytest.importorskip("pyarrow")

	class HF:
		_indices = None
		column_names = ("input_ids",)
		format = None  # no custom transform

		def __init__(self):
			self.rows = [[list(range(n))] for n in (5, 300)]  # return_tensors="pt" per row: [[ids]]
			self.data = pa.table({"input_ids": self.rows})

		def __len__(self):
			return 2

		def __getitem__(self, i):
			return {"input_ids": Shape(1, len(self.rows[i][0]))}

	assert measure(HF())["dataset_max_tokens"] == 300
