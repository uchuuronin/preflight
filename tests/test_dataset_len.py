"""The training set's row lengths for B's memory rule (B, Oct 30), on fake datasets, no torch."""

import pytest

from preflight.recorder.dataset_len import measure, percentile


class Shape:
	def __init__(self, *dims):
		self.shape = dims


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
	assert 64 <= got["dataset_rows_scanned"] < 200 and got["dataset_rows_total"] == 100_000


@pytest.mark.parametrize(
	"ds",
	[Rows([{"text": "raw text, not tokens"}] * 3), Rows([]), Rows([Shape(4, 128)] * 2)],
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
