"""engine/quality.py: a dataset's own quality column, binned and handed to the repetition model."""

from preflight.engine import quality as Q


def _fineweb(rows):
	col = Q.column_for("HuggingFaceFW/fineweb-edu:sample-10BT", ["text", "int_score", "score", "token_count"])
	assert col is not None and col.name == "int_score"
	return Q.profile(rows, col)


def test_fineweb_edu_bins_by_int_score_and_reads_its_token_column():
	rows = [{"int_score": 1, "token_count": 300}, {"int_score": 3, "token_count": 700}, {"int_score": None}]
	p = _fineweb(rows)
	assert p.tokens == {"int_score 1": 300.0, "int_score 3": 700.0} and p.missing == 1
	assert p.low == {"int_score 1"}  # under 3: the authors' edu cut-off


def test_low_quality_repeated_past_twice_is_flagged():
	p = _fineweb([{"int_score": 1, "token_count": 1000}, {"int_score": 4, "token_count": 1000}])
	rows = {r.name: r for r in Q.table(p, budget_tokens=3_000)}  # 1.5 passes: nothing flagged
	assert rows["int_score 1"].epochs == 1.5 and rows["int_score 1"].flag is None
	rows = {r.name: r for r in Q.table(p, budget_tokens=6_000)}  # 3 passes: the low bin is seen 3 times
	assert "low-quality" in rows["int_score 1"].flag and rows["int_score 4"].flag is None
	assert rows["int_score 1"].seen_per_epoch == 1000  # each pass sees the bin once


def test_unknown_datasets_and_missing_columns_read_nothing():
	assert Q.column_for("roneneldan/TinyStories", ["text"]) is None
	assert Q.column_for("HuggingFaceFW/fineweb-edu", ["text"]) is None


def test_redpajama_reads_its_ccnet_bucket_from_the_signals():
	col = Q.column_for("togethercomputer/RedPajama-Data-V2", ["raw_content", "quality_signals"])
	p = Q.profile([{"quality_signals": '{"ccnet_bucket": [[0, 10, "tail"]]}'}], col, tokens_of=lambda r: 10)
	assert p.tokens == {"ccnet tail": 10.0} and p.low == {"ccnet tail"}


def test_only_the_exact_dataset_matches_and_dclm_flags_no_decile():
	assert Q.column_for("HuggingFaceFW/fineweb-edu-llama3-annotations", ["int_score"]) is None
	assert Q.column_for("HuggingFaceFW/fineweb-edu-score-2", ["int_score", "token_count"]).low_below == 3
	col = Q.column_for("mlfoundations/dclm-baseline-1.0", [Q.KNOWN[2].name])
	p = Q.profile([{col.name: 0.1}, {col.name: 0.95}], col, tokens_of=lambda r: 100)
	assert p.low == set() and len(p.tokens) == 2
