"""Data repetition (engine/repetition.py): Muennighoff's effective data, and per-bin exposure."""

import math

import pytest

from preflight.engine import repetition as R


def test_effective_tokens_follow_muennighoff():
	u = 1e9
	assert R.effective_tokens(u, 1) == u
	assert R.effective_tokens(u, 0.5) == 0.5 * u
	# 4 epochs: R = 3, D' = U (1 + 15.4 (1 - e^(-3/15.4)))
	assert R.effective_tokens(u, 4) == pytest.approx(u * (1 + 15.4 * (1 - math.exp(-3 / 15.4))))
	assert R.value_per_token(u, 4) > 0.9  # still close to new data
	assert R.value_per_token(u, 40) < 0.4  # repeats barely help
	assert R.effective_tokens(u, 1000) == pytest.approx(u * (1 + R.R_STAR))  # bounded by U (1 + R*)


def test_exposure_counts_and_flags():
	bins = [
		R.Bin("en/high", 8e9, language="en"),
		R.Bin("en/low", 2e9, low_quality=True, language="en"),
		R.Bin("sw", 0.1e9, weight=20, language="sw"),  # upsampled
	]
	ex = {e.name: e for e in R.exposure(bins, 30e9)}
	assert ex["en/high"].epochs == pytest.approx(ex["en/low"].epochs)  # same weight, same epochs
	assert ex["en/low"].flag and "seen" in ex["en/low"].flag
	assert ex["sw"].epochs == pytest.approx(20 * ex["en/high"].epochs)
	assert ex["sw"].flag and "past" in ex["sw"].flag
	langs = R.by_language(bins, 30e9)
	assert langs["sw"] > R.FLAG_EPOCHS > langs["en"]
	tot = sum(e.epochs * b.tokens for e, b in zip(R.exposure(bins, 30e9), bins, strict=True))
	assert tot == pytest.approx(30e9)  # every budgeted token is drawn from some bin


def test_split_table_and_bad_inputs():
	rows = R.split_table(1e9, [1e9, 4e9, 16e9])
	assert [round(e) for _, e, _ in rows] == [1, 4, 16]
	with pytest.raises(ValueError):
		R.exposure([R.Bin("x", 0)], 1e9)
