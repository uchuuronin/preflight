"""Power analysis (engine/power.py): seeds per arm from seed-to-seed spread."""

import math
import random
import statistics

import pytest

from preflight.engine import power


def test_wilson_hilferty_is_close_to_the_chi_square_table():
	# chi-square quantiles from the standard table (df, p, value)
	for df, p, v in (
		(4, 0.05, 0.711),
		(4, 0.95, 9.488),
		(9, 0.05, 3.325),
		(9, 0.95, 16.919),
		(29, 0.95, 42.557),
	):
		assert power._chi2_quantile(p, df) == pytest.approx(v, rel=0.03)


def test_seeds_needed_is_the_textbook_two_sample_formula():
	s = power.SeedSpread(sd=0.02, lo=0.02, hi=0.02, n=10)
	n = power.seeds_needed(s, gap=0.02)
	# 2 (1.96 + 0.84)^2 = 15.7 seeds per arm for a gap of one sd
	assert n.point == math.ceil(2 * (1.959964 + 0.841621) ** 2)
	assert power.seeds_needed(s, gap=0.04).point == math.ceil(2 * (1.959964 + 0.841621) ** 2 / 4)


def test_few_seeds_give_a_wide_answer_and_the_interval_covers():
	rng = random.Random(0)
	covered = 0
	for _ in range(2000):
		xs = [rng.gauss(0, 1.0) for _ in range(5)]
		sp = power.seed_spread(xs)
		covered += sp.lo <= 1.0 <= sp.hi
	assert covered / 2000 == pytest.approx(0.90, abs=0.02)  # a real 90% interval on the sd
	wide = power.seeds_needed(power.seed_spread([1.0, 1.1, 0.95, 1.05]), gap=0.05)
	assert wide.lo < wide.point < wide.hi and "from 4 seeds" in wide.reason


def test_bad_inputs_raise():
	with pytest.raises(ValueError):
		power.seed_spread([1.0])
	with pytest.raises(ValueError):
		power.seeds_needed(power.SeedSpread(1, 1, 1, 3), gap=0)
	assert power.seed_spread([1.0, float("nan"), 2.0]).n == 2
	assert statistics.stdev([1.0, 2.0]) == power.seed_spread([1.0, 2.0]).sd
