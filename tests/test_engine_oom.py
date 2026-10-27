import numpy as np
import pytest

from preflight.engine import oom


def test_bins():
	assert [oom.bin_label(p) for p in (0.0, 0.04, 0.05, 0.5, 0.7, 1.0)] == [
		"unlikely", "unlikely", "possible", "likely", "very likely", "very likely",
	]  # fmt: skip


def test_empirical_p_oom_counts_residuals_past_the_margin():
	res = [-0.1, 0.0, 0.05, 0.2, 0.3]
	# capacity 10% above the prediction: margin log(1.1) ~ 0.095, two residuals exceed it
	assert oom.empirical_p_oom(res, 1e9, 1.1e9) == pytest.approx(2.5 / 6)
	assert 0 < oom.empirical_p_oom(res, 1e9, 10e9) < 0.1  # never exactly 0 on a small sample
	assert oom.empirical_p_oom(res, 1e9, 0) == 1.0
	with pytest.raises(ValueError):
		oom.empirical_p_oom([], 1e9, 2e9)


def test_reliability_diagram_flags_overconfidence():
	g = np.random.default_rng(0)
	honest = [(p, bool(g.random() < p)) for p in g.uniform(0, 1, 2000)]
	assert oom.reliable(oom.reliability(honest))
	overconfident = [(min(1.0, p * 3), y) for p, y in honest]  # says 3x the real chance
	assert not oom.reliable(oom.reliability(overconfident))
	assert not oom.reliable(oom.reliability([]))  # nothing measured: not reliable
