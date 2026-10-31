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


# ---------------------------------------------------------------- Chiya, Oct 31: a number only where verified


def _bins(middle_n: int, middle_ok: bool = True):
	return (
		(0.0, 0.05, 587, True),
		(0.05, 0.30, middle_n, middle_ok),
		(0.30, 0.70, middle_n, middle_ok),
		(0.70, 0.95, middle_n, middle_ok),
		(0.95, 1.0, 10, True),
	)


def test_middle_probabilities_switch_to_numbers_once_every_middle_bin_is_verified():
	from preflight.engine import oom

	assert oom.shown_as_number(0.01, _bins(0)) and oom.shown_as_number(0.99, _bins(0))  # the ends always show
	for p in (0.05, 0.3, 0.5, 0.95):
		assert not oom.shown_as_number(p, _bins(9))  # one point short in every middle bin
		assert oom.shown_as_number(p, _bins(10))  # the switch
	thin = ((0.0, 0.05, 587, True), (0.05, 0.30, 40, True), (0.30, 0.70, 9, True), (0.70, 0.95, 40, True))
	assert not oom.shown_as_number(0.1, thin)  # each middle bin, not the middle as a whole
	assert not oom.shown_as_number(
		0.5, _bins(50, middle_ok=False)
	)  # enough points that disagree with outcomes
	assert not oom.shown_as_number(0.5, ())  # no table: nothing is verified


def test_the_engine_reports_the_uncertain_range_until_the_switch(monkeypatch):
	from dataclasses import replace

	from preflight.engine import api, conformal, memory

	est = memory.MemoryEstimate(
		1.0e10, 0.9e10, 1.1e10, "bytes", "predicted", memory.MODEL_ID, 0.4, "tight", 0.1, None
	)
	monkeypatch.setattr(conformal, "TABLE", replace(conformal.TABLE, oom_bins=_bins(5)))
	oom = api._from_memory_estimate(est, memory.MODEL_ID).oom
	assert (oom.lo, oom.hi) == (0.05, 0.95) and "uncertain (5–95%)" in oom.reason
	assert oom.point == 0.4  # the fit map and the ranking still read the model's value
	monkeypatch.setattr(conformal, "TABLE", replace(conformal.TABLE, oom_bins=_bins(10)))
	oom = api._from_memory_estimate(est, memory.MODEL_ID, "from 19 held-out runs").oom
	assert oom.lo == oom.hi == 0.4 and "uncertain" not in oom.reason
	# a P(OOM) from the simulated band is never what the bins verified: it stays uncertain
	oom = api._from_memory_estimate(est, memory.MODEL_ID, "simulated band").oom
	assert (oom.lo, oom.hi) == (0.05, 0.95)


def test_the_installed_table_carries_the_bins_the_switch_reads():
	from preflight.engine import conformal, oom

	edges = [(b[0], b[1]) for b in conformal.TABLE.oom_bins]
	assert edges == [(lo, min(hi, 1.0)) for lo, hi in zip(oom.SHOWN_EDGES, oom.SHOWN_EDGES[1:], strict=False)]
