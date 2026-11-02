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
	assert (oom.lo, oom.hi) == (0.05, 0.95) and oom.reason.startswith("tight: P(OOM) between 5% and 95%")
	assert oom.point == 0.4  # the fit map and the ranking still read the model's value
	monkeypatch.setattr(conformal, "TABLE", replace(conformal.TABLE, oom_bins=_bins(10)))
	oom = api._from_memory_estimate(est, memory.MODEL_ID, "from 19 held-out runs").oom
	assert oom.lo == oom.hi == 0.4 and oom.reason.startswith("tight: P(OOM) 40% (from 19 held-out runs)")
	# a P(OOM) from the simulated band is never what the bins verified: it stays uncertain
	oom = api._from_memory_estimate(est, memory.MODEL_ID, "simulated band").oom
	assert (oom.lo, oom.hi) == (0.05, 0.95)


def test_the_installed_table_carries_the_bins_the_switch_reads():
	from preflight.engine import conformal, oom

	edges = [(b[0], b[1]) for b in conformal.TABLE.oom_bins]
	assert edges == [(lo, min(hi, 1.0)) for lo, hi in zip(oom.SHOWN_EDGES, oom.SHOWN_EDGES[1:], strict=False)]


def test_an_unverified_top_bin_shows_its_own_counts_until_it_verifies(monkeypatch):
	# Chiya, Oct 31: not a number, not "above 95%", but what similar runs did; PM, Nov 1: led by the S2b word
	from dataclasses import replace

	from preflight.engine import api, conformal, memory, oom

	failing = ((0.0, 0.05, 587, True, 2), (0.95, 1.0, 10, False, 9))
	assert oom.end_evidence(0.99, failing) == (9, 10) and not oom.shown_as_number(0.99, failing)
	assert oom.end_evidence(0.01, failing) is None and oom.shown_as_number(
		0.01, failing
	)  # the bottom verifies
	est = memory.MemoryEstimate(
		3.0e10, 2.8e10, 3.2e10, "bytes", "predicted", memory.MODEL_ID, 0.99, "fails", 0.1, None
	)
	monkeypatch.setattr(conformal, "TABLE", replace(conformal.TABLE, oom_bins=failing))
	r = api._from_memory_estimate(est, memory.MODEL_ID, "from 19 held-out runs").oom
	assert r.reason.startswith("fails: 9 of 10 similar runs ran out of memory (from 19 held-out runs)")
	lo, hi = conformal.wilson(9, 10)
	assert r.lo == pytest.approx(lo) and r.hi == pytest.approx(
		max(hi, 0.99)
	)  # the range holds the model's point
	# the switch: once that bin holds 10 runs that agree with outcomes, a number again
	passing = ((0.0, 0.05, 587, True, 2), (0.95, 1.0, 12, True, 12))
	monkeypatch.setattr(conformal, "TABLE", replace(conformal.TABLE, oom_bins=passing))
	r = api._from_memory_estimate(est, memory.MODEL_ID, "from 19 held-out runs").oom
	assert r.lo == r.hi == 0.99 and r.reason.startswith("fails: P(OOM) 99%")


def test_the_installed_table_shows_the_top_bin_as_evidence():
	from preflight.engine import conformal, oom

	top = next(b for b in conformal.TABLE.oom_bins if b[0] >= 0.95 - 1e-9)
	assert len(top) == 5 and oom.end_evidence(0.99, conformal.TABLE.oom_bins) == (top[4], top[2])


def test_every_oom_reason_leads_with_the_s2b_word(monkeypatch):
	"""PM, Nov 1: "tight: out of memory unlikely" read as two verdicts. The reason leads with fits, tight or fails,
	and a tight P(OOM) below 5% says why it isn't fits, and when it is the lowest the held-out runs can show."""
	from dataclasses import replace

	from preflight.engine import api, conformal, memory

	verified = ((0.0, 0.05, 587, True, 2), (0.05, 0.3, 10, True, 1), (0.3, 0.7, 10, True, 5))
	verified += ((0.7, 0.95, 10, True, 8), (0.95, 1.0, 12, True, 12))
	monkeypatch.setattr(conformal, "TABLE", replace(conformal.TABLE, oom_bins=verified))
	for p, word in ((0.004, "fits"), (0.025, "tight"), (0.5, "tight"), (0.9, "fails")):
		est = memory.MemoryEstimate(
			1.0e10, 0.9e10, 1.1e10, "bytes", "predicted", memory.MODEL_ID, p, memory.oom_status(p), 0.1, None
		)
		r = api._from_memory_estimate(est, memory.MODEL_ID, "from 19 held-out runs", floor=0.025).oom
		assert r.reason.split(":")[0] == word
		for other in ("unlikely", "likely", "possible", "won't fit"):
			assert other not in r.reason
	est = memory.MemoryEstimate(
		1.0e10, 0.9e10, 1.1e10, "bytes", "predicted", memory.MODEL_ID, 0.025, "tight", 0.1, None
	)
	at = api._from_memory_estimate(est, memory.MODEL_ID, "from 19 held-out runs", floor=0.025).oom
	assert "the lowest these runs can show, and fits needs below 1%" in at.reason
	above = api._from_memory_estimate(est, memory.MODEL_ID, "from 47 held-out runs", floor=0.0104).oom
	assert "lowest" not in above.reason and "fits needs below 1%" in above.reason


def test_the_pool_only_lowers_p_oom_below_the_cell_s_floor():
	"""Review, Nov 1: a pool value above the cell's floor made P(OOM) fall as memory rose (10.4% at 91.8% of the
	card, 7.5% at 92%). The pool decides only where it is lower than the floor."""
	from preflight.engine import oom

	cell = [0.0] * 19  # no residual over any positive margin
	pool = [0.0] * 400 + [0.2] * 87
	low, used = oom.pooled_p_oom(cell, pool, 5.0, 22.0, 0.01)
	assert used and low == pytest.approx(0.5 / 488)
	near, used = oom.pooled_p_oom(cell, pool, 19.0, 22.0, 0.01)  # margin 0.147: the pool's 87 runs are over
	assert not used and near == pytest.approx(0.5 / 20)
	ps = [oom.pooled_p_oom(cell, pool, m, 22.0, 0.01)[0] for m in (5.0, 10.0, 15.0, 19.0, 21.0)]
	assert ps == sorted(ps)
