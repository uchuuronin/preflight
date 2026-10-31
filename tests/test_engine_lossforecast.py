"""Loss forecast (engine/lossforecast.py): a power law fitted to the probe's loss, with a bootstrap band."""

import numpy as np
import pytest

from preflight.engine import lossforecast as L


def _curve(a=4.0, b=0.4, c=1.8, n=300, noise=0.02, seed=0):
	rng = np.random.default_rng(seed)
	t = np.arange(1, n + 1, dtype=float)
	return list(c + a * t ** (-b) + rng.normal(0, noise, n)), (a, b, c)


def test_a_clean_power_law_is_recovered_and_extrapolated():
	ys, (a, b, c) = _curve(noise=0.0)
	f = L.forecast(ys, last_step=3000)
	assert f.status == "extrapolated"
	assert f.point == pytest.approx(c + a * 3000 ** (-b), rel=0.01)


def test_the_band_covers_the_truth_on_noisy_curves():
	hits, n = 0, 0
	for seed in range(40):
		ys, (a, b, c) = _curve(seed=seed)
		f = L.forecast(ys, last_step=1500, draws=100, seed=seed)
		if f.status == "abstain":
			continue
		n += 1
		hits += f.lo <= c + a * 1500 ** (-b) <= f.hi
	assert n >= 30 and hits / n >= 0.8  # nominal 90%; a bootstrap of a nonlinear fit runs a little narrow


def test_it_abstains_when_it_should():
	assert L.forecast([3.0] * 10, last_step=1000).status == "abstain"  # too few steps
	assert "isn't falling" in L.forecast([3.0 + 0.001 * i for i in range(100)], last_step=1000).reason
	noisy, _ = _curve(noise=1.0, n=40)
	f = L.forecast(noisy, last_step=100_000)
	assert f.status == "abstain"  # a band wider than +-50%
	ys, _ = _curve(noise=0.0, n=100)
	assert L.forecast(ys, last_step=50).point == pytest.approx(ys[49])  # already past it: the loss seen there
