"""Proxy campaigns (engine/proxy.py): rank agreement and proxy settings."""

import random

import pytest

from preflight.engine import proxy as P


def test_kendall_tau_known_values():
	assert P.kendall_tau([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
	assert P.kendall_tau([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0
	assert P.kendall_tau([1, 2, 3], [1, 3, 2]) == pytest.approx(1 / 3)
	with pytest.raises(ValueError):
		P.kendall_tau([1], [1])


def test_agreement_verdicts():
	rng = random.Random(0)
	full = [rng.random() for _ in range(12)]
	good = P.agreement([f + rng.gauss(0, 0.02) for f in full], full)
	assert good.verdict == "agrees" and good.lo > P.AGREES
	noise = P.agreement([rng.random() for _ in full], full)
	assert noise.verdict == "no evidence" and noise.lo <= 0 <= noise.hi


def test_proxy_settings_shrink_width_and_steps_only():
	full = {"d_model": 2048, "max_steps": 10000, "lr": 3e-4, "batch_size": 32}
	p = P.proxy_settings(full)
	assert p["d_model"] == 512 and p["max_steps"] == 1000 and p["lr"] == 3e-4 and p["batch_size"] == 32
	assert P.proxy_settings({"d_model": 100}, width_fraction=0.1)["d_model"] == 64
