"""Proxy campaigns: rank choices (data mixtures, learning rates) with small proxy runs before paying for full ones
(main plan: "proxy campaigns"; validation row: the proxy ranking agrees with full-size runs where both exist).

Two parts:
- **What a proxy campaign costs:** each choice runs at a fraction of the full model's width or steps. The cost
  comes from the campaign planner on the proxy settings, so it is the engine's own number.
- **Whether the proxies can be trusted:** where both proxy and full results exist, the ranks are compared with
  Kendall's tau and a bootstrap interval. A tau whose interval sits above 0.5 is "agrees"; one that crosses 0 is
  "no evidence". The plan says to abstain from ranking splits that differ in quality as well as quantity, and the
  caller decides that, because only the data profile knows it.

Pure Python: it runs in Pyodide.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

MODEL_ID = "proxy-rank@0.1.0"
AGREES = 0.5  # guess: a proxy ranking is trusted when the 90% interval on Kendall's tau sits above 0.5


def kendall_tau(a: list[float], b: list[float]) -> float:
	"""Kendall's tau-b between two score lists for the same choices."""
	if len(a) != len(b) or len(a) < 2:
		raise ValueError("needs two equal-length lists of at least two choices")
	conc = disc = ta = tb = 0
	for i, j in itertools.combinations(range(len(a)), 2):
		da, db = a[i] - a[j], b[i] - b[j]
		if da == 0 and db == 0:
			continue
		if da == 0:
			ta += 1
		elif db == 0:
			tb += 1
		elif (da > 0) == (db > 0):
			conc += 1
		else:
			disc += 1
	denom = ((conc + disc + ta) * (conc + disc + tb)) ** 0.5
	return (conc - disc) / denom if denom else 0.0


@dataclass(frozen=True)
class Agreement:
	tau: float
	lo: float
	hi: float
	n: int
	verdict: str  # "agrees", "no evidence" or "unclear"


def agreement(proxy: list[float], full: list[float], draws: int = 2000, seed: int = 0) -> Agreement:
	"""How well the proxies' ranking matches the full runs', with a 90% bootstrap interval over the choices."""
	tau = kendall_tau(proxy, full)
	rng = np.random.default_rng(seed)
	n = len(proxy)
	taus = []
	for _ in range(draws):
		idx = rng.integers(0, n, n)
		pa, fa = [proxy[i] for i in idx], [full[i] for i in idx]
		if len(set(pa)) > 1 and len(set(fa)) > 1:
			taus.append(kendall_tau(pa, fa))
	taus.sort()
	lo = taus[int(0.05 * len(taus))] if taus else -1.0
	hi = taus[int(0.95 * len(taus)) - 1] if taus else 1.0
	verdict = "agrees" if lo > AGREES else ("no evidence" if lo <= 0 <= hi else "unclear")
	return Agreement(tau, min(lo, tau), max(hi, tau), n, verdict)


def proxy_settings(full: dict, width_fraction: float = 0.25, step_fraction: float = 0.1) -> dict:
	"""Knob values for a proxy of a full run: narrower and shorter, everything else the same. The planner prices
	them like any other grid."""
	out = dict(full)
	for k in ("d_model", "hidden_size", "n_embd"):
		if out.get(k):
			out[k] = max(64, round(out[k] * width_fraction / 64) * 64)
	for k in ("max_steps", "steps"):
		if out.get(k):
			out[k] = max(1, int(out[k] * step_fraction))
	return out
