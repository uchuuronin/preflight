"""Loss forecast: where a run's loss lands at its last step, from the loss the probes saw (main plan: "Power-law fit
and LC-PFN posterior, ensembled; abstains when the band is too wide").

This is the power-law half. The loss follows L(t) = c + a t^(-b) (Kaplan et al.), fitted to the probe's loss series.
For each candidate floor c, log(L - c) is linear in log t, so the fit is a one-dimensional search over c with
least squares inside. The interval comes from a residual block bootstrap: the series is refitted on resampled noise,
and every refit is extrapolated to the last step. It abstains when the probe saw too few steps, or when the 90% band
is wider than +-50% of the forecast. The LC-PFN half runs with torch outside the engine (corpus/lcpfn.py, the MIT
checkpoints ported, Nov 1), and `ensemble` combines the two from plain numbers. Pure numpy: it runs in Pyodide.

Known limit (review, Oct 31): with strongly autocorrelated noise (AR 0.8) the band covered 20 of 30, against 89 of 100
with independent noise. It stays out of `predict` until it covers on real probe losses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MODEL_ID = "loss-powerlaw@0.1.0"
MIN_POINTS = 30  # guess: fewer steps than this and the exponent is a guess too
MAX_HALF_WIDTH = math.log(1.5)  # main plan: abstain when the band is wider than +-50%
BLOCK = 10  # bootstrap block length in steps: loss noise is autocorrelated over a few steps


@dataclass(frozen=True)
class Forecast:
	point: float | None
	lo: float | None
	hi: float | None
	status: str  # "extrapolated" or "abstain"
	reason: str
	params: tuple[float, float, float] | None = None  # (a, b, c)


def _fit(t: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
	"""(a, b, c, sse) of L = c + a t^-b, searching c below the smallest loss seen."""
	lt = np.log(t)
	best = None
	y_min = float(y.min())
	for c in np.linspace(0.0, y_min * 0.999, 60):
		z = y - c
		if (z <= 0).any():
			continue
		slope, icept = np.polyfit(lt, np.log(z), 1)
		b = -slope
		if b <= 0:
			continue
		pred = c + math.exp(icept) * t ** (-b)
		sse = float(((pred - y) ** 2).sum())
		if best is None or sse < best[3]:
			best = (math.exp(icept), b, float(c), sse)
	if best is None:
		raise ValueError("the loss isn't falling, so no power law fits")
	return best


def forecast(
	losses: list[float | None], last_step: int, first_step: int = 1, draws: int = 200, seed: int = 0
) -> Forecast:
	"""The loss at `last_step`, from a series that starts at `first_step` (one value per step, None for missing)."""
	pts = [(first_step + i, float(v)) for i, v in enumerate(losses) if v is not None and math.isfinite(v)]
	if len(pts) < MIN_POINTS:
		return Forecast(
			None, None, None, "abstain", f"only {len(pts)} loss values; the fit needs {MIN_POINTS}"
		)
	t = np.asarray([p[0] for p in pts], dtype=float)
	y = np.asarray([p[1] for p in pts], dtype=float)
	if last_step <= t[-1]:
		v = float(y[int(np.argmin(np.abs(t - last_step)))])
		return Forecast(
			v, v, v, "extrapolated", f"the probe already reached step {last_step}: its loss there"
		)
	try:
		a, b, c, _ = _fit(t, y)
	except ValueError as exc:
		return Forecast(None, None, None, "abstain", str(exc))
	fitted = c + a * t ** (-b)
	resid = y - fitted
	rng = np.random.default_rng(seed)
	outs = []
	n = len(y)
	for _ in range(draws):
		starts = rng.integers(0, max(1, n - BLOCK + 1), size=math.ceil(n / BLOCK))
		boot = np.concatenate([resid[s : s + BLOCK] for s in starts])[:n]
		try:
			ab, bb, cb, _ = _fit(t, fitted + boot)
		except ValueError:
			continue
		outs.append(cb + ab * last_step ** (-bb))
	point = c + a * last_step ** (-b)
	if len(outs) < draws // 2 or point <= 0:
		return Forecast(None, None, None, "abstain", "the fit is unstable under resampling", (a, b, c))
	lo, hi = (float(x) for x in np.quantile(outs, [0.05, 0.95]))
	lo, hi = min(lo, point), max(hi, point)
	if lo <= 0 or max(math.log(point / lo), math.log(hi / point)) > MAX_HALF_WIDTH:
		return Forecast(
			None, None, None, "abstain",
			f"the 90% band ({lo:.3g} to {hi:.3g}) is wider than +-50% of the forecast; probe for more steps",
			(a, b, c),
		)  # fmt: skip
	why = (
		f"power law L = {c:.3g} + {a:.3g} t^-{b:.2f} fitted on steps {int(t[0])} to {int(t[-1])}, "
		f"extrapolated to step {last_step}; band from a block bootstrap of the residuals"
	)
	return Forecast(point, lo, hi, "extrapolated", why, (a, b, c))


# ---------------------------------------------------------------- the ensemble with LC-PFN (corpus/lcpfn.py)

MODEL_ID_ENSEMBLE = "loss-ensemble@0.1.0"


def ensemble(power: Forecast, pfn: tuple[float, float, float] | None, pfn_reason: str = "") -> Forecast:
	"""The power law and LC-PFN together (main plan: "ensembled; abstains when the band is too wide"). `pfn` is
	LC-PFN's (5%, 50%, 95%) loss at the last step, which the runner computes with torch (corpus/lcpfn.py); this
	half is plain numbers so it runs anywhere. The point is the mean of the two medians and the band the union of the
	two bands, so the ensemble is never surer than either half; it abstains past +-50%, as the power law does.
	With one half missing it is the other half alone, and says so."""
	if pfn is None:
		return power
	lo_p, mid_p, hi_p = (float(v) for v in pfn)
	if not (0 < lo_p <= mid_p <= hi_p):
		return power
	why_pfn = pfn_reason or "LC-PFN posterior"
	if power.point is None and power.params is not None:
		# review, Nov 1: the power law fitted but its band was too wide or unstable; LC-PFN alone would be surer
		# than "I don't know", so the ensemble abstains too
		return power
	if power.point is None:  # too few points for the power law: LC-PFN alone
		point, lo, hi = mid_p, lo_p, hi_p
		why = f"{why_pfn}; the power law abstained ({power.reason})"
	else:
		point = (power.point + mid_p) / 2
		lo, hi = min(power.lo, lo_p, point), max(power.hi, hi_p, point)
		why = f"mean of the power law and LC-PFN, band the union of theirs: {power.reason}; {why_pfn}"
	if max(math.log(point / lo), math.log(hi / point)) > MAX_HALF_WIDTH:
		return Forecast(
			None, None, None, "abstain",
			f"the 90% band ({lo:.3g} to {hi:.3g}) is wider than +-50% of the forecast; probe for more steps",
			power.params,
		)  # fmt: skip
	return Forecast(point, lo, hi, "extrapolated", why, power.params)
