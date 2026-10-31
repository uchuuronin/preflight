"""Power analysis: how many seeds a comparison needs (main plan, Prediction engine: "Power analysis").

Two configurations are compared by their final loss, averaged over seeds. Seed-to-seed spread decides how many seeds
it takes to tell a real gap from noise. Short multi-seed probes measure that spread at matched steps; this turns it
into seeds per arm for a chosen gap, significance level and power, using the two-sample normal approximation
n = 2 (z_(1-a/2) + z_(1-b))^2 s^2 / d^2. With few seeds, s itself is uncertain, so the answer carries an interval
from the chi-square interval on s (Wilson-Hilferty). Validation row: predicted seed variance within 25% of
measured. Pure Python (statistics.NormalDist), so it runs in Pyodide.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass

MODEL_ID = "power-seeds@0.1.0"
_N = statistics.NormalDist()


def _chi2_quantile(p: float, df: int) -> float:
	"""Wilson-Hilferty: chi-square quantiles to within about 3% for df >= 4 (5 seeds), closer as df grows."""
	z = _N.inv_cdf(p)
	a = 2.0 / (9.0 * df)
	return df * (1 - a + z * math.sqrt(a)) ** 3


@dataclass(frozen=True)
class SeedSpread:
	sd: float
	lo: float  # 90% interval on the seed-to-seed sd
	hi: float
	n: int


def seed_spread(values: list[float], level: float = 0.9) -> SeedSpread:
	"""Seed-to-seed standard deviation of a metric (one value per seed, at the same step), with its interval."""
	xs = [float(v) for v in values if v is not None and math.isfinite(v)]
	if len(xs) < 2:
		raise ValueError("needs at least two seeds")
	sd = statistics.stdev(xs)
	df = len(xs) - 1
	tail = (1 - level) / 2
	lo = sd * math.sqrt(df / _chi2_quantile(1 - tail, df))
	hi = sd * math.sqrt(df / _chi2_quantile(tail, df))
	return SeedSpread(sd, lo, hi, len(xs))


@dataclass(frozen=True)
class SeedsNeeded:
	point: int
	lo: int
	hi: int
	reason: str


def seeds_needed(spread: SeedSpread, gap: float, alpha: float = 0.05, power: float = 0.8) -> SeedsNeeded:
	"""Seeds per configuration to detect a true difference of `gap` in the mean, two-sided at `alpha`."""
	if gap <= 0:
		raise ValueError("the gap to detect must be positive")
	k = (_N.inv_cdf(1 - alpha / 2) + _N.inv_cdf(power)) ** 2

	def n(sd: float) -> int:
		return max(2, math.ceil(2 * k * sd * sd / (gap * gap)))

	why = (
		f"to see a {gap:g} gap at {alpha:g} significance with {power:.0%} power; seed-to-seed sd {spread.sd:.3g} "
		f"(90%: {spread.lo:.3g} to {spread.hi:.3g}) from {spread.n} seeds"
	)
	return SeedsNeeded(n(spread.sd), n(spread.lo), n(spread.hi), why)
