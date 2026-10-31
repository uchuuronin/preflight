"""P(OOM) from empirical residuals, and the reliability check it must pass (Review, Oct 27).

With held-out runs, P(OOM) is the share of past log residuals log(actual / predicted) that would
put this job's peak above the GPU's capacity: P(r > log(capacity / predicted)). Without them it
falls back to a log-normal on the simulated band, and says so. Either way the screen shows a bin,
not a percentage, until a reliability diagram on real outcomes shows the probabilities can be
read as probabilities.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

from preflight.engine.conformal import wilson

# guess: four bands for display; revisited once the reliability diagram on real outcomes exists (target Oct 31)
BINS = (
	(0.0, 0.05, "unlikely"),
	(0.05, 0.30, "possible"),
	(0.30, 0.70, "likely"),
	(0.70, 1.0001, "very likely"),
)


def bin_label(p: float) -> str:
	for lo, hi, label in BINS:
		if lo <= p < hi:
			return label
	return BINS[-1][2]


def empirical_p_oom(residuals: tuple[float, ...] | list[float], predicted: float, capacity: float) -> float:
	"""Share of residuals that would push the peak past capacity, with a half-count either side (Jeffreys),
	so a small sample never says exactly 0 or 1."""
	n = len(residuals)
	if n == 0 or predicted <= 0:
		raise ValueError("needs residuals and a positive prediction")
	if capacity <= 0:
		return 1.0
	margin = math.log(capacity / predicted)
	over = sum(1 for r in residuals if r > margin)
	return (over + 0.5) / (n + 1)


def lognormal_p_oom(predicted: float, capacity: float, sigma_log: float) -> float:
	if capacity <= 0:
		return 1.0
	if sigma_log <= 0:
		return float(predicted > capacity)
	z = (math.log(capacity) - math.log(predicted)) / sigma_log
	return 0.5 * math.erfc(z / math.sqrt(2))


@dataclass(frozen=True)
class Bin:
	lo: float
	hi: float
	n: int
	mean_predicted: float
	observed: float
	wilson_lo: float
	wilson_hi: float

	@property
	def consistent(self) -> bool:
		"""The bin's mean prediction sits inside the Wilson interval of what actually happened."""
		return self.n == 0 or self.wilson_lo <= self.mean_predicted <= self.wilson_hi


def reliability(
	pairs: list[tuple[float, bool]], edges: tuple[float, ...] = (0.0, 0.05, 0.3, 0.7, 1.0001)
) -> list[Bin]:
	"""Reliability diagram: predicted P(OOM) against how often those jobs really ran out of memory."""
	out = []
	for lo, hi in itertools.pairwise(edges):
		inside = [(p, y) for p, y in pairs if lo <= p < hi]
		n = len(inside)
		hits = sum(1 for _, y in inside if y)
		w = wilson(hits, n)
		mean_p = sum(p for p, _ in inside) / n if n else float("nan")
		out.append(Bin(lo, min(hi, 1.0), n, mean_p, hits / n if n else float("nan"), w[0], w[1]))
	return out


MIN_OOM_EVENTS = (
	10  # guess: real out-of-memory runs needed before the upper bins mean anything; revisit with the corpus
)


# Chiya, Oct 31: P(OOM) shows as a number only where the reliability diagram verifies it. Below 5% and above 95%
# it is shown; in between it reads "uncertain (5-95%)" until every middle bin holds MIN_SHOWN points that agree with
# what happened, then it switches to a number by itself when the table is rebuilt
SHOWN_EDGES = (0.0, 0.05, 0.30, 0.70, 0.95, 1.0001)
UNCERTAIN = (0.05, 0.95)
MIN_SHOWN = 10


def middle_verified(bins) -> bool:
	"""Every bin between 5% and 95% has at least MIN_SHOWN points and agrees with outcomes.

	Takes Bin objects or the table's (lo, hi, n, consistent) tuples."""
	rows = [(b.lo, b.hi, b.n, b.consistent) if isinstance(b, Bin) else tuple(b) for b in bins]
	middle = [r for r in rows if r[0] >= UNCERTAIN[0] - 1e-9 and r[1] <= UNCERTAIN[1] + 1e-9]
	return bool(middle) and all(n >= MIN_SHOWN and ok for _, _, n, ok in middle)


def shown_as_number(p: float, bins) -> bool:
	return p < UNCERTAIN[0] or p > UNCERTAIN[1] or middle_verified(bins)


def reliable(bins: list[Bin], min_per_bin: int = 10, min_ooms: int = MIN_OOM_EVENTS) -> bool:
	"""Probabilities can be shown as numbers only when every populated bin agrees with what happened, at least two
	bins are populated, and real out-of-memory runs were seen: a check made only on jobs that fit says nothing about
	the probabilities that matter."""
	populated = [b for b in bins if b.n >= min_per_bin]
	ooms = sum(round(b.observed * b.n) for b in bins if b.n)
	return len(populated) >= 2 and ooms >= min_ooms and all(b.consistent for b in populated)
