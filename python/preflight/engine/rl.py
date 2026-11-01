"""RL post-training: the share of GRPO groups that teach nothing, and how long rollouts take (main plan, RL rollouts).

Zero-variance groups. A group of G responses to one prompt has no advantage signal when every response gets the same
reward: all pass or all fail. For a prompt with pass rate p that happens with probability p^G + (1 - p)^G (DAPO;
Dr. GRPO). A's probe measures each prompt's pass rate from the k responses it generated. So p is uncertain: a
prompt that passed 4 of 4 may have a true rate of 0.9, and then p^16 is 0.19, not 1. Each prompt's rate gets a
Jeffreys Beta posterior, and the fraction is averaged over prompts, with prompts resampled for the interval.

Rollout time. Batched generation runs until the longest response in the batch stops, or hits the completion cap
(HybridFlow; DAPO's overlong analysis). So a step's rollout follows E[max of N lengths, capped], not the mean length.
The probe measured one rollout at its own N and cap. The length distribution is fitted from the probe's mean and
95th percentile (a log-normal, which is what response lengths look like), and the time is scaled by the ratio of
expected batch maxima. With raw lengths it bootstraps them directly. Decoding is memory-bound, so seconds per
decoding step are taken as flat in N, and the band widens when N moves far from the probe.

Pure Python and numpy, no I/O: it runs in Pyodide.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MODEL_ID_ZERO = "rl-zero-spread@0.1.0"
MODEL_ID_ROLLOUT = "rl-rollout-bootstrap@0.1.0"
Z90 = 1.6448536269514722
DRAWS = 2000  # guess: enough for a stable 90% interval of a mean over prompts; about 5 ms
# guess: seconds per decoding step drift as the batch grows (memory-bound until it isn't); 0.1 in log per doubling
# of the sequences per call away from the probe's, revisited with A's GRPO probes at several group sizes
N_DRIFT_PER_DOUBLING = 0.10
# fitted on one pair of GRPO runs, until the unseen TRL script tests it on Nov 2. Response lengths differ from the probe's sample (other prompts, drift as RL training goes on). A's
# GRPO zoo runs 2 and 3, same settings and different prompts, differ by 0.155 in log rollout time per call while
# their expected longest responses predict run 3 from run 2 within 6%; 0.16 covers that gap (was a 0.10 guess). One
# pair of runs, so the GRPO zoo script is calibration for the rollout row, not its validation
LENGTH_DRIFT = 0.16
# checked, Nov 1, once: from run 2's 3 prompts the zero-spread band (0.06 to 0.71) missed run 3's 17 new prompts
# (0.82); from 8 prompts it covered the other 9. Fewer prompts than this and the readout abstains
MIN_PROMPTS = 8


@dataclass(frozen=True)
class Interval:
	point: float
	lo: float
	hi: float
	reason: str


# ---------------------------------------------------------------- zero-variance groups


def zero_spread(
	pass_rates: list[float], group_size: int, samples: int | None, draws: int = DRAWS, seed: int = 0
) -> Interval:
	"""Expected share of prompts whose group of `group_size` responses all pass or all fail.

	`pass_rates` are per-prompt rates the probe measured from `samples` responses each. With `samples` known, each
	prompt's rate has a Jeffreys posterior Beta(s + 1/2, samples - s + 1/2). Without it, the measured rates are
	used as they are, and the interval covers only the choice of prompts."""
	rates = np.clip(
		np.asarray([r for r in pass_rates if r is not None and math.isfinite(r)], dtype=float), 0, 1
	)
	if rates.size == 0:
		raise ValueError("needs at least one measured pass rate")
	if group_size < 1:
		raise ValueError("group size must be at least 1")
	rng = np.random.default_rng(seed)
	idx = rng.integers(0, rates.size, size=(draws, rates.size))  # prompts resampled
	if samples:
		s = np.rint(rates * samples)
		a, b = s + 0.5, samples - s + 0.5
		p = rng.beta(a[idx], b[idx])
		point = float(np.mean(_beta_moment(a, b, group_size)))
		how = f"each prompt's rate from {samples} responses, with its uncertainty"
	else:
		p = rates[idx]
		point = float(np.mean(rates**group_size + (1 - rates) ** group_size))
		how = "the measured rates taken as exact (the probe didn't say how many responses each came from)"
	frac = np.mean(p**group_size + (1 - p) ** group_size, axis=1)
	lo, hi = (float(x) for x in np.quantile(frac, [0.05, 0.95]))
	why = (
		f"{point:.0%} of groups of {group_size} would all pass or all fail, from {rates.size} prompts; {how}"
	)
	return Interval(point, min(lo, point), max(hi, point), why)


def _beta_moment(a: np.ndarray, b: np.ndarray, g: int) -> np.ndarray:
	"""E[p^g + (1-p)^g] for p ~ Beta(a, b), exactly: B(a+g, b)/B(a, b) + B(a, b+g)/B(a, b)."""

	def lbeta(x, y):
		return np.vectorize(math.lgamma)(x) + np.vectorize(math.lgamma)(y) - np.vectorize(math.lgamma)(x + y)

	base = lbeta(a, b)
	return np.exp(lbeta(a + g, b) - base) + np.exp(lbeta(a, b + g) - base)


# ---------------------------------------------------------------- response lengths


def lognormal_from(mean: float, p95: float) -> tuple[float, float] | None:
	"""(mu, sigma) of the log-normal with this mean and 95th percentile. None for a non-positive input; a 95th
	percentile too far above the mean for any log-normal gets the widest one, sigma = 1.645."""
	if mean <= 0 or p95 <= 0:
		return None
	gap = math.log(p95) - math.log(mean)  # = 1.645 sigma - sigma^2 / 2
	disc = Z90**2 - 2 * gap
	# a 95th percentile far above the mean has no log-normal: use the widest one
	sigma = Z90 if disc < 0 else Z90 - math.sqrt(disc)
	sigma = max(sigma, 1e-6)
	return math.log(mean) - sigma * sigma / 2, sigma


def expected_batch_max(lengths: np.ndarray, n: int, cap: float | None) -> np.ndarray | float:
	"""E[max of n responses drawn from `lengths`, each capped]: what a batched generate call waits for.

	Exact for drawing with replacement: the i-th smallest of m lengths is the maximum with probability
	(i/m)^n - ((i-1)/m)^n. `lengths` may be one sample (1-D) or a stack of bootstrap replicates (2-D, one per row)."""
	x = np.sort(np.asarray(lengths, dtype=float), axis=-1)
	if cap:
		x = np.minimum(x, cap)
	m = x.shape[-1]
	i = np.arange(1, m + 1, dtype=float)
	w = (i / m) ** max(1, int(n)) - ((i - 1) / m) ** max(1, int(n))
	out = (x * w).sum(axis=-1)
	return float(out) if np.ndim(out) == 0 else out


@dataclass(frozen=True)
class LengthSample:
	"""What the probe saw: raw lengths when it kept them, else a log-normal fitted to its mean and p95."""

	mean: float
	p95: float
	max: float
	n: int
	raw: tuple[float, ...] = ()

	def draws(self, size: int, rng: np.random.Generator) -> np.ndarray:
		if self.raw:
			return rng.choice(np.asarray(self.raw, dtype=float), size=size, replace=True)
		fit = lognormal_from(self.mean, self.p95)
		if fit is None:
			return np.full(size, self.mean)
		return rng.lognormal(fit[0], fit[1], size=size)

	def resampled(self, rng: np.random.Generator) -> np.ndarray:
		"""One bootstrap replicate of the probe's sample: n lengths, as the probe might have seen them."""
		return self.draws(max(self.n, 1), rng)


def truncation_share(sample: LengthSample, cap: float, draws: int = 20000, seed: int = 0) -> float:
	"""Share of responses that would hit `cap`."""
	return float(np.mean(sample.draws(draws, np.random.default_rng(seed)) >= cap))


def truncation_band(
	sample: LengthSample, cap: float, reps: int = 200, seed: int = 0
) -> tuple[float, float, float]:
	"""(share, 5%, 95%): the share hitting `cap`, with the probe's length sample bootstrapped for the band."""
	rng = np.random.default_rng(seed)
	point = truncation_share(sample, cap, seed=seed)
	shares = [float(np.mean(sample.resampled(rng) >= cap)) for _ in range(reps)]
	lo, hi = np.quantile(shares, [0.05, 0.95])
	return point, min(float(lo), point), max(float(hi), point)


def rollout_seconds(
	measured_s: float,
	sample: LengthSample,
	n_probe: int,
	cap_probe: float | None,
	n_query: int,
	cap_query: float | None,
	draws: int = 400,
	seed: int = 0,
) -> Interval:
	"""A step's rollout at n_query sequences per call and a cap of cap_query, from the probe's measured rollout.

	Time follows the expected longest capped response, so it is the measured time times the ratio of expected batch
	maxima. The length sample is bootstrapped for the interval; a big move in N, and the drift in lengths as RL
	training goes on, widen it."""
	if measured_s <= 0 or n_probe < 1 or n_query < 1:
		raise ValueError("needs a measured rollout and at least one sequence per call")
	rng = np.random.default_rng(seed)
	reps = np.stack(
		[sample.resampled(rng) for _ in range(draws)]
	)  # bootstrap replicates of the probe's lengths
	base = expected_batch_max(reps, n_probe, cap_probe)
	new = expected_batch_max(reps, n_query, cap_query)
	r = np.where(base > 0, new / np.maximum(base, 1e-12), 1.0)
	point = float(measured_s * np.median(r))
	lo_r, hi_r = np.quantile(r, [0.05, 0.95])
	# the bootstrap covers the length sample; widen for decode speed away from the probe's N and length drift
	extra = N_DRIFT_PER_DOUBLING * abs(math.log2(n_query / n_probe)) + LENGTH_DRIFT
	lo = float(measured_s * lo_r * math.exp(-extra))
	hi = float(measured_s * hi_r * math.exp(extra))
	parts = [f"the longest of {n_query} responses sets the pace"]
	if cap_query:
		parts.append(f"capped at {cap_query:g} tokens")
	if n_query != n_probe or cap_query != cap_probe:
		parts.append(
			f"scaled from the probe's {n_probe} per call"
			+ (f" at a {cap_probe:g}-token cap" if cap_probe else "")
		)
	parts.append(
		"lengths from "
		+ ("the probe's responses" if sample.raw else "a log-normal fitted to the probe's mean and p95")
	)
	return Interval(point, min(lo, point), max(hi, point), "; ".join(parts))
