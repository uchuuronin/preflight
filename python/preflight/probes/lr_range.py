"""The learning-rate range test (Smith, "Cyclical Learning Rates for Training Neural Networks",
2017, section 3.3; D, Nov 2, behind probes.lr_range, off).

About 100 optimizer steps with the learning rate rising exponentially from 1e-7 to 1, the loss
recorded at each. Read from the curve: the learning rate where the smoothed loss falls fastest
against log(lr) (the suggestion), the one where it is lowest, and the one where it diverges.
Stored in the probe's extras, floats only (no contract change):

	lr_range.suggested_lr      steepest fall
	lr_range.min_loss_lr       lowest smoothed loss
	lr_range.diverged_lr       where it diverged (absent if it never did)
	lr_range.steps             points read
	lr_range.flat              1.0 when the loss never fell by FLAT_DROP: no suggestion then
	lr_range.lr.NNN, lr_range.loss.NNN   the curve, raw loss, at most MAX_POINTS points

Diverged (`Divergence`, the same rule in the recorder, which stops the probe there): a loss that
isn't finite, or the raw loss above DIVERGE_X times the best of a running (look-behind) average,
two steps running. The curve is read only up to there, so the blow-up never leaks into the
smoothing. A step whose loss the recorder couldn't read (no one-element loss) is skipped, not
taken as divergence.

The rates are the ramp's own (`lr_at`), which the recorder sets on the parameter group with the
largest rate (others keep their ratio to it), so the suggestion is in the units of the script's
largest rate. The suggestion is a starting point for the script's own schedule, not a measured
optimum: the range test runs from the model's first steps, without the script's warm-up or decay.
Not caught: an optimizer that ignores the rate it's given (Adafactor with `relative_step`), and a
step mixed precision skipped (its loss joins the next step's).
"""

import math

LR_START = 1e-7  # cited: Smith's range and fastai's lr_find defaults
LR_END = 1.0
STEPS = 100  # cited: fastai's lr_find default (num_it=100)
BETA = 0.9  # guess: EMA smoothing for 100 points (fastai uses 0.98 with its default)
DIVERGE_X = 4.0  # cited: fastai stops when the smoothed loss passes 4x its best
DIVERGE_RUN = 2  # guess: two steps running, so one spike doesn't end the curve
SKIP_START = 5  # guess: the first few points are the optimizer's start-up, not the curve
FLAT_DROP = 0.05  # guess: below a 5% fall from its start the curve has no slope to read
MAX_POINTS = 128


def lr_at(step: int, steps: int = STEPS, start: float = LR_START, end: float = LR_END) -> float:
	"""The learning rate for optimizer step `step` (0-based) of `steps`, rising exponentially (flat
	when start equals end: a run at one fixed rate)."""
	if steps <= 1 or start == end:
		return start
	return start * (end / start) ** (min(step, steps - 1) / (steps - 1))


class Divergence:
	"""Feed losses in step order; `add` says whether the run has diverged at this step."""

	def __init__(self):
		self.avg, self.n, self.best, self.bad = 0.0, 0, math.inf, 0

	def add(self, loss: float | None, nonfinite: bool = False) -> bool:
		if nonfinite or (loss is not None and not math.isfinite(loss)):
			return True
		if loss is None:
			return False  # no loss read at this step: nothing to judge
		if self.n >= SKIP_START and self.best > 0 and loss > DIVERGE_X * self.best:
			self.bad += 1
		else:
			self.bad = 0
		self.n += 1
		self.avg = BETA * self.avg + (1 - BETA) * loss
		self.best = min(self.best, self.avg / (1 - BETA**self.n))
		return self.bad >= DIVERGE_RUN


def _ema(xs: list[float]) -> list[float]:
	out, avg = [], 0.0
	for i, v in enumerate(xs, 1):
		avg = BETA * avg + (1 - BETA) * v
		out.append(avg / (1 - BETA**i))  # bias-corrected
	return out


def _smooth(losses: list[float]) -> list[float]:
	"""Forward then backward: no lag, so the curve's features stay at their learning rates (a forward
	EMA alone moves them about 1 / (1 - BETA) steps to the right)."""
	return _ema(_ema(losses)[::-1])[::-1]


def analyse(points: list[tuple]) -> dict[str, float]:
	"""Extras from (learning rate, loss[, nonfinite]) per step, in order. `nonfinite` true: the step's
	loss wasn't finite (the recorder stores such a loss as None)."""
	extras: dict[str, float] = {}
	lrs, losses, diverged = [], [], None
	div = Divergence()
	for pt in points:
		lr, loss = pt[0], pt[1]
		bad = bool(pt[2]) if len(pt) > 2 else False
		if lr is None or not (lr > 0) or not math.isfinite(lr):
			continue
		if div.add(loss, bad):
			diverged = float(lr)
			break
		if loss is None:
			continue
		lrs.append(float(lr))
		losses.append(float(loss))
	if diverged is not None:
		extras["lr_range.diverged_lr"] = diverged
		# the steps that led into it are the blow-up's start, not the curve
		while losses and len(losses) > SKIP_START and losses[-1] > DIVERGE_X * min(losses):
			lrs.pop()
			losses.pop()
	extras["lr_range.steps"] = float(len(lrs))
	stride = max(1, math.ceil(len(lrs) / MAX_POINTS))
	for j, i in enumerate(range(0, len(lrs), stride)):
		extras[f"lr_range.lr.{j:03d}"] = lrs[i]
		extras[f"lr_range.loss.{j:03d}"] = losses[i]
	end = len(lrs)
	if end < SKIP_START + 3:
		extras["lr_range.flat"] = 1.0
		return extras
	ys = _smooth(losses)
	xs = [math.log10(x) for x in lrs]
	lo = min(range(SKIP_START, end), key=lambda i: ys[i])
	extras["lr_range.min_loss_lr"] = lrs[lo]
	start_level = ys[SKIP_START]
	if start_level <= 0 or (start_level - ys[lo]) / start_level < FLAT_DROP:
		extras["lr_range.flat"] = 1.0
		return extras
	# steepest fall of the smoothed loss against log10(lr), from the start up to the lowest point
	best_i, best_slope = None, 0.0
	for i in range(SKIP_START + 1, lo + 1):
		dx = xs[i] - xs[i - 1]
		if dx <= 0:
			continue
		slope = (ys[i] - ys[i - 1]) / dx
		if slope < best_slope:
			best_i, best_slope = i, slope
	if best_i is None:
		extras["lr_range.flat"] = 1.0
		return extras
	extras["lr_range.suggested_lr"] = lrs[best_i]
	return extras


def curve(extras: dict[str, float]) -> list[tuple[float, float]]:
	"""The stored curve back as (lr, loss) pairs, for the guide and the cockpit."""
	out, j = [], 0
	while f"lr_range.lr.{j:03d}" in extras:
		out.append((extras[f"lr_range.lr.{j:03d}"], extras[f"lr_range.loss.{j:03d}"]))
		j += 1
	return out
