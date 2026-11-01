"""The hindsight divergence label (S3 D9, docs/decisions/engine-settings.md), with no torch: the corpus worker
labels each finished run with it, and the classifier's fit and tests read it without the GPU stack."""

from __future__ import annotations

import math
import statistics
from typing import Any

LABEL_WINDOW = 25  # simulated: S3 D9
LABEL_TAIL = 50  # simulated: S3 D9
LABEL_RATIO = 1.25  # simulated: S3 D9
LABEL_FLOOR = 0.1  # simulated: S3 D9
COLLAPSE_SPIKE = 2.0  # simulated: S3 D9
COLLAPSE_NO_LEARNING = 0.9  # simulated: S3 D9


def _rolling_median(xs: list[float], win: int) -> list[float]:
	pad = [xs[0]] * (win - 1) + xs
	return [statistics.median(pad[i : i + win]) for i in range(len(xs))]


def label_divergence(losses: list[float], vocab_size: int | None = None) -> dict[str, Any]:
	"""Hindsight label for a finished run (docs/decisions/engine-settings.md, S3 chosen D9).

	Diverged if the loss goes non-finite; or the run ends well above its best (the median of the
	last 50 steps exceeds 1.25x the best 25-step rolling median plus 0.1); or it spiked past 2x its
	starting loss and never learned below 90% of it (a collapse to the trivial baseline). A spike
	that recovers is not divergence. The onset is where the rolling median last rose above the line
	for good, or the spike for a collapse; the classifier learns to warn about it early.

	The no-learning line is 90% of the trivial baseline: the starting loss, or ln(vocab) when a large
	init starts the run above uniform prediction.
	"""
	for i, x in enumerate(losses):
		if x is None or not math.isfinite(x):
			return {"diverged": True, "diverged_step": i, "reason": "non-finite loss", "rule": "D9"}
	if len(losses) < 2 * LABEL_WINDOW:
		return {"diverged": False, "diverged_step": None, "reason": "too short to label", "rule": "D9"}
	med = _rolling_median(losses, LABEL_WINDOW)
	best, thresholds = math.inf, []
	for v in med:
		best = min(best, v)
		thresholds.append(LABEL_RATIO * best + LABEL_FLOOR)
	tail = statistics.median(losses[-min(LABEL_TAIL, len(losses) // 2) :])
	if tail > thresholds[-1]:
		below = [i for i, (v, t) in enumerate(zip(med, thresholds, strict=True)) if v <= t]
		onset = below[-1] + 1 if below else 0
		return {
			"diverged": True,
			"diverged_step": onset,
			"reason": "the run ends well above its best loss",
			"rule": "D9",
		}
	peak = max(range(len(losses)), key=losses.__getitem__)
	baseline = min(losses[0], math.log(vocab_size)) if vocab_size else losses[0]
	if losses[peak] > COLLAPSE_SPIKE * losses[0] and tail >= COLLAPSE_NO_LEARNING * baseline:
		return {
			"diverged": True,
			"diverged_step": peak,
			"reason": "it blew up and never learned afterwards",
			"rule": "D9",
		}
	return {"diverged": False, "diverged_step": None, "reason": "", "rule": "D9"}
