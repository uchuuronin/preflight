"""How much the recorder slows a step down, from paired step times (ground truth, Oct 28).

Two numbers, both medians of paired step ratios with a percentile bootstrap interval:

  sync overhead   one recorder run alternating clean steps (no recorder sync point) and synced
                  steps (the phase steps' sync points): each synced step against the mean of the
                  clean steps either side of it, so drift over the run cancels.
  hook overhead   the recorder's clean steps against the same step of a bare run (same script,
                  seed and data order; only a CUDA event at each optimizer step), paired by
                  step index.

Step times come from CUDA events at step boundaries only, warmup steps are dropped, and the
inputs are the probes' own step logs. Pure Python, so it is tested without a GPU.
"""

import random
import statistics
from dataclasses import dataclass

BOOTSTRAP_RESAMPLES = 4000  # derived: enough that the 95% interval's ends move under 0.1% between seeds
BOOTSTRAP_SEED = 20261028  # derived: fixed so the reported interval is reproducible


@dataclass
class Ratio:
	median: float
	low: float  # 2.5th percentile of the bootstrap medians
	high: float  # 97.5th
	pairs: int

	def as_dict(self) -> dict:
		return {"median": round(self.median, 5), "ci95": [round(self.low, 5), round(self.high, 5)],
			"pairs": self.pairs, "source": "measured"}  # fmt: skip


def median_ratio(
	ratios: list[float], *, resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED
) -> Ratio:
	if not ratios:
		raise ValueError("no pairs to compare")
	rng = random.Random(seed)
	n = len(ratios)
	boots = sorted(statistics.median(rng.choices(ratios, k=n)) for _ in range(resamples))
	lo = boots[int(0.025 * (resamples - 1))]
	hi = boots[int(0.975 * (resamples - 1))]
	return Ratio(statistics.median(ratios), lo, hi, n)


def _walls(step_log: list[dict]) -> list[tuple[str, float | None]]:
	return [(s.get("kind"), s.get("wall_s")) for s in step_log]


def sync_ratios(step_log: list[dict]) -> list[float]:
	"""Synced ('phase') steps against the mean of the clean ('timed') steps on both sides."""
	w = _walls(step_log)
	out = []
	for i in range(1, len(w) - 1):
		kind, t = w[i]
		(k0, a), (k1, b) = w[i - 1], w[i + 1]
		if kind == "phase" and k0 == "timed" and k1 == "timed" and t and a and b:
			out.append(t / ((a + b) / 2))
	return out


def hook_ratios(recorder_log: list[dict], bare_walls: list[float | None], *, warmup: int) -> list[float]:
	"""The recorder's clean steps against the bare run's same step (index from the first step)."""
	out = []
	for i, s in enumerate(recorder_log):
		if i < warmup or s.get("kind") != "timed" or i >= len(bare_walls):
			continue
		t, b = s.get("wall_s"), bare_walls[i]
		if t and b:
			out.append(t / b)
	return out


def report(recorder_log: list[dict], bare_walls: list[float | None], *, warmup: int) -> dict:
	sync = sync_ratios(recorder_log)
	hooks = hook_ratios(recorder_log, bare_walls, warmup=warmup)
	return {
		"sync_overhead": median_ratio(sync).as_dict() if sync else None,
		"hook_overhead": median_ratio(hooks).as_dict() if hooks else None,
		"note": "ratio of step times; 1.03 means the recorder makes a step 3% slower",
	}
