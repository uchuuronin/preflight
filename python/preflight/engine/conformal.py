"""Split-conformal intervals on log ratios (Review, Oct 27).

Scores are |log(actual / predicted)| on held-out real runs. Calibration is split by model
architecture, so no architecture sits in both calibration and test, and kept separately per
GPU and per lens. The 90% half-width for a cell is the ceil((n+1)(1-alpha))-th smallest score
of its n calibration runs; a cell with too few runs falls back to a coarser one (lens only, then
the target alone), and with none at all to the simulated constants, labelled as such.

Pure and Pyodide-safe: the table is data (a dict), built offline by corpus/calibrate.py and
shipped as a generated module; nothing here reads files.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field

LEVEL = 0.9  # cited: contract, 90% intervals
MIN_CELL = 9  # derived: the smallest n where ceil((n+1)*0.9) <= n, so a 90% quantile exists
Z95 = 1.959963984540054  # derived: two-sided 95% normal quantile, for Wilson intervals


@dataclass(frozen=True)
class Score:
	"""One held-out run: what the engine predicted and what the run measured."""

	target: str  # memory.peak, time.step, ...
	gpu: str  # engine.gpus table name
	lens: str
	group: str  # architecture group: runs of one architecture never straddle calibration and test
	predicted: float
	actual: float

	@property
	def log_ratio(self) -> float:
		return math.log(self.actual / self.predicted)

	@property
	def value(self) -> float:
		return abs(self.log_ratio)


def conformal_half_width(scores: list[float], level: float = LEVEL) -> float | None:
	"""Split-conformal quantile: the ceil((n+1)*level)-th smallest score, or None when n is too small."""
	n = len(scores)
	k = math.ceil((n + 1) * level)
	if n == 0 or k > n:
		return None
	return sorted(scores)[k - 1]


def wilson(hits: int, n: int, z: float = Z95) -> tuple[float, float]:
	"""Wilson score interval for a proportion (the review's rule for reporting coverage)."""
	if n <= 0:
		return 0.0, 1.0
	p = hits / n
	den = 1 + z * z / n
	mid = (p + z * z / (2 * n)) / den
	half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
	return max(0.0, mid - half), min(1.0, mid + half)


@dataclass(frozen=True)
class Cell:
	half_width: float
	n: int
	groups: int
	source: str  # "held_out" or "simulated"
	key: tuple[str, ...]
	residuals: tuple[
		float, ...
	] = ()  # signed log(actual / predicted), sorted: P(OOM) reads these (engine.oom)


@dataclass(frozen=True)
class Table:
	"""Half-widths by (target, gpu, lens), with coarser fallbacks; built from held-out runs."""

	cells: dict[tuple[str, ...], Cell] = field(default_factory=dict)
	built_from: str = ""
	version: str = "0"
	oom_reliable: bool = False  # P(OOM) passed its reliability diagram on real outcomes (engine.oom)

	def lookup(self, target: str, gpu: str | None, lens: str | None) -> Cell | None:
		"""The most specific cell with enough runs: (target, gpu, lens), then (target, lens), then (target,)."""
		for key in ((target, gpu or "", lens or ""), (target, "", lens or ""), (target, "", "")):
			c = self.cells.get(key)
			if c is not None:
				return c
		return None

	def to_dict(self) -> dict:
		return {
			"version": self.version,
			"built_from": self.built_from,
			"oom_reliable": self.oom_reliable,
			"cells": [
				{
					"key": list(c.key),
					"half_width": c.half_width,
					"n": c.n,
					"groups": c.groups,
					"source": c.source,
					"residuals": [round(r, 5) for r in c.residuals],
				}
				for c in self.cells.values()
			],
		}

	@classmethod
	def from_dict(cls, d: dict) -> Table:
		cells = {
			tuple(c["key"]): Cell(
				c["half_width"],
				c["n"],
				c["groups"],
				c["source"],
				tuple(c["key"]),
				tuple(c.get("residuals", ())),
			)
			for c in d.get("cells", [])
		}
		return cls(cells, d.get("built_from", ""), d.get("version", "0"), bool(d.get("oom_reliable", False)))


def calibrate(scores: Iterable[Score], level: float = LEVEL, built_from: str = "") -> Table:
	"""A table from held-out scores: every cell with at least MIN_CELL runs at each level of detail."""
	by: dict[tuple[str, ...], list[Score]] = defaultdict(list)
	for s in scores:
		by[(s.target, s.gpu, s.lens)].append(s)
		by[(s.target, "", s.lens)].append(s)
		by[(s.target, "", "")].append(s)
	cells = {}
	for key, rows in by.items():
		h = conformal_half_width([r.value for r in rows], level) if len(rows) >= MIN_CELL else None
		if h is not None:
			res = tuple(sorted(r.log_ratio for r in rows))
			cells[key] = Cell(h, len(rows), len({r.group for r in rows}), "held_out", key, res)
	return Table(cells, built_from)


@dataclass(frozen=True)
class Coverage:
	hits: int
	n: int
	lo: float
	hi: float
	median_half_width: float

	@property
	def rate(self) -> float:
		return self.hits / self.n if self.n else float("nan")


def evaluate(scores: list[Score], level: float = LEVEL) -> dict[str, Coverage]:
	"""Leave-one-architecture-out coverage per target: calibrate on the other groups, test on this one."""
	out: dict[str, Coverage] = {}
	for target in sorted({s.target for s in scores}):
		rows = [s for s in scores if s.target == target]
		hits, n, widths = 0, 0, []
		for g in sorted({s.group for s in rows}):
			table = calibrate([s for s in rows if s.group != g], level)
			for s in (s for s in rows if s.group == g):
				cell = table.lookup(s.target, s.gpu, s.lens)
				if cell is None:
					continue
				n += 1
				hits += s.value <= cell.half_width
				widths.append(cell.half_width)
		lo, hi = wilson(hits, n)
		widths.sort()
		out[target] = Coverage(hits, n, lo, hi, widths[len(widths) // 2] if widths else float("nan"))
	return out


# The table the engine uses. Empty until corpus/calibrate.py builds one from held-out real runs; with no
# cell, every band falls back to the simulated constants and its model card says calibration: simulated.
try:  # written by corpus/calibrate.py once held-out real runs exist; a plain module, so Pyodide imports it too
	from preflight.engine._calibration_table import DATA as _DATA

	TABLE = Table.from_dict(_DATA)
except ImportError:
	TABLE = Table(built_from="none: no held-out real runs yet")


def half_width(target: str, gpu: str | None, lens: str | None, simulated: float, table: Table | None = None):
	"""(half-width, source, cell) for a band: the table's held-out cell when there is one, else the simulated one."""
	cell = (table or TABLE).lookup(target, gpu, lens)
	if cell is None:
		return simulated, "simulated", None
	return cell.half_width, cell.source, cell
