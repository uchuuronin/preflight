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

	def label(self) -> str:
		"""Which runs set this band, saying when a GPU and job type had too few and was merged."""
		_, gpu, lens = (*self.key, "", "")[:3]
		if gpu and lens:
			return f"{gpu}, {lens} (n = {self.n})"
		if lens:
			return f"{lens}, merged across GPUs (n = {self.n})"
		return f"merged across GPUs and job types (n = {self.n})"


@dataclass(frozen=True)
class Table:
	"""Half-widths by (target, gpu, lens), with coarser fallbacks; built from held-out runs."""

	cells: dict[tuple[str, ...], Cell] = field(default_factory=dict)
	built_from: str = ""
	version: str = "0"
	oom_reliable: bool = False  # P(OOM) passed its reliability diagram on real outcomes (engine.oom)
	coverage: dict[str, tuple[int, int]] = field(
		default_factory=dict
	)  # target -> leave-one-arch-out (hits, n)
	holdout: dict[str, dict[str, tuple[int, int]]] = field(
		default_factory=dict
	)  # target -> gpu -> (hits, n), once
	# P(OOM)'s reliability diagram on oom.SHOWN_EDGES: (lo, hi, points, agrees with outcomes) per bin
	oom_bins: tuple[tuple, ...] = ()  # (lo, hi, n, consistent, ooms); older tables have no ooms

	def describe(self, target: str) -> str:
		"""For a model card: the cells this target has, how pairs below MIN_CELL merge, coverage and holdout."""
		own = sorted(
			(c for k, c in self.cells.items() if k[0] == target), key=lambda c: (-len(c.key[1]), c.key)
		)
		if not own:
			return "no held-out cells"
		parts = [f"Own cells: {'; '.join(c.label() for c in own)}."]
		pooled = (target, "", "") in self.cells
		parts.append(
			f"A GPU and job type with fewer than {MIN_CELL} held-out runs has no band of its own: it uses its job "
			+ (
				"type merged across GPUs, then all runs."
				if pooled
				else "type merged across GPUs. A job type with no cell keeps the simulated band."
			)
		)
		if self.coverage.get(target, (0, 0))[1]:
			h, n = self.coverage[target]
			lo, hi = wilson(h, n)
			parts.append(f"Leave-one-architecture-out coverage {h} of {n} (Wilson 95% {lo:.0%} to {hi:.0%}).")
		for gpu, (h, n) in sorted(self.holdout.get(target, {}).items()):
			if not n:
				continue
			lo, hi = wilson(h, n)
			parts.append(
				f"Holdout, never tuned on: {gpu} {h} of {n} inside the band (Wilson 95% {lo:.0%} to {hi:.0%})."
			)
		return " ".join(parts)

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
			"coverage": {t: list(v) for t, v in self.coverage.items()},
			"holdout": {t: {g: list(v) for g, v in d.items()} for t, d in self.holdout.items()},
			"oom_bins": [list(b) for b in self.oom_bins],
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
		return cls(
			cells,
			d.get("built_from", ""),
			d.get("version", "0"),
			bool(d.get("oom_reliable", False)),
			{t: (int(v[0]), int(v[1])) for t, v in d.get("coverage", {}).items()},
			{t: {g: (int(v[0]), int(v[1])) for g, v in gs.items()} for t, gs in d.get("holdout", {}).items()},
			tuple(
				(float(b[0]), float(b[1]), int(b[2]), bool(b[3]), *((int(b[4]),) if len(b) > 4 else ()))
				for b in d.get("oom_bins", ())
			),
		)


def calibrate(scores: Iterable[Score], level: float = LEVEL, built_from: str = "") -> Table:
	"""A table from held-out scores: every cell with at least MIN_CELL runs at each level of detail."""
	by: dict[tuple[str, ...], list[Score]] = defaultdict(list)
	for s in scores:
		# a score with no lens (the cross-GPU band) would land in the same pooled cell twice
		for key in dict.fromkeys(((s.target, s.gpu, s.lens), (s.target, "", s.lens), (s.target, "", ""))):
			by[key].append(s)
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
	import preflight.engine._calibration_table as _MOD  # absent: ModuleNotFoundError, quietly simulated

	TABLE = Table.from_dict(_MOD.DATA)  # a module without DATA raises AttributeError: warned below
except ModuleNotFoundError:
	TABLE = Table(built_from="none: no held-out real runs yet")
except (SyntaxError, NameError, AttributeError, ImportError, KeyError, TypeError, ValueError) as _exc:
	# a broken generated table must never stop the engine importing, but it must not pass silently either
	import warnings

	warnings.warn(
		f"engine/_calibration_table.py is unusable ({_exc!r}); bands fall back to simulated", stacklevel=1
	)
	TABLE = Table(built_from=f"none: the generated table is unusable ({type(_exc).__name__})")


def half_width(target: str, gpu: str | None, lens: str | None, simulated: float, table: Table | None = None):
	"""(half-width, source, cell) for a band: the table's held-out cell when there is one, else the simulated one."""
	cell = (table or TABLE).lookup(target, gpu, lens)
	if cell is None:
		return simulated, "simulated", None
	return cell.half_width, cell.source, cell
