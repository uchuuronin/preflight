"""Monte Carlo for the dials' fixed scale: which fixed full-scale value per record keeps the
needle readable and moving as a visitor flies the knobs?

Five candidates, each fixed once per record (no running maximum):
  A  max hi over the record's grid cells (linear)
  B  95th percentile of hi over the cells (linear), past it the dial says "off scale"
  C  2 x the record's starting hi (linear), past it "off scale"
  D  1.25 x the largest hi within one knob step of the start (linear), past it "off scale"
  E  log scale from the smallest lo to the largest hi over the cells, at most 1,000x wide
     (LOG_SPAN_MAX in web/src/workspace/record/scale.ts)

Visits: the start, then random single-knob steps (what a visitor does), 40 steps per walk,
2,000 walks per record and target. Measured per candidate:
  off   share of visits whose value is past full scale
  tiny  share of visits whose needle sits under 5% of the arc (reads as empty)
  moved share of steps that move the needle by at least 2% of the arc (about 4 px)

Usage: python web/scripts/mc_dial_scale.py [record.json ...]
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = [
	ROOT / "web/src/workspace/fixtures/dev-finetune.record.json",
	ROOT / "schema/examples/finetune.json",
	ROOT / "schema/examples/pretrain.json",
]
DIAL_TARGETS = ("time.total", "cost.total", "throughput.tokens_per_s", "memory.peak")
WALKS = 2000
STEPS = 40
LOG_SPAN_MAX = 1000
rng = np.random.default_rng(7)


def f16(data: str, n: int, scale: float) -> np.ndarray:
	raw = np.frombuffer(base64.b64decode(data), dtype="<f2").astype(np.float64)
	assert raw.size == n, (raw.size, n)
	return raw * scale


def mask(data: str | None, n: int) -> np.ndarray:
	if not data:
		return np.zeros(n, dtype=bool)
	bits = np.unpackbits(np.frombuffer(base64.b64decode(data), dtype=np.uint8), bitorder="little")
	return bits[:n].astype(bool)


def start_index(grid: dict, knobs: dict) -> tuple[int, ...] | None:
	idx = []
	for axis in grid["axes"]:
		v = knobs.get(axis["knob_id"])
		if v not in axis["values"]:
			return None
		idx.append(axis["values"].index(v))
	return tuple(idx)


def walk(shape: tuple[int, ...], start: tuple[int, ...]) -> list[tuple[int, ...]]:
	at = list(start)
	out = [tuple(at)]
	for _ in range(STEPS):
		axis = int(rng.integers(len(shape)))
		step = int(rng.choice([-1, 1]))
		at[axis] = min(shape[axis] - 1, max(0, at[axis] + step))
		out.append(tuple(at))
	return out


def needle(v: float, opt: str, s: dict) -> tuple[float, bool]:
	if opt == "E":
		lo, hi = s["E"]
		if v <= lo:
			return 0.0, False
		f = np.log(v / lo) / np.log(hi / lo)
		return min(1.0, f), f > 1.0
	full = s[opt]
	return min(1.0, v / full), v > full * (1 + 1e-9)


def run(path: Path) -> list[dict]:
	rec = json.loads(path.read_text())
	knobs = {k["id"]: k["current"] for k in rec["knobs"]}
	rows = []
	for target in DIAL_TARGETS:
		grids = [g for g in rec["predictions"]["grids"] if g["target_id"] == target]
		grid = next(
			(g for g in grids if all(knobs.get(k) == v for k, v in (g.get("fixed") or {}).items())), None
		)
		if not grid:
			continue
		shape = tuple(len(a["values"]) for a in grid["axes"])
		n = int(np.prod(shape))
		sc = grid.get("scale") or 1
		hi = f16(grid["hi"], n, sc).reshape(shape)
		lo = f16(grid["lo"], n, sc).reshape(shape)
		ab = mask(grid.get("abstain"), n).reshape(shape)
		ok = ~ab & np.isfinite(hi) & (hi > 0)
		start = start_index(grid, knobs)
		if start is None or not ok[start]:
			continue
		near = [hi[start]]
		for a in range(len(shape)):
			for d in (-1, 1):
				j = list(start)
				j[a] += d
				if 0 <= j[a] < shape[a] and ok[tuple(j)]:
					near.append(hi[tuple(j)])
		s = {
			"A": float(hi[ok].max()),
			"B": float(np.percentile(hi[ok], 95)),
			"C": float(2 * hi[start]),
			"D": float(1.25 * max(near)),
			"E": (float(max(lo[ok][lo[ok] > 0].min(), hi[ok].max() / LOG_SPAN_MAX)), float(hi[ok].max())),
		}
		stats = {o: {"off": 0, "tiny": 0, "moved": 0, "visits": 0, "steps": 0} for o in "ABCDE"}
		for _ in range(WALKS):
			path_ = [p for p in walk(shape, start) if ok[p]]
			for o in "ABCDE":
				prev = None
				for p in path_:
					f, off = needle(float(hi[p]), o, s)
					st = stats[o]
					st["visits"] += 1
					st["off"] += off
					st["tiny"] += (f < 0.05) and not off
					if prev is not None and p != prev[0]:
						st["steps"] += 1
						st["moved"] += abs(f - prev[1]) >= 0.02
					prev = (p, f)
		for o, st in stats.items():
			rows.append(
				{
					"record": path.stem.replace(".record", ""),
					"target": target,
					"option": o,
					"off": st["off"] / st["visits"],
					"tiny": st["tiny"] / st["visits"],
					"moved": st["moved"] / max(1, st["steps"]),
				}
			)
	return rows


def main() -> None:
	paths = [Path(p) for p in sys.argv[1:]] or DEFAULT
	rows = [r for p in paths for r in run(p)]
	print(f"{'record':<14}{'target':<26}{'opt':<4}{'off':>7}{'tiny':>7}{'moved':>7}")
	for r in rows:
		print(
			f"{r['record']:<14}{r['target']:<26}{r['option']:<4}{r['off']:>7.1%}{r['tiny']:>7.1%}{r['moved']:>7.1%}"
		)
	print()
	print("Mean over records and targets")
	for o in "ABCDE":
		sel = [r for r in rows if r["option"] == o]
		m = {k: sum(r[k] for r in sel) / len(sel) for k in ("off", "tiny", "moved")}
		worst_off = max(r["off"] for r in sel)
		print(
			f"  {o}: off {m['off']:.1%} (worst {worst_off:.1%}), tiny {m['tiny']:.1%}, moved {m['moved']:.1%}"
		)


if __name__ == "__main__":
	main()
