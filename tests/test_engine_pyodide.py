"""Pyodide parity (Review, Oct 27): the engine gives the same answers in the browser as in CPython.

tests/pyodide/parity.mjs runs tests/pyodide/cases.py in Pyodide on the three golden records; this test
runs the same cases here and compares. It needs node and `npm ci` in tests/pyodide, and Pyodide fetches
numpy and pydantic from its CDN. Without a route there it skips, unless PREFLIGHT_PYODIDE=1 (CI), where
it fails instead.
"""

import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from preflight.schema.encoding import decode_f16

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "tests" / "pyodide"
REQUIRED = os.environ.get("PREFLIGHT_PYODIDE") == "1"
GRID_FIELDS = {"point", "lo", "hi"}
REL = 1e-9  # floats: same IEEE arithmetic; transcendental functions may differ in the last bits
F16_REL = (
	2e-3  # grid cells: one float16 ulp (~0.1%) either side when a last-bit difference crosses a rounding edge
)


def _pyodide_results() -> dict:
	if shutil.which("node") is None or not (RUNNER / "node_modules" / "pyodide").exists():
		msg = "node or tests/pyodide/node_modules missing (run npm ci in tests/pyodide)"
		if REQUIRED:
			pytest.fail(msg)
		pytest.skip(msg)
	proc = subprocess.run(["node", "parity.mjs"], cwd=RUNNER, capture_output=True, text=True, timeout=900)
	if proc.returncode == 3 and not REQUIRED:
		pytest.skip(f"Pyodide's packages are unreachable here: {proc.stderr.strip()[:200]}")
	assert proc.returncode == 0, proc.stderr[-2000:]
	return json.loads(proc.stdout)


def _close(a, b, path: str, problems: list[str]) -> None:
	if isinstance(a, dict) and isinstance(b, dict):
		if a.keys() != b.keys():
			problems.append(f"{path}: keys differ {sorted(set(a) ^ set(b))}")
			return
		n = a.get("axes") and len(a.get("axes") or [])
		cells = math.prod(len(ax["values"]) for ax in a["axes"]) if n else None
		for k in a:
			if cells and k in GRID_FIELDS:
				x, y = decode_f16(a[k], cells) * a["scale"], decode_f16(b[k], cells) * b["scale"]
				bad = [
					(i, u, v)
					for i, (u, v) in enumerate(zip(x, y, strict=True))
					if not math.isclose(u, v, rel_tol=F16_REL, abs_tol=1e-12)
				]
				if bad:
					problems.append(f"{path}.{k}: {len(bad)} cells differ, first {bad[0]}")
			else:
				_close(a[k], b[k], f"{path}.{k}", problems)
	elif isinstance(a, list) and isinstance(b, list):
		if len(a) != len(b):
			problems.append(f"{path}: lengths {len(a)} vs {len(b)}")
			return
		for i, (u, v) in enumerate(zip(a, b, strict=True)):
			_close(u, v, f"{path}[{i}]", problems)
	elif isinstance(a, float) or isinstance(b, float):
		if not (
			isinstance(a, int | float)
			and isinstance(b, int | float)
			and math.isclose(a, b, rel_tol=REL, abs_tol=1e-15)
		):
			problems.append(f"{path}: {a!r} vs {b!r}")
	elif a != b:
		problems.append(f"{path}: {a!r} vs {b!r}")


def test_engine_matches_in_pyodide():
	sys.path.insert(0, str(RUNNER))
	try:
		import cases
	finally:
		sys.path.remove(str(RUNNER))
	here = json.loads(json.dumps(cases.run(str(ROOT))))
	there = _pyodide_results()
	assert there["python"].startswith("3.14")
	problems: list[str] = []
	_close(here, there["results"], "results", problems)
	assert not problems, "\n".join(problems[:20])
