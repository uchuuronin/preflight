"""Every tracked path must stay unique when case is ignored.

Windows and macOS (by default) treat `Orbit.ts` and `orbit.ts` as one file, so an import resolves to the
wrong module and the app fails to load there while Linux CI stays green (found Oct 31: web/src/workspace/map
had Orbit.tsx beside orbit.ts). Module paths are also compared without their extension, since `./Orbit`
resolves across .ts and .tsx.
"""

import subprocess
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_EXTS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".py")


def _tracked() -> list[str]:
	try:
		out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
	except (OSError, subprocess.CalledProcessError):
		pytest.skip("not a git checkout")
	return [p for p in out.splitlines() if p]


def test_no_two_paths_differ_only_in_case():
	seen = defaultdict(set)
	for p in _tracked():
		seen[p.lower()].add(p)
	assert not [v for v in seen.values() if len(v) > 1]


def test_no_two_modules_differ_only_in_case():
	seen = defaultdict(set)
	for p in _tracked():
		stem, dot, ext = p.rpartition(".")
		if dot and "." + ext in MODULE_EXTS:
			seen[stem.lower()].add(stem)
	assert not [sorted(v) for v in seen.values() if len(v) > 1]
