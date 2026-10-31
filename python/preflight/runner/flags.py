"""Feature flags for the runner, read the way C's API reads them (api/preflight_api/flags.py).

Defaults come from the repo's flags.json (flat `name: bool`, every feature off until its
validation row passes). Overrides, later wins: flags.<PREFLIGHT_ENV>.json next to it, then
PREFLIGHT_FLAGS (a JSON object). In the repo, flags.json is at the root; in the worker image it
sits beside the package at /opt/preflight/flags.json (runner/modal_app.py ships it).
"""

import contextlib
import json
import os
from functools import lru_cache
from pathlib import Path


def _root() -> Path:
	env = os.environ.get("PREFLIGHT_ROOT")
	if env:
		return Path(env)
	here = Path(__file__).resolve()
	for up in here.parents[1:4]:  # <repo>/python/preflight/runner, or /opt/preflight/preflight/runner
		if (up / "flags.json").is_file():
			return up
	return here.parents[2]


def _bools(raw) -> dict[str, bool]:
	if not isinstance(raw, dict):
		return {}
	return {
		k: v for k, v in raw.items() if isinstance(k, str) and not k.startswith("$") and isinstance(v, bool)
	}


def _read(path: Path) -> dict[str, bool]:
	try:
		return _bools(json.loads(path.read_text()))
	except (OSError, ValueError):
		return {}


@lru_cache(maxsize=1)
def _resolved() -> dict[str, bool]:
	root = _root()
	flags = _read(root / "flags.json")
	env = os.environ.get("PREFLIGHT_ENV")
	if env:
		flags.update(_read(root / f"flags.{env}.json"))
	raw = os.environ.get("PREFLIGHT_FLAGS")
	if raw:
		with contextlib.suppress(ValueError):
			flags.update(_bools(json.loads(raw)))
	return flags


def flag(name: str) -> bool:
	return _resolved().get(name, False)


def reload() -> None:
	_resolved.cache_clear()
