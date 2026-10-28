"""Feature flags for the API.

Defaults come from the repo's flags.json (flat `name: bool`, every feature off until its
validation row passes). Overrides, later wins: flags.<PREFLIGHT_ENV>.json at the repo root,
then PREFLIGHT_FLAGS (a JSON object).
"""

from __future__ import annotations

import contextlib
import json
import os
from functools import lru_cache
from pathlib import Path

ROOT = Path(os.environ.get("PREFLIGHT_ROOT", Path(__file__).resolve().parents[2]))


def _bools(raw: dict) -> dict[str, bool]:
	return {k: v for k, v in raw.items() if not k.startswith("$") and isinstance(v, bool)}


def _read(path: Path) -> dict[str, bool]:
	try:
		return _bools(json.loads(path.read_text()))
	except (OSError, ValueError, AttributeError):
		return {}


@lru_cache(maxsize=1)
def _resolved() -> dict[str, bool]:
	flags = _read(ROOT / "flags.json")
	env = os.environ.get("PREFLIGHT_ENV")
	if env:
		flags.update(_read(ROOT / f"flags.{env}.json"))
	raw = os.environ.get("PREFLIGHT_FLAGS")
	if raw:
		with contextlib.suppress(ValueError, AttributeError):
			flags.update(_bools(json.loads(raw)))
	return flags


def flag(name: str) -> bool:
	return _resolved().get(name, False)


def reload() -> None:
	_resolved.cache_clear()
