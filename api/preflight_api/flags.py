"""Feature flags for the API: the repo's flags.json, overridden by PREFLIGHT_FLAGS (JSON)."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

_FLAGS_FILE = Path(__file__).resolve().parents[2] / "flags.json"


@lru_cache(maxsize=1)
def _resolved() -> dict[str, bool]:
	flags: dict[str, bool] = {}
	try:
		flags.update(json.loads(_FLAGS_FILE.read_text())["flags"])
	except (OSError, KeyError, ValueError):
		pass
	raw = os.environ.get("PREFLIGHT_FLAGS")
	if raw:
		try:
			flags.update({k: bool(v) for k, v in json.loads(raw).items()})
		except ValueError:
			pass
	return flags


def flag(name: str) -> bool:
	return _resolved().get(name, False)


def reload() -> None:
	_resolved.cache_clear()
