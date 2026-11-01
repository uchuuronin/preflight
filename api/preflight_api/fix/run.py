"""One fix race, end to end, inside the `preflight-fix` app's `race_fix` function.

The record's code is read again exactly as A's runner reads it for a probe on a record
(python/preflight/runner/record_jobs.py `resolve`, then sources.py `fetch`), so the API never
handles the visitor's files. The race itself is race.py's: Muse Code edits in a sandbox, one
worktree per candidate, diffs back.

What comes back is the public part only: each candidate's `public()` (never stderr), the cost
from Muse Code's reported tokens, and whether the box stopped at the cost cap.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from . import race as race_mod

# Source: derived, the two smoke runs on Oct 31 (41246585559 $0.12, 41250977670 about $0.09,
# docs/status/c.md), rounded up: what one candidate is allowed before the race stops.
USD_PER_CANDIDATE = 0.13


def stop_usd(n_candidates: int) -> float:
	"""The race's cost cap: every candidate's allowance, stopped as one box."""
	return round(USD_PER_CANDIDATE * n_candidates, 2)


def source_files(record_ref: str, *, reader, blobs) -> dict[str, str]:
	"""The record's code, read the way A's runner reads it for a probe on a record."""
	from preflight.runner.record_jobs import resolve
	from preflight.runner.sources import fetch

	origin = resolve({"kind": "record", "record_ref": record_ref}, reader=reader, blobs=blobs)
	return fetch(origin.fetch_source, blobs=blobs).files


def run_race(
	record_ref: str,
	candidates: list[dict],
	measured: str | None,
	*,
	reader,
	blobs,
	open_box: Callable[[], Any],
	key: str,
	path: str = race_mod.SYSTEM_PATH,
) -> dict:
	"""Read the code, open the box (a sandbox, only once the code is in hand), race, close it.

	`candidates` are `{template_id, value}` dicts, already checked by the API; checked again here
	by race.race. `open_box()` returns (box, close)."""
	t0 = time.monotonic()
	try:
		files = source_files(record_ref, reader=reader, blobs=blobs)
	except Exception as e:  # A's InvalidSubmission carries a visitor-safe code
		return {"error": getattr(e, "code", None) or "source_unreadable", "candidates": []}
	box, close = open_box()
	try:
		cands = race_mod.race(
			box,
			files,
			[(c["template_id"], c.get("value")) for c in candidates],
			key,
			measured=measured,
			path=path,
		)
	except race_mod.GateBroken:
		return {"error": "gate_not_seen", "candidates": []}
	except race_mod.BadUpload:
		return {"error": "nothing_to_fix", "candidates": []}
	finally:
		close()
	return {
		"seconds": round(time.monotonic() - t0, 1),
		"usd_from_reported_tokens": round(getattr(box, "spent", 0.0), 4),
		"stopped_at_cap": bool(getattr(box, "stopped", False)),
		"candidates": [c.public() for c in cands],
	}
