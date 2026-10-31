"""Everything about a job expires with its record, 30 days on (Chiya, Oct 30).

The deployed runner's daily cleanup (modal_app.cleanup, started by tick) runs these beside records.prune:

  uploads       the files a visitor checked in (preflight-uploads, one file per content hash),
                30 days after the newest job that used them (DirBlobs.put touches a reused file),
                and never while a kept record or a queued or running job still names them
  probe cache   cached probe results, their error text included (the preflight-probe-cache Dict),
                30 days after they were cached; an entry without a time is from before this rule
                and goes on the first pass
  job index     what the job store keeps of a finished job (status, record id, error, its
                idempotency key), 30 days after it was submitted (InMemoryJobStore.expire)

A repo's code is never kept at all: it is deleted as soon as it is read (runner/sources.py).
Nothing here touches a file or entry dated in the future: a clock gone wrong keeps data, it
never deletes it early.
"""

import re
import time
from collections.abc import Iterable
from pathlib import Path

from .records import RECORD_TTL_S

TTL_S = RECORD_TTL_S
TEMP_TTL_S = 86400  # a half-written upload from an interrupted job, a day old


_UPLOAD_REF = re.compile(r'"(?:uri|files_ref)"\s*:\s*"(?:upload://)?sha256:([0-9a-f]{16,128})"')


def upload_refs(texts: Iterable[str]) -> set[str]:
	"""The uploads named in records or job requests (their `upload://sha256:...` uri or
	`files_ref`), as the file names DirBlobs stores them under."""
	return {f"{h}.json" for t in texts for h in _UPLOAD_REF.findall(t)}


def records_text(root: Path | str) -> Iterable[str]:
	for p in sorted(Path(root).glob("*.json")):
		try:
			yield p.read_text(errors="replace")
		except OSError:
			continue


def prune_uploads(root: Path | str, *, max_age_s: float = TTL_S, now: float | None = None,
	keep: set[str] = frozenset()) -> list[str]:  # fmt: skip
	"""Delete uploads last used more than max_age_s ago, and stale temp files; returns their names.
	keep: uploads a kept record or a live job still names (a re-probe reads its parent's files by
	reference, so the file lives as long as the newest record made from it)."""
	now = time.time() if now is None else now
	root = Path(root)
	if not root.is_dir():
		return []
	removed = []
	for p in sorted(root.iterdir()):
		if not p.is_file() or p.is_symlink():
			continue
		age = now - p.stat().st_mtime
		if age < 0:
			continue
		if p.name in keep:
			continue
		if (p.suffix == ".json" and age > max_age_s) or (p.suffix == ".tmp" and age > TEMP_TTL_S):
			p.unlink(missing_ok=True)
			removed.append(p.name)
	return removed


def cache_entry(probe: dict, *, now: float | None = None) -> dict:
	"""How the probe cache stores a result: the probe and when it was cached."""
	return {"cached_at": time.time() if now is None else now, "probe": probe}


def cached_probe(value) -> dict | None:
	"""The probe in a cache entry; None for an entry without a time (it is pruned, not used)."""
	if isinstance(value, dict) and isinstance(value.get("probe"), dict) and "cached_at" in value:
		return value["probe"]
	return None


def prune_cache(entries: Iterable[tuple[str, object]], delete, *, max_age_s: float = TTL_S,
	now: float | None = None) -> list[str]:  # fmt: skip
	"""Delete cache entries cached more than max_age_s ago, or with no time; returns their keys.
	entries: (key, value) pairs (a Modal Dict's items()); delete: removes one key."""
	now = time.time() if now is None else now
	removed = []
	for key, value in entries:
		at = value.get("cached_at") if isinstance(value, dict) else None
		if isinstance(at, int | float) and not isinstance(at, bool) and 0 <= now - at <= max_age_s:
			continue
		if isinstance(at, int | float) and not isinstance(at, bool) and at > now:
			continue  # dated in the future: kept
		delete(key)
		removed.append(key)
	return removed
