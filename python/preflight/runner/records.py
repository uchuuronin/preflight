"""Where finished and in-progress flight records go.

C's API reads records by share token (`GET /v1/records/{share_token}`) from a
RecordStore; in development that is a directory of `<share_token>.json`. The
writer here produces exactly that layout, so C's LocalRecordStore reads what
A writes, and on Modal both sides mount one Volume (A read-write, C
read-only) until the R2 store lands. Writes are atomic (temp file, then
rename): a reader never sees half a record.
"""

import contextlib
import os
import re
import tempfile
from pathlib import Path
from typing import Protocol

from preflight.schema import FlightRecord, dump_record, load_record

_TOKEN = re.compile(r"[A-Za-z0-9_-]{16,64}")
MAX_READ_BYTES = 20 * 2**20  # load_record's own raw cap; a stored record is under 1 MB gzipped


class RecordWriter(Protocol):
	def put(self, rec: FlightRecord) -> int:
		"""Store the record under its share token; returns the new revision (1, 2, ...)."""
		...


class RecordReader(Protocol):
	def get(self, share_token: str) -> FlightRecord | None:
		"""The stored record for a share token, or None if there is none (or it doesn't read)."""
		...


class LocalRecordWriter:
	def __init__(self, root: Path | str, *, on_write=None, on_read=None):
		self.root = Path(root)
		self.root.mkdir(parents=True, exist_ok=True)
		self._revs: dict[str, int] = {}
		self._on_write = on_write  # e.g. a Modal Volume's commit()
		self._on_read = on_read  # e.g. its reload(), to see other containers' commits

	def get(self, share_token: str) -> FlightRecord | None:
		if not isinstance(share_token, str) or not _TOKEN.fullmatch(share_token):
			return None
		p = self.root / f"{share_token}.json"
		if self._on_read is not None:
			# always: this container's view may hold an older revision of a record another wrote
			with contextlib.suppress(Exception):
				self._on_read()
		try:
			if p.stat().st_size > MAX_READ_BYTES:
				return None
			return load_record(p.read_bytes())
		except (OSError, ValueError):
			return None

	def put(self, rec: FlightRecord) -> int:
		data = dump_record(rec)  # raises if the record breaks the 1 MB gzipped budget
		fd, tmp = tempfile.mkstemp(dir=self.root, prefix=".tmp-", suffix=".json")
		try:
			with os.fdopen(fd, "wb") as f:
				f.write(data)
			os.chmod(tmp, 0o644)  # mkstemp makes it 0600; the API reads it as another process
			os.replace(tmp, self.root / f"{rec.share_token}.json")
		except BaseException:
			Path(tmp).unlink(missing_ok=True)
			raise
		if self._on_write is not None:
			self._on_write()
		rev = self._revs.get(rec.record_id, 0) + 1
		self._revs[rec.record_id] = rev
		return rev


RECORD_TTL_S = 30 * 86400  # PM decision (Oct 28): records live 30 days on the preflight-records volume


PRUNE_MAX_SHARE = 0.25  # guess: a daily pass should remove one day's records; more means the dates are wrong
PRUNE_MIN_GUARD = 20  # below this many records the share check is too noisy to trust either way


_CREATED = re.compile(r'"created_at"\s*:\s*"([^"]{10,40})"')


def _created_at(p: Path) -> float | None:
	"""The record's own creation time (its `created_at`), not the file's: volume file times
	aren't guaranteed to survive a reload, and a wrong one must never delete a record."""
	import json
	from datetime import datetime

	try:
		with p.open("rb") as f:
			head = f.read(8192).decode("utf-8", "replace")
		m = _CREATED.search(head)  # dumps put it near the top
		at = m.group(1) if m else json.loads(p.read_text(encoding="utf-8")).get("created_at")
		return datetime.fromisoformat(str(at).replace("Z", "+00:00")).timestamp() if at else None
	except (OSError, ValueError, AttributeError, TypeError):
		return None


def prune(root: Path | str, *, max_age_s: float = RECORD_TTL_S, now: float | None = None) -> list[str]:
	"""Delete stored records older than max_age_s by their own `created_at`, and temp files left
	over from interrupted writes (by file time, a day old); returns the names removed.

	Only `<share_token>.json` and `.tmp-*.json` directly in root are considered. A record whose date
	can't be read, or is in the future, is kept. One pass removes at most PRUNE_MAX_SHARE of the
	records (at least PRUNE_MIN_GUARD), oldest first: a backlog drains over a few days, and dates
	gone wrong can't empty the volume in one go."""
	import time

	root = Path(root)
	now = time.time() if now is None else now
	records, expired, temps = 0, [], []
	for p in sorted(root.iterdir()) if root.is_dir() else []:
		name = p.name
		if p.is_symlink() or not p.is_file() or not name.endswith(".json"):
			continue
		if name.startswith(".tmp-"):
			with contextlib.suppress(FileNotFoundError):
				if now - p.stat().st_mtime > 86_400:
					temps.append(p)
			continue
		if not _TOKEN.fullmatch(name[:-5]):
			continue
		records += 1
		at = _created_at(p)
		if at is not None and max_age_s < now - at and at <= now + 86_400:
			expired.append((at, p))
	cap = max(PRUNE_MIN_GUARD, int(PRUNE_MAX_SHARE * records))
	doomed = [p for _, p in sorted(expired)[:cap]] + temps
	removed = []
	for p in doomed:
		try:
			p.unlink()
			removed.append(p.name)
		except FileNotFoundError:
			continue  # another container removed it first
	return sorted(removed)
