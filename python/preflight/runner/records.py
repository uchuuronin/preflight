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
