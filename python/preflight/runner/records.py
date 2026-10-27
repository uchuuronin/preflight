"""Where finished and in-progress flight records go.

C's API reads records by share token (`GET /v1/records/{share_token}`) from a
RecordStore; in development that is a directory of `<share_token>.json`. The
writer here produces exactly that layout, so C's LocalRecordStore reads what
A writes, and on Modal both sides mount one Volume (A read-write, C
read-only) until the R2 store lands. Writes are atomic (temp file, then
rename): a reader never sees half a record.
"""

import os
import tempfile
from pathlib import Path
from typing import Protocol

from preflight.schema import FlightRecord, dump_record


class RecordWriter(Protocol):
	def put(self, rec: FlightRecord) -> int:
		"""Store the record under its share token; returns the new revision (1, 2, ...)."""
		...


class LocalRecordWriter:
	def __init__(self, root: Path | str, *, on_write=None):
		self.root = Path(root)
		self.root.mkdir(parents=True, exist_ok=True)
		self._revs: dict[str, int] = {}
		self._on_write = on_write  # e.g. a Modal Volume's commit()

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
