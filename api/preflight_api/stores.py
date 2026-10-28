"""Record storage behind an interface, so a host swap is one file (gap 1).

Public reads go through a random share token, never the record's ULID. Records live on A's
"preflight-records" Modal volume (PM decision, Oct 28; R2 dropped), which LocalRecordStore reads.
"""

from __future__ import annotations

import json
import logging
import re
import secrets
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

log = logging.getLogger("preflight.records")

_TOKEN = re.compile(r"^[A-Za-z0-9_-]{16,64}$")


def new_share_token() -> str:
	return secrets.token_urlsafe(18)


def valid_token(token: str) -> bool:
	return bool(_TOKEN.match(token))


class RecordStore(Protocol):
	def by_share_token(self, token: str) -> dict | None: ...
	def healthy(self) -> bool: ...


class LocalRecordStore:
	"""<root>/<share_token>.json: a directory in development and tests, A's "preflight-records"
	Modal volume (mounted read-only) in production.

	On Modal, `reload` is the volume's reload(): without it this container never sees what A's
	workers committed after it started. A container sees the volume empty while it reloads, and a
	reload fails while a file on it is open, so one lock covers every reload and every read here
	(reads are a single small file each). Reloads are at most one per RELOAD_MIN_S; a failed one
	serves the view the container already has.
	"""

	# Source: guess, a record changes at most once per probe (tens of seconds apart) and boarding
	# reads it after the job's done event, so a second-old view is current enough; replaced by the
	# reload time measured in the end-to-end run if reloads turn out slow.
	RELOAD_MIN_S = 1.0
	# Source: derived, A's reader cap (runner/records.py MAX_READ_BYTES); a stored record is under
	# 1 MB gzipped (schema MAX_RECORD_GZIP_BYTES).
	MAX_READ_BYTES = 20 * 2**20

	def __init__(
		self,
		root: Path,
		*,
		reload: Callable[[], None] | None = None,
		clock: Callable[[], float] = time.monotonic,
	) -> None:
		self.root = root
		self._reload = reload
		self._clock = clock
		self._lock = threading.Lock()
		self._last_reload: float | None = None

	def _maybe_reload(self) -> None:
		if self._reload is None:
			return
		now = self._clock()
		if self._last_reload is not None and now - self._last_reload < self.RELOAD_MIN_S:
			return
		self._last_reload = now
		try:
			self._reload()
		except Exception:
			log.warning("record volume reload failed; serving the current view", exc_info=True)

	def by_share_token(self, token: str) -> dict | None:
		if not valid_token(token):
			return None
		path = self.root / f"{token}.json"
		with self._lock:
			self._maybe_reload()
			try:
				if path.stat().st_size > self.MAX_READ_BYTES:
					return None
				data = path.read_bytes()
			except OSError:
				return None
		try:
			rec = json.loads(data)
		except ValueError:
			log.warning("record %s… doesn't parse", token[:6])
			return None
		return rec if isinstance(rec, dict) else None

	def healthy(self) -> bool:
		return self.root.is_dir()


#: Sample flights served from the repo, so every visitor can try the cockpit and the guide
#: without running anything. Ids match web/src/workspace/record/load.ts SAMPLES.
SAMPLES: dict[str, str] = {
	"lora-7b": "web/src/workspace/fixtures/dev-finetune.record.json",
	"lora-qwen": "schema/examples/finetune.json",
	"pretrain": "schema/examples/pretrain.json",
	"agent": "schema/examples/agent.json",
}


class SampleStore:
	"""Bundled sample records by id. Read once and kept: they never change while running."""

	def __init__(self, root: Path) -> None:
		self.root = root
		self._cache: dict[str, dict] = {}

	def get(self, sample_id: str) -> dict | None:
		if sample_id not in SAMPLES:
			return None
		if sample_id not in self._cache:
			path = self.root / SAMPLES[sample_id]
			if not path.is_file():
				return None
			rec = json.loads(path.read_text())
			# A sample's key is its id, so each visitor's session on it is one run for the caps.
			rec["record_id"] = f"sample:{sample_id}"
			self._cache[sample_id] = rec
		return self._cache[sample_id]


def resolve_record(ref: str, samples: SampleStore, records: RecordStore) -> dict | None:
	"""A record by sample id or share token. The run id comes from the record, never the request."""
	return samples.get(ref) or records.by_share_token(ref)
