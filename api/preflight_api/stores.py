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


class StoreBusy(RuntimeError):
	"""The record store couldn't take the read in time (a volume reload is still running)."""


class LocalRecordStore:
	"""<root>/<share_token>.json: a directory in development and tests, A's "preflight-records"
	Modal volume (mounted read-only) in production.

	On Modal, `reload` is the volume's reload(): without it this container never sees what A's
	workers committed after it started. A container sees the volume empty while it reloads, and a
	reload fails while a file on it is open, so one lock covers every reload and every read here
	(reads are a single small file each). Reloads are at most one per RELOAD_MIN_S, except that a
	miss reloads once more if no reload started after the request arrived and the last is at
	least MISS_RELOAD_MIN_S old: a record committed a moment ago is found, and a stream of misses
	can't reload back to back. A failed reload serves the view the container already has; a reload that holds the lock past LOCK_WAIT_S turns other reads into StoreBusy,
	which the API answers with a retryable 503 instead of piling up threads.
	"""

	# Source: guess, a record changes at most once per probe (tens of seconds apart), so a
	# second-old view of a record that exists is current enough; replaced by the reload time
	# measured in the end-to-end run if reloads turn out slow.
	RELOAD_MIN_S = 1.0
	# Source: guess, boarding asks for a record seconds after its job's done event, so a miss only
	# needs a reload when the last one is a quarter second old; the floor stops a stream of misses
	# (expired links, scanners) from reloading back to back and starving real reads.
	MISS_RELOAD_MIN_S = 0.25
	# Source: guess, a reload of a small volume is well under a second; five seconds of waiting
	# means one is stuck; the cockpit retries a 503 once after Retry-After (record/load.ts), which is
	# cheaper than a held thread.
	LOCK_WAIT_S = 5.0
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
		self._last_reload: float | None = None  # when the latest reload started

	def _reload_now(self) -> None:
		self._last_reload = self._clock()
		try:
			self._reload()  # type: ignore[misc]
		except Exception:
			log.warning("record volume reload failed; serving the current view", exc_info=True)

	def _read(self, path: Path) -> bytes | None:
		try:
			if path.stat().st_size > self.MAX_READ_BYTES:
				return None
			return path.read_bytes()
		except OSError:
			return None

	def by_share_token(self, token: str) -> dict | None:
		if not valid_token(token):
			return None
		path = self.root / f"{token}.json"
		arrived = self._clock()
		if not self._lock.acquire(timeout=self.LOCK_WAIT_S):
			raise StoreBusy("the record store is reloading")
		try:
			if self._reload is not None and (
				self._last_reload is None or arrived - self._last_reload >= self.RELOAD_MIN_S
			):
				self._reload_now()
			data = self._read(path)
			if (
				data is None
				and self._reload is not None
				and (
					self._last_reload is None
					or (
						self._last_reload < arrived
						and self._clock() - self._last_reload >= self.MISS_RELOAD_MIN_S
					)
				)
			):
				# A miss: look again, unless a reload already started after we arrived or the last one
				# is under MISS_RELOAD_MIN_S old.
				self._reload_now()
				data = self._read(path)
		finally:
			self._lock.release()
		if data is None:
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
	# Every sample the web lists is a measured run (PM, Nov 3). The hand-made LoRA 7B and agent records
	# left the web's list but stay here, so old links, logbook entries and the fix loop's tests still open.
	"lora-7b": "web/src/workspace/fixtures/dev-finetune.record.json",
	"agent": "schema/examples/agent.json",
	# The live record of zoo/finetune_lora_hf on an L4, its share token removed (PF102, Nov 1).
	"lora-qwen": "web/src/workspace/fixtures/lora-live-nov1.record.json",
	"pretrain": "web/src/workspace/fixtures/pretrain_nanogpt-a10-nov2.record.json",
	"agent-eval": "web/src/workspace/fixtures/agent-eval-oct31.record.json",
	"campaign-grid": "web/src/workspace/fixtures/campaign-grid.record.json",
	"grpo": "web/src/workspace/fixtures/grpo-l4.record.json",
	"retrieval": "web/src/workspace/fixtures/retrieval_st-a10-nov2.record.json",
	"gnn": "web/src/workspace/fixtures/gnn_pyg-a10-nov2.record.json",
	"vision": "web/src/workspace/fixtures/vision_timm-a10-nov2.record.json",
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
