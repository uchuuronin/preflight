"""Record storage behind an interface, so a host swap is one file (gap 1).

Public reads go through a random share token, never the record's ULID. Production is
Cloudflare R2 (G1); LocalRecordStore serves a directory for development and tests.
"""

from __future__ import annotations

import json
import re
import secrets
from pathlib import Path
from typing import Protocol

_TOKEN = re.compile(r"^[A-Za-z0-9_-]{16,64}$")


def new_share_token() -> str:
	return secrets.token_urlsafe(18)


def valid_token(token: str) -> bool:
	return bool(_TOKEN.match(token))


class RecordStore(Protocol):
	def by_share_token(self, token: str) -> dict | None: ...
	def healthy(self) -> bool: ...


class LocalRecordStore:
	"""<root>/<share_token>.json. Development and tests only."""

	def __init__(self, root: Path) -> None:
		self.root = root

	def by_share_token(self, token: str) -> dict | None:
		if not valid_token(token):
			return None
		path = self.root / f"{token}.json"
		if not path.is_file():
			return None
		return json.loads(path.read_text())

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
