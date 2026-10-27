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
