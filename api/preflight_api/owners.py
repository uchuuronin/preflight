"""Who checked a record in, so a fix race on uploaded code runs only for them.

A record carries no owner (the schema has none), and a share link is meant to be passed on. A
fix race returns diffs of the record's code, so on a record made from an upload (code that isn't
public) only the visitor whose job made it may start one. The API learns that the first time a
share token reaches a visitor on their own job's status or events (job ids are 80 random bits,
known only to whoever checked the job in), and keeps it here: token -> visitor hash.

Repo and sample records need no owner: their code is public already.
"""

from __future__ import annotations

import os
from typing import Protocol

ENV = os.environ.get("PREFLIGHT_MODAL_ENV", "main")


class Owners(Protocol):
	async def claim(self, share_token: str, visitor: str) -> None:
		"""Record the owner if the token has none yet (the first claim stands)."""
		...

	async def owner(self, share_token: str) -> str | None: ...


class MemoryOwners:
	def __init__(self) -> None:
		self._d: dict[str, str] = {}

	async def claim(self, share_token: str, visitor: str) -> None:
		self._d.setdefault(share_token, visitor)

	async def owner(self, share_token: str) -> str | None:
		return self._d.get(share_token)


class ModalOwners:
	"""A Modal Dict, so every API container sees the same owners."""

	def __init__(self, name: str = "preflight-record-owners") -> None:
		import modal

		self._d = modal.Dict.from_name(name, create_if_missing=True, environment_name=ENV)

	async def claim(self, share_token: str, visitor: str) -> None:
		await self._d.put.aio(share_token, visitor, skip_if_exists=True)

	async def owner(self, share_token: str) -> str | None:
		return await self._d.get.aio(share_token)
