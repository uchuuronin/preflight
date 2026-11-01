"""Each session's confirmed limits, kept on the server (one copy per visitor and record).

The browser sends the limits the visitor confirmed whenever they change (PUT /v1/guide/limits/
state). The fix route reads them from here and never from its own request body, so a fix
request with the limits stripped out is still held to them; the guide's routes hold every move
to these and to whatever the browser sends, whichever is stricter. Kept for a day: limits are a
visit's, and nothing here outlives the session cookie by much.
"""

from __future__ import annotations

import os
import time
from typing import Protocol

from .guide.types import Limits

ENV = os.environ.get("PREFLIGHT_MODAL_ENV", "main")
# Source: guess, a visit; the session cookie lasts longer, but limits set yesterday are stale.
TTL_S = 24 * 3600


class SessionLimits(Protocol):
	async def put(self, visitor: str, record_key: str, limits: Limits) -> None: ...

	async def get(self, visitor: str, record_key: str) -> Limits | None: ...


def _key(visitor: str, record_key: str) -> str:
	return f"{visitor}:{record_key}"


class MemorySessionLimits:
	def __init__(self) -> None:
		self._d: dict[str, tuple[float, dict]] = {}

	async def put(self, visitor: str, record_key: str, limits: Limits) -> None:
		self._d[_key(visitor, record_key)] = (time.time(), limits.model_dump())

	async def get(self, visitor: str, record_key: str) -> Limits | None:
		hit = self._d.get(_key(visitor, record_key))
		if not hit or time.time() - hit[0] > TTL_S:
			return None
		return Limits.model_validate(hit[1])


class ModalSessionLimits:
	"""A Modal Dict, so every API container sees the same limits."""

	def __init__(self, name: str = "preflight-session-limits") -> None:
		import modal

		self._d = modal.Dict.from_name(name, create_if_missing=True, environment_name=ENV)

	async def put(self, visitor: str, record_key: str, limits: Limits) -> None:
		await self._d.put.aio(_key(visitor, record_key), {"at": time.time(), "limits": limits.model_dump()})

	async def get(self, visitor: str, record_key: str) -> Limits | None:
		hit = await self._d.get.aio(_key(visitor, record_key))
		if not hit or time.time() - float(hit.get("at", 0)) > TTL_S:
			return None
		return Limits.model_validate(hit["limits"])


def default_session_limits() -> SessionLimits:
	return ModalSessionLimits() if os.environ.get("PREFLIGHT_LEDGER") == "modal" else MemorySessionLimits()
