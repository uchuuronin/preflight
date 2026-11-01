"""How the API starts and follows a fix race: a Modal function in the `preflight-fix` app
(api/fix_app.py `race_fix`), spawned by name, followed by its call id.

The API never handles the visitor's code: the function reads the record and fetches its source
the way A's runner does (python/preflight/runner/record_jobs.py `refetch`). The API holds only
the call id, the visitor and the spend reservation, in a small store keyed by the patch id.
"""

from __future__ import annotations

import logging
import os
import secrets
import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol

APP = os.environ.get("PREFLIGHT_FIX_APP", "preflight-fix")
FUNCTION = "race_fix"
log = logging.getLogger("preflight.fix")
ENV = os.environ.get("PREFLIGHT_MODAL_ENV", "main")


@dataclass
class Race:
	patch_id: str
	visitor: str
	record_ref: str
	call_id: str
	reservation: str | None
	worst_usd: float
	started: float
	settled: bool = False
	result: dict | None = None


class Fixer(Protocol):
	async def start(
		self, *, record_ref: str, candidates: list[dict], measured: str | None, stop_usd: float
	) -> str:
		"""Returns the call id."""
		...

	async def poll(self, call_id: str) -> dict | None:
		"""The race's result once it has one, else None."""
		...

	async def cancel(self, call_id: str) -> None: ...


class RaceStore(Protocol):
	async def put(self, race: Race) -> None: ...

	async def get(self, patch_id: str) -> Race | None: ...


class MemoryRaces:
	def __init__(self) -> None:
		self._d: dict[str, dict] = {}

	async def put(self, race: Race) -> None:
		self._d[race.patch_id] = asdict(race)

	async def get(self, patch_id: str) -> Race | None:
		raw = self._d.get(patch_id)
		return Race(**raw) if raw else None


class ModalRaces:
	"""A Modal Dict, so every API container sees the same races."""

	def __init__(self, name: str = "preflight-fixes") -> None:
		import modal

		self._d = modal.Dict.from_name(name, create_if_missing=True, environment_name=ENV)

	async def put(self, race: Race) -> None:
		await self._d.put.aio(race.patch_id, asdict(race))

	async def get(self, patch_id: str) -> Race | None:
		raw = await self._d.get.aio(patch_id)
		return Race(**raw) if raw else None


class ModalFixer:
	async def start(
		self, *, record_ref: str, candidates: list[dict], measured: str | None, stop_usd: float
	) -> str:
		import modal

		fn = modal.Function.from_name(APP, FUNCTION, environment_name=ENV)
		call = await fn.spawn.aio(
			record_ref=record_ref, candidates=candidates, measured=measured, stop_usd=stop_usd
		)
		return call.object_id

	async def poll(self, call_id: str) -> dict | None:
		import modal

		call = modal.FunctionCall.from_id(call_id)
		try:
			return await call.get.aio(timeout=0)
		except TimeoutError:  # the built-in one: not finished yet
			return None
		except modal.exception.OutputExpiredError:
			return {"error": "expired", "candidates": []}
		except (modal.exception.ConnectionError, modal.exception.ServiceError, modal.exception.InternalError):
			return None  # Modal itself didn't answer: ask again on the next poll
		except Exception:  # the Function raised (its own timeout, a sandbox that wouldn't start)
			log.exception("fix race %s failed", call_id)
			return {"error": "race_failed", "candidates": []}

	async def cancel(self, call_id: str) -> None:
		import modal

		await modal.FunctionCall.from_id(call_id).cancel.aio()


def new_patch_id() -> str:
	return "fx_" + secrets.token_urlsafe(18)


def now() -> float:
	return time.time()


def public(race: Race) -> dict[str, Any]:
	"""What the visitor sees: never the call id or the reservation."""
	out: dict[str, Any] = {"patch_id": race.patch_id, "status": "running", "candidates": []}
	if race.result is not None:
		res = race.result
		out["status"] = "failed" if res.get("error") else "done"
		out["candidates"] = res.get("candidates", [])
		out["spent_usd"] = res.get("usd_from_reported_tokens")
		if res.get("error"):
			out["error"] = res["error"]
	return out
