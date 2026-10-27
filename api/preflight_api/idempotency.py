"""Idempotency-Key for every write that spends money.

A write repeated with the same key returns the first response instead of running again,
so a retry never buys a second probe. The same key with a different body is a conflict.
The store sits behind an interface; production uses Modal Dict (one file to swap).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Protocol

from fastapi import Request

from .errors import ApiError

TTL_S = 24 * 3600


@dataclass
class Stored:
	body_hash: str
	status: int
	response: dict
	at: float


class IdempotencyStore(Protocol):
	def get(self, key: str) -> Stored | None: ...
	def put(self, key: str, value: Stored) -> None: ...


class MemoryIdempotencyStore:
	def __init__(self) -> None:
		self._items: dict[str, Stored] = {}

	def get(self, key: str) -> Stored | None:
		item = self._items.get(key)
		if item and time.time() - item.at > TTL_S:
			del self._items[key]
			return None
		return item

	def put(self, key: str, value: Stored) -> None:
		self._items[key] = value


def body_hash(body: object) -> str:
	return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()


def require_key(request: Request) -> str:
	key = request.headers.get("Idempotency-Key", "").strip()
	if not key or len(key) > 128:
		raise ApiError(
			400,
			"idempotency_key_required",
			"This request spends money, so it needs an Idempotency-Key header of at most 128 characters.",
		)
	return key


def replay_or_none(store: IdempotencyStore, scope: str, key: str, body: object) -> Stored | None:
	"""The stored response for a repeated request, or None for a new one."""
	stored = store.get(f"{scope}:{key}")
	if stored is None:
		return None
	if stored.body_hash != body_hash(body):
		raise ApiError(
			409, "idempotency_conflict", "This Idempotency-Key was already used with a different request."
		)
	return stored


def remember(
	store: IdempotencyStore, scope: str, key: str, body: object, status: int, response: dict
) -> None:
	store.put(f"{scope}:{key}", Stored(body_hash(body), status, response, time.time()))
