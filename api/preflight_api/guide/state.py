"""The guide conversation, carried by the browser but written only by the server.

The server returns the conversation as an opaque blob signed with HMAC-SHA256 over the
visitor's session id and the record's key, and accepts it back only if the signature holds.
So the browser can't edit what the model said or what a tool returned, can't replay another
visitor's conversation, and can't move one conversation to another record. The API stays
stateless across containers.

Bounded: at most MAX_ITEMS items (oldest dropped after the first user message) and MAX_BYTES.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import zlib
from dataclasses import asdict, dataclass, field
from typing import Any

from .. import session

MAX_ITEMS = 80  # source: guess, about 12 rounds of a tour with its tool calls
MAX_BYTES = 120_000  # source: derived, MAX_ITEMS x about 1.5 KB per item
VERSION = 1


class BadState(ValueError):
	pass


@dataclass
class Pending:
	call_id: str
	name: str
	arguments: dict[str, Any]
	needs_approval: bool = False


@dataclass
class GuideState:
	items: list[dict[str, Any]] = field(default_factory=list)
	pending: list[Pending] = field(default_factory=list)
	#: knob the guide moved and the visitor hasn't decided on yet (one move per round)
	move: str | None = None
	#: index into `items` where the current visitor turn began (for the number check)
	turn_start: int = 0
	rounds: int = 0
	#: numbers the server injected this turn (decisions, server tool results), for the number check
	evidence: list[float] = field(default_factory=list)

	def add(self, *items: dict[str, Any]) -> None:
		self.items.extend(items)
		if len(self.items) > MAX_ITEMS:
			first_user = next((i for i, it in enumerate(self.items) if it.get("role") == "user"), None)
			keep_head = self.items[: first_user + 1] if first_user is not None else []
			tail = self.items[-(MAX_ITEMS - len(keep_head)) :]
			dropped = len(self.items) - len(keep_head) - len(tail)
			self.items = keep_head + tail
			self.turn_start = max(len(keep_head), self.turn_start - dropped)


def _mac(key: bytes, blob: bytes, sid: str, record_key: str) -> bytes:
	msg = b"guide-state." + sid.encode() + b"." + record_key.encode() + b"." + blob
	return hmac.new(key, msg, hashlib.sha256).digest()


def seal(state: GuideState, *, sid: str, record_key: str) -> str:
	raw = json.dumps({"v": VERSION, **asdict(state)}, separators=(",", ":")).encode()
	blob = zlib.compress(raw, 6)
	if len(blob) > MAX_BYTES:
		raise BadState("conversation too long")
	mac = _mac(session._keys()[0], blob, sid, record_key)
	return base64.urlsafe_b64encode(mac + blob).decode()


def open_state(token: str | None, *, sid: str, record_key: str) -> GuideState:
	if not token:
		return GuideState()
	try:
		data = base64.urlsafe_b64decode(token.encode())
	except (ValueError, TypeError) as e:
		raise BadState("unreadable state") from e
	mac, blob = data[:32], data[32:]
	if len(mac) != 32 or not any(
		hmac.compare_digest(mac, _mac(k, blob, sid, record_key)) for k in session._keys()
	):
		raise BadState("state signature doesn't match this visitor and record")
	try:
		inflater = zlib.decompressobj()
		plain = inflater.decompress(blob, MAX_BYTES * 8)  # bounded: no decompression bombs
		if inflater.unconsumed_tail:
			raise BadState("state too large")
		raw = json.loads(plain)
	except (ValueError, zlib.error) as e:
		raise BadState("unreadable state") from e
	if raw.get("v") != VERSION:
		raise BadState("old state version")
	return GuideState(
		items=list(raw.get("items", [])),
		pending=[Pending(**p) for p in raw.get("pending", [])],
		move=raw.get("move"),
		turn_start=int(raw.get("turn_start", 0)),
		rounds=int(raw.get("rounds", 0)),
		evidence=[float(x) for x in raw.get("evidence", [])],
	)
