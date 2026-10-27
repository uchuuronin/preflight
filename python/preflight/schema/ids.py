"""Identifiers: ULIDs for internal ids, random tokens for anything shared."""

import os
import secrets
import time

_CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def new_ulid(ms: int | None = None) -> str:
	"""A 26-character ULID: 48 bits of milliseconds, then 80 random bits.

	ULIDs sort by creation time, which is why they stay internal: a link built
	from one would tell a stranger when the record was made.
	"""
	t = int(time.time() * 1000) if ms is None else ms
	value = (t << 80) | int.from_bytes(os.urandom(10), "big")
	out = []
	for _ in range(26):
		out.append(_CROCKFORD[value & 31])
		value >>= 5
	return "".join(reversed(out))


def new_share_token() -> str:
	"""Read-only share token for links. 128 random bits, URL-safe."""
	return secrets.token_urlsafe(16)


def new_write_token() -> str:
	"""Per-record write token for CLI and in-flight uploads. Never stored in the record."""
	return secrets.token_urlsafe(32)
