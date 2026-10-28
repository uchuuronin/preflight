"""Anonymous visitor sessions in a signed cookie.

The cookie is `__Host-pf_session` = `v1.<sid>.<mac>`: a random session id and an HMAC-SHA256
over it. The `__Host-` prefix makes the browser require Secure, Path=/ and no Domain, so the
cookie is first-party to the site that serves the app (the API sits behind the same origin,
see edge.py). A cookie whose MAC doesn't verify is ignored and a fresh session is issued, so
nobody can pick another visitor's id or mint ids to dodge per-visitor caps without also
paying a round trip per id (and the IP bucket in the ledger bounds that).

Keys: PREFLIGHT_SESSION_KEYS, comma-separated; the first signs, all verify (rotation without
logging everyone out). Required in production; development gets a per-process key.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import logging
import os
import re
import secrets
from functools import lru_cache

log = logging.getLogger("preflight.session")

COOKIE = "__Host-pf_session"
LEGACY_COOKIE = "pf_session"  # unsigned cookie from before Oct 27; ignored, then replaced
VERSION = "v1"
MAX_AGE_S = 30 * 86_400  # source: guess, one month of returning visits; revisit with analytics
_SID = re.compile(r"^[A-Za-z0-9_-]{22}$")


class MissingSessionKey(RuntimeError):
	pass


@lru_cache(maxsize=1)
def _keys() -> tuple[bytes, ...]:
	raw = [k.strip() for k in os.environ.get("PREFLIGHT_SESSION_KEYS", "").split(",") if k.strip()]
	if raw:
		short = [k for k in raw if len(k) < 32]
		if short:
			raise MissingSessionKey("PREFLIGHT_SESSION_KEYS entries must be at least 32 characters.")
		return tuple(k.encode() for k in raw)
	if os.environ.get("PREFLIGHT_ENV") == "production":
		raise MissingSessionKey("PREFLIGHT_SESSION_KEYS is required in production.")
	log.warning("PREFLIGHT_SESSION_KEYS unset: sessions are signed with a per-process key (dev only).")
	return (secrets.token_bytes(32),)


def reset_keys() -> None:
	"""For tests that change the environment."""
	_keys.cache_clear()


def _mac(key: bytes, sid: str) -> str:
	digest = hmac.new(key, f"{VERSION}.{sid}".encode(), hashlib.sha256).digest()
	return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def new_session_id() -> str:
	return secrets.token_urlsafe(16)  # 22 characters, 128 bits


def sign(sid: str) -> str:
	return f"{VERSION}.{sid}.{_mac(_keys()[0], sid)}"


def verify(value: str | None) -> str | None:
	"""The session id inside a cookie value, or None if it is missing, malformed or forged."""
	if not value or len(value) > 128:
		return None
	parts = value.split(".")
	if len(parts) != 3 or parts[0] != VERSION or not _SID.match(parts[1]):
		return None
	sid, mac = parts[1], parts[2]
	return sid if any(hmac.compare_digest(mac, _mac(k, sid)) for k in _keys()) else None


def needs_resign(value: str | None) -> bool:
	"""True when the cookie verifies only under an older key, so it should be reissued."""
	if not value:
		return False
	parts = value.split(".")
	return len(parts) == 3 and not hmac.compare_digest(parts[2], _mac(_keys()[0], parts[1]))
