"""The API is served from the web app's own origin (same site), through a Cloudflare Pages
Function at /v1/* that forwards to Modal (web/functions/v1/[[path]].ts). The function adds:

- X-Preflight-Edge: a shared secret (PREFLIGHT_EDGE_SECRET on both sides), so the Modal URL
  can't be called directly to skip the edge;
- X-Preflight-Client-IP: the visitor's address from CF-Connecting-IP, for the IP bucket.

With PREFLIGHT_REQUIRE_EDGE=1 (production) a request without the secret is refused, except
health checks. Without it (development, tests) the socket's peer address is the client IP.
"""

from __future__ import annotations

import hmac
import ipaddress
import os

from fastapi import Request

EDGE_HEADER = "X-Preflight-Edge"
CLIENT_IP_HEADER = "X-Preflight-Client-IP"
OPEN_PATHS = ("/livez", "/readyz")


def edge_secret() -> str | None:
	value = os.environ.get("PREFLIGHT_EDGE_SECRET", "")
	return value or None


def require_edge() -> bool:
	return os.environ.get("PREFLIGHT_REQUIRE_EDGE") == "1"


def from_edge(request: Request) -> bool:
	secret = edge_secret()
	got = request.headers.get(EDGE_HEADER, "")
	return bool(secret and got and hmac.compare_digest(got, secret))


def allowed(request: Request) -> bool:
	if not require_edge() or request.url.path in OPEN_PATHS:
		return True
	return from_edge(request)


def _valid_ip(value: str) -> str | None:
	try:
		return str(ipaddress.ip_address(value.strip()))
	except ValueError:
		return None


def client_ip(request: Request) -> str:
	"""The visitor's address: the edge's word for it when the edge is verified, else the peer."""
	if from_edge(request):
		ip = _valid_ip(request.headers.get(CLIENT_IP_HEADER, ""))
		if ip:
			return ip
	return request.client.host if request.client else "unknown"


def ip_key(ip: str) -> str:
	"""IPv6 visitors get a /64 each (one household or phone), IPv4 a single address."""
	try:
		addr = ipaddress.ip_address(ip)
	except ValueError:
		return ip
	if addr.version == 6:
		return str(ipaddress.ip_network(f"{addr}/64", strict=False))
	return str(addr)
