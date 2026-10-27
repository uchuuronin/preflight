"""Muse Spark client for the guide. The only file that knows the wire format.

The Meta Model API is OpenAI-compatible; this sends an OpenAI-style Responses request.
NOT YET VERIFIED against the live API (its docs weren't reachable from the build
sandbox), which is why c.guide.live is off. If the shape differs, fix it here only.

Config (Modal secret "preflight-meta"): META_API_KEY, META_API_BASE.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import httpx

MODEL = "muse-spark-1.3"
TIMEOUT_S = 30.0  # contract: 30 s per guide turn, then the rule-based tour

# Per million tokens, from a third-party listing; confirm on the Meta Usage page.
PRICE_IN = 1.25
PRICE_OUT = 4.25
PRICE_CACHED = 0.15


@dataclass
class FunctionCall:
	call_id: str
	name: str
	arguments: dict


@dataclass
class TurnResult:
	text: str = ""
	calls: list[FunctionCall] = field(default_factory=list)
	input_tokens: int = 0
	output_tokens: int = 0
	cached_tokens: int = 0
	request_id: str | None = None
	response_id: str | None = None

	@property
	def cost_usd(self) -> float:
		fresh = max(0, self.input_tokens - self.cached_tokens)
		return (fresh * PRICE_IN + self.cached_tokens * PRICE_CACHED + self.output_tokens * PRICE_OUT) / 1e6


class GuideModelError(Exception):
	def __init__(self, message: str, retryable: bool) -> None:
		super().__init__(message)
		self.retryable = retryable


def configured() -> bool:
	return bool(os.environ.get("META_API_KEY") and os.environ.get("META_API_BASE"))


def _parse(body: dict, request_id: str | None) -> TurnResult:
	out = TurnResult(request_id=request_id, response_id=body.get("id"))
	for item in body.get("output", []):
		kind = item.get("type")
		if kind == "message":
			for part in item.get("content", []):
				if part.get("type") in ("output_text", "text"):
					out.text += part.get("text", "")
		elif kind == "function_call":
			try:
				args = json.loads(item.get("arguments") or "{}")
			except ValueError:
				args = {"_malformed": item.get("arguments")}
			out.calls.append(
				FunctionCall(item.get("call_id") or item.get("id", ""), item.get("name", ""), args)
			)
	usage = body.get("usage") or {}
	out.input_tokens = int(usage.get("input_tokens", 0))
	out.output_tokens = int(usage.get("output_tokens", 0))
	out.cached_tokens = int((usage.get("input_tokens_details") or {}).get("cached_tokens", 0))
	return out


async def create_turn(
	*,
	instructions: str,
	input_items: list[dict],
	tools: list[dict],
	max_output_tokens: int,
	reasoning_effort: str = "low",
	client: httpx.AsyncClient | None = None,
) -> TurnResult:
	base = os.environ["META_API_BASE"].rstrip("/")
	payload = {
		"model": MODEL,
		"instructions": instructions,
		"input": input_items,
		"tools": tools,
		"tool_choice": "auto",
		"parallel_tool_calls": True,
		"max_output_tokens": max_output_tokens,
		"reasoning": {"effort": reasoning_effort},
		"store": False,
	}
	headers = {"Authorization": f"Bearer {os.environ['META_API_KEY']}"}
	own = client is None
	c = client or httpx.AsyncClient(timeout=TIMEOUT_S)
	try:
		res = await c.post(f"{base}/responses", json=payload, headers=headers)
	except httpx.TimeoutException as e:
		raise GuideModelError("The guide model took too long to answer.", retryable=True) from e
	except httpx.HTTPError as e:
		raise GuideModelError("The guide model couldn't be reached.", retryable=True) from e
	finally:
		if own:
			await c.aclose()
	if res.status_code == 429 or res.status_code >= 500:
		raise GuideModelError(f"The guide model is unavailable ({res.status_code}).", retryable=True)
	if res.status_code >= 400:
		raise GuideModelError(f"The guide model rejected the request ({res.status_code}).", retryable=False)
	return _parse(res.json(), res.headers.get("x-request-id"))
