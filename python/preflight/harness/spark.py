"""A small client for OpenAI-compatible chat endpoints: Meta Model API (Muse Spark),
Muse Spark Open on Baseten, Fireworks.

Contract rules it enforces (Failure and degradation table, gap 11):
  - every call has a timeout (60 s by default for script maps and diagnosis);
  - transient failures (429, 5xx, timeouts) retry twice with backoff and jitter,
    honouring Retry-After; other errors fail at once;
  - malformed structured output gets exactly one re-ask carrying the validation error;
  - a per-call input-token cap refuses oversized prompts before any money is spent;
  - every call, failed or not, is recorded as a ModelCall for the record's provenance;
  - visitor code never goes to a "-contributor" model, which lets Meta train on it.
"""

import copy
import os
import random
import time
from datetime import UTC, datetime
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from preflight.schema import ModelCall

META_BASE_URL = "https://api.meta.ai/v1"
# USD per million tokens: (input, cached input, output). From public listings on Oct 27;
# confirm against the Meta Usage page (open question in the Contracts tab).
PRICES: dict[str, tuple[float, float, float]] = {
	"muse-spark-1.3": (1.25, 0.15, 4.25),
	"muse-spark-1.1": (1.25, 0.15, 4.25),
	"muse-glimmer-30b": (0.25, 0.03, 1.05),  # B's figures from Qubrid (Nov 1)
	# Spark Open on Baseten: the rate B confirmed on Baseten's own page (Nov 1), $0.15 cached
	"meta-models/muse-spark-open-1026": (1.25, 0.15, 4.25),
}
T = TypeVar("T", bound=BaseModel)
# Retry only where a repeat can't double-bill: rate limits and gateway errors before generation.
RETRY_STATUS = {429, 500, 502, 503, 504}
# Transport errors before the request reached the provider. Read and write timeouts are not here:
# the provider may have run (and billed) the call.
RETRY_TRANSPORT = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)


class TokenCapExceeded(ValueError):
	pass


class ModelRefused(ValueError):
	pass


def _strictify(node):
	if isinstance(node, dict):
		node.pop("default", None)
		node.pop("title", None)
		for k in (
			"pattern",
			"format",
			"minLength",
			"maxLength",
			"minItems",
			"maxItems",
			"minimum",
			"maximum",
			"exclusiveMinimum",
			"exclusiveMaximum",
		):
			node.pop(k, None)
		if node.get("type") == "object" and "properties" in node:
			node["additionalProperties"] = False
			node["required"] = list(node["properties"])
		for v in node.values():
			_strictify(v)
	elif isinstance(node, list):
		for v in node:
			_strictify(v)
	return node


def _json_or_none(resp: httpx.Response) -> dict | None:
	try:
		data = resp.json()
		return data if isinstance(data, dict) else None
	except ValueError:
		return None


def strict_schema(model: type[BaseModel]) -> dict:
	"""JSON Schema in the strict structured-output dialect: every property required
	(optional ones stay nullable), no extras, no defaults or numeric/string bounds."""
	return _strictify(copy.deepcopy(model.model_json_schema()))


def estimate_tokens(messages: list[dict]) -> int:
	return sum(len(str(m.get("content", ""))) for m in messages) // 4 + 8 * len(messages)


class ModelClient:
	def __init__(
		self,
		*,
		base_url: str = META_BASE_URL,
		api_key: str | None = None,
		model: str = "muse-spark-1.3",
		provider: str = "meta",
		timeout_s: float = 60.0,
		retries: int = 2,
		backoff_s: float = 1.0,
		deadline_s: float | None = None,
		max_input_tokens: int = 120_000,
		max_output_tokens: int = 8_192,
		prices: dict[str, tuple[float, float, float]] | None = None,
		visitor_data: bool = True,
		transport: httpx.BaseTransport | None = None,
	):
		if visitor_data and model.endswith("-contributor"):
			raise ModelRefused(f"{model} lets the provider train on prompts; visitor code never goes to it")
		self.model = model
		self.provider = provider
		self.retries = retries
		self.backoff_s = backoff_s
		self.timeout_s = timeout_s
		# one overall budget per call, re-ask included: the contract's timeout, plus room for one retry
		self.deadline_s = deadline_s if deadline_s is not None else 1.5 * timeout_s
		self.max_input_tokens = max_input_tokens
		self.max_output_tokens = max_output_tokens
		self.prices = PRICES if prices is None else prices
		self.calls: list[ModelCall] = []
		self._deadline: float | None = None  # set by structured() so the re-ask shares one budget
		key = api_key if api_key is not None else os.environ.get("MODEL_API_KEY", "")
		self._http = httpx.Client(
			base_url=base_url.rstrip("/"),
			timeout=httpx.Timeout(timeout_s, connect=10.0),
			headers={"authorization": f"Bearer {key}", "content-type": "application/json"},
			transport=transport,
		)

	@classmethod
	def meta(cls, **kw) -> "ModelClient":
		"""Muse Spark on the Meta Model API, configured from the `preflight-meta` secret.

		META_API_BASE and META_API_KEY are the canonical names (docs/setup/cloud.md);
		META_BASE_URL and MODEL_API_KEY are read as fallbacks for one release.
		"""
		env = os.environ
		kw.setdefault("base_url", env.get("META_API_BASE") or env.get("META_BASE_URL") or META_BASE_URL)
		kw.setdefault("api_key", env.get("META_API_KEY") or env.get("MODEL_API_KEY") or "")
		return cls(provider="meta", **kw)

	@staticmethod
	def meta_configured() -> bool:
		return bool(os.environ.get("META_API_KEY") or os.environ.get("MODEL_API_KEY"))

	@classmethod
	def baseten_spark_open(cls, **kw) -> "ModelClient":
		return cls(
			base_url=os.environ.get("BASETEN_BASE_URL", "https://inference.baseten.co/v1"),
			api_key=os.environ.get("BASETEN_API_KEY", ""),
			model=os.environ.get("SPARK_OPEN_MODEL", "meta-models/muse-spark-open-1026"),
			provider="baseten",
			timeout_s=20.0,
			**kw,
		)

	# ------------------------------------------------------------- transport

	def _post(self, body: dict, purpose: str, deadline: float) -> httpx.Response:
		"""POST with bounded retries inside an overall deadline. Every failed attempt goes to provenance."""
		attempt = 0
		while True:
			remaining = deadline - time.monotonic()
			if remaining <= 0:
				raise TimeoutError(f"{purpose}: no time left in the {self.deadline_s:.0f} s budget")
			started = time.monotonic()
			try:
				resp = self._http.post("/chat/completions", json=body, timeout=min(self.timeout_s, remaining))
			except RETRY_TRANSPORT:
				self._record(purpose, None, started, False, None)
				if attempt >= self.retries:
					raise
				wait = None
			except httpx.TransportError:
				self._record(purpose, None, started, False, None)
				raise
			else:
				if resp.status_code < 400:
					return resp
				self._record(purpose, _json_or_none(resp), started, False, resp.headers.get("x-request-id"))
				if resp.status_code not in RETRY_STATUS or attempt >= self.retries:
					resp.raise_for_status()
				wait = resp.headers.get("retry-after")
			attempt += 1
			delay = (
				float(wait)
				if wait and wait.replace(".", "", 1).isdigit()
				else self.backoff_s * 2 ** (attempt - 1)
			)
			delay = min(delay, 30.0) + random.uniform(0, self.backoff_s)
			if time.monotonic() + delay >= deadline:
				raise TimeoutError(f"{purpose}: retry would run past the {self.deadline_s:.0f} s budget")
			time.sleep(delay)

	def _record(
		self, purpose: str, data: dict | None, started: float, ok: bool, request_id: str | None
	) -> ModelCall:
		usage = (data or {}).get("usage") or {}
		inp = int(usage.get("prompt_tokens", 0))
		out = int(usage.get("completion_tokens", 0))
		cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0))
		p_in, p_cached, p_out = self.prices.get(self.model, (0.0, 0.0, 0.0))
		cost = ((inp - cached) * p_in + cached * p_cached + out * p_out) / 1e6
		call = ModelCall(
			provider=self.provider,
			model=self.model,
			purpose=purpose,
			input_tokens=inp,
			output_tokens=out,
			cached_input_tokens=cached,
			cost_usd=round(cost, 6),
			request_id=request_id,
			at=datetime.now(UTC).replace(microsecond=0),
			latency_s=round(time.monotonic() - started, 3),
			ok=ok,
		)
		self.calls.append(call)
		return call

	# ------------------------------------------------------------- API

	def chat(
		self, messages: list[dict], *, purpose: str, reasoning_effort: str | None = None, **extra
	) -> tuple[str, ModelCall]:
		if estimate_tokens(messages) > self.max_input_tokens:
			raise TokenCapExceeded(
				f"prompt is about {estimate_tokens(messages)} tokens; cap is {self.max_input_tokens}"
			)
		body = {
			"model": self.model,
			"messages": messages,
			"max_completion_tokens": self.max_output_tokens,
			**extra,
		}
		if reasoning_effort:
			body["reasoning_effort"] = reasoning_effort
		deadline = self._deadline if self._deadline is not None else time.monotonic() + self.deadline_s
		started = time.monotonic()
		resp = self._post(body, purpose, deadline)
		data = resp.json()
		content = data["choices"][0]["message"].get("content") or ""
		call = self._record(purpose, data, started, True, resp.headers.get("x-request-id") or data.get("id"))
		return content, call

	def structured(
		self, messages: list[dict], schema: type[T], *, purpose: str, reasoning_effort: str | None = None
	) -> tuple[T, ModelCall]:
		fmt = {
			"type": "json_schema",
			"json_schema": {"name": schema.__name__, "schema": strict_schema(schema), "strict": True},
		}
		msgs = list(messages)
		self._deadline = time.monotonic() + self.deadline_s
		try:
			return self._structured(msgs, schema, fmt, purpose, reasoning_effort)
		finally:
			self._deadline = None

	def _structured(self, msgs, schema, fmt, purpose, reasoning_effort):
		for attempt in range(2):
			content, call = self.chat(
				msgs, purpose=purpose, reasoning_effort=reasoning_effort, response_format=fmt
			)
			try:
				return schema.model_validate_json(content), call
			except (ValidationError, ValueError) as exc:
				err = str(exc)[:2000]
				self.calls[-1] = call.model_copy(update={"ok": False})
				if attempt == 1:
					raise ValueError(
						f"{schema.__name__} output did not validate after a re-ask: {err}"
					) from exc
				msgs = [
					*msgs,
					{"role": "assistant", "content": content[:4000]},
					{
						"role": "user",
						"content": f"Your output did not validate against the schema:\n{err}\nReturn corrected JSON only.",
					},
				]
		raise AssertionError("unreachable")
