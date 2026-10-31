"""The agent recorder: an OpenAI-compatible proxy that measures an agent's model calls, task by task.

An agent or API run is measured by running a sample of its tasks (the sampler, next) with the
script's client pointed at this proxy instead of the provider:

	proxy = AgentProxy(upstream="https://api.llama.com/compat/v1", api_key=..., rate=token_rate)
	host, port = proxy.start()
	# the task's process gets OPENAI_BASE_URL=http://127.0.0.1:<port>/t/<task_id>/v1
	...
	task = proxy.task("t017", success=True)   # an AgentTask for the record's agent_sample
	spans = proxy.spans("t017")               # OTel GenAI spans, counts only

What it keeps per call: the model, token counts as the provider reports them (input, cached
input, output, the reasoning part of the output), tool calls asked for, the finish reason, the
latency and the HTTP status. It never keeps prompts or completions: the spans follow the OTel
GenAI semantic conventions (`gen_ai.*`) without the content attributes.

What it guards: the visitor's script never holds the provider key (the proxy drops whatever
Authorization the script sends and sets its own), it binds to 127.0.0.1 only, it forwards only
the generation endpoints, and it stops a task that runs away (too many calls, too many dollars,
or the same request over and over) with a 429 the client sees as a rate limit, so a runaway in a
sample costs at most its cap. The caps are guesses until the agent validation run.
"""

import contextlib
import hashlib
import json
import re
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx

from preflight.schema import AgentTask

# guess: a coding-agent task in the golden record makes 6–12 calls; 200 is far past any honest one
MAX_CALLS_PER_TASK = 200
MAX_USD_PER_TASK = 2.0  # guess: 40x the golden record's median task ($0.046)
MAX_USD_TOTAL = 20.0  # guess: a 50-task sample at the per-task cap is $100; 20 is the sampler's purse
REPEATS_FOR_LOOP = 5  # the same request body this many times in a row is a loop, not a retry
MAX_BODY_BYTES = 8 * 2**20
FORWARDED = {"chat/completions", "responses", "completions"}
_PATH = re.compile(r"^/t/(?P<task>[A-Za-z0-9_.-]{1,64})/v1/(?P<rest>[a-z/]{1,40})$")
_DROP_HEADERS = {"authorization", "host", "content-length", "connection", "transfer-encoding",
	"accept-encoding", "x-api-key", "api-key", "openai-organization", "openai-project", "cookie"}  # fmt: skip
_HOP = {"connection", "transfer-encoding", "content-length", "content-encoding", "keep-alive"}


@dataclass
class Call:
	start: float  # time.time(), for the spans
	seconds: float
	status: int
	model: str | None = None
	input_tokens: int = 0
	cached_input_tokens: int = 0
	output_tokens: int = 0
	reasoning_tokens: int = 0
	tool_calls: int = 0
	finish_reason: str | None = None
	error: str | None = None  # the provider's error type, never its message
	usd: float = 0.0
	stream: bool = False
	span_id: str = field(default_factory=lambda: secrets.token_hex(8))


@dataclass
class _Task:
	calls: list[Call] = field(default_factory=list)
	last_hash: str | None = None
	repeats: int = 0
	stopped: str | None = None  # why the proxy stopped it: max_turns, loop, budget
	trace_id: str = field(default_factory=lambda: secrets.token_hex(16))


def _int(v) -> int:
	return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else 0


def read_usage(usage: dict | None) -> dict:
	"""Token counts from a chat-completions or a responses-API usage object."""
	u = usage if isinstance(usage, dict) else {}
	in_d = u.get("prompt_tokens_details") or u.get("input_tokens_details") or {}
	out_d = u.get("completion_tokens_details") or u.get("output_tokens_details") or {}
	inp = _int(u.get("prompt_tokens", u.get("input_tokens")))
	out = _int(u.get("completion_tokens", u.get("output_tokens")))
	return {
		"input_tokens": inp,
		"output_tokens": out,
		"cached_input_tokens": min(_int(in_d.get("cached_tokens") if isinstance(in_d, dict) else 0), inp),
		"reasoning_tokens": min(_int(out_d.get("reasoning_tokens") if isinstance(out_d, dict) else 0), out),
	}


def read_response(data: dict) -> dict:
	"""What a non-streamed reply says: usage, tool calls asked for, finish reason, model."""
	out = read_usage(data.get("usage"))
	out["model"] = data.get("model") if isinstance(data.get("model"), str) else None
	choices = data.get("choices")
	if isinstance(choices, list) and choices and isinstance(choices[0], dict):  # chat completions
		msg = choices[0].get("message") or {}
		calls = msg.get("tool_calls") if isinstance(msg, dict) else None
		out["tool_calls"] = len(calls) if isinstance(calls, list) else 0
		out["finish_reason"] = choices[0].get("finish_reason")
	elif isinstance(data.get("output"), list):  # responses API
		items = [i for i in data["output"] if isinstance(i, dict)]
		out["tool_calls"] = sum(1 for i in items if i.get("type") in ("function_call", "custom_tool_call"))
		out["finish_reason"] = "tool_calls" if out["tool_calls"] else data.get("status")
	return out


class _StreamReader:
	"""Reads usage, tool calls and the finish reason out of a server-sent-events reply as it passes."""

	def __init__(self):
		self.buf = b""
		self.found: dict = {"tool_calls": 0}
		self._tool_ids: set = set()

	def feed(self, chunk: bytes) -> None:
		self.buf += chunk
		while b"\n" in self.buf:
			line, self.buf = self.buf.split(b"\n", 1)
			line = line.strip()
			if not line.startswith(b"data:"):
				continue
			body = line[5:].strip()
			if not body or body == b"[DONE]":
				continue
			try:
				ev = json.loads(body)
			except ValueError:
				continue
			if isinstance(ev, dict):
				self._event(ev)

	def _event(self, ev: dict) -> None:
		if isinstance(ev.get("model"), str):
			self.found["model"] = ev["model"]
		if ev.get("usage"):
			self.found.update(read_usage(ev["usage"]))
		resp = ev.get("response")
		if ev.get("type") == "response.completed" and isinstance(resp, dict):  # responses API
			done = read_response(resp)
			self.found.update({k: v for k, v in done.items() if v is not None})
			return
		for ch in ev.get("choices") or []:
			if not isinstance(ch, dict):
				continue
			if ch.get("finish_reason"):
				self.found["finish_reason"] = ch["finish_reason"]
			for tc in (ch.get("delta") or {}).get("tool_calls") or []:
				if isinstance(tc, dict):
					self._tool_ids.add(tc.get("index", tc.get("id")))
					self.found["tool_calls"] = len(self._tool_ids)


class AgentProxy:
	def __init__(
		self,
		upstream: str,
		api_key: str,
		*,
		rate: Callable[[str], object | None] | None = None,
		max_calls_per_task: int = MAX_CALLS_PER_TASK,
		max_usd_per_task: float = MAX_USD_PER_TASK,
		max_usd_total: float = MAX_USD_TOTAL,
		timeout_s: float = 300.0,
		transport: httpx.BaseTransport | None = None,
	):
		self.upstream = upstream.rstrip("/")
		self._key = api_key
		self._rate = rate
		self.max_calls_per_task = max_calls_per_task
		self.max_usd_per_task = max_usd_per_task
		self.max_usd_total = max_usd_total
		self._client = httpx.Client(timeout=httpx.Timeout(timeout_s, connect=10.0), transport=transport)
		self._lock = threading.Lock()
		self._tasks: dict[str, _Task] = {}
		self._server: ThreadingHTTPServer | None = None

	# ------------------------------------------------------------- lifecycle

	def start(self) -> tuple[str, int]:
		proxy = self

		class Handler(BaseHTTPRequestHandler):
			protocol_version = "HTTP/1.1"

			def log_message(self, *a):  # requests are counted, never logged
				pass

			def do_POST(self):
				proxy._handle(self)

			def do_GET(self):
				self._send_json(404, {"error": {"type": "not_found", "message": "preflight: POST only"}})

			def _send_json(self, status, obj):
				_reply(self, status, obj)

		self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
		self._server.daemon_threads = True
		threading.Thread(target=self._server.serve_forever, name="agent-proxy", daemon=True).start()
		return self._server.server_address[0], self._server.server_address[1]

	def base_url(self, task_id: str) -> str:
		if self._server is None:
			raise RuntimeError("start() the proxy first")
		host, port = self._server.server_address[:2]
		return f"http://{host}:{port}/t/{task_id}/v1"

	def stop(self) -> None:
		if self._server is not None:
			self._server.shutdown()
			self._server.server_close()
			self._server = None
		self._client.close()

	# ------------------------------------------------------------- accounting

	def spent_usd(self) -> float:
		with self._lock:
			return sum(c.usd for t in self._tasks.values() for c in t.calls)

	def _price(self, model: str | None, u: dict) -> float:
		r = self._rate(model) if (self._rate is not None and model) else None
		if r is None:
			return 0.0
		return float(
			r.usd(u.get("input_tokens", 0), u.get("output_tokens", 0), u.get("cached_input_tokens", 0))
		)

	def _admit(self, task_id: str, body: bytes) -> str | None:
		"""Whether this call may go out; when not, the reason the task is stopped."""
		h = hashlib.sha256(body).hexdigest()
		with self._lock:
			t = self._tasks.setdefault(task_id, _Task())
			if t.stopped:
				return t.stopped
			t.repeats = t.repeats + 1 if h == t.last_hash else 1
			t.last_hash = h
			spent_task = sum(c.usd for c in t.calls)
			spent = sum(c.usd for x in self._tasks.values() for c in x.calls)
			if t.repeats >= REPEATS_FOR_LOOP:
				t.stopped = "loop"
			elif len(t.calls) >= self.max_calls_per_task:
				t.stopped = "max_turns"
			elif spent_task >= self.max_usd_per_task or spent >= self.max_usd_total:
				t.stopped = "budget"
			return t.stopped

	def _log(self, task_id: str, call: Call) -> None:
		with self._lock:
			self._tasks.setdefault(task_id, _Task()).calls.append(call)

	# ------------------------------------------------------------- one request

	def _handle(self, h: BaseHTTPRequestHandler) -> None:
		m = _PATH.match(h.path.split("?", 1)[0])
		if m is None or m["rest"] not in FORWARDED:
			_reply(
				h, 404, {"error": {"type": "not_found", "message": "preflight: not a generation endpoint"}}
			)
			return
		task_id, rest = m["task"], m["rest"]
		try:
			n = int(h.headers.get("Content-Length") or 0)
		except ValueError:
			n = -1
		if not 0 < n <= MAX_BODY_BYTES:
			_reply(
				h,
				413,
				{"error": {"type": "invalid_request", "message": "preflight: body missing or too large"}},
			)
			return
		body = h.rfile.read(n)
		try:
			req = json.loads(body)
		except ValueError:
			req = None
		if not isinstance(req, dict):
			_reply(
				h,
				400,
				{"error": {"type": "invalid_request", "message": "preflight: body is not a JSON object"}},
			)
			return
		why = self._admit(task_id, body)
		if why:
			_reply(h, 429, {"error": {"type": "preflight_task_stopped", "code": why,
				"message": f"preflight: this task was stopped ({why}); it counts as a runaway in the sample"}})  # fmt: skip
			return
		stream = bool(req.get("stream"))
		if stream and rest == "chat/completions":
			# the usage arrives only when asked for; the script's own chunks are unchanged otherwise
			opts = req.get("stream_options") if isinstance(req.get("stream_options"), dict) else {}
			req["stream_options"] = {**opts, "include_usage": True}
			body = json.dumps(req).encode()
		headers = {k: v for k, v in h.headers.items() if k.lower() not in _DROP_HEADERS}
		headers.update({"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"})
		started, t0 = time.time(), time.monotonic()
		call = Call(start=started, seconds=0.0, status=0, model=req.get("model") if isinstance(req.get("model"), str) else None,
			stream=stream)  # fmt: skip
		try:
			if stream:
				self._relay_stream(h, f"{self.upstream}/{rest}", headers, body, call)
			else:
				self._relay(h, f"{self.upstream}/{rest}", headers, body, call)
		except httpx.HTTPError as exc:
			call.status, call.error = 502, type(exc).__name__
			if not getattr(h, "_started", False):
				_reply(
					h,
					502,
					{
						"error": {
							"type": "upstream_unreachable",
							"message": "preflight: the provider didn't answer",
						}
					},
				)
		finally:
			call.seconds = time.monotonic() - t0
			call.usd = self._price(call.model, call.__dict__)
			self._log(task_id, call)

	def _relay(self, h, url, headers, body, call: Call) -> None:
		r = self._client.post(url, content=body, headers=headers)
		call.status = r.status_code
		data = None
		with contextlib.suppress(ValueError):
			data = r.json()
		if isinstance(data, dict):
			if r.status_code < 400:
				for k, v in read_response(data).items():
					if v is not None:
						setattr(call, k, v)
			else:
				err = data.get("error")
				call.error = (
					str((err or {}).get("code") or (err or {}).get("type") or r.status_code)[:64]
					if isinstance(err, dict)
					else str(r.status_code)
				)
		elif r.status_code >= 400:
			call.error = str(r.status_code)
		h.send_response(r.status_code)
		for k, v in r.headers.items():
			if k.lower() not in _HOP:
				h.send_header(k, v)
		h.send_header("Content-Length", str(len(r.content)))
		h.end_headers()
		h.wfile.write(r.content)

	def _relay_stream(self, h, url, headers, body, call: Call) -> None:
		reader = _StreamReader()
		with self._client.stream("POST", url, content=body, headers=headers) as r:
			call.status = r.status_code
			h.send_response(r.status_code)
			for k, v in r.headers.items():
				if k.lower() not in _HOP:
					h.send_header(k, v)
			h.send_header("Transfer-Encoding", "chunked")
			h.end_headers()
			h._started = True
			for chunk in r.iter_raw():
				if not chunk:
					continue
				reader.feed(chunk)
				h.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
				h.wfile.flush()
			h.wfile.write(b"0\r\n\r\n")
			h.wfile.flush()
		if r.status_code >= 400:
			call.error = str(r.status_code)
		for k, v in reader.found.items():
			if v is not None:
				setattr(call, k, v)

	# ------------------------------------------------------------- results

	def calls(self, task_id: str) -> list[Call]:
		with self._lock:
			return list(self._tasks.get(task_id, _Task()).calls)

	def task(self, task_id: str, *, success: bool | None = None, wall_s: float | None = None,
		timed_out: bool = False) -> AgentTask:  # fmt: skip
		"""The task as the record's AgentTask: turns are the calls the provider answered."""
		with self._lock:
			t = self._tasks.get(task_id, _Task())
			calls, stopped = list(t.calls), t.stopped
		ok = [c for c in calls if c.status and c.status < 400]
		models = [c.model for c in ok if c.model]
		if stopped in ("loop", "max_turns"):
			reason = stopped
		elif stopped == "budget":
			reason = "max_turns"  # the record's closest word: stopped for running on, not for failing
		elif timed_out:
			reason = "timeout"
		elif any(c.error and "context" in c.error for c in calls) or (
			ok and ok[-1].finish_reason == "length" and ok[-1].output_tokens == 0
		):
			reason = "context_overflow"
		elif calls and not ok:
			reason = "error"
		else:
			reason = "done" if ok else None
		span = (max(c.start + c.seconds for c in calls) - min(c.start for c in calls)) if calls else 0.0
		return AgentTask(
			task_id=task_id,
			model=max(set(models), key=models.count) if models else None,
			turns=len(ok),
			input_tokens=sum(c.input_tokens for c in ok),
			output_tokens=sum(c.output_tokens for c in ok),
			reasoning_tokens=sum(c.reasoning_tokens for c in ok),
			cached_input_tokens=sum(c.cached_input_tokens for c in ok),
			tool_calls=sum(c.tool_calls for c in ok),
			errors=sum(1 for c in calls if not c.status or c.status >= 400),
			latency_s=round(wall_s if wall_s is not None else span, 3),
			success=success,
			context_tokens_by_turn=[c.input_tokens for c in ok],
			cost_usd=round(sum(c.usd for c in calls), 6) if self._rate is not None else None,
			stop_reason=reason,
		)

	def spans(self, task_id: str) -> list[dict]:
		"""OTel GenAI spans for the task's calls (counts and timings, no content), one trace per task."""
		with self._lock:
			t = self._tasks.get(task_id, _Task())
			calls, trace = list(t.calls), t.trace_id
		out = []
		for c in calls:
			attrs = {
				"gen_ai.operation.name": "chat",
				"gen_ai.request.model": c.model,
				"gen_ai.response.model": c.model,
				"gen_ai.usage.input_tokens": c.input_tokens,
				"gen_ai.usage.output_tokens": c.output_tokens,
				"gen_ai.usage.cached_input_tokens": c.cached_input_tokens,
				"gen_ai.usage.reasoning_tokens": c.reasoning_tokens,
				"gen_ai.response.finish_reasons": [c.finish_reason] if c.finish_reason else [],
				"http.response.status_code": c.status,
				"preflight.task_id": task_id,
				"preflight.tool_calls": c.tool_calls,
				"preflight.cost_usd": round(c.usd, 6),
				"preflight.stream": c.stream,
			}
			if c.error:
				attrs["error.type"] = c.error
			out.append({
				"trace_id": trace,
				"span_id": c.span_id,
				"name": f"chat {c.model or 'unknown'}",
				"kind": "CLIENT",
				"start_time_unix_nano": int(c.start * 1e9),
				"end_time_unix_nano": int((c.start + c.seconds) * 1e9),
				"status": {"code": "ERROR" if c.error else "OK"},
				"attributes": {k: v for k, v in attrs.items() if v is not None},
			})  # fmt: skip
		return out


def _reply(h: BaseHTTPRequestHandler, status: int, obj: dict) -> None:
	data = json.dumps(obj).encode()
	h.send_response(status)
	h.send_header("Content-Type", "application/json")
	h.send_header("Content-Length", str(len(data)))
	h.end_headers()
	h.wfile.write(data)
