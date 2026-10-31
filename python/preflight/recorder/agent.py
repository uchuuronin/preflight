"""The agent recorder: an OpenAI-compatible proxy that measures an agent's model calls, task by task.

An agent or API run is measured by running a sample of its tasks (the sampler, next) with the
script's client pointed at this proxy instead of the provider:

	proxy = AgentProxy(upstream="https://api.llama.com/compat/v1", api_key=..., rate=token_rate)
	proxy.start()
	url = proxy.open_task("t017")   # http://127.0.0.1:<port>/t/<random token>/v1
	# the task's process gets OPENAI_BASE_URL=url
	...
	task = proxy.task("t017", success=True)   # an AgentTask for the record's agent_sample
	spans = proxy.spans("t017")               # OTel GenAI spans, counts only

What it keeps per call: the model, token counts as the provider reports them (input, cached
input, output, the reasoning part of the output), tool calls asked for, the finish reason, the
latency and the HTTP status. It never keeps prompts or completions: the spans follow the OTel
GenAI semantic conventions (`gen_ai.*`) without the content attributes.

What it guards: the visitor's script never holds the provider key (the proxy drops whatever
Authorization the script sends and sets its own; run the proxy in another process or uid than the
script, or the script could read the key from /proc), it binds to 127.0.0.1 only, a task is
reached only through the random token open_task gave it, it forwards only the generation
endpoints, and it stops a task that runs away (too many calls, too many dollars, or the same
request over and over) with a 429 the client sees as a rate limit. Dollars are reserved before a
call goes out (its input and its max output at the model's price), so parallel calls can't spend
past a cap, and a model without a price is refused: a sample that can't be costed isn't run. The
caps are guesses until the agent validation run.
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
MAX_EVENT_BYTES = 4 * 2**20  # one server-sent event; a stream that never ends an event is cut off
# guess: the output a call may produce when it doesn't say (max_tokens), for the dollar reservation
DEFAULT_MAX_OUTPUT_TOKENS = 8192
SETTLE_S = 10.0  # guess: a call's tail (logging after the provider's last byte) takes milliseconds
CHARS_PER_TOKEN = 3  # guess, on the high side for tokens: the reservation over-, never under-counts
FORWARDED = {"chat/completions", "responses", "completions"}
_PATH = re.compile(r"^/t/(?P<token>[A-Za-z0-9_-]{16,64})/v1/(?P<rest>[a-z/]{1,40})$")
_TASK_ID = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}")
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
	response_id: str | None = None  # responses API: the next turn names it as previous_response_id
	span_id: str = field(default_factory=lambda: secrets.token_hex(8))


@dataclass
class _Run:
	run_id: str
	max_tasks: int
	skip_keys: dict = field(default_factory=dict)  # opening key -> tasks already sampled with it (a top-up)
	skipped_keys: dict = field(default_factory=dict)  # opening key -> tasks refused with it so far
	skipped: list[list[str]] = field(default_factory=list)  # first calls of the tasks refused
	order: list[str] = field(default_factory=list)
	last: dict[str, list[str]] = field(default_factory=dict)  # task -> its last call's message hashes
	by_response: dict[str, str] = field(default_factory=dict)  # response id -> task (responses API)
	keys: dict[str, int] = field(default_factory=dict)  # opening key -> tasks opened with it
	refused: int = 0  # calls refused because the sample was full
	system: dict[str, str | None] = field(default_factory=dict)  # task -> its last system prompt hash
	last_call_at: float | None = None  # monotonic: any call through the run, refused ones too
	budget_hit: bool = False  # a task was refused because the proxy's dollar cap was reached


def _h(x) -> str:
	return hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _split(req: dict) -> tuple[str | None, list[str]]:
	"""A call's system prompt (hashed) and the rest of its conversation as hashes of its messages
	(chat `messages`, responses `input`, legacy completions `prompt`). The system prompt is kept
	apart: agents often rebuild it every call (a timestamp, the tool list), which would make every
	turn look like a new conversation."""
	items = req.get("messages")
	if items is None:
		items = req.get("input")
	if items is None:
		items = req.get("prompt")
	if isinstance(items, str):
		items = [{"role": "user", "content": items}]
	if not isinstance(items, list):
		return None, []
	system = [x for x in items if isinstance(x, dict) and x.get("role") in ("system", "developer")]
	if isinstance(req.get("instructions"), str):
		system.append({"instructions": req["instructions"]})
	rest = [x for x in items if not (isinstance(x, dict) and x.get("role") in ("system", "developer"))]
	return (_h(system) if system else None), [_h(x) for x in rest]


def _conversation(req: dict) -> list[str]:
	return _split(req)[1]


def opening_key(msgs: list[str]) -> str:
	"""A task's key: a hash of its first call's message hashes (no content). A rerun of the same
	script opens the same task with the same key, so a top-up can refuse it by content."""
	return hashlib.sha256(",".join(msgs).encode()).hexdigest()[:12]


def task_key(task_id: str) -> str | None:
	"""The opening key inside a run-mode task id (`<run>-<key>` or `<run>-<key>.<n>`)."""
	tail = task_id.rsplit("-", 1)[-1].split(".", 1)[0]
	return tail if re.fullmatch(r"[0-9a-f]{12}", tail) else None


@dataclass
class _Task:
	calls: list[Call] = field(default_factory=list)
	last_hash: str | None = None
	repeats: int = 0
	stopped: str | None = None  # why the proxy stopped it: max_turns, loop, budget
	in_flight: int = 0
	open: list = field(default_factory=list)  # calls in flight; their status is set once the provider answers
	reserved_usd: float = 0.0
	unpriced: bool = False
	trace_id: str = field(default_factory=lambda: secrets.token_hex(16))


def _int(v) -> int:
	return v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else 0


def read_usage(usage: dict | None) -> dict:
	"""Token counts from a chat-completions or a responses-API usage object."""
	u = usage if isinstance(usage, dict) else {}
	in_d = u.get("prompt_tokens_details") or u.get("input_tokens_details") or {}
	out_d = u.get("completion_tokens_details") or u.get("output_tokens_details") or {}
	inp = _int(u.get("prompt_tokens") if u.get("prompt_tokens") is not None else u.get("input_tokens"))
	out = _int(
		u.get("completion_tokens") if u.get("completion_tokens") is not None else u.get("output_tokens")
	)
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
	"""Reads usage, tool calls and the finish reason out of a server-sent-events reply as it passes,
	event by event, and says which events to pass on (the usage-only chunk the proxy asked for on
	the script's behalf is read and dropped, so the script sees the stream it asked for)."""

	ENDS = ("response.completed", "response.incomplete", "response.failed")

	def __init__(self, drop_usage_chunk: bool = False):
		self.buf = b""
		self._cr = b""
		self.found: dict = {"tool_calls": 0}
		self._tool_ids: set = set()
		self.drop_usage_chunk = drop_usage_chunk
		self.overflow = False

	def feed(self, chunk: bytes) -> bytes:
		"""The bytes of the whole events in buf so far that go on to the script."""
		# a chunk may end between a \r and its \n: that \r waits for the next chunk
		chunk = self._cr + chunk
		self._cr = b""
		if chunk.endswith(b"\r"):
			chunk, self._cr = chunk[:-1], b"\r"
		pos = 0
		start = max(0, len(self.buf) - 1)  # an event's end can straddle the old buffer's last byte
		self.buf += chunk.replace(b"\r\n", b"\n")
		out = []
		while (i := self.buf.find(b"\n\n", start)) != -1:
			event = self.buf[pos:i]
			pos = start = i + 2
			if self._event_bytes(event):
				out.append(event + b"\n\n")
		self.buf = self.buf[pos:]  # one copy per chunk, however many events it held
		if len(self.buf) > MAX_EVENT_BYTES:
			self.overflow = True
		return b"".join(out)

	def rest(self) -> bytes:
		left, self.buf, self._cr = self.buf + self._cr, b"", b""
		if left.strip():
			self._event_bytes(left)
		return left

	def _event_bytes(self, event: bytes) -> bool:
		keep = True
		for line in event.split(b"\n"):
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
				if self.drop_usage_chunk and ev.get("usage") and ev.get("choices") == []:
					keep = False
		return keep

	def _event(self, ev: dict) -> None:
		if isinstance(ev.get("model"), str):
			self.found["model"] = ev["model"]
		if ev.get("usage"):
			self.found.update(read_usage(ev["usage"]))
		err = ev.get("error")
		if isinstance(err, dict):
			self.found["error"] = str(err.get("code") or err.get("type") or "error")[:64]
		resp = ev.get("response")
		if ev.get("type") in self.ENDS and isinstance(resp, dict):  # responses API
			if isinstance(resp.get("id"), str):
				self.found["response_id"] = resp["id"][:128]
			done = read_response(resp)
			self.found.update({k: v for k, v in done.items() if v is not None})
			if ev["type"] != "response.completed":
				self.found["finish_reason"] = ev["type"].split(".", 1)[1]
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


def _error_code(data) -> str | None:
	err = data.get("error") if isinstance(data, dict) else None
	if isinstance(err, dict):
		return str(err.get("code") or err.get("type") or "error")[:64]
	return None


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
		# identity: the reply is relayed as it comes, and the script must get bytes it can read
		# whether this thread's call left for the provider: the request hook fires as it is sent, so a
		# call that failed before it (a header that won't encode, no free connection) is never billed
		self._went = threading.local()
		self._client = httpx.Client(timeout=httpx.Timeout(timeout_s, connect=10.0), transport=transport,
			headers={"Accept-Encoding": "identity"},
			event_hooks={"request": [lambda _r: setattr(self._went, "out", True)]})  # fmt: skip
		self._lock = threading.Lock()
		self._settled = threading.Condition(self._lock)
		self._tasks: dict[str, _Task] = {}
		self._tokens: dict[str, str] = {}  # path token -> task id
		self._server: ThreadingHTTPServer | None = None
		self.public_base: str | None = None  # e.g. a Modal tunnel's https URL
		self._last_activity = 0.0  # monotonic: the last call logged

	# ------------------------------------------------------------- lifecycle

	def _url(self, token: str) -> str:
		if self.public_base:  # reached through a tunnel (a sandbox on another machine)
			return f"{self.public_base.rstrip('/')}/t/{token}/v1"
		host, port = self._server.server_address[:2]
		return f"http://{host}:{port}/t/{token}/v1"

	def start(self, bind: str = "127.0.0.1") -> tuple[str, int]:
		"""bind: 127.0.0.1 for a script on this machine; 0.0.0.0 behind a tunnel (the token in the
		path is what admits a caller, and the proxy forwards only generation endpoints)."""
		proxy = self

		class Handler(BaseHTTPRequestHandler):
			protocol_version = "HTTP/1.1"
			timeout = 60  # a slow or silent client (the tunnel is public) can't hold a thread forever

			def log_message(self, *a):  # requests are counted, never logged
				pass

			def do_POST(self):
				proxy._handle(self)

			def do_GET(self):
				_reply(
					self, 404, {"error": {"type": "not_found", "message": "preflight: POST only"}}, close=True
				)

		self._server = ThreadingHTTPServer((bind, 0), Handler)
		self._server.daemon_threads = True
		threading.Thread(target=self._server.serve_forever, name="agent-proxy", daemon=True).start()
		return self._server.server_address[0], self._server.server_address[1]

	def open_task(self, task_id: str) -> str:
		"""The base URL for one task's process: only it knows the token in the path."""
		if self._server is None:
			raise RuntimeError("start() the proxy first")
		if not _TASK_ID.fullmatch(task_id):
			raise ValueError(f"task id {task_id!r}: letters, digits, '_', '.', '-', up to 64")
		token = secrets.token_urlsafe(24)
		with self._lock:
			self._tokens[token] = task_id
			self._tasks.setdefault(task_id, _Task())
		return self._url(token)

	def open_run(self, run_id: str, *, max_tasks: int, skip_keys=()) -> str:
		"""The base URL for a script that runs many tasks in one process (the usual agent eval
		loop): its calls are split into tasks by conversation (a call whose messages extend a
		task's last messages, or name its last response, belongs to that task; anything else
		starts one), and once max_tasks tasks have started, a call that would start another is
		refused with a 429 that tells the SDK not to retry, so the script's later tasks end at once
		and the sample costs only its own tasks. Task ids are `<run_id>-<opening key>`. skip_keys:
		tasks whose opening key is here are refused at their first call (a top-up after a first
		sample: a script that can't be told where to start runs its first tasks again, and they
		cost nothing this way, in whatever order they start; a key given twice skips two tasks
		opened that way)."""
		if not _TASK_ID.fullmatch(run_id):
			raise ValueError(f"run id {run_id!r}: letters, digits, '_', '.', '-', up to 64")
		if self._server is None:
			raise RuntimeError("start() the proxy first")
		token = secrets.token_urlsafe(24)
		with self._lock:
			if any(isinstance(r, _Run) and r.run_id == run_id for r in self._tokens.values()):
				raise ValueError(f"run id {run_id!r} is already open on this proxy")
			counts: dict = {}
			for k in skip_keys:  # a key twice: two sampled tasks opened the same way, both skipped
				counts[k] = counts.get(k, 0) + 1
			self._tokens[token] = _Run(run_id, max_tasks, skip_keys=counts)
		return self._url(token)

	def _run(self, run_id: str):
		return next((r for r in self._tokens.values() if isinstance(r, _Run) and r.run_id == run_id), None)

	def run_tasks(self, run_id: str) -> list[str]:
		"""The task ids a run's calls were split into, in the order they started."""
		with self._lock:
			run = self._run(run_id)
			return list(run.order) if run else []

	def run_stats(self, run_id: str) -> dict:
		"""What the selection line says: tasks skipped and calls refused, tasks with a call in
		flight now, and tasks of one call (side calls, or a history trimmed mid-task, split tasks)."""
		with self._lock:
			run = self._run(run_id)
			if run is None:
				return {}
			ts = {tid: self._tasks.get(tid, _Task()) for tid in run.order}
			return {"skipped": len(run.skipped), "refused": run.refused,
				"in_flight": [tid for tid, t in ts.items() if t.in_flight > 0],
				# a task whose last answer asked for a tool was mid-task (running the tool) if cut now
				"mid_tool": [tid for tid, t in ts.items() if t.calls and t.calls[-1].finish_reason == "tool_calls"],
				# a task waiting on its tools: seconds since its last answer (that asked for a tool) ended
				"tool_wait_s": {tid: time.time() - (t.calls[-1].start + t.calls[-1].seconds) for tid, t in ts.items()
					if t.calls and not t.in_flight and t.calls[-1].finish_reason == "tool_calls"},
				"single_call": sum(1 for t in ts.values() if len(t.calls) <= 1),
				"budget_hit": run.budget_hit,
				# since the last call started or ended; never idle while a call is in flight
				"idle_s": None if run.last_call_at is None or any(t.in_flight > 0 for t in ts.values())
					else time.monotonic() - max(run.last_call_at, self._last_activity)}  # fmt: skip

	def _task_for(self, run: "_Run", req: dict) -> str | None:
		"""The task this call belongs to, opening one if it starts a conversation; None when the
		run already has max_tasks tasks and this would be a new one."""
		sys_h, msgs = _split(req)
		prev = req.get("previous_response_id") if isinstance(req.get("previous_response_id"), str) else None
		with self._lock:
			run.last_call_at = time.monotonic()
			if prev and prev in run.by_response:
				return run.by_response[prev]
			best, best_key = None, None
			for tid, last in run.last.items():
				if not last or len(last) > len(msgs) or msgs[: len(last)] != last:
					continue
				t = self._tasks.get(tid, _Task())
				if t.stopped and len(last) == len(msgs):
					return tid  # a retry of the call that stopped it: refused again, not a fresh task
				# the last answer failed, logged or still being relayed (a client can retry on the status
				# line, before the proxy has logged the call)
				failed = any(c.status >= 400 or c.error for c in t.open if c.status) or (
					bool(t.calls)
					and bool(t.calls[-1].error or not t.calls[-1].status or t.calls[-1].status >= 400)
				)
				# a retry of a failed opening is the same task only under the same system prompt: the
				# same opening under another one is another task (task text in the system prompt)
				if len(last) == len(msgs) and len(msgs) == 1 and run.system.get(tid) != sys_h:
					failed = False
				if len(last) == len(msgs) and not failed:
					# the same call again, unless its last answer failed: a repeat (pass@k, or n samples
					# sent at once), not a retry; a client's own retry after a timeout opens a task too
					continue
				# the longest match; among equals, the same system prompt, then a task with nothing in
				# flight (its turn is next)
				k = (len(last), run.system.get(tid) == sys_h, t.in_flight == 0)
				if best_key is None or k > best_key:
					best, best_key = tid, k
			if best is not None:
				run.last[best] = msgs
				run.system[best] = sys_h
				return best
			for k_sys, k in run.skipped:
				if not k or len(k) > len(msgs) or msgs[: len(k)] != k:
					continue
				if len(k) < len(msgs) or k_sys == sys_h:
					# a later turn of a skipped task, or a retry of its opening (same system prompt):
					# still skipped. Tasks that open with the very same call can't be told from
					# retries of it, so a top-up of such a script runs nothing new and says so
					return ""
			# the same messages under another system prompt: tasks whose text is in the system prompt
			# (or a prompt rebuilt each call), skipped by their key's count below
			key = opening_key(msgs)
			if run.skipped_keys.get(key, 0) < run.skip_keys.get(key, 0):
				run.skipped_keys[key] = run.skipped_keys.get(key, 0) + 1
				run.skipped.append((sys_h, msgs))
				return ""
			if len(run.order) >= run.max_tasks:
				run.refused += 1
				return None
			spent = sum(c.usd for x in self._tasks.values() for c in x.calls) + sum(
				x.reserved_usd for x in self._tasks.values()
			)
			if spent + self._worst_case_usd(req, json.dumps(req).encode()) > self.max_usd_total:
				# the purse is spent: a new task would be a stopped, empty one that lowers the
				# sample's cost per task; refused like a task past the sample, and said so
				run.refused += 1
				run.budget_hit = True
				return None
			n = run.keys.get(key, 0) + 1
			run.keys[key] = n
			tid = f"{run.run_id}-{key}" + (f".{n}" if n > 1 else "")
			run.order.append(tid)
			run.last[tid] = msgs
			run.system[tid] = sys_h
			self._tasks.setdefault(tid, _Task())
			return tid

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

	def _rate_for(self, model: str | None):
		return self._rate(model) if (self._rate is not None and model) else None

	def _price(self, model: str | None, u: dict) -> float:
		r = self._rate_for(model)
		if r is None:
			return 0.0
		return float(
			r.usd(u.get("input_tokens", 0), u.get("output_tokens", 0), u.get("cached_input_tokens", 0))
		)

	def _worst_case_usd(self, req: dict, body: bytes) -> float:
		"""What the call can cost at most: its whole body as input, its max output for each of its n
		(or best_of) choices, uncached. Not counted: images sent by URL (their tokens aren't in the
		body); a call that bills more than this still counts in full once its usage arrives."""
		cap = next((req[k] for k in ("max_tokens", "max_completion_tokens", "max_output_tokens")
			if _int(req.get(k))), DEFAULT_MAX_OUTPUT_TOKENS)  # fmt: skip
		choices = max(_int(req.get("n")) or 1, _int(req.get("best_of")) or 1)
		return self._price(
			req.get("model"),
			{"input_tokens": len(body) // CHARS_PER_TOKEN + 1, "output_tokens": cap * choices},
		)

	def _admit(self, task_id: str, req: dict, body: bytes) -> tuple[str | None, float]:
		"""Whether this call may go out (None) or why the task is stopped, and the dollars reserved."""
		h = hashlib.sha256(body).hexdigest()
		priced = self._rate is None or self._rate_for(req.get("model")) is not None
		reserve = self._worst_case_usd(req, body) if self._rate is not None and priced else 0.0
		with self._lock:
			t = self._tasks.setdefault(task_id, _Task())
			if t.stopped:
				return t.stopped, 0.0
			if not priced:
				t.unpriced = True
				return "unpriced_model", 0.0  # this call only; the task may go on with a priced model
			t.repeats = t.repeats + 1 if h == t.last_hash else 1
			t.last_hash = h
			spent_task = sum(c.usd for c in t.calls) + t.reserved_usd
			spent = sum(c.usd for x in self._tasks.values() for c in x.calls) + sum(
				x.reserved_usd for x in self._tasks.values()
			)
			if t.repeats >= REPEATS_FOR_LOOP:
				t.stopped = "loop"
			elif len(t.calls) + t.in_flight >= self.max_calls_per_task:
				t.stopped = "max_turns"
			elif spent_task + reserve > self.max_usd_per_task or spent + reserve > self.max_usd_total:
				t.stopped = "budget"
			if t.stopped:
				return t.stopped, 0.0
			t.in_flight += 1
			t.reserved_usd += reserve
			return None, reserve

	def _log(self, task_id: str, call: Call, reserved: float, *, uncosted: bool = False) -> None:
		with self._lock:
			t = self._tasks.setdefault(task_id, _Task())
			t.in_flight -= 1
			self._last_activity = time.monotonic()
			with contextlib.suppress(ValueError):
				t.open.remove(call)
			t.reserved_usd = max(0.0, t.reserved_usd - reserved)
			t.unpriced = t.unpriced or uncosted
			t.calls.append(call)
			self._settled.notify_all()

	# ------------------------------------------------------------- one request

	def _handle(self, h: BaseHTTPRequestHandler) -> None:
		m = _PATH.match(h.path.split("?", 1)[0])
		with self._lock:
			task_id = self._tokens.get(m["token"]) if m else None
		run = task_id if isinstance(task_id, _Run) else None
		if m is None or m["rest"] not in FORWARDED or task_id is None:
			_reply(h, 404, {"error": {"type": "not_found", "message": "preflight: not a generation endpoint"}},
				close=True)  # fmt: skip
			return
		rest = m["rest"]
		try:
			n = int(h.headers.get("Content-Length") or 0)
		except ValueError:
			n = -1
		if not 0 < n <= MAX_BODY_BYTES or h.headers.get("Transfer-Encoding"):
			_reply(h, 413, {"error": {"type": "invalid_request", "message": "preflight: body missing or too large"}},
				close=True)  # fmt: skip
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
		if run is not None:
			task_id = self._task_for(run, req)
			if task_id == "":
				_reply(h, 429, {"error": {"type": "preflight_task_skipped", "code": "already_sampled",
					"message": "preflight: this task was in the first sample; skipped in the top-up"}},
					headers=NO_RETRY)  # fmt: skip
				return
			if task_id is None:
				_reply(h, 429, {"error": {"type": "preflight_sample_complete", "code": "sample_complete",
					"message": f"preflight: the sample's {run.max_tasks} tasks are running; this task isn't one"}},
					headers=NO_RETRY)  # fmt: skip
				return
		why, reserved = self._admit(task_id, req, body)
		if why == "unpriced_model":
			self._note(task_id, req, why)  # counted before the script hears back
			_reply(h, 400, {"error": {"type": "preflight_unpriced_model", "code": why,
				"message": "preflight: this model has no price, so the sample can't be costed"}})  # fmt: skip
			return
		if why:
			_reply(h, 429, {"error": {"type": "preflight_task_stopped", "code": why,
				"message": f"preflight: this task was stopped ({why}); it counts as a runaway in the sample"}},
				headers=NO_RETRY)  # fmt: skip
			return
		started, t0 = time.time(), time.monotonic()
		model = req.get("model") if isinstance(req.get("model"), str) else None
		stream = req.get("stream") is True
		call = Call(start=started, seconds=0.0, status=0, model=model, stream=stream)
		with self._lock:
			self._tasks.setdefault(task_id, _Task()).open.append(call)
		sent = [False]  # this request's reply has started: a failure now can only close the connection
		last: list = []  # the reply's last bytes (or the headers' end), sent once the call is logged
		broke, never_sent, upstream_status, fail_reply = False, False, 0, False
		self._went.out = False
		try:  # from here on the admitted call is always logged, so its slot and reservation are freed
			opts = req.get("stream_options")
			opts = opts if isinstance(opts, dict) else {}
			asked = opts.get("include_usage") is True
			inject = stream and rest in ("chat/completions", "completions") and not asked
			if inject:
				# the usage arrives only when asked for; the extra chunk it adds is read and dropped
				req["stream_options"] = {**opts, "include_usage": True}
				body = json.dumps(req).encode()
			headers = {k: v for k, v in h.headers.items() if k.lower() not in _DROP_HEADERS}
			headers.update({"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"})
			if stream:
				self._relay_stream(
					h, f"{self.upstream}/{rest}", headers, body, call, sent, last, drop_usage=inject
				)
			else:
				self._relay(h, f"{self.upstream}/{rest}", headers, body, call, sent, last)
		except Exception as exc:  # the script gets a 502 or a closed connection; the call is counted
			broke = True
			# never reached the provider: nothing to bill
			never_sent = not self._went.out or isinstance(
				exc,
				httpx.ConnectError
				| httpx.ConnectTimeout
				| httpx.PoolTimeout
				| httpx.ProxyError
				| httpx.LocalProtocolError,
			)
			upstream_status = call.status  # 0 when the provider's reply never began
			call.status = call.status or 502
			call.error = call.error or type(exc).__name__
			if sent[0]:
				h.close_connection = True
			else:
				fail_reply = True  # sent once the call is logged, like any reply's last byte
		finally:
			call.seconds = time.monotonic() - t0
			# priced by the reply's model, else the request's (which _admit found a price for)
			priced_as = call.model if self._rate_for(call.model) is not None else model
			call.usd = self._price(priced_as, call.__dict__)
			no_usage = call.input_tokens == call.output_tokens == 0
			# broke off with no usage seen, or answered without saying what it used: the provider may
			# still bill it, so its reservation stands in for the cost and the task isn't exactly costed
			# broken: billable unless it never reached the provider or its reply was an error (status 0:
			# no reply began, so it may have run); answered: a 4xx or 5xx the provider sent isn't billed
			billable = (not never_sent and upstream_status < 400) if broke else call.status < 400
			uncosted = self._rate is not None and no_usage and billable
			if uncosted:
				call.usd = reserved
			if run is not None and call.response_id:
				with self._lock:
					run.by_response[call.response_id] = task_id
			# logged before the script gets its last byte: once its call returns, the task is counted
			self._log(task_id, call, reserved, uncosted=uncosted)
		if fail_reply:
			with contextlib.suppress(OSError):
				_reply(h, 502, {"error": {"type": "upstream_unreachable",
					"message": "preflight: the provider didn't answer"}}, close=True)  # fmt: skip
		if last:
			with contextlib.suppress(OSError):
				for part in last:
					if callable(part):
						part()
					else:
						h.wfile.write(part)
				h.wfile.flush()

	def _note(self, task_id: str, req: dict, error: str) -> None:
		"""A call refused before it went out, counted as an error."""
		model = req.get("model") if isinstance(req.get("model"), str) else None
		with self._lock:
			self._tasks.setdefault(task_id, _Task()).calls.append(
				Call(start=time.time(), seconds=0.0, status=400, model=model, error=error)
			)

	def _relay(self, h, url, headers, body, call: Call, sent: list, last: list) -> None:
		r = self._client.post(url, content=body, headers=headers)
		call.status = r.status_code
		data = None
		with contextlib.suppress(ValueError):
			data = r.json()
		if r.status_code < 400 and isinstance(data, dict):
			for k, v in read_response(data).items():
				if v is not None:
					setattr(call, k, v)
			if isinstance(data.get("id"), str) and data.get("object") == "response":
				call.response_id = data["id"][:128]
		elif r.status_code >= 400:
			call.error = _error_code(data) or str(r.status_code)
		sent[0] = True
		h.send_response(r.status_code)
		for k, v in r.headers.items():
			if k.lower() not in _HOP:
				h.send_header(k, v)
		h.send_header("Content-Length", str(len(r.content)))
		if r.content:
			h.end_headers()
			h.wfile.write(r.content[:-1])
			last.append(r.content[-1:])
		else:  # no body to hold a byte of: the headers' end waits for the log instead
			last.append(h.end_headers)

	def _relay_stream(
		self, h, url, headers, body, call: Call, sent: list, last: list, *, drop_usage: bool
	) -> None:
		reader = _StreamReader(drop_usage_chunk=drop_usage)

		def write(data: bytes) -> None:
			if data:
				h.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
				h.wfile.flush()

		try:
			with self._client.stream("POST", url, content=body, headers=headers) as r:
				call.status = r.status_code
				sent[0] = True
				h.send_response(r.status_code)
				for k, v in r.headers.items():
					if k.lower() not in _HOP:
						h.send_header(k, v)
				h.send_header("Transfer-Encoding", "chunked")
				h.end_headers()
				if r.status_code >= 400:  # an error body, not events: pass it on and read its code
					raw = r.read()
					with contextlib.suppress(ValueError):
						call.error = _error_code(json.loads(raw)) or str(r.status_code)
					call.error = call.error or str(r.status_code)
					write(raw)
				else:
					for chunk in r.iter_bytes():
						write(reader.feed(chunk))
						if reader.overflow:
							raise OSError("an event past the size cap")
					write(reader.rest())
				last.append(b"0\r\n\r\n")
		finally:
			for k, v in reader.found.items():  # what was read before any failure is still billed
				if v is not None and not (k == "error" and call.error):
					setattr(call, k, v)

	# ------------------------------------------------------------- results

	def calls(self, task_id: str) -> list[Call]:
		with self._lock:
			return list(self._tasks.get(task_id, _Task()).calls)

	def task(self, task_id: str, *, success: bool | None = None, wall_s: float | None = None,
		timed_out: bool = False, stalled: bool = False) -> AgentTask:  # fmt: skip
		"""The task as the record's AgentTask: turns are the calls the provider answered. Waits (up
		to SETTLE_S) for calls still in flight, so a script that exits at [DONE] is counted whole."""
		with self._lock:
			t = self._tasks.get(task_id, _Task())
			self._settled.wait_for(lambda: t.in_flight <= 0, timeout=SETTLE_S)
			calls, stopped, unpriced = list(t.calls), t.stopped, t.unpriced
		ok = [c for c in calls if c.status and c.status < 400]
		models = [c.model for c in ok if c.model]
		if stopped in ("loop", "max_turns"):
			reason = stopped
		elif stopped == "budget":
			reason = "max_turns"  # the record's closest word: stopped for running on, not for failing
		elif stalled:
			reason = "loop"  # waited on its tools past the stall cutoff with no model call: a tool stuck or looping
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
			cost_usd=round(sum(c.usd for c in calls), 6) if self._rate is not None and not unpriced else None,
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


# the OpenAI SDKs retry a 429 unless the server says not to (x-should-retry: false); a refusal
# of the proxy's own is final, so a script past its sample ends at once rather than backing off
NO_RETRY = {"x-should-retry": "false"}


def _reply(h: BaseHTTPRequestHandler, status: int, obj: dict, *, close: bool = False, headers=None) -> None:
	"""A reply of the proxy's own. close: the request's body may be unread, so the connection can't
	carry another request."""
	data = json.dumps(obj).encode()
	h.send_response(status)
	h.send_header("Content-Type", "application/json")
	for k, v in (headers or {}).items():
		h.send_header(k, v)
	h.send_header("Content-Length", str(len(data)))
	if close:
		h.send_header("Connection", "close")
		h.close_connection = True
	h.end_headers()
	h.wfile.write(data)
