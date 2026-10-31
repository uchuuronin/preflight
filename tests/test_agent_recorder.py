"""The agent recorder against a fake OpenAI-compatible provider, through real sockets."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from preflight.engine.prices import token_rate
from preflight.recorder.agent import AgentProxy, read_usage


class Provider:
	"""A fake provider: answers chat completions (plain or streamed) and the responses API, and
	remembers what it was sent."""

	def __init__(self):
		self.seen: list[dict] = []
		self.status = 200
		provider = self

		class H(BaseHTTPRequestHandler):
			protocol_version = "HTTP/1.1"

			def log_message(self, *a):
				pass

			def do_POST(self):
				body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
				provider.seen.append({"path": self.path, "auth": self.headers.get("Authorization"),
					"cookie": self.headers.get("Cookie"), "body": body})  # fmt: skip
				if provider.status != 200:
					self._json(provider.status, {"error": {"type": "invalid_request_error",
						"code": "context_length_exceeded", "message": "secret detail"}})  # fmt: skip
				elif self.path.endswith("/responses"):
					self._json(200, {"model": body["model"], "status": "completed",
						"output": [{"type": "function_call", "name": "ls"}],
						"usage": {"input_tokens": 900, "output_tokens": 40,
							"input_tokens_details": {"cached_tokens": 100}, "output_tokens_details": {"reasoning_tokens": 10}}})  # fmt: skip
				elif body.get("stream"):
					self.send_response(200)
					self.send_header("Content-Type", "text/event-stream")
					self.send_header("Transfer-Encoding", "chunked")
					self.end_headers()
					events = [
						{"model": body["model"], "choices": [{"delta": {"content": "hi"}}]},
						{"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1"}]}}]},
						{
							"choices": [
								{"delta": {"tool_calls": [{"index": 0}]}, "finish_reason": "tool_calls"}
							]
						},
					]
					if (body.get("stream_options") or {}).get("include_usage"):
						events.append(
							{"choices": [], "usage": {"prompt_tokens": 1200, "completion_tokens": 30}}
						)
					for ev in [*map(json.dumps, events), "[DONE]"]:
						data = f"data: {ev}\n\n".encode()
						self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
					self.wfile.write(b"0\r\n\r\n")
				else:
					n = len(provider.seen)
					self._json(200, {"model": body["model"], "choices": [{"message": {"content": "ok",
						"tool_calls": [{"id": "a"}, {"id": "b"}] if n == 1 else None},
						"finish_reason": "tool_calls" if n == 1 else "stop"}],
						"usage": {"prompt_tokens": 1000 * n, "completion_tokens": 50,
							"prompt_tokens_details": {"cached_tokens": 200},
							"completion_tokens_details": {"reasoning_tokens": 20}}})  # fmt: skip

			def _json(self, status, obj):
				data = json.dumps(obj).encode()
				self.send_response(status)
				self.send_header("Content-Type", "application/json")
				self.send_header("Content-Length", str(len(data)))
				self.end_headers()
				self.wfile.write(data)

		self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
		threading.Thread(target=self.server.serve_forever, daemon=True).start()
		self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"


@pytest.fixture
def setup():
	prov = Provider()
	proxy = AgentProxy(prov.url, "our-key", rate=token_rate)
	proxy.start()
	yield prov, proxy
	proxy.stop()
	prov.server.shutdown()


def _chat(proxy, task, **extra):
	body = {"model": "muse-spark-1.3", "messages": [{"role": "user", "content": "do it"}], **extra}
	return httpx.post(f"{proxy.base_url(task)}/chat/completions", json=body,
		headers={"Authorization": "Bearer visitors-key", "Cookie": "s=1"}, timeout=10)  # fmt: skip


def test_a_task_becomes_an_agent_task_and_the_key_is_ours(setup):
	prov, proxy = setup
	assert _chat(proxy, "t1").status_code == 200
	assert _chat(proxy, "t1", messages=[{"role": "user", "content": "next"}]).json()["choices"][0]["message"]
	assert {s["auth"] for s in prov.seen} == {"Bearer our-key"} and not any(s["cookie"] for s in prov.seen)
	t = proxy.task("t1", success=True)
	assert (t.turns, t.input_tokens, t.output_tokens) == (2, 3000, 100)
	assert (t.cached_input_tokens, t.reasoning_tokens, t.tool_calls) == (400, 40, 2)
	assert t.context_tokens_by_turn == [1000, 2000] and t.stop_reason == "done" and t.success
	rate = token_rate("muse-spark-1.3")
	assert t.cost_usd == pytest.approx(rate.usd(1000, 50, 200) + rate.usd(2000, 50, 200), abs=1e-6)
	spans = proxy.spans("t1")
	assert len(spans) == 2 and len({s["trace_id"] for s in spans}) == 1
	assert spans[0]["attributes"]["gen_ai.usage.input_tokens"] == 1000
	assert "do it" not in json.dumps(spans)  # counts and timings only, never content


def test_a_streamed_call_is_counted_and_the_usage_is_asked_for(setup):
	prov, proxy = setup
	r = _chat(proxy, "t2", stream=True)
	assert r.status_code == 200 and "data: [DONE]" in r.text  # the script still gets its stream
	assert prov.seen[-1]["body"]["stream_options"] == {"include_usage": True}
	c = proxy.calls("t2")[0]
	assert (c.input_tokens, c.output_tokens, c.tool_calls, c.finish_reason) == (1200, 30, 1, "tool_calls")


def test_the_responses_api_is_read_too(setup):
	_, proxy = setup
	r = httpx.post(
		f"{proxy.base_url('t3')}/responses", json={"model": "muse-spark-1.3", "input": "x"}, timeout=10
	)
	assert r.status_code == 200
	t = proxy.task("t3")
	assert (t.input_tokens, t.cached_input_tokens, t.reasoning_tokens, t.tool_calls) == (900, 100, 10, 1)


def test_a_task_that_repeats_itself_is_stopped_as_a_loop(setup):
	prov, proxy = setup
	codes = [_chat(proxy, "t4").status_code for _ in range(7)]
	assert codes == [200] * 4 + [429] * 3
	assert len(prov.seen) == 4  # nothing past the stop reaches the provider
	assert proxy.task("t4").stop_reason == "loop"
	assert _chat(proxy, "t5").status_code == 200  # other tasks go on


def test_call_and_dollar_caps_stop_a_runaway(setup):
	prov, _ = setup
	proxy = AgentProxy(prov.url, "k", rate=token_rate, max_calls_per_task=2)
	proxy.start()
	try:
		codes = [
			_chat(proxy, "t6", messages=[{"role": "user", "content": str(i)}]).status_code for i in range(3)
		]
		assert codes == [200, 200, 429] and proxy.task("t6").stop_reason == "max_turns"
	finally:
		proxy.stop()
	proxy = AgentProxy(prov.url, "k", rate=token_rate, max_usd_total=1e-9)
	proxy.start()
	try:
		assert _chat(proxy, "a").status_code == 200
		assert _chat(proxy, "b").status_code == 429  # the sample's purse is spent
		assert proxy.task("b").stop_reason == "max_turns"
	finally:
		proxy.stop()


def test_provider_errors_are_counted_without_their_message(setup):
	prov, proxy = setup
	prov.status = 400
	r = _chat(proxy, "t7")
	assert r.status_code == 400  # the script sees what the provider said
	t = proxy.task("t7")
	assert t.errors == 1 and t.turns == 0 and t.stop_reason == "context_overflow"
	assert "secret detail" not in json.dumps(proxy.spans("t7"))


@pytest.mark.parametrize("path", ["/t/x/v1/files", "/t/../v1/chat/completions", "/v1/chat/completions",
	"/t/" + "a" * 65 + "/v1/chat/completions"])  # fmt: skip
def test_only_generation_endpoints_are_forwarded(setup, path):
	prov, proxy = setup
	host, port = proxy._server.server_address[:2]
	r = httpx.post(f"http://{host}:{port}{path}", json={"model": "m"}, timeout=10)
	assert r.status_code == 404 and not prov.seen


def test_usage_shapes_are_read_defensively():
	assert read_usage({"prompt_tokens": -3, "completion_tokens": True}) == {
		"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0, "reasoning_tokens": 0}  # fmt: skip
	got = read_usage({"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 99}})
	assert got["cached_input_tokens"] == 10  # never more than the input
