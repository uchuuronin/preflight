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


_URLS: dict = {}


def _url(proxy, task):
	key = (id(proxy), task)
	if key not in _URLS:
		_URLS[key] = proxy.open_task(task)
	return _URLS[key]


def _chat(proxy, task, client=httpx, **extra):
	body = {"model": "muse-spark-1.3", "messages": [{"role": "user", "content": "do it"}], **extra}
	return client.post(f"{_url(proxy, task)}/chat/completions", json=body,
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
	assert '"choices": []' not in r.text  # the usage chunk it didn't ask for is not passed on
	r = _chat(proxy, "t2b", stream=True, stream_options={"include_usage": True})
	assert '"choices": []' in r.text  # asked for, so kept
	c = proxy.calls("t2")[0]
	assert (c.input_tokens, c.output_tokens, c.tool_calls, c.finish_reason) == (1200, 30, 1, "tool_calls")


def test_the_responses_api_is_read_too(setup):
	_, proxy = setup
	r = httpx.post(
		f"{_url(proxy, 't3')}/responses", json={"model": "muse-spark-1.3", "input": "x"}, timeout=10
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
	# a total purse with room for one call's worst case: the first goes out, the next can't fit
	small = {"max_tokens": 50}
	probe = AgentProxy(prov.url, "k", rate=token_rate)
	body = {"model": "muse-spark-1.3", "messages": [{"role": "user", "content": "do it"}], **small}
	one = probe._worst_case_usd(body, json.dumps(body).encode())
	probe.stop()
	proxy = AgentProxy(prov.url, "k", rate=token_rate, max_usd_total=one * 1.5)
	proxy.start()
	try:
		assert _chat(proxy, "a", **small).status_code == 200
		assert _chat(proxy, "b", **small).status_code == 429  # the sample's purse is spent
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


@pytest.mark.parametrize("path", ["/t/{tok}/v1/files", "/t/notarealtokenatall123/v1/chat/completions",
	"/v1/chat/completions", "/t/t1/v1/chat/completions"])  # fmt: skip
def test_only_generation_endpoints_of_opened_tasks_are_forwarded(setup, path):
	prov, proxy = setup
	tok = _url(proxy, "t1").split("/t/", 1)[1].split("/", 1)[0]
	host, port = proxy._server.server_address[:2]
	r = httpx.post(f"http://{host}:{port}{path.format(tok=tok)}", json={"model": "m"}, timeout=10)
	assert r.status_code == 404 and not prov.seen
	with pytest.raises(ValueError):
		proxy.open_task("../t1")


def test_parallel_calls_cant_spend_past_the_cap(setup):
	prov, _proxy = setup
	proxy = AgentProxy(prov.url, "k", rate=token_rate)
	body = {
		"model": "muse-spark-1.3",
		"max_tokens": 50,
		"messages": [{"role": "user", "content": "x"}],
		"n_": 10,
	}
	one = proxy._worst_case_usd(body, json.dumps(body).encode())  # one call's reservation
	proxy.stop()
	proxy = AgentProxy(prov.url, "k", rate=token_rate, max_usd_per_task=one * 3.5)
	proxy.start()
	try:
		from concurrent.futures import ThreadPoolExecutor

		url = proxy.open_task("par")

		def go(i):
			return httpx.post(f"{url}/chat/completions", json={**body, "n_": 10 + i}, timeout=10).status_code

		with ThreadPoolExecutor(20) as pool:
			codes = list(pool.map(go, range(20)))
		assert codes.count(200) <= 3 and len(prov.seen) == codes.count(200)
	finally:
		proxy.stop()


def test_a_model_without_a_price_is_refused_and_the_task_is_not_costed(setup):
	prov, proxy = setup
	r = _chat(proxy, "u1", model="someone-elses-model")
	assert r.status_code == 400 and r.json()["error"]["code"] == "unpriced_model" and not prov.seen
	assert proxy.task("u1").cost_usd is None and proxy.task("u1").errors == 1


def test_a_failure_after_the_stream_started_closes_cleanly_and_keeps_the_usage():

	def handler(request):
		def gen():
			yield b'data: {"model": "muse-spark-1.3", "choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3}}\n\n'
			raise httpx.ReadError("provider went away")

		return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=_Gen(gen()))

	proxy = AgentProxy("http://up/v1", "k", rate=token_rate, transport=httpx.MockTransport(handler))
	proxy.start()
	try:
		with httpx.Client(timeout=5) as c:
			with pytest.raises(httpx.HTTPError):
				_chat(proxy, "s1", client=c, stream=True, stream_options={"include_usage": True}).read()
			call = proxy.calls("s1")[0]
			assert (call.input_tokens, call.output_tokens) == (7, 3) and call.error
			# the same client goes on: the broken connection was closed, not left hanging
			assert _chat(proxy, "s2", client=c).status_code in (200, 502)
	finally:
		proxy.stop()


class _Gen(httpx.SyncByteStream):
	def __init__(self, it):
		self.it = it

	def __iter__(self):
		yield from self.it


def test_a_compressed_reply_reaches_the_script_readable():
	import gzip

	events = b'data: {"model": "m", "choices": [{"delta": {}, "finish_reason": "stop"}]}\n\ndata: {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 1}}\n\ndata: [DONE]\n\n'
	seen = {}

	def handler(request):
		seen["enc"] = request.headers.get("accept-encoding")
		if seen["enc"] == "identity":
			return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=events)
		return httpx.Response(200, headers={"content-type": "text/event-stream", "content-encoding": "gzip"},
			content=gzip.compress(events))  # fmt: skip

	proxy = AgentProxy("http://up/v1", "k", rate=token_rate, transport=httpx.MockTransport(handler))
	proxy.start()
	try:
		r = _chat(proxy, "z1", stream=True)
		assert seen["enc"] == "identity" and r.text.startswith("data: ")
		assert proxy.calls("z1")[0].input_tokens == 5
	finally:
		proxy.stop()


def test_an_early_refusal_doesnt_poison_the_connection(setup):
	_prov, proxy = setup
	tok = _url(proxy, "k1").split("/t/", 1)[1].split("/", 1)[0]
	host, port = proxy._server.server_address[:2]
	with httpx.Client(timeout=5) as c:
		assert (
			c.post(f"http://{host}:{port}/t/{tok}/v1/embeddings", json={"input": "x" * 1000}).status_code
			== 404
		)
		assert _chat(proxy, "k1", client=c).status_code == 200


def test_usage_shapes_are_read_defensively():
	assert read_usage({"prompt_tokens": -3, "completion_tokens": True}) == {
		"input_tokens": 0, "output_tokens": 0, "cached_input_tokens": 0, "reasoning_tokens": 0}  # fmt: skip
	got = read_usage({"prompt_tokens": 10, "prompt_tokens_details": {"cached_tokens": 99}})
	assert got["cached_input_tokens"] == 10  # never more than the input


def test_a_bad_stream_options_doesnt_leak_the_calls_slot(setup):
	_prov, proxy = setup
	_chat(proxy, "b1", stream=True, stream_options=["x"])
	t = proxy._tasks[proxy._tokens[_url(proxy, "b1").split("/t/", 1)[1].split("/", 1)[0]]]
	assert t.in_flight == 0 and t.reserved_usd == 0.0


def test_a_call_that_breaks_before_its_usage_is_charged_its_reservation():
	def handler(request):
		def gen():
			yield b'data: {"model": "muse-spark-1.3", "choices": [{"delta": {"content": "hi"}}]}\n\n'
			raise httpx.ReadError("provider went away")

		return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=_Gen(gen()))

	proxy = AgentProxy("http://up/v1", "k", rate=token_rate, transport=httpx.MockTransport(handler))
	proxy.start()
	try:
		with pytest.raises(httpx.HTTPError):
			_chat(proxy, "x1", stream=True).read()
		call = proxy.calls("x1")[0]
		assert call.usd > 0  # the reservation stands in: the provider may bill what it ran
		assert proxy.task("x1").cost_usd is None  # and the task says it isn't exactly costed
	finally:
		proxy.stop()


def test_a_reply_naming_an_unpriced_model_is_priced_by_the_requested_one():
	def handler(request):
		return httpx.Response(200, json={"model": "some-internal-name", "choices": [{"message": {}, "finish_reason": "stop"}],
			"usage": {"prompt_tokens": 50_000, "completion_tokens": 1000}})  # fmt: skip

	proxy = AgentProxy("http://up/v1", "k", rate=token_rate, transport=httpx.MockTransport(handler))
	proxy.start()
	try:
		assert _chat(proxy, "m1").status_code == 200
		assert proxy.task("m1").cost_usd == pytest.approx(
			token_rate("muse-spark-1.3").usd(50_000, 1000), abs=1e-6
		)
	finally:
		proxy.stop()


def test_crlf_split_across_chunks_doesnt_merge_events():
	from preflight.recorder.agent import _StreamReader

	r = _StreamReader(drop_usage_chunk=True)
	stream = (b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\r\n\r\n'
		b'data: {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 1}}\r\n\r\ndata: [DONE]\r\n\r\n')  # fmt: skip
	cut = stream.index(b"\r\n\r\n") + 1  # between a \r and its \n
	out = r.feed(stream[:cut]) + r.feed(stream[cut:]) + r.rest()
	assert b'"finish_reason": "stop"' in out and b"usage" not in out and b"[DONE]" in out
	assert r.found["finish_reason"] == "stop" and r.found["input_tokens"] == 3


def test_the_reservation_counts_every_choice():
	proxy = AgentProxy("http://up/v1", "k", rate=token_rate)
	base = {"model": "muse-spark-1.3", "max_tokens": 1000, "messages": []}
	one = proxy._worst_case_usd(base, b"{}")
	assert proxy._worst_case_usd({**base, "n": 8}, b"{}") > 7 * one
	proxy.stop()
