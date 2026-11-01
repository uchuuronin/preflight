"""Spark Open checks each diagnosis claim through a tool call (item 30): code decides what stands."""

import json
import time
from pathlib import Path

import httpx
import pytest

from preflight.harness.diagnosis_check import check_diagnosis
from preflight.harness.spark import ModelClient
from preflight.schema import Evidence, Finding, load_record

GRPO = Path(__file__).resolve().parents[1] / "zoo" / "grpo_trl" / "record_nov1_a10.json"
PATH = "probes[p1].memory.peak_allocated_bytes"


@pytest.fixture
def rec():
	return load_record(GRPO.read_text())


def claim(i=1):
	return Finding(id=f"diagnosis.{i}", severity="warn", code="diagnosis.disagrees",
		title="The memory timeline and its numbers disagree", detail="The plot peaks below 22 GB.",
		source="spark_diagnosis", evidence=[Evidence(metric=PATH, value=22293533696.0, probe_id="p1")])  # fmt: skip


def spark_open(script: list, seen: list):
	"""Answers each request with the next scripted assistant message."""
	it = iter(script)

	def handler(request: httpx.Request) -> httpx.Response:
		seen.append(json.loads(request.content))
		msg = next(it)
		return httpx.Response(200, json={"id": f"r{len(seen)}", "choices": [{"message": msg}],
			"usage": {"prompt_tokens": 800, "completion_tokens": 60}})  # fmt: skip

	return ModelClient(api_key="k", model="meta-models/muse-spark-open-1026", provider="baseten",
		transport=httpx.MockTransport(handler))  # fmt: skip


def fetch(path, i="t1"):
	return {"content": "", "tool_calls": [{"id": i, "type": "function",
		"function": {"name": "probe_number", "arguments": json.dumps({"path": path})}}]}  # fmt: skip


def answer(verdict, path):
	return {"content": json.dumps({"verdict": verdict, "path": path, "reason": "r"})}


def test_a_claim_stands_when_the_number_it_cites_was_fetched_through_the_tool(rec):
	seen = []
	(f,), calls = check_diagnosis(spark_open([fetch(PATH), answer("supported", PATH)], seen), rec, [claim()])
	assert f.code == "diagnosis.disagrees" and "22293533696" in f.detail
	assert f.evidence[-1].metric == "spark_open_check" and f.evidence[-1].value == PATH
	assert [c.purpose for c in calls] == ["diagnosis_check", "diagnosis_check"] and calls[
		0
	].provider == "baseten"
	assert seen[0]["tools"][0]["function"]["name"] == "probe_number"
	assert "22293533696" not in json.dumps(seen[0]["messages"])  # the numbers come only through the tool
	tool_reply = seen[1]["messages"][-1]
	assert tool_reply["role"] == "tool" and json.loads(tool_reply["content"])["value"] == 22293533696


@pytest.mark.parametrize("script", [
	[answer("supported", PATH)],  # claims support without fetching anything
	[fetch(PATH), answer("unsupported", None)],
	[fetch("probes[p1].memory.nope"), answer("supported", "probes[p1].memory.nope")],  # fetched, doesn't resolve
	[fetch(PATH), answer("supported", "probes[p1].steps")],  # cites a path it didn't fetch
	[fetch(PATH), {"content": "I think it's fine"}],  # no JSON
	[fetch(PATH, "a"), fetch(PATH, "b"), fetch(PATH, "c"), fetch(PATH, "d")],  # never answers
])  # fmt: skip
def test_anything_short_of_a_fetched_supporting_number_abstains(rec, script):
	(f,), calls = check_diagnosis(spark_open(script, []), rec, [claim()])
	assert f.code == "diagnosis.abstain" and f.severity == "info" and "abstain, or run a probe" in f.title
	assert calls and all(c.purpose == "diagnosis_check" for c in calls)


def test_a_failed_call_abstains_and_stays_in_provenance(rec):
	def handler(request):
		return httpx.Response(400, json={"error": {"message": "bad"}})

	client = ModelClient(api_key="k", provider="baseten", transport=httpx.MockTransport(handler), retries=0)
	(f,), calls = check_diagnosis(client, rec, [claim()])
	assert f.code == "diagnosis.abstain" and len(calls) == 1 and calls[0].ok is False


def test_spark_open_never_sees_the_number_the_diagnosis_cited(rec):
	f = claim().model_copy(update={"detail": "Peaks below 22 GB. (plot: memory_timeline, steps 30-35; number: "
		f"{PATH} = 22293533696)"})  # fmt: skip
	seen = []
	check_diagnosis(spark_open([answer("unsupported", None)], seen), rec, [f])
	sent = json.dumps(seen[0]["messages"])
	assert "steps 30-35" in sent and PATH not in sent and "22293533696" not in sent


def test_a_list_shaped_answer_is_read_not_a_crash(rec):
	parts = {
		"content": [
			{"type": "text", "text": json.dumps({"verdict": "supported", "path": PATH, "reason": "r"})}
		]
	}
	(f,), _ = check_diagnosis(spark_open([fetch(PATH), parts], []), rec, [claim()])
	assert f.code == "diagnosis.disagrees"


def test_the_check_is_bounded_in_claims_and_time(rec, monkeypatch):
	from preflight.harness import diagnosis_check

	seen = []
	script = [m for _ in range(10) for m in (fetch(PATH), answer("supported", PATH))]
	out, calls = check_diagnosis(spark_open(script, seen), rec, [claim(i) for i in range(1, 10)])
	assert len(out) == 9 and len(calls) == 2 * diagnosis_check.MAX_CLAIMS
	assert [f.code for f in out[diagnosis_check.MAX_CLAIMS :]] == ["diagnosis.abstain"] * 3
	monkeypatch.setattr(diagnosis_check, "DEADLINE_S", 0.0)
	out, calls = check_diagnosis(spark_open(script, []), rec, [claim()])
	assert calls == [] and out[0].code == "diagnosis.abstain" and "ran out of time" in out[0].detail


def test_one_real_run_diagnoses_then_checks_inside_its_cap(rec):
	pytest.importorskip("matplotlib")  # the diagnosis renders its plots; CI's Python job has no matplotlib
	from preflight.harness.diagnosis import numbers
	from preflight.harness.diagnosis_run import run_once

	def client_answering(checks, seen):
		def handler(request):
			seen.append(json.loads(request.content))
			body = {"id": "r1", "choices": [{"message": {"content": json.dumps({"checks": checks})}}],
				"usage": {"prompt_tokens": 5000, "completion_tokens": 300}}  # fmt: skip
			return httpx.Response(200, json=body)

		return ModelClient(api_key="k", transport=httpx.MockTransport(handler))

	nums = numbers(rec)
	diag = [{"verdict": "consistent", "plot": "memory_timeline", "plot_region": "whole run", "path": PATH,
		"value": nums["probes"]["p1"]["memory"]["peak_allocated_bytes"], "detail": "agree"}]  # fmt: skip
	out = run_once(
		rec, client_answering(diag, []), spark_open([fetch(PATH), answer("supported", PATH)], []), 0.5
	)
	assert [f["code"] for f in out["checked"]] == ["diagnosis.consistent"]
	assert [c["purpose"] for c in out["calls"]] == ["diagnosis", "diagnosis_check", "diagnosis_check"]
	assert out["usd"] == round(sum(c["cost_usd"] for c in out["calls"]), 6) and out["usd"] < 0.5
	# the cap holds (review, Nov 1): a diagnosis whose worst case passes it never starts
	seen = []
	skipped = run_once(rec, client_answering(diag, seen), spark_open([], []), 0.001)
	assert seen == [] and skipped["usd"] == 0 and "over the $0.00 cap" in skipped["skipped"]
	# and a check call whose worst case won't fit what's left never starts
	pricey = spark_open([fetch(PATH), answer("supported", PATH)], [])
	pricey.prices = {pricey.model: (1_000.0, 1_000.0, 1_000.0)}  # one call's worst case is dollars
	tight = run_once(rec, client_answering(diag, []), pricey, 0.5)
	assert [c["purpose"] for c in tight["calls"]] == ["diagnosis"]
	assert (
		tight["checked"][0]["code"] == "diagnosis.abstain" and "dollar cap" in tight["checked"][0]["detail"]
	)
	assert tight["usd"] <= 0.5


def test_the_glimmer_client_sends_modals_proxy_headers_and_its_reasoning_strength(monkeypatch):
	monkeypatch.setenv("GLIMMER_BASE_URL", "https://example.modal.run")
	monkeypatch.setenv("GLIMMER_MODAL_KEY", "k1")
	monkeypatch.setenv("GLIMMER_MODAL_SECRET", "s1")
	seen = []

	def handler(request):
		seen.append((request.url, dict(request.headers), json.loads(request.content)))
		return httpx.Response(200, json={"id": "g1", "choices": [{"message": {"content": "ok"}}],
			"usage": {"prompt_tokens": 100, "completion_tokens": 10}})  # fmt: skip

	assert ModelClient.glimmer_configured()
	g = ModelClient.glimmer(transport=httpx.MockTransport(handler))
	_, call = g.chat([{"role": "user", "content": "hi"}], purpose="triage", reasoning_effort="low")
	url, headers, body = seen[0]
	assert str(url) == "https://example.modal.run/v1/chat/completions"
	assert headers["modal-key"] == "k1" and headers["modal-secret"] == "s1"
	assert body["chat_template_kwargs"] == {"reasoning_strength": "low"} and "reasoning_effort" not in body
	assert body["max_completion_tokens"] >= 1500 and body["model"] == "muse-glimmer"
	assert call.provider == "modal-glimmer" and 0 <= call.cost_usd < 0.01  # GPU time, not tokens


def test_an_unpriced_check_model_is_never_called_under_a_cap(rec):
	seen = []
	so = spark_open([fetch(PATH), answer("supported", PATH)], seen)
	so.prices = {}
	(f,), calls = check_diagnosis(so, rec, [claim()], budget_usd=0.5)
	assert seen == [] and calls == [] and f.code == "diagnosis.abstain" and "no price" in f.detail


def test_triage_reads_and_can_quote_the_probes_recorded_error_not_just_stdout():
	# Nov 1 production run: the eval OOM was caught by the recorder, so stdout had no traceback
	from preflight.harness.triage import triage_probe
	from preflight.schema import Probe

	oom = "Ran out of GPU memory during the eval pass. OutOfMemoryError: CUDA out of memory. Tried to allocate 530.00 MiB."
	probe = Probe.model_construct(id="p1-eval", status="oom", error_code="oom.eval", steps=3, error=oom)
	seen = []

	def handler(request):
		seen.append(json.loads(request.content))
		ans = {"category": "oom", "summary": "the eval pass ran out of GPU memory",
			"evidence_line": "Ran out of GPU memory during the eval pass. OutOfMemoryError: CUDA out of memory. Tried to allocate 530.00 MiB."}  # fmt: skip
		return httpx.Response(200, json={"id": "t1", "choices": [{"message": {"content": json.dumps(ans)}}],
			"usage": {"prompt_tokens": 300, "completion_tokens": 50}})  # fmt: skip

	client = ModelClient(api_key="k", model="meta-models/muse-spark-open-1026", provider="baseten",
		transport=httpx.MockTransport(handler))  # fmt: skip
	f, _calls = triage_probe(client, probe, "{'loss': 1.5, 'epoch': 0.0003}\n", node_id="eval")
	assert "CUDA out of memory" in seen[0]["messages"][1]["content"]
	assert (
		f.code == "triage.oom"
		and f.node_id == "eval"
		and "CUDA out of memory" in f.detail
		and "no log line quoted" not in f.detail
	)
	assert f.title.startswith("Spark Open on probe p1-eval")
	# a private job's triage names the model that made it (Glimmer, 0.6.2)
	glim = ModelClient(api_key="k", model="muse-glimmer", provider="modal-glimmer",
		transport=httpx.MockTransport(handler))  # fmt: skip
	g, _ = triage_probe(glim, probe, "")
	assert g.title.startswith("Muse Glimmer on probe p1-eval") and g.code == "triage.oom"


def test_a_reply_without_a_completion_is_still_a_logged_priced_call_with_its_reason():
	# Glimmer's first triage (Nov 1): 150 s, no finding, and no call in provenance
	def handler(request):
		time.sleep(0.01)
		return httpx.Response(200, text="<html>cold start at https://secret-host.example/x</html>")

	client = ModelClient(api_key="k", model="muse-glimmer", provider="modal-glimmer", usd_per_second=0.001,
		transport=httpx.MockTransport(handler))  # fmt: skip
	with pytest.raises(ValueError):
		client.chat([{"role": "user", "content": "hi"}], purpose="triage")
	assert len(client.calls) == 1 and not client.calls[0].ok
	assert "HTTP 200 without a completion" in client.last_error and "cold start" in client.last_error
	assert "secret-host" not in client.last_error and client.calls[0].cost_usd > 0  # priced by wall time

	def denied(request):
		return httpx.Response(401, json={"error": "bad Modal-Key"})

	c2 = ModelClient(api_key="k", base_url="https://secret-host.example/v1", model="m", provider="modal-glimmer",
		transport=httpx.MockTransport(denied))  # fmt: skip
	with pytest.raises(httpx.HTTPStatusError):
		c2.chat([{"role": "user", "content": "hi"}], purpose="triage")
	assert c2.last_error.startswith("HTTP 401") and "bad Modal-Key" in c2.last_error

	def down(request):
		raise httpx.ConnectError("can't reach https://secret-host.example/v1/chat/completions")

	c3 = ModelClient(
		api_key="k", model="m", provider="modal-glimmer", retries=0, transport=httpx.MockTransport(down)
	)
	with pytest.raises(httpx.ConnectError):
		c3.chat([{"role": "user", "content": "hi"}], purpose="triage")
	assert c3.last_error.startswith("ConnectError") and "secret-host" not in c3.last_error
