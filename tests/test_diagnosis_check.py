"""Spark Open checks each diagnosis claim through a tool call (item 30): code decides what stands."""

import json
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
