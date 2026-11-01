"""Spark diagnosis (item 31): plots and raw numbers in one call, citations checked by code."""

import json
from pathlib import Path

import httpx
import pytest

pytest.importorskip("matplotlib")

from preflight.harness.diagnosis import diagnose_record, numbers, render, resolve
from preflight.harness.spark import ModelClient, estimate_tokens
from preflight.schema import load_record

GRPO = Path(__file__).resolve().parents[1] / "zoo" / "grpo_trl" / "record_nov1_a10.json"


@pytest.fixture
def rec():
	return load_record(GRPO.read_text())


def client_answering(checks: list[dict], seen: list):
	def handler(request: httpx.Request) -> httpx.Response:
		seen.append(json.loads(request.content))
		body = {"id": "r1", "choices": [{"message": {"content": json.dumps({"checks": checks})}}],
			"usage": {"prompt_tokens": 5000, "completion_tokens": 300}}  # fmt: skip
		return httpx.Response(200, json=body)

	return ModelClient(api_key="k", transport=httpx.MockTransport(handler))


def test_the_plots_are_drawn_from_the_numbers_sent(rec):
	nums = numbers(rec)
	assert list(nums["probes"]) == ["p1"]
	p1 = nums["probes"]["p1"]
	assert len(p1["step_log"]) == 52 and p1["stability"]["series"]["loss"]
	pngs = render(rec)
	assert set(pngs) == {"memory_timeline", "loss"} and all(b.startswith(b"\x89PNG") for b in pngs.values())


def test_a_citation_resolves_only_to_a_number_in_the_record(rec):
	nums = numbers(rec)
	row = nums["probes"]["p1"]["step_log"][2]
	assert resolve(nums, "probes[p1].step_log[2].peak_allocated_bytes") == row["peak_allocated_bytes"]
	assert resolve(nums, "probes[p1].memory.peak_allocated_bytes") == 22293533696
	assert resolve(nums, "probes[p2].memory.peak_allocated_bytes") is None  # no such probe
	assert resolve(nums, "probes[p1].step_log[999].wall_s") is None
	assert resolve(nums, "probes[p1].step_log[2].kind") is None  # not a number
	assert resolve(nums, "__import__('os')") is None


def test_one_call_with_both_plots_and_the_numbers_and_only_checked_findings_kept(rec):
	nums = numbers(rec)
	row = nums["probes"]["p1"]["step_log"][30]
	checks = [
		{"verdict": "disagrees", "plot": "memory_timeline", "plot_region": "steps 28-32, the reserved line",
			"path": "probes[p1].step_log[30].peak_reserved_bytes", "value": row["peak_reserved_bytes"],
			"detail": "The plot shows reserved flat while the number rises."},
		{"verdict": "disagrees", "plot": "loss", "plot_region": "step 7",
			"path": "probes[p1].stability.series.loss[7]", "value": 12345.0, "detail": "a made-up number"},
		{"verdict": "disagrees", "plot": "loss", "plot_region": "step 7",
			"path": "probes[p9].stability.series.loss[7]", "value": 0.1, "detail": "no such probe"},
	]  # fmt: skip
	seen = []
	findings, calls, stats = diagnose_record(client_answering(checks, seen), rec)
	assert len(seen) == 1 and len(calls) == 1 and calls[0].purpose == "diagnosis"
	parts = seen[0]["messages"][1]["content"]
	assert sum(p["type"] == "image_url" for p in parts) == 2
	assert '"peak_allocated_bytes"' in parts[0]["text"]  # the raw numbers go with the plots
	assert stats == {"checks": 3, "kept": 1, "unresolved": 1, "mismatched": 1}
	(f,) = findings
	assert f.source == "spark_diagnosis" and f.severity == "warn" and f.code == "diagnosis.disagrees"
	assert "steps 28-32" in f.detail and "probes[p1].step_log[30].peak_reserved_bytes" in f.detail
	assert f.evidence[0].probe_id == "p1" and f.evidence[0].step == row["index"]


def test_agreement_is_one_consistent_finding(rec):
	nums = numbers(rec)
	checks = [{"verdict": "consistent", "plot": "memory_timeline", "plot_region": "whole run",
		"path": "probes[p1].memory.peak_allocated_bytes", "value": nums["probes"]["p1"]["memory"]["peak_allocated_bytes"],
		"detail": "The allocated line peaks where the probe's peak is."}]  # fmt: skip
	findings, _, _ = diagnose_record(client_answering(checks, []), rec)
	assert [(f.code, f.severity) for f in findings] == [("diagnosis.consistent", "info")]


def test_a_record_with_no_finished_probe_makes_no_call(rec):
	bare = rec.model_copy(update={"probes": [rec.probes[0].model_copy(update={"status": "error"})]})
	seen = []
	assert diagnose_record(client_answering([], seen), bare) == (
		[],
		[],
		{"checks": 0, "kept": 0, "unresolved": 0, "mismatched": 0},
	)
	assert seen == []


def test_an_image_counts_as_an_image_not_its_base64_length():
	msgs = [{"role": "user", "content": [{"type": "text", "text": "x" * 400},
		{"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 400_000}}]}]  # fmt: skip
	assert estimate_tokens(msgs) < 2_000


def test_the_pipeline_diagnoses_only_with_its_flag_on(rec, tmp_path, monkeypatch):
	from preflight.runner import flags
	from preflight.runner.pipeline import Deps, _diagnose
	from preflight.runner.records import LocalRecordWriter

	nums = numbers(rec)
	checks = [{"verdict": "consistent", "plot": "loss", "plot_region": "whole run",
		"path": "probes[p1].stability.series.loss[3]", "value": nums["probes"]["p1"]["stability"]["series"]["loss"][3],
		"detail": "agree"}]  # fmt: skip
	seen = []
	deps = Deps(records=LocalRecordWriter(tmp_path), spark=client_answering(checks, seen))
	monkeypatch.delenv("PREFLIGHT_FLAGS", raising=False)
	flags.reload()
	assert _diagnose(rec, deps) is rec and seen == []  # harness.spark_diagnosis is off by default
	monkeypatch.setenv("PREFLIGHT_FLAGS", json.dumps({"harness.spark_diagnosis": True}))
	flags.reload()
	try:
		out = _diagnose(rec, deps)
		again = _diagnose(out, deps)
	finally:
		monkeypatch.delenv("PREFLIGHT_FLAGS")
		flags.reload()
	assert len(seen) == 2
	assert [f.code for f in out.findings if f.source == "spark_diagnosis"] == ["diagnosis.consistent"]
	assert len([f for f in again.findings if f.source == "spark_diagnosis"]) == 1  # replaced, not stacked
	assert [c.purpose for c in out.provenance][-1] == "diagnosis"


def test_a_huge_or_padded_index_never_raises(rec):
	nums = numbers(rec)
	assert resolve(nums, "probes[p1].step_log[" + "9" * 5000 + "].wall_s") is None
	assert resolve(nums, "probes[p1]" + ".memory" * 50) is None


def test_the_diagnosis_calls_are_in_the_records_cost(rec, tmp_path, monkeypatch):
	from preflight.runner import flags
	from preflight.runner.pipeline import Deps, _diagnose
	from preflight.runner.records import LocalRecordWriter

	nums = numbers(rec)
	checks = [{"verdict": "consistent", "plot": "loss", "plot_region": "all", "path": "probes[p1].steps",
		"value": nums["probes"]["p1"]["steps"], "detail": "agree"}]  # fmt: skip
	deps = Deps(records=LocalRecordWriter(tmp_path), spark=client_answering(checks, []))
	monkeypatch.setenv("PREFLIGHT_FLAGS", json.dumps({"harness.spark_diagnosis": True}))
	flags.reload()
	try:
		out = _diagnose(rec, deps)
	finally:
		monkeypatch.delenv("PREFLIGHT_FLAGS")
		flags.reload()
	call = out.provenance[-1]
	assert call.cost_usd > 0
	assert out.costs.llm_usd == round((rec.costs.llm_usd or 0) + call.cost_usd, 6)
	assert out.costs.breakdown["model_calls"] == round(
		(rec.costs.breakdown or {}).get("model_calls", 0) + call.cost_usd, 6
	)
