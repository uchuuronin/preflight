"""scripts/spark_label_all.py: batches map back to their rows, the budget guard holds, and the score compares."""

import gzip
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("spark_label_all", ROOT / "scripts/spark_label_all.py")
sla = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sla)


def _items():
	mem = [
		{
			"row_id": f"kernels/L4/k{i}",
			"task": "memory",
			"status": "ok",
			"evidence": {"i": i},
			"rule": "activations",
		}
		for i in range(10)
	]
	div = [
		{
			"row_id": f"divergence/L4/d{i}",
			"task": "divergence",
			"status": "stable",
			"evidence": {"i": i},
			"rule": "stable",
		}
		for i in range(5)
	]
	return mem + div


class FakeClient:
	"""Answers every run id it is shown, except the last of each batch, and bills a fixed amount per call."""

	prices: ClassVar[dict] = {sla.MODEL: (1.25, 0.15, 4.25)}
	model = sla.MODEL

	def __init__(self, **kw):
		self.calls = []

	def structured(self, messages, schema, *, purpose, reasoning_effort):
		user = messages[-1]["content"]
		ids = [line.split(":", 1)[0] for line in user.split("Runs:\n", 1)[1].splitlines()]
		task_mem = "peak GPU memory" in user
		lab = "activations" if task_mem else "stable"
		ans = schema(labels=[{"run": r, "label": lab, "confidence": "high", "why": "x"} for r in ids[:-1]])
		self.calls.append(
			SimpleNamespace(
				model_dump=lambda mode=None: {
					"ok": True,
					"cost_usd": 0.01,
					"input_tokens": 100,
					"output_tokens": 50,
				}
			)
		)
		return ans, None


@pytest.fixture
def staged(tmp_path, monkeypatch):
	inputs = tmp_path / "inputs.jsonl.gz"
	with gzip.open(inputs, "wt") as fh:
		fh.write("".join(json.dumps(i) + "\n" for i in _items()))
	monkeypatch.setattr(sla, "INPUTS", inputs)
	monkeypatch.setattr(sla, "REPORT", tmp_path / "report.json")
	monkeypatch.setattr(sla, "PROVENANCE", tmp_path / "prov.jsonl")
	monkeypatch.setattr(sla, "FIRST_ROWS", tmp_path / "none.jsonl")
	from preflight.harness import spark

	monkeypatch.setattr(
		spark.ModelClient, "baseten_spark_open", classmethod(lambda cls, **kw: FakeClient(**kw))
	)
	return tmp_path


def test_batches_keep_one_kind_and_map_answers_back_to_their_rows(staged):
	chunks = sla._chunks(_items(), 4)
	assert sorted(t for t, _ in chunks) == ["divergence"] * 2 + ["memory"] * 3
	assert (
		chunks[0][0] != chunks[1][0] or chunks[1][0] != chunks[2][0]
	)  # interleaved, not one kind after the other
	sla.label(str(staged / "out"), None, batch=4, workers=2, budget=5.0)
	rows = [json.loads(x) for x in (staged / "out/spark_all_labels.jsonl").read_text().splitlines()]
	assert sorted(r["row_id"] for r in rows) == sorted(i["row_id"] for i in _items())
	# the fake leaves the last run of each batch unanswered: those rows are unlabelled, never shifted onto another
	unanswered = {r["row_id"] for r in rows if r["label"] is None}
	assert unanswered == {
		"kernels/L4/k3",
		"kernels/L4/k7",
		"kernels/L4/k9",
		"divergence/L4/d3",
		"divergence/L4/d4",
	}
	sla.score(str(staged / "out/spark_all_labels.jsonl"), str(staged / "out/spark_all_calls.jsonl"), "test")
	rep = json.loads((staged / "report.json").read_text())
	assert rep["memory"]["all"]["agree"] == rep["memory"]["all"]["compared"] == 7
	assert rep["divergence"]["diverged_or_not"]["compared"] == 3 and rep["cost_usd"] == pytest.approx(0.05)


def test_the_guard_counts_every_call_in_flight_at_its_worst(staged):
	# one call's worst case is $0.114 at batch 4 (twice the output cap and the prompt): with $0.13, a call starts
	# only while what is spent plus its worst case fits, so two start ($0 and $0.01 spent) and the third doesn't
	sla.label(str(staged / "out"), None, batch=4, workers=1, budget=0.13)
	calls = (staged / "out/spark_all_calls.jsonl").read_text().splitlines()
	assert len(calls) == 2
	with pytest.raises(SystemExit):
		sla.main(["label", "--budget", "12"])


class SlowClient(FakeClient):
	def structured(self, *a, **kw):
		import time

		time.sleep(0.05)
		return super().structured(*a, **kw)


def test_the_guard_waits_for_calls_in_flight_instead_of_dropping_chunks(staged, monkeypatch):
	"""Review, Nov 1: with room for two calls at once, six workers dropped the other chunks at the start."""
	from preflight.harness import spark

	monkeypatch.setattr(
		spark.ModelClient, "baseten_spark_open", classmethod(lambda cls, **kw: SlowClient(**kw))
	)
	sla.label(str(staged / "out"), None, batch=4, workers=6, budget=0.30)
	calls = (staged / "out/spark_all_calls.jsonl").read_text().splitlines()
	assert len(calls) == 5  # every chunk ran: $0.05 spent leaves room for two at a time throughout


def test_a_timed_out_call_counts_at_one_generation_s_worst(staged, monkeypatch):
	from preflight.harness import spark

	class TimingOut(FakeClient):
		def structured(self, *a, **kw):
			self.calls.append(
				SimpleNamespace(
					model_dump=lambda mode=None: {
						"ok": False,
						"cost_usd": 0.0,
						"input_tokens": 0,
						"output_tokens": 0,
					}
				)
			)
			raise TimeoutError("read timeout")

	monkeypatch.setattr(
		spark.ModelClient, "baseten_spark_open", classmethod(lambda cls, **kw: TimingOut(**kw))
	)
	sla.label(str(staged / "out"), None, batch=4, workers=1, budget=5.0)
	summary = json.loads((staged / "out/spark_all_summary.json").read_text())
	assert summary["counted_usd"] == pytest.approx(5 * summary["worst_call_usd"] / 2, abs=1e-3)


def test_the_pilot_takes_both_kinds_and_refuses_too_few_rows():
	items = _items()
	got = sla._pilot(items, 4)
	assert len(got) == 4 and {i["task"] for i in got} == {"memory", "divergence"}
	for bad in (0, 1):
		with pytest.raises(SystemExit):
			sla._pilot(items, bad)
	assert sla._run_id(" R3 ") == sla._run_id("3") == "r3"


def test_a_worker_that_fails_outside_the_call_releases_its_place(staged, monkeypatch):
	"""Re-check, Nov 1: a client that can't be built left in_flight raised, and every waiting worker hung."""
	from preflight.harness import spark

	built = {"n": 0}

	def make(cls, **kw):
		built["n"] += 1
		if built["n"] == 2:  # the first worker's own client (the first build is the model check)
			raise RuntimeError("no client")
		return FakeClient(**kw)

	monkeypatch.setattr(spark.ModelClient, "baseten_spark_open", classmethod(make))
	with pytest.raises(SystemExit):
		sla.label(str(staged / "out"), None, batch=4, workers=2, budget=0.30)
	summary = json.loads((staged / "out/spark_all_summary.json").read_text())
	assert "no client" in summary["worker_error"] and summary["counted_usd"] >= summary["worst_call_usd"]
