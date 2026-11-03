"""harness/claim_check.py: numbers in a text against a record's raw probe numbers."""

from pathlib import Path

from preflight.harness.claim_check import _Picks, check_findings, check_text, numbers_in
from preflight.schema import load_record

ROOT = Path(__file__).resolve().parents[1]


def _rec():
	return load_record((ROOT / "schema" / "examples" / "finetune.json").read_text())


def test_numbers_carry_units_and_the_precision_they_were_written_at():
	ns = {
		n.text: n
		for n in numbers_in("peak 21.9 GB (20.4 GiB) in 189 s, $0.042, 13% padding, 1,024 tokens, 2 seeds")
	}
	assert ns["21.9 GB"].kind == "bytes" and ns["21.9 GB"].value == 21.9e9
	assert ns["20.4 GiB"].value == 20.4 * 2**30 and ns["189 s"].kind == "seconds"
	assert ns["$0.042"].kind == "usd" and ns["13%"].value == 0.13 and ns["1,024"].value == 1024
	assert "2" not in ns  # a small bare count is words, not a measurement


def test_a_number_the_probes_show_is_supported_and_one_they_dont_is_flagged():
	rec = _rec()
	p = rec.probes[0]
	peak = p.memory.peak_allocated_bytes
	gb = f"{peak / 1e9:.1f} GB"
	checks, calls = check_text(f"The run peaked at {gb}, and at 999.9 GB in eval.", rec)
	by = {c.number: c for c in checks}
	assert by[gb].verdict in ("supported", "ambiguous") and calls == []
	assert by["999.9 GB"].verdict == "unsupported"


class Fake:
	def __init__(self, field):
		self.field, self.seen = field, None

	def structured(self, msgs, schema, *, purpose):
		self.seen = msgs
		text = msgs[1]["content"]
		number = text.split("\n- ", 1)[1].split(":", 1)[0]
		return _Picks.model_validate(
			{"picks": [{"number": number, "field": self.field, "not_a_measurement": False}]}
		), "call"


def test_the_model_only_chooses_among_the_fields_the_code_found():
	rec = _rec()
	p = rec.probes[0]
	# a value two fields share, so the code can't decide alone
	rec.probes[0] = p.model_copy(
		update={"extras": {**p.extras, "tokens_per_step": 4096.0, "samples_per_step": 4096.0}}
	)
	ok, calls = check_text("4096 tokens a step", rec, client=Fake(f"probes.{p.id}.extras.tokens_per_step"))
	assert ok[0].verdict == "supported" and ok[0].field.endswith("tokens_per_step") and calls == ["call"]
	made_up, _ = check_text("4096 tokens a step", rec, client=Fake("probes.p9.invented"))
	assert made_up[0].verdict == "ambiguous"  # the model's pick wasn't a candidate: not supported


def test_findings_are_checked_by_id():
	rec = _rec()
	out, _ = check_findings(rec)
	assert all(isinstance(k, str) for k in out)


def test_numbers_inside_names_and_versions_are_not_claims():
	texts = [
		n.text
		for n in numbers_in(
			"H100 in bf16 with Qwen2.5-1.5B on CUDA 12.4 took 1.5 hr, -12% and 4,096 tokens/s"
		)
	]
	assert texts == ["1.5 hr", "-12%", "4,096"]
	ns = numbers_in("1.5 hr, 4,096 tokens/s, 0%")
	assert ns[0].kind == "seconds" and ns[0].value == 5400 and ns[1].kind == "rate" and ns[2].value == 0


def test_phase_shares_and_rates_have_their_own_kinds():
	from preflight.harness.claim_check import _kind_of

	assert _kind_of("probes.p1.step_time.by_phase.optimizer") == "fraction"
	assert _kind_of("probes.p1.throughput.tokens_per_s") == "rate"
	assert _kind_of("probes.p1.extras.encode_items_per_s") == "rate"
	assert _kind_of("probes.p1.memory.peak_allocated_bytes") == "bytes"
	assert _kind_of("probes.p1.step_time.p50_s") == "seconds"


def test_knob_values_never_back_a_claim():
	from preflight.harness.claim_check import raw_numbers

	assert not any(".knobs." in k for k in raw_numbers(_rec()))


def test_a_number_ending_a_sentence_keeps_its_unit_and_dev_is_not_a_version():
	ns = numbers_in("It took 189 s. Peak was 21.9 GB. Padding is 13%. On dev 12 GB.")
	assert [(n.text, n.kind) for n in ns] == [("189 s", "seconds"), ("21.9 GB", "bytes"), ("13%", "fraction"),
		("12 GB", "bytes")]  # fmt: skip
	from preflight.harness.claim_check import _kind_of

	assert _kind_of("probes.p1.generation.truncation_rate") == "fraction"
	assert _kind_of("probes.p1.throughput.tokens_per_s") == "rate"


def test_a_zero_reads_true_at_the_precision_written():
	from preflight.harness.claim_check import candidates

	(n,) = numbers_in("loader waits 0.0% of the step")
	raw = {"probes.p1.data.loader_wait_fraction": 0.0004, "probes.p1.data.other_fraction": 0.002}
	assert candidates(n, raw) == ["probes.p1.data.loader_wait_fraction"]  # 0.04% rounds to 0.0%; 0.2% doesn't


def test_annotate_marks_an_unsupported_number_and_leaves_rule_findings_alone():
	from preflight.harness.claim_check import UNSUPPORTED, annotate_findings
	from preflight.schema import Finding

	rec = _rec()
	model = Finding(id="t1", severity="warn", code="triage.oom", title="Eval ran out at 999.9 GB",
		source="spark_open_triage")  # fmt: skip
	rule = Finding(
		id="r1", severity="warn", code="eval.no_grad", title="Eval keeps 999.9 GB of grads", source="rule"
	)
	rec = rec.model_copy(update={"findings": [model, rule]})
	out, calls = annotate_findings(rec)
	by = {f.id: f for f in out}
	assert calls == [] and by["r1"] == rule
	assert by["t1"].detail == f"(999.9 GB: {UNSUPPORTED})"
	assert [e.metric for e in by["t1"].evidence] == ["claim_check.checked", "claim_check.unsupported"]
	again, _ = annotate_findings(rec.model_copy(update={"findings": out}))  # checked once: kept as it was
	assert {f.id: f for f in again}["t1"] == by["t1"]


# ---------------------------------------------------------------- privacy and the pipeline (A's review, Nov 2)


class _Picky:
	"""A Spark Open stand-in: records what it was sent, bills every attempt, the first one failing."""

	def __init__(self):
		self.calls, self.sent = [], []

	def structured(self, msgs, schema, **kw):
		from datetime import UTC, datetime

		from preflight.schema import ModelCall

		self.sent.append(msgs[-1]["content"])
		for ok in (False, True):  # a failed first answer, then the re-ask: both billed
			self.calls.append(ModelCall(provider="baseten", model="m", purpose="claim_check", cost_usd=0.01,
				at=datetime.now(UTC), ok=ok))  # fmt: skip
		return _Picks(picks=[]), self.calls[-1]


def _twins(rec):
	"""The record with probe p1's reserved peak equal to its allocated one: one number, two fields."""
	p = rec.probes[0]
	mem = p.memory.model_copy(update={"peak_reserved_bytes": p.memory.peak_allocated_bytes})
	return rec.model_copy(update={"probes": [p.model_copy(update={"memory": mem}), *rec.probes[1:]]})


def _ambiguous_finding(rec, *, private=False):
	from preflight.schema import Evidence, Finding

	p = rec.probes[0]
	num = f"{p.memory.peak_allocated_bytes / 1e9:.1f} GB"  # matches both twins, so Spark Open is asked
	ev = [Evidence(metric="private_job", value=1.0)] if private else []
	return Finding(id="t9", severity="info", code="triage.oom", title=f"Probe peaked near {num}",
		detail='Out of memory. (log: "Tried to allocate 79.15 GiB")', source="spark_open_triage", evidence=ev)  # fmt: skip


def test_a_quoted_log_line_is_not_a_claim_and_every_billed_attempt_is_kept():
	from preflight.harness.claim_check import annotate_findings

	rec = _twins(_rec())
	rec = rec.model_copy(update={"findings": [_ambiguous_finding(rec)]})
	client = _Picky()
	out, calls = annotate_findings(rec, client=client)
	(f,) = out
	assert "79.15 GiB" not in " ".join(client.sent) and "unsupported" not in (f.detail or "")
	assert len(calls) == 2 and sum(c.cost_usd for c in calls) == 0.02  # the failed first answer too


def test_a_private_jobs_finding_never_reaches_a_hosted_model_on_any_later_job():
	from preflight.harness.claim_check import annotate_findings

	rec = _twins(_rec())
	rec = rec.model_copy(update={"findings": [_ambiguous_finding(rec, private=True)]})
	client = _Picky()
	out, calls = annotate_findings(rec, client=client)  # a later, non-private job on the same record
	assert client.sent == [] and calls == []
	assert any(e.metric == "claim_check.checked" and e.value == "code only" for e in out[0].evidence)
	again, _ = annotate_findings(rec.model_copy(update={"findings": out}), client=client)
	assert client.sent == [] and again == out  # checked once, never sent


def test_the_pipeline_keeps_a_record_with_private_history_off_meta_and_baseten(monkeypatch):
	import json

	from preflight.runner import flags, pipeline

	rec = _twins(_rec())
	marked = rec.model_copy(update={"probes": [rec.probes[0].model_copy(update={"extras": {"private_job": 1.0}}),
		*rec.probes[1:]], "findings": [_ambiguous_finding(rec)]})  # fmt: skip
	assert pipeline._private_history(marked) and not pipeline._private_history(rec)
	client = _Picky()

	class Deps:
		triage = client
		spark = None

	monkeypatch.setenv("PREFLIGHT_FLAGS", json.dumps({"harness.spark_claim_check": True}))
	flags.reload()
	try:
		out = pipeline._claim_check(marked, Deps, private=pipeline._private_history(marked))
		assert client.sent == [] and out.costs == marked.costs
		on = pipeline._claim_check(
			rec.model_copy(update={"findings": [_ambiguous_finding(rec)]}), Deps, private=False
		)
		assert client.sent and on.costs.llm_usd == round((rec.costs.llm_usd or 0) + 0.02, 6)
	finally:
		monkeypatch.delenv("PREFLIGHT_FLAGS")
		flags.reload()
	off = pipeline._claim_check(rec, Deps, private=False)  # flag off: unchanged
	assert off is rec


def test_a_jobs_spend_stops_at_its_cap_and_the_rest_are_code_only():
	from preflight.harness.claim_check import annotate_findings

	rec = _twins(_rec())
	f1 = _ambiguous_finding(rec)
	f2 = f1.model_copy(update={"id": "t10"})
	rec = rec.model_copy(update={"findings": [f1, f2]})
	client = _Picky()  # $0.02 a finding
	out, calls = annotate_findings(rec, client=client, cap_usd=0.02)
	assert len(client.sent) == 1 and sum(c.cost_usd for c in calls) == 0.02
	assert [e.value for f in out for e in f.evidence if e.metric == "claim_check.checked"] == [
		"Spark Open picked",
		"code only",
	]


def test_a_diagnosis_is_checked_by_code_only_and_an_earlier_note_is_not_a_claim():
	from preflight.harness.claim_check import UNSUPPORTED, annotate_findings

	rec = _twins(_rec())
	diag = _ambiguous_finding(rec).model_copy(update={"source": "spark_diagnosis"})
	noted = _ambiguous_finding(rec).model_copy(
		update={"id": "t11", "detail": f"Out of memory. (999.9 GB: {UNSUPPORTED})"}
	)
	rec = rec.model_copy(update={"findings": [diag, noted]})
	client = _Picky()
	out, _ = annotate_findings(rec, client=client)
	by = {f.id: f for f in out}
	assert len(client.sent) == 1 and "999.9" not in client.sent[0]  # only t11 is sent, without the old note
	assert any(e.value == "code only" for e in by["t9"].evidence if e.metric == "claim_check.checked")
	assert "999.9 GB" not in (by["t11"].detail or "")


def test_the_pipeline_keeps_the_record_when_the_check_raises(monkeypatch):
	import json

	import preflight.harness.claim_check as cc
	from preflight.runner import flags, pipeline

	rec = _twins(_rec())
	rec = rec.model_copy(update={"findings": [_ambiguous_finding(rec)]})

	def boom(*a, **k):
		raise RuntimeError("a bug in the check")

	class Deps:
		triage = _Picky()
		spark = None

	monkeypatch.setattr(cc, "annotate_findings", boom)
	monkeypatch.setenv("PREFLIGHT_FLAGS", json.dumps({"harness.spark_claim_check": True}))
	flags.reload()
	try:
		assert pipeline._claim_check(rec, Deps, private=False) is rec
	finally:
		monkeypatch.delenv("PREFLIGHT_FLAGS")
		flags.reload()


def test_baseten_down_bills_what_it_billed_and_the_finding_reads_code_only(monkeypatch):
	import json
	from datetime import UTC, datetime

	from preflight.runner import flags, pipeline
	from preflight.schema import ModelCall

	rec = _twins(_rec())
	rec = rec.model_copy(update={"findings": [_ambiguous_finding(rec)]})

	class Broken:
		def __init__(self):
			self.calls = []

		def structured(self, *a, **k):
			self.calls.append(ModelCall(provider="baseten", model="m", purpose="claim_check", cost_usd=0.01,
				at=datetime.now(UTC), ok=False))  # fmt: skip
			raise RuntimeError("Baseten down")

	class Deps:
		triage = Broken()
		spark = None

	monkeypatch.setenv("PREFLIGHT_FLAGS", json.dumps({"harness.spark_claim_check": True}))
	flags.reload()
	try:
		out = pipeline._claim_check(rec, Deps, private=False)
	finally:
		monkeypatch.delenv("PREFLIGHT_FLAGS")
		flags.reload()
	assert out is not rec and out.costs.llm_usd == round((rec.costs.llm_usd or 0) + 0.01, 6)
	assert any(c.purpose == "claim_check" and not c.ok for c in out.provenance)
	assert [e.value for e in out.findings[0].evidence if e.metric == "claim_check.checked"] == ["code only"]
