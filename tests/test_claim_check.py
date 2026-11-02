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
	assert [e.metric for e in by["t1"].evidence] == ["claim_check.unsupported"]
	again, _ = annotate_findings(rec.model_copy(update={"findings": out}))  # checked once: kept as it was
	assert {f.id: f for f in again}["t1"] == by["t1"]
