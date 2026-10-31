"""Step time scales with the tokens a step really processes (A's probes record them), not the seq_len setting.

The record is A's L4 smoke run (remote-runs a-smoke-l4-main-4, Oct 28; C's coverage fixture): its rows
never reach 512 tokens, so the seq_len 512 probe did the same work as the 1024 one.
"""

from pathlib import Path

import pytest

from preflight.engine import api, conformal
from preflight.schema import load_record

REC = Path(__file__).parent / "fixtures" / "b-a-smoke-l4-main-4.record.json"


@pytest.fixture(scope="module")
def smoke():
	return load_record(REC.read_text())


def _without(rec, pid):
	return rec.model_copy(
		update={"probes": [p for p in rec.probes if p.id != pid and not p.id.startswith(pid + "-")]}
	)


def test_a_seq_len_above_the_rows_changes_no_work(smoke):
	short = api._step_work(smoke, api._values(smoke, {"seq_len": 512}))
	long = api._step_work(smoke, api._values(smoke, {"seq_len": 1024}))
	assert short == pytest.approx(long)  # 16 samples x ~295 padded tokens either way
	assert short == pytest.approx(4721.6, rel=0.01)
	tiny = api._step_work(smoke, api._values(smoke, {"seq_len": 128}))
	assert tiny == pytest.approx(16 * 128)  # a limit below the rows does cut the work


def test_held_out_step_time_on_the_real_record(smoke):
	"""C's coverage check, leave-one-probe-out: a sanity check on three probes, not a score."""
	hits, rows = 0, []
	for p in [p for p in smoke.probes if p.status == "ok"]:
		pred = api.predict(_without(smoke, p.id), api._probe_values(smoke, p)).root["time.step"]
		err = abs(pred.point / p.step_time.p50_s - 1)
		rows.append((p.id, round(err, 3), pred.lo <= p.step_time.p50_s <= pred.hi))
		hits += err <= 0.15
	lo, hi = conformal.wilson(hits, len(rows))
	print(rows, hits, len(rows), lo, hi)
	assert hits >= 2  # p1 and p3 are read off the other batch-4 probe; p2 (batch 2) extrapolates the padding


def test_records_without_measured_tokens_keep_the_setting(smoke):
	probes = [
		p.model_copy(update={"extras": {k: v for k, v in (p.extras or {}).items() if k != "tokens_per_step"}})
		for p in smoke.probes
	]
	rec = smoke.model_copy(update={"probes": probes})
	assert api._step_work(rec, api._values(rec, {"seq_len": 512})) < api._step_work(
		rec, api._values(rec, {"seq_len": 1024})
	)


def _drop_knob(rec, kid):
	return rec.model_copy(update={"knobs": [k for k in rec.knobs if k.id != kid]})


def test_accumulation_comes_from_the_probes_when_the_script_declares_none(smoke):
	rec = _drop_knob(smoke, "grad_accum")  # the script still accumulates 4 micro-steps
	assert api._step_work(rec, api._values(rec, {"batch_size": 8})) == pytest.approx(
		api._step_work(smoke, api._values(smoke, {"batch_size": 8}))
	)
	t = api.predict(rec, {"batch_size": 8}).root["time.step"].point
	assert t > 2.5  # twice the batch-4 work, not a quarter of it


def test_both_sides_leave_the_token_axis_together(smoke):
	for rec in (
		_drop_knob(smoke, "batch_size"),
		smoke.model_copy(update={"job": smoke.job.model_copy(update={"lens": "rl"})}),
	):
		assert not api._on_token_axis(rec)
		p = next(p for p in rec.probes if p.id == "p1")
		assert api._probe_work(rec, p) == pytest.approx(api._step_x(rec.job.lens, api._probe_values(rec, p)))


@pytest.mark.parametrize("bad", [float("inf"), float("nan"), True, 0, -5])
def test_unusable_token_counts_are_refused(smoke, bad):
	p = smoke.probes[1].model_copy(update={"extras": {**smoke.probes[1].extras, "tokens_per_step": bad}})
	assert api._measured_tokens(p) is None


def test_a_declared_accumulation_moves_the_measured_one_by_its_ratio(smoke):
	base = api._step_work(smoke, api._values(smoke, {}))
	assert api._step_work(smoke, api._values(smoke, {"grad_accum": 8})) == pytest.approx(2 * base)
	# the knob says 1 but the probes ran 4 micro-steps (accumulation set elsewhere): the measured scale wins
	knobs = [k.model_copy(update={"current": 1}) if k.id == "grad_accum" else k for k in smoke.knobs]
	off = smoke.model_copy(update={"knobs": knobs})
	assert api._step_work(off, api._values(off, {})) == pytest.approx(base)


def test_probes_are_fitted_on_their_rows_but_queries_keep_seq_len(smoke):
	p1 = next(p for p in smoke.probes if p.id == "p1")
	p3 = next(p for p in smoke.probes if p.id == "p3")
	assert api._padded_seq(smoke, {}, probe=p1) == 517  # its warm-up's longest row
	assert api._padded_seq(smoke, {}, probe=p3) == 512  # a row reached its limit
	assert (
		api._padded_seq(smoke, api._values(smoke, {"seq_len": 1024})) is None
	)  # a full run may meet longer rows
	short = api.predict(smoke, {"seq_len": 512}).root["memory.peak"].point
	long = api.predict(smoke, {"seq_len": 768}).root["memory.peak"].point
	assert long > short  # sized by the setting: never "fits" for rows the probes happened not to see


def test_held_out_memory_errs_high_never_low(smoke):
	"""C's memory check, leave-one-probe-out, each probe predicted as it ran. Queries are sized by seq_len, so on
	data shorter than the limit the engine over-predicts; it must not under-predict."""
	from preflight.engine.memory import DEFAULT_CONTEXT_BYTES

	for p in [p for p in smoke.probes if p.status == "ok"]:
		pk = api.predict_as_run(_without(smoke, p.id), p).root["memory.peak"]
		assert pk.hi - DEFAULT_CONTEXT_BYTES >= p.memory.peak_reserved_bytes


def test_one_rule_for_whether_a_probe_ran_eval(smoke):
	p = next(p for p in smoke.probes if p.id == "p1")
	assert api.probe_ran_eval(p) is (p.eval is not None)  # no eval_off recorded: the eval stats decide
	on = p.model_copy(update={"extras": {**p.extras, "eval_off": False}})
	off = p.model_copy(update={"extras": {**p.extras, "eval_off": True}, "eval": None})
	assert api.probe_ran_eval(on) and not api.probe_ran_eval(off)


@pytest.mark.parametrize(("stored", "ran"), [(1.0, False), (0.0, True), (1, False), (0, True), (True, False)])
def test_eval_off_as_a_loaded_record_stores_it(smoke, stored, ran):
	"""A's pipeline marks an eval-off re-run with extras["eval_off"] = 1.0 (runner/pipeline.py _marked_eval_off),
	and records store extras as numbers, so the marker arrives as 1.0 or 0.0 after a round trip, not a bool."""
	from preflight.schema import dump_record

	probes = [
		p.model_copy(update={"extras": {**p.extras, "eval_off": stored}}) if p.id == "p1" else p
		for p in smoke.probes
	]
	rec = load_record(dump_record(smoke.model_copy(update={"probes": probes})))
	p1 = next(p for p in rec.probes if p.id == "p1")
	assert api.probe_ran_eval(p1) is ran
