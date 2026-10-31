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
