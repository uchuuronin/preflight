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
