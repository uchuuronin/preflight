"""engine/api.py against the golden records: predict, build_grids, sensitivity, fill_predictions."""

import gzip
import json
import math
import time
from pathlib import Path

import numpy as np
import pytest

from preflight.engine import api, gpus
from preflight.schema import FlightRecord, load_record
from preflight.schema.encoding import decode_mask
from preflight.schema.prediction import Objective

EX = Path(__file__).resolve().parents[1] / "schema" / "examples"


def _rec(name: str) -> FlightRecord:
	return load_record((EX / f"{name}.json").read_text())


@pytest.fixture(scope="module")
def ft():
	return _rec("finetune")


@pytest.fixture(scope="module")
def pt():
	return _rec("pretrain")


@pytest.fixture(scope="module")
def ag():
	return _rec("agent")


def _agent_with(ag, n: int):
	base = ag.agent_sample.tasks
	tasks = [base[i % len(base)].model_copy(update={"task_id": f"x{i}"}) for i in range(n)]
	return ag.model_copy(update={"agent_sample": ag.agent_sample.model_copy(update={"tasks": tasks})})


# ---------------------------------------------------------------- predict


@pytest.mark.parametrize("name", ["finetune", "pretrain", "agent"])
def test_every_lens_target_comes_back_valid(name):
	rec = _rec(name)
	pred = api.predict(rec).root
	assert {t.id for t in api.targets_for(rec)} <= set(pred)
	for tid, e in pred.items():
		assert e.unit == api._TARGETS[tid][0]
		if e.status == "abstain":
			assert e.reason
		else:
			assert e.lo <= e.point <= e.hi and all(math.isfinite(v) for v in (e.point, e.lo, e.hi))


def test_predict_is_pure_and_fast(ft):
	before = ft.model_dump_json()
	t = time.perf_counter()
	for _ in range(20):
		api.predict(ft, {"batch_size": 2})
	assert (time.perf_counter() - t) / 20 < 0.05  # the knob-response budget is 400 ms end to end
	assert ft.model_dump_json() == before


def test_unknown_knobs_are_refused_but_gpu_what_ifs_are_not(ft):
	with pytest.raises(ValueError, match="unknown knob"):
		api.predict(ft, {"not_a_knob": 3})
	assert api.predict(ft, {"gpu_type": "H100-SXM-80GB"}).root["time.step"].status == "predicted"


def test_measured_only_when_a_probe_saw_the_peak(ft, pt):
	assert api.predict(pt).root["memory.peak"].status == "measured"  # p2 ran these knobs
	e = api.predict(ft).root["memory.peak"]  # p2 matches, but the peak is the eval no probe ran
	assert e.status == "extrapolated" and "no probe ran" in e.reason


def test_eval_settings_move_the_oom_risk(ft):
	bad = api.predict(ft).root["risk.oom"]
	good = api.predict(ft, {"eval_batch_size": 1, "eval_no_grad": True}).root["risk.oom"]
	assert bad.point >= 0.8 and good.point < bad.point
	assert "eval" in bad.reason


def test_below_and_above_the_probes(ft):
	low = api.predict(ft, {"batch_size": 1, "eval_batch_size": 1, "eval_no_grad": True}).root
	assert low["time.step"].status == "extrapolated" and "run a probe here" in low["time.step"].reason
	high = api.predict(ft, {"batch_size": 64}).root
	assert high["time.step"].status == "abstain" and high["memory.peak"].status == "abstain"


def test_precision_change_scales_time_with_tf32_in_mind(pt):
	base = api.predict(pt).root["time.step"].point
	bf16 = api.predict(pt, {"precision": "bf16"}).root
	assert bf16["time.step"].status == "extrapolated"
	assert 1.5 < base / bf16["time.step"].point < 3.0  # TF32 was on, so bf16 is ~2x, not 16x
	assert bf16["memory.peak"].status == "abstain"  # no model shape: the probes' line can't follow precision


def test_other_gpus(ft):
	l4 = api.predict(ft).root["time.step"]
	h100 = api.predict(ft, {"gpu_type": "H100-SXM-80GB"}).root["time.step"]
	mi = api.predict(ft, {"gpu_type": "MI300X"}).root["time.step"]
	t4 = api.predict(ft, {"gpu_type": "T4"}).root["time.step"]
	assert h100.point < l4.point and h100.model_id.startswith("steptime-roofline")
	assert mi.status == "predicted" and (mi.hi / mi.lo) > (h100.hi / h100.lo)  # cross-vendor band is wider
	assert t4.status == "abstain" and "bf16" in t4.reason


def test_cost_is_time_times_the_dated_price(ft):
	p = api.predict(ft).root
	rate = gpus.price("L4", "modal").usd_per_hour / 3600
	assert p["cost.total"].point == pytest.approx(p["time.total"].point * rate)
	assert "L4 at $" in p["cost.total"].reason


def test_wall_cap_and_preemption(pt):
	p = api.predict(pt).root
	assert 0 <= p["risk.wall_cap"].point <= 1
	assert p["risk.preemption_loss"].status == "abstain"  # a policy in words, no rate
	c = pt.job.constraints.model_copy(
		update={
			"preemption": pt.job.constraints.preemption.model_copy(
				update={"kind": "poisson", "mtti_s": 4 * 3600.0}
			)
		}
	)
	rec = pt.model_copy(update={"job": pt.job.model_copy(update={"constraints": c})})
	loss = api.predict(rec).root["risk.preemption_loss"]
	assert loss.status != "abstain" and loss.point > 0


# ---------------------------------------------------------------- agents


def test_agent_needs_ten_tasks(ag):
	p = api.predict(ag).root
	assert p["cost.total"].status == "abstain" and "at least 10" in p["cost.total"].reason


def test_agent_totals_with_thirty_tasks(ag):
	rec = _agent_with(ag, 30)
	p = api.predict(rec).root
	assert p["cost.total"].status == "predicted" and p["cost.total"].point > 0
	assert p["time.total"].status == "predicted" and "bound by" in p["time.total"].reason
	assert p["risk.runaway"].unit == "probability"
	slow = api.predict(rec, {"concurrency": 1}).root["time.total"].point
	fast = api.predict(rec, {"concurrency": 16}).root["time.total"].point
	assert fast <= slow
	other = api.predict(rec, {"model": "muse-glimmer"}).root["cost.total"]
	assert other.status == "abstain" and "muse-glimmer" in other.reason


def test_context_fill_follows_the_limit(ag):
	rec = _agent_with(ag, 30)
	tight = api.predict(rec).root["memory.context_fill"]
	roomy = api.predict(rec, {"context_limit": 262144}).root["memory.context_fill"]
	assert roomy.point < tight.point
	assert tight.status == "extrapolated"  # a sampled task overflowed, so its real need is unknown


# ---------------------------------------------------------------- grids


def test_grids_decode_back_to_predict(ft):
	grids = api.build_grids(ft, ["batch_size", "seq_len"])
	assert {g.target_id for g in grids} == {"memory.peak", "risk.oom", "time.total", "cost.total"}
	g = next(g for g in grids if g.target_id == "time.total")
	bs, sl = g.axes[0].values, g.axes[1].values
	assert ft.knob("batch_size").current in bs and ft.knob("seq_len").current in sl
	vals, mask = g.values("point"), decode_mask(g.abstain, g.cells).reshape(len(bs), len(sl))
	i, j = bs.index(2), sl.index(1024)
	direct = api.predict(ft, {"batch_size": 2, "seq_len": 1024}).root["time.total"]
	assert not mask[i, j] and vals[i, j] == pytest.approx(direct.point, rel=2e-3)  # float16
	assert mask.any() and not mask.all()  # far cells abstain, near ones don't
	assert g.fixed["grad_accum"] == 4 and "batch_size" not in g.fixed


def test_grid_limits(ft):
	with pytest.raises(ValueError):
		api.build_grids(ft, [])
	with pytest.raises(ValueError):
		api.build_grids(ft, ["batch_size", "seq_len", "lora_r", "grad_accum", "eval_batch_size"])


def test_axis_values():
	k = _rec("finetune").knob("batch_size")
	v = api.axis_values(k)
	assert v == sorted(v) and k.current in v and all(isinstance(x, int) for x in v) and len(v) <= 7
	assert api.axis_values(_rec("finetune").knob("packing")) == [False, True]
	assert set(api.axis_values(_rec("finetune").knob("precision"))) == {"bf16", "fp16", "fp32"}


# ---------------------------------------------------------------- sensitivity


def test_sensitivity_ranks_levers_not_the_workload(pt, ft):
	rows = api.sensitivity(pt, Objective(kind="min_time"))
	ids = [r.knob_id for r in rows]
	assert "max_steps" not in ids and rows == sorted(rows, key=lambda r: -r.impact)
	assert all(0 <= r.impact <= 1 for r in rows) and rows[0].impact == 1
	cost = {r.knob_id: r for r in api.sensitivity(ft, Objective(kind="min_cost"))}
	assert "check quality" in cost["seq_len"].reason
	mem = api.sensitivity(ft, Objective(kind="fit_memory"))
	assert mem[0].knob_id == "eval_batch_size" and mem[0].best_value == 1


# ---------------------------------------------------------------- the predictions block


def test_fill_predictions_replaces_the_fixtures(ft, pt, ag):
	for rec in (ft, pt, _agent_with(ag, 30)):
		out = api.fill_predictions(rec)
		again = FlightRecord.model_validate(json.loads(out.model_dump_json()))
		assert not any(m.id.startswith("golden-fixture") for m in again.models)
		used = {e.model_id for e in again.predictions.estimates.values()}
		assert used <= {m.id for m in again.models}
		assert len(gzip.compress(out.model_dump_json().encode())) < 1_000_000
	assert api.fill_predictions(ft).knob("batch_size").probed_range.max == 4.0


def test_grid_values_stay_finite_after_scaling(ft):
	for g in api.build_grids(ft, ["batch_size", "seq_len"]):
		for name in ("point", "lo", "hi"):
			assert np.isfinite(g.values(name)).all()
