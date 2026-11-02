"""engine/api.py against the golden records: predict, build_grids, sensitivity, fill_predictions."""

import gzip
import json
import math
import statistics
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
		assert e.unit == api._meta(tid)[0]
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


@pytest.fixture
def no_cross_band(monkeypatch):
	"""K1's own behaviour: engine.steptime_cross_gpu off and its held-out cross-GPU band gone."""
	from dataclasses import replace as dc_replace

	from preflight.engine import conformal

	monkeypatch.setitem(api.FEATURES, "steptime_cross_gpu", False)
	cells = {k: c for k, c in conformal.TABLE.cells.items() if k[0] != "time.step.cross"}
	monkeypatch.setattr(conformal, "TABLE", dc_replace(conformal.TABLE, cells=cells))


def test_unknown_knobs_are_refused_but_gpu_what_ifs_are_not(ft, no_cross_band):
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


def test_other_gpus(ft, no_cross_band):
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
	assert {g.target_id for g in grids} == {
		"memory.peak",
		"risk.oom",
		"time.total",
		"cost.total",
		"cost.gpu_hours",
	}
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
	# a bigger GPU reaches the pooled cell's lower floor; at the shown resolution it ties, and the cheaper lever leads
	gpu = next(r for r in mem if r.knob_id == "gpu_type")
	assert gpu.impact == pytest.approx(mem[0].impact)


def test_sampled_rates_keep_their_resolution_in_the_ranking(ag):
	# only P(OOM) is clamped at the smallest cell's floor; a runaway rate is a sampled rate, ranked as measured
	rec = _agent_with(ag, 30)
	(turns,) = [r for r in api.sensitivity(rec, Objective(kind="min_risk")) if r.knob_id == "max_turns"]
	assert turns.impact == pytest.approx(api.predict(rec).root["risk.runaway"].point, abs=1e-6)


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


# ---------------------------------------------------------------- review fixes (Oct 27)


def test_unknown_gpu_never_reads_as_measured(ft):
	e = api.predict(ft, {"gpu_type": "TPU-v5"}).root["time.step"]
	assert e.status == "abstain" and "TPU-v5" in e.reason
	no_gpu = ft.model_copy(update={"env": ft.env.model_copy(update={"gpu": None, "backend": "cpu"})})
	e = api.predict(no_gpu, {"gpu_type": "H100-SXM-80GB"}).root["time.step"]
	assert e.status == "abstain" and "without a known GPU" in e.reason


def test_an_oom_probe_only_counts_on_its_own_gpu(ft):
	oom = ft.probes[0].model_copy(
		update={"id": "p9", "status": "oom", "knobs": {"batch_size": 16, "seq_len": 1024}}
	)
	rec = ft.model_copy(update={"probes": [*ft.probes, oom]})
	assert api.predict(rec, {"batch_size": 16}).root["risk.oom"].status == "measured"
	h100 = api.predict(rec, {"batch_size": 16, "gpu_type": "H100-SXM-80GB"}).root["risk.oom"]
	assert h100.status != "measured"


def test_unknown_precision_abstains_instead_of_crashing(ft):
	pc = ft.knob("precision")
	knob = pc.model_copy(
		update={"domain": pc.domain.model_copy(update={"values": [*pc.domain.values, "int8"]})}
	)
	rec = ft.model_copy(update={"knobs": [knob if k.id == "precision" else k for k in ft.knobs]})
	p = api.predict(rec, {"precision": "int8"}).root
	assert all(e.status == "abstain" for e in p.values()) and "int8" in p["time.step"].reason
	assert api.build_grids(rec, ["precision"]) and api.sensitivity(rec, Objective(kind="min_cost"))


def test_bad_inputs(ft):
	assert api.predict(ft, {"batch_size": 0}).root["memory.peak"].reason == "batch_size must be at least 1"
	with pytest.raises(ValueError, match="takes a number"):
		api.predict(ft, {"batch_size": "abc"})
	with pytest.raises(ValueError, match="true or false"):
		api.predict(ft, {"packing": "yes"})


def test_max_throughput_prefers_bigger_batches(ft):
	rows = {r.knob_id: r for r in api.sensitivity(ft, Objective(kind="max_throughput"))}
	assert rows["batch_size"].direction == "increase" and rows["batch_size"].best_value > 1


def test_sensitivity_drops_knobs_inside_the_band(ft):
	ids = {r.knob_id for r in api.sensitivity(ft, Objective(kind="min_cost"))}
	assert not ids & {"packing", "eval_no_grad", "lora_r"}


def test_more_gpus_add_communication(ft):
	# compare at a batch no probe ran, so both sides come from the same fit
	one = api.predict(ft, {"batch_size": 3}).root
	eight = api.predict(ft, {"batch_size": 3, "gpus": 8}).root
	assert eight["time.step"].status == "extrapolated" and "all-reduce" in eight["time.step"].reason
	assert eight["time.step"].point > one["time.step"].point  # the all-reduce costs something
	assert eight["time.total"].point > one["time.total"].point / 8  # not a perfect 8x speed-up
	# the all-reduce's own error adds to the step's (LoRA gradients are small, so only just)
	assert (eight["time.step"].hi - eight["time.step"].lo) >= (one["time.step"].hi - one["time.step"].lo)


def test_preemption_without_a_checkpoint_in_the_run_restarts_it_from_zero(pt):
	m = 4 * 3600.0
	c = pt.job.constraints
	c = c.model_copy(update={"preemption": c.preemption.model_copy(update={"kind": "poisson", "mtti_s": m})})
	rec = pt.model_copy(update={"job": pt.job.model_copy(update={"constraints": c})})
	p = api.predict(
		rec, {"checkpoint_every": 100000}
	).root  # an interval longer than the run: capped at the run
	t, r = p["time.total"].point, 60 + (c.preemption.grace_s or 0)
	# restart from zero, interruptions during redone work included: (M + R)(e^(T/M) - 1) - T
	assert p["risk.preemption_loss"].point == pytest.approx((m + r) * math.expm1(t / m) - t)
	assert (
		"restarts the run" not in p["risk.preemption_loss"].reason
	)  # the script does checkpoint, just rarely
	# with frequent checkpoints it is close to Young/Daly's first order: T/M x (half an interval + restart)
	q = api.predict(rec, {"checkpoint_every": 100}).root
	step = q["time.step"].point
	assert q["risk.preemption_loss"].point == pytest.approx(
		q["time.total"].point / m * (50 * step + r), rel=0.05
	)


def test_log_axis_from_zero():
	from preflight.schema.record import Domain, Knob

	k = Knob(
		id="wd", label_key="knob.wd", kind="float", current=0.1, domain=Domain(min=0.0, max=1.0, log=True)
	)
	v = api.axis_values(k)
	assert v[0] > 0 and 0.1 in v


def test_agent_turn_cap_and_untagged_tasks(ag):
	rec = _agent_with(ag, 30)
	base = api.predict(rec).root["cost.total"]
	capped = api.predict(rec, {"max_turns": 2}).root["cost.total"]
	assert (
		capped.point < base.point and capped.status == "extrapolated" and "stop at 2 turns" in capped.reason
	)
	retry = api.predict(rec, {"tool_retry_cap": 0}).root["cost.total"]
	assert retry.status == "extrapolated" and "tool_retry_cap" in retry.reason
	untagged = rec.model_copy(
		update={
			"agent_sample": rec.agent_sample.model_copy(
				update={"tasks": [t.model_copy(update={"model": None}) for t in rec.agent_sample.tasks]}
			)
		}
	)
	assert api.predict(untagged, {"model": "muse-glimmer"}).root["cost.total"].status == "abstain"
	assert api.predict(untagged).root["cost.total"].status == "predicted"


def test_context_limit_below_the_sample_abstains(ag):
	e = api.predict(_agent_with(ag, 30), {"context_limit": 4096}).root["memory.context_fill"]
	assert e.status == "abstain" and "would overflow" in e.reason


# ---------------------------------------------------------------- C's requests (Oct 27)


def test_memory_terms_add_up_to_the_peak_with_eval_as_its_own_wedge(ft):
	p = api.predict(ft).root
	terms = {k: v for k, v in p.items() if k.startswith("memory.") and k != "memory.peak"}
	assert set(terms) == {f"memory.{t}" for t in api.MEMORY_TERMS}
	assert sum(v.point for v in terms.values()) == pytest.approx(p["memory.peak"].point)
	assert terms["memory.eval_spike"].point > terms["memory.activations"].point  # the planted eval OOM
	calm = api.predict(ft, {"eval_batch_size": 1, "eval_no_grad": True}).root
	assert calm["memory.eval_spike"].point == 0


def test_phase_times_add_up_to_the_step(ft, no_cross_band):
	p = api.predict(ft).root
	phases = {k: v for k, v in p.items() if k.startswith("time.phase.")}
	assert {"time.phase.forward", "time.phase.backward"} <= set(phases)
	assert sum(v.point for v in phases.values()) == pytest.approx(p["time.step"].point)
	h100 = api.predict(ft, {"gpu_type": "H100-SXM-80GB"}).root
	assert h100["time.phase.backward"].point < phases["time.phase.backward"].point


def test_sensitivity_skips_pinned_knobs_and_starts_from_the_given_setting(ft):
	rows = api.sensitivity(ft, Objective(kind="min_cost"), pinned=["precision"], knobs={"batch_size": 2})
	assert "precision" not in {r.knob_id for r in rows}
	assert {r.knob_id for r in rows} <= {k.id for k in ft.knobs}


def test_eval_batch_is_held_in_every_grid(ft):
	for g in api.build_grids(ft, ["batch_size", "seq_len"]):
		assert g.fixed["eval_batch_size"] == 32 and g.fixed["eval_no_grad"] is False


# ---------------------------------------------------------------- Review, Oct 27


def test_oom_risk_is_measured_only_from_an_oom_probe(pt):
	p = api.predict(pt).root
	assert p["memory.peak"].status == "measured" and p["risk.oom"].status == "predicted"


def test_knobs_off_the_probes_make_estimates_extrapolated(ft, pt):
	p = api.predict(ft, {"lora_r": 64, "eval_batch_size": 1, "eval_no_grad": True}).root
	assert p["memory.peak"].status == "extrapolated" and "lora_r" in p["memory.peak"].reason
	assert p["time.step"].status == "extrapolated"
	assert (
		api.predict(pt, {"max_steps": 1000}).root["time.step"].status == "measured"
	)  # run length isn't a step knob


def test_one_dimensional_grids_for_every_knob_around_the_current_point(ft):
	out = api.fill_predictions(ft)
	one_d = {
		g.axes[0].knob_id: g
		for g in out.predictions.grids
		if len(g.axes) == 1 and g.target_id == "memory.peak"
	}
	assert set(one_d) == {k.id for k in ft.knobs}
	bs = one_d["batch_size"].axes[0].values
	assert 4 in bs and min(bs) >= 1 and max(bs) <= 32  # 4 x 1/8 .. 4 x 8, inside the domain


def test_model_cards_say_where_their_calibration_comes_from(ft):
	from preflight.engine import conformal

	cards = {c.id: c for c in api.fill_predictions(ft).models}
	gpu, lens = api._env_gpu(ft), ft.job.lens
	for mid, c in cards.items():
		target = api._CARD_TARGET.get(mid)
		# held out only where this record's own band is: a fine-tune's step time has no cell, so its card says simulated
		if target and conformal.TABLE.lookup(target, gpu, lens) is not None and mid in api._CARD_SCORED:
			assert c.notes.startswith("calibration: held_out") and c.coverage is not None
		else:
			assert c.notes.startswith("calibration: simulated") and c.coverage is None
	assert cards[api.memory.MODEL_ID].notes.startswith("calibration: held_out")
	assert cards[api.steptime.MODEL_ID_SAME].notes.startswith("calibration: simulated")


def test_unknown_probe_gpu_uses_its_reported_memory(ft):
	rec = _as_harness_builds_it(
		ft, "NVIDIA Weird 9000", 24 * 10**9
	)  # knob current = env.gpu.name, as A's merge sets
	p = api.predict(rec, {"eval_batch_size": 1, "eval_no_grad": True}).root
	assert p["memory.peak"].status != "abstain" and p["risk.oom"].status != "abstain"
	assert p["time.step"].status in ("measured", "predicted", "extrapolated")
	assert p["cost.total"].status == "abstain" and "no price" in p["cost.total"].reason
	moved = api.predict(rec, {"gpu_type": "H100-SXM-80GB"}).root["time.step"]
	assert moved.status == "abstain" and "isn't in the GPU table" in moved.reason


def test_capacity_prefers_the_probes_reported_total(ft):
	# the golden fine-tune's probe reported 23,034,920,960 bytes on its L4; other GPUs use the table
	assert api._capacity(ft, "L4") == ft.env.gpu.memory_bytes
	assert api._capacity(ft, "A100-SXM-80GB") == gpus.get_gpu("A100-SXM-80GB").capacity_bytes()


def _as_harness_builds_it(rec, env_name, memory_bytes, knob_current=None):
	"""The record the way A's pipeline leaves it: harness.skeleton.gpu_type_knob, then the merge's
	`current = env.gpu.name` (runner/pipeline.py), unless knob_current pins a stale value."""
	from preflight.harness.skeleton import gpu_type_knob

	gk = gpu_type_knob(rec.job.script_map, gpu="L4")
	seen = env_name if knob_current is None else knob_current
	values = list(gk.domain.values) + ([seen] if seen not in gk.domain.values else [])
	gk = gk.model_copy(update={"current": seen, "domain": gk.domain.model_copy(update={"values": values})})
	knobs = [k for k in rec.knobs if k.id != "gpu_type"] + [gk]
	env = rec.env.model_copy(
		update={"gpu": rec.env.gpu.model_copy(update={"name": env_name, "memory_bytes": memory_bytes})}
	)
	return rec.model_copy(update={"env": env, "knobs": knobs})


def test_the_records_own_gpu_is_never_a_move(ft):
	weird = _as_harness_builds_it(ft, "NVIDIA Weird 9000", 24 * 10**9)
	p = api.predict(weird).root
	assert p["time.step"].status != "abstain"  # the probed, unlisted card keeps its measured step time
	assert api.predict(weird, {"gpu_type": "H100-SXM-80GB"}).root["time.step"].status == "abstain"
	# the same GPU under another name is no move either
	l4 = _as_harness_builds_it(ft, "L4", ft.env.gpu.memory_bytes)
	assert api.predict(l4, {"gpu_type": "NVIDIA L4"}).root["time.step"] == api.predict(l4).root["time.step"]


def test_a_stale_knob_predicts_for_the_gpu_it_shows(ft):
	# the probe stepped up to an L40S after an OOM but the knob still says L4: the dial shows L4, so predict for L4
	rec = _as_harness_builds_it(ft, "L40S", 48 * 10**9, knob_current="L4")
	assert api._target_gpu(rec, api._values(rec, None)) == "L4"
	assert api._target_gpu(rec, api._values(rec, {"gpu_type": "L40S"})) == "L40S"


def test_records_pin_their_models_and_say_when_one_has_moved_on(ft):
	# gap 15: a shared record keeps the numbers it was predicted with, and names the model that made them
	out = api.fill_predictions(ft)
	assert api.stale_models(out) == []
	now = api.current_models()
	assert now["memory-analytical"] == api.memory.MODEL_ID
	old = out.model_copy(
		update={
			"models": [
				m.model_copy(update={"id": "memory-analytical@0.1.0"}) if m.id == api.memory.MODEL_ID else m
				for m in out.models
			]
		}
	)
	assert ("memory-analytical@0.1.0", api.memory.MODEL_ID) in api.stale_models(old)
	assert all("@" in mid for mid in now.values())


def test_nodes_split_the_gpus_and_the_step_says_analytical_only(pt):
	from preflight.schema.record import Domain, Knob

	knobs = [
		*pt.knobs,
		Knob(id="nodes", label_key="knob.nodes", kind="int", current=1, domain=Domain(min=1, max=16)),
	]
	if not any(k.id == "gpus" for k in pt.knobs):
		knobs.append(
			Knob(id="gpus", label_key="knob.gpus", kind="int", current=1, domain=Domain(min=1, max=64))
		)
	rec = pt.model_copy(update={"knobs": knobs})
	one = api.predict(rec, {"gpus": 16, "nodes": 1}).root["time.step"]
	two = api.predict(rec, {"gpus": 16, "nodes": 2}).root["time.step"]
	if one.status == "abstain":
		pytest.skip(one.reason)
	assert two.point > one.point and "analytical only" in two.reason
	assert api.predict(rec, {"gpus": 15, "nodes": 2}).root["time.step"].status == "abstain"


def test_calibration_summary_is_what_the_coverage_page_needs():
	out = api.calibration_summary()
	mem = out["targets"]["memory.peak"]
	assert (
		mem["coverage"]["n"] > 0
		and mem["coverage"]["wilson95"][0] <= mem["coverage"]["rate"] <= mem["coverage"]["wilson95"][1]
	)
	assert "A100-SXM-80GB" in mem["holdout"] and "finetune" in mem["lenses"]
	assert "finetune" not in out["targets"]["time.step"]["lenses"]  # HF fine-tune step time stays simulated
	assert out["oom_bins"] and out["oom_middle_verified"] is False
	import json

	json.dumps(out)  # plain data


def test_a_record_banded_with_an_older_table_is_reported(ft):
	from preflight.engine import conformal

	out = api.fill_predictions(ft)
	assert api.stale_models(out) == []
	cards = [
		m.model_copy(
			update={"training_data": m.training_data.replace(f"table {conformal.TABLE.version}", "table 0")}
		)
		if m.training_data and "; table " in m.training_data
		else m
		for m in out.models
	]
	old = out.model_copy(update={"models": cards})
	assert ("conformal-table@0", f"conformal-table@{conformal.TABLE.version}") in api.stale_models(old)


def test_the_summary_says_what_the_engine_shows():
	bins = api.calibration_summary()["oom_bins"]
	ends = [b for b in bins if b["hi"] <= 0.05 + 1e-9 or b["lo"] >= 0.95 - 1e-9]
	assert ends and all(
		b["shown_as_number"] == b["verified"] for b in ends
	)  # an end shows a number once verified
	middle = [b for b in bins if b not in ends]
	assert middle and not any(b["shown_as_number"] for b in middle)
	top = next(b for b in bins if b["lo"] >= 0.95 - 1e-9)
	assert (
		top["verified"] is False and top["ooms"] == 11 and top["n"] == 12
	)  # shown as "11 of 12 similar runs did" (table 67ab8edae6: P(OOM) scored as the engine shows it, Nov 1)


def test_one_threads_quiet_grid_doesnt_silence_another():
	import threading

	seen = []
	with api._quiet_serving(True):
		t = threading.Thread(target=lambda: seen.append(api._QUIET.get()))
		t.start()
		t.join()
		assert api._QUIET.get() is True
	assert seen == [False] and api._QUIET.get() is False


def test_cross_gpu_time_takes_the_held_out_band_and_its_card_says_so(ft, monkeypatch):
	monkeypatch.setitem(api.FEATURES, "steptime_cross_gpu", True)
	from preflight.engine import conformal, steptime

	cell = conformal.TABLE.lookup("time.step.cross", "A100-SXM-80GB", None)
	assert cell is not None and cell.key == ("time.step.cross", "", "")
	moved = api.predict(ft, {"gpu_type": "A100-SXM-80GB"}).root["time.step"]
	assert "held-out corpus pairs" in moved.reason  # with this band the move abstains (the test below)
	(card,) = api._cards({steptime.MODEL_ID_CROSS}, ft)
	assert card.notes.startswith("calibration: held_out") and card.coverage == pytest.approx(
		2130 / 2286, abs=0.02
	)


def test_without_the_feature_a_gpu_move_keeps_k1s_own_band(ft, monkeypatch):
	monkeypatch.setitem(api.FEATURES, "steptime_cross_gpu", False)
	e = api.predict(ft, {"gpu_type": "A100-SXM-80GB"}).root["time.step"]
	assert e.status != "abstain" and "held-out corpus pairs" not in (e.reason or "")
	(card,) = api._cards({api.steptime.MODEL_ID_CROSS}, ft)
	assert card.notes.startswith("calibration: simulated")


def test_with_the_held_out_band_a_gpu_move_is_rough_with_a_probe_offer(ft, monkeypatch):
	# Chiya, Oct 31 (option b): the calibrated band is shown past +-50%, labelled rough, with "run a probe here"
	monkeypatch.setitem(api.FEATURES, "steptime_cross_gpu", True)
	e = api.predict(ft, {"gpu_type": "A100-SXM-80GB"}).root["time.step"]
	assert e.status == "extrapolated" and "rough, no probe on this GPU; run a probe here" in e.reason
	assert math.log(e.hi / e.point) > math.log(1.5)  # past the usual +-50% abstention, on purpose
	total = api.predict(ft, {"gpu_type": "A100-SXM-80GB"}).root["time.total"]
	assert total.status != "abstain"
	# abstains only where it did: no bf16 path on T4, and another vendor until corpus rows from it exist
	assert api.predict(ft, {"gpu_type": "T4"}).root["time.step"].status == "abstain"
	amd = api.predict(ft, {"gpu_type": "MI300X"}).root["time.step"]
	assert (
		amd.status == "abstain"
		and amd.reason.startswith("no AMD measurements")
		and "another vendor" in amd.reason
	)
	(card,) = api._cards({api.steptime.MODEL_ID_CROSS}, ft)
	assert "Rough, no probe on this GPU" in card.notes and "26 of 28" in card.notes
	assert card.coverage == pytest.approx(2130 / 2286)


def test_without_a_kernel_table_a_gpu_move_still_takes_the_held_out_band(ft, monkeypatch):
	# review, Oct 31: the class-scaling fallback once kept a narrower band than the card's held_out claims
	monkeypatch.setitem(api.FEATURES, "steptime_cross_gpu", True)
	bare = ft.model_copy(update={"probes": [p.model_copy(update={"kernels": None}) for p in ft.probes]})
	e = api.predict(bare, {"gpu_type": "A100-SXM-80GB"}).root["time.step"]
	cell = api.conformal.TABLE.lookup("time.step.cross", "A100-SXM-80GB", None)
	assert e.status == "extrapolated" and "rough, no probe on this GPU" in e.reason
	assert math.log(e.hi / e.point) >= cell.half_width * 0.999


def test_with_the_feature_off_another_vendor_keeps_k1s_band(ft, monkeypatch):
	monkeypatch.setitem(api.FEATURES, "steptime_cross_gpu", False)
	amd = api.predict(ft, {"gpu_type": "MI300X"}).root["time.step"]
	assert amd.status == "predicted" and "another vendor" not in (amd.reason or "")


def test_the_step_band_covers_a_few_step_median_on_varying_batches():
	# Chiya, Nov 1: the LoRA full run's median (1.87 s) sat outside a band built from the probe's 1.68 s; checked on
	# that ground truth only (tests/fixtures/validation), never on the unseen scripts
	from preflight.schema import load_record

	rec = load_record((Path(__file__).parent / "fixtures" / "b-a-smoke-l4-main-4.record.json").read_text())
	gt = json.loads(
		(Path(__file__).parent / "fixtures" / "validation" / "ground_truth_finetune_lora_hf.json").read_text()
	)
	from preflight.corpus.validate import steady_step_s

	e = api.predict(rec).root["time.step"]
	assert e.lo <= steady_step_s(gt) <= e.hi
	assert "batches vary in length" in e.reason


def test_the_vectorised_normal_quantile():
	from statistics import NormalDist

	import numpy as np

	ps = np.asarray([1e-9, 0.001, 0.02, 0.3, 0.5, 0.9, 0.99, 1 - 1e-9])
	assert np.allclose(api._norm_ppf(ps), [NormalDist().inv_cdf(x) for x in ps], atol=1e-7)


def test_padded_step_follows_median_and_mean_and_stops_at_seq_len():
	med, mean, spread = api._padded_step(150, 200, 1024, 4, 1)
	assert med > 150 and mean > med and spread > 0  # each micro-batch pads to its longest row
	assert api._padded_step(150, 200, 160, 4, 1)[2] < spread  # nearly every batch hits seq_len
	assert api._padded_step(150, 150, None, 4, 1)[2] == 0.0  # every row the median: no spread
	assert api._padded_step(150, 200, 1024, 4 * 8, 1)[0] > med  # 8 ranks: a step waits for the slowest
	acc = api._padded_step(150, 200, 1024, 4, 4)[2]
	assert acc == pytest.approx(spread / 2, rel=0.15)  # 4 micro-batches a step average out


def _lora():
	from preflight.schema import load_record

	return load_record((Path(__file__).parent / "fixtures" / "b-a-smoke-l4-main-4.record.json").read_text())


def _with_lengths(rec, **extras):
	probes = [p.model_copy(update={"extras": {**(p.extras or {}), **extras}}) for p in rec.probes]
	return rec.model_copy(update={"probes": probes})


def test_each_probe_is_corrected_by_its_own_padding_before_the_fit():
	# review, Nov 1: a factor taken from whichever probe matched jumped 4.5% when a knob moved onto another probe
	lens = _with_lengths(_lora(), dataset_p50_tokens=200.0, dataset_mean_tokens=260.0)
	for pid in ("p1", "p2", "p3"):
		p = next(q for q in lens.probes if q.id == pid)
		pv = api._probe_values(lens, p)
		f = api._length_factor(lens, p)
		med = api._padded_step(
			200.0, 260.0, float(pv["seq_len"]), int(pv["batch_size"]), int(pv.get("grad_accum") or 1)
		)[0]
		seen = [x.tokens / x.samples for x in p.step_log if x.kind == "timed" and x.tokens and x.samples]
		assert f == pytest.approx(med / statistics.median(seen))  # medians: the probe's p50 is a median step
		ph = api.steptime.phases_seconds(p.step_time.by_phase, p.step_time.p50_s)
		want = sum(
			v * f if k in api.TOKEN_PHASES else v for k, v in ph.items()
		)  # its token-bound phases corrected
		got = api.predict(lens, {k: pv[k] for k in ("batch_size", "seq_len")}).root["time.step"]
		assert got.point == pytest.approx(want, rel=1e-6) and got.status == "predicted"


def test_a_knob_move_between_probes_doesnt_jump():
	lens = _with_lengths(_lora(), dataset_p50_tokens=200.0, dataset_mean_tokens=260.0)
	pts = [api.predict(lens, {"batch_size": b, "seq_len": 1024}).root["time.step"].point for b in (2, 3, 4)]
	assert pts[0] < pts[1] < pts[2]  # the fit interpolates between the corrected probes


def test_without_the_datasets_lengths_the_probes_spread_sets_both_terms():
	rec = _lora()
	p = next(q for q in rec.probes if q.id == "p1")
	lt = api._length_term(rec, {}, p, 1000)
	k = p.step_time.n_steps
	seen = math.log(p.step_time.p95_s / p.step_time.p50_s) / api.steptime.Z90
	assert lt.total_scale == 1.0
	assert lt.step_sigma == pytest.approx(api.MEDIAN_SE * seen / math.sqrt(k))
	assert lt.total_sigma == pytest.approx(seen / math.sqrt(k))


def test_a_long_tail_moves_memory_not_time():
	# Chiya, Nov 1: time follows the median and the mean; p99.9 and the longest row belong to peak memory
	base = _with_lengths(_lora(), dataset_p50_tokens=140.0, dataset_mean_tokens=152.0)
	tail = _with_lengths(base, dataset_p999_tokens=1000.0, dataset_max_tokens=4000.0)
	a, b = api.predict(base).root["time.step"], api.predict(tail).root["time.step"]
	assert a.point == b.point and a.hi == b.hi
	assert math.log(a.hi / a.point) < math.log(1.5)  # a long-tailed record doesn't push time to abstain


def _lora_steps(n=1000):
	rec = _lora()
	sm = rec.job.script_map
	sm = sm.model_copy(update={"training": sm.training.model_copy(update={"max_steps": n})})
	return rec.model_copy(update={"job": rec.job.model_copy(update={"script_map": sm})})


def test_a_total_follows_the_mean_and_its_length_error_shrinks_with_the_steps():
	rec = _with_lengths(_lora_steps(1000), dataset_p50_tokens=200.0, dataset_mean_tokens=260.0)
	p = next(q for q in rec.probes if q.id == "p1")
	lt = api._length_term(rec, {}, p, 1000)
	pv = api._probe_values(rec, p)
	_, _, spread = api._padded_step(
		200.0, 260.0, float(pv["seq_len"]), int(pv["batch_size"]), int(pv.get("grad_accum") or 1)
	)
	assert lt.total_sigma == pytest.approx(spread / math.sqrt(1000)) and lt.total_scale > 1
	out = api.predict(rec).root
	t, st = out["time.total"], out["time.step"]
	assert t.point > 1000 * st.point  # the mean step is above the median step
	assert math.log(t.hi / t.point) < math.log(st.hi / st.point)


def test_the_cold_cost_doesnt_grow_with_accumulation():
	import time

	api._padded_step.cache_clear()
	t0 = time.perf_counter()
	api._padded_step(150.0, 200.0, 1024.0, 8, 1024)
	assert time.perf_counter() - t0 < 0.5  # about 4,000 quantiles whatever the accumulation


def test_rl_and_packed_rows_take_no_padding_term():
	rec = _with_lengths(_lora(), dataset_p50_tokens=200.0, dataset_mean_tokens=260.0)
	p = next(q for q in rec.probes if q.id == "p1")
	assert api._length_term(rec, {"packing": True}, p).total_scale == 1.0
	rl = rec.model_copy(update={"job": rec.job.model_copy(update={"lens": "rl"})})
	assert api._length_term(rl, {}, p) is None


def _at_dataset(rec, med_tokens=200.0, mean_tokens=260.0, log=True):
	"""Each probe's timed steps padded to exactly the run's median, so the probe saw the run's batches."""
	probes = []
	for p in rec.probes:
		pv = api._probe_values(rec, p)
		run_med, run_mean, _ = api._padded_step(
			med_tokens,
			mean_tokens,
			float(pv["seq_len"]),
			int(pv["batch_size"]),
			int(pv.get("grad_accum") or 1),
		)
		upd = {
			"extras": {
				**(p.extras or {}),
				"dataset_p50_tokens": med_tokens,
				"dataset_mean_tokens": mean_tokens,
			}
		}
		if log:
			upd["step_log"] = [
				x.model_copy(update={"tokens": round(run_med * x.samples)})
				if x.kind == "timed" and x.samples
				else x
				for x in p.step_log
			]
		else:
			upd["step_log"] = []
			upd["data"] = p.data.model_copy(
				update={"tokens_per_sample_mean": run_mean * (1 - p.data.padding_fraction)}
			)
		probes.append(p.model_copy(update=upd))
	return rec.model_copy(update={"probes": probes})


def test_a_probe_that_saw_the_runs_data_is_left_alone():
	# review, Nov 1: a median-over-mean factor cut such a probe by 16%
	for log in (True, False):  # medians from the step log, or the mean-over-mean fallback without one
		rec = _at_dataset(_lora(), log=log)
		for p in rec.probes:
			if p.status in api.OK_PROBE:
				assert api._length_factor(rec, p) == pytest.approx(1.0, abs=0.01)


def test_one_uncorrectable_probe_leaves_every_probe_uncorrected():
	# review, Nov 1: break p2, query between the probes (batch 3), where a mixed fit would show
	lens = _with_lengths(_lora(), dataset_p50_tokens=200.0, dataset_mean_tokens=260.0)
	mixed = lens.model_copy(
		update={
			"probes": [
				q.model_copy(update={"data": q.data.model_copy(update={"padding_fraction": None})})
				if q.id == "p2"
				else q
				for q in lens.probes
			]
		}
	)
	q = {"batch_size": 3}
	assert api.predict(mixed, q).root["time.step"].point == pytest.approx(
		api.predict(_lora(), q).root["time.step"].point
	)
	assert api.predict(lens, q).root["time.step"].point != pytest.approx(
		api.predict(_lora(), q).root["time.step"].point
	)


def test_dollars_are_gpu_hours_times_the_rate_so_the_browser_can_recompute(ft):
	# Chiya, Nov 1: C recomputes cost.total at a user's own rate; the interval is the quantity's
	p = api.predict(ft).root
	hours, usd = p["cost.gpu_hours"], p["cost.total"]
	rate = gpus.price("L4", "modal").usd_per_hour
	assert hours.unit == "gpu_hours"
	for a, b in ((hours.point, usd.point), (hours.lo, usd.lo), (hours.hi, usd.hi)):
		assert a * rate == pytest.approx(b)
	assert "cost.gpu_hours" in {t.id for t in api.targets_for(ft)}


def test_no_amd_measurements_on_every_engine_surface(ft):
	# Chiya, Nov 1: the account is on a verification hold and the AMD slice is a config list
	assert api.calibration_summary()["amd"] == "no AMD measurements"
	(card,) = api._cards({api.steptime.MODEL_ID_CROSS}, ft)
	assert "No AMD measurements" in card.notes
	amd = api.predict(ft, {"gpu_type": "MI300X"}).root["time.step"]
	assert amd.status == "abstain" and amd.reason.startswith("no AMD measurements")


def test_one_fit_uses_one_estimator():
	# review, Nov 1: a probe without a step log beside ones with it once mixed the median and mean factors
	lens = _with_lengths(_lora(), dataset_p50_tokens=200.0, dataset_mean_tokens=260.0)
	no_log = lens.model_copy(
		update={"probes": [q.model_copy(update={"step_log": []}) if q.id == "p2" else q for q in lens.probes]}
	)
	for p in no_log.probes:
		if p.id in ("p1", "p3"):
			assert api._length_factor(no_log, p, False) != api._length_factor(no_log, p, True)
	# with one probe lacking a log, every probe takes the mean factor
	timed = [p for p in no_log.probes if p.status in api.OK_PROBE and p.step_time and p.step_time.p50_s]
	assert not all(api._probe_rows(p) for p in timed)


def test_no_gpu_no_gpu_hours(ft):
	cpu = ft.model_copy(update={"env": ft.env.model_copy(update={"gpu": None})})
	assert api.predict(cpu).root["cost.gpu_hours"].status == "abstain"


def test_a_missing_step_log_sends_every_probe_to_the_mean_factor():
	lens = _with_lengths(_lora(), dataset_p50_tokens=200.0, dataset_mean_tokens=260.0)
	one = lens.model_copy(
		update={"probes": [q.model_copy(update={"step_log": []}) if q.id == "p2" else q for q in lens.probes]}
	)
	f_one = {
		p.id: api._length_factor(
			one, p, all(api._probe_rows(q) for q in one.probes if q.status in api.OK_PROBE)
		)
		for p in one.probes
		if p.status in api.OK_PROBE
	}
	f_mean = {p.id: api._length_factor(lens, p, False) for p in lens.probes if p.status in api.OK_PROBE}
	assert f_one == pytest.approx(f_mean)


def test_the_cuda_context_is_measured_not_guessed():
	# C, Nov 1: a guessed 0.75 GiB on a GRPO probe's reserved peak read 23.8 GB on a 23.7 GB L4 for a run that fit
	rec = _lora()
	gaps = [
		x.device_used_bytes - x.peak_reserved_bytes
		for p in rec.probes
		if p.status in api.OK_PROBE
		for x in p.step_log
		if x.kind == "timed" and x.device_used_bytes and x.peak_reserved_bytes
	]
	assert api._context_bytes(rec) == pytest.approx(
		max(gaps)
	)  # the step that released nothing shows it whole
	assert api._context_bytes(rec) < api.memory.DEFAULT_CONTEXT_BYTES / 2
	bare = rec.model_copy(update={"probes": [p.model_copy(update={"step_log": []}) for p in rec.probes]})
	assert api._context_bytes(bare) == api.memory.DEFAULT_CONTEXT_BYTES


def test_every_memory_path_takes_the_measured_context(monkeypatch):
	rec = _lora()
	q = {"batch_size": 2, "seq_len": 1024}
	measured = api.predict(rec, q).root["memory.peak"].point
	monkeypatch.setattr(api, "_context_bytes", lambda r, gpu=None: float(api.memory.DEFAULT_CONTEXT_BYTES))
	guessed = api.predict(rec, q).root["memory.peak"].point
	real = max(
		x.device_used_bytes - x.peak_reserved_bytes
		for p in rec.probes
		if p.status in api.OK_PROBE
		for x in p.step_log
		if x.kind == "timed" and x.device_used_bytes and x.peak_reserved_bytes
	)
	assert guessed - measured == pytest.approx(api.memory.DEFAULT_CONTEXT_BYTES - real, rel=0.05)


def _with_gaps(rec, gaps_mib, rest=None):
	"""The LoRA record with its timed steps' context gaps set, in order, to gaps_mib, then `rest` (the first if None)."""
	rest = gaps_mib[0] if rest is None else rest
	probes, k = [], 0
	for p in rec.probes:
		log = []
		for x in p.step_log or []:
			if x.kind == "timed" and x.peak_reserved_bytes:
				x = x.model_copy(
					update={
						"device_used_bytes": x.peak_reserved_bytes
						+ int((gaps_mib[k] if k < len(gaps_mib) else rest) * 2**20)
					}
				)
				k += 1
			log.append(x)
		probes.append(p.model_copy(update={"step_log": log}))
	return rec.model_copy(update={"probes": probes})


def test_the_context_is_the_largest_gap_two_steps_agree_on():
	# reviews, Nov 1: a release after the peak lowers a step's gap; a co-tenant's spike has no partner
	rec = _with_gaps(_lora(), [252, 120, -900, 252, 2999, 1500], rest=200)
	ctx = api._context_bytes(rec) / 2**20
	assert ctx == pytest.approx(252, abs=0.5)  # not the median (200), not the one-off 1,500 MiB spike
	assert api._context_bytes(_with_gaps(_lora(), [-50])) == api.memory.DEFAULT_CONTEXT_BYTES  # no usable gap


def test_another_gpu_takes_the_guess():
	rec = _lora()
	assert api._context_bytes(rec, "H100-SXM-80GB") == api.memory.DEFAULT_CONTEXT_BYTES
	assert api._context_bytes(rec, api._env_gpu(rec)) < api.memory.DEFAULT_CONTEXT_BYTES


def test_every_engine_cost_estimate_carries_its_rate(ft):
	# contract 0.6.0: quantity x rate, so the browser re-prices at the visitor's rate
	p = api.predict(ft).root
	r = p["cost.total"].rate
	assert r.quantity == "cost.gpu_hours" and r.unit == "gpu_hours" and r.price_row.startswith("gpu/")
	assert p["cost.total"].point == pytest.approx(p["cost.gpu_hours"].point * r.usd_per_unit)
	_, jc = api.run_cost(ft) if hasattr(api, "run_cost") else (None, None)
	if jc is not None and jc.total_usd is not None:
		assert jc.total_usd.rate.quantity == "job_cost.gpu_hours"
		assert jc.total_usd.point == pytest.approx(jc.gpu_hours.point * jc.total_usd.rate.usd_per_unit)
		for line in jc.lines:
			if line.usd is not None and line.usd.status != "abstain":
				assert line.usd.rate.price_row == line.price_row


def test_money_at_risk_carries_the_same_rate_as_the_total(ft):
	_, jc = api.run_cost(ft)
	if jc.at_risk_usd is not None and jc.at_risk_usd.status != "abstain":
		r = jc.at_risk_usd.rate
		assert r.quantity == "job_cost.at_risk_gpu_hours" and r.usd_per_unit == jc.total_usd.rate.usd_per_unit
		assert jc.at_risk_usd.point == pytest.approx(jc.at_risk_gpu_hours.point * r.usd_per_unit)


def test_a_stalled_sample_reaches_the_runaway_readout(ag):
	"""Chiya, Nov 1: the readout counts a stalled run's open tasks up to the worker limit, through predict."""
	rec = _agent_with(ag, 30)
	tasks = [
		t.model_copy(update={"stop_reason": None if i < 24 else "done", "cost_usd": 0.01, "latency_s": 20.0})
		for i, t in enumerate(rec.agent_sample.tasks)
	]
	sample = rec.agent_sample.model_copy(update={"tasks": tasks, "ended": "stalled"})
	r = api.predict(rec.model_copy(update={"agent_sample": sample})).root["risk.runaway"]
	assert r.status != "abstain" and "stalled with 24 tasks waiting on a tool" in r.reason
	# uniform tasks, so no other rule flags any; no worker count before contract 0.6.1: the limit is one
	plain = api.predict(
		rec.model_copy(update={"agent_sample": sample.model_copy(update={"ended": None})})
	).root
	assert r.point == pytest.approx(plain["risk.runaway"].point + 1 / 30)
	finished = sample.model_copy(update={"ended": "finished"})
	assert (
		"stalled"
		not in api.predict(rec.model_copy(update={"agent_sample": finished})).root["risk.runaway"].reason
	)


def _grpo_l4():
	return load_record(
		(Path(__file__).resolve().parents[1] / "web/src/workspace/fixtures/grpo-l4.record.json").read_text()
	)


def test_a_gpu_move_from_the_probes_settings_is_predicted_not_measured():
	"""Review, Nov 1: the probe's own peak is measured only on the probe's own GPU."""
	rec = _grpo_l4()
	here = api.predict(rec).root["memory.peak"]
	assert here.status == "measured"
	moved = api.predict(rec, {"gpu_type": "A10"}).root["memory.peak"]
	assert moved.status != "measured"


def test_size_set_terms_keep_their_modelled_values():
	"""C, Nov 1: a 494M-parameter GRPO model's weights read 9.9 GB when every term was scaled to the peak."""
	rec = _grpo_l4()
	p = api.predict(rec).root
	terms = {k.split(".")[1]: v.point for k, v in p.items() if k.startswith("memory.") and k != "memory.peak"}
	assert sum(terms.values()) == pytest.approx(p["memory.peak"].point)
	assert terms["weights"] < 2.5e9 and terms["gradients"] < 2.5e9  # fp32 494M params: about 1.98 GB each
	assert terms["activations"] > terms["weights"]  # the rest of the measured peak


def _grpo_a10():
	return load_record(
		(Path(__file__).resolve().parents[1] / "zoo/grpo_trl/record_nov1_a10.json").read_text()
	)


def test_a_probe_s_parameter_count_stands_in_when_the_script_map_has_none():
	"""Nov 1: A's A10 GRPO record names Qwen2.5-0.5B-Instruct with no size; its probe counted 494,032,768."""
	rec = _grpo_a10()
	assert rec.job.script_map.model.param_count is None
	p = api.predict(rec).root
	assert p["memory.peak"].status != "abstain"
	assert p["memory.weights"].point == pytest.approx(494_032_768 * 4, rel=0.01)  # fp32 weights, as measured


def test_a_record_without_its_gpu_says_which_gpu_it_assumed():
	# A's record names its A10 since round six (0874c0f), so the no-GPU case is built here (PM, Nov 1)
	rec = _grpo_a10()
	knob_gpu = next(k.current for k in rec.knobs if k.id == "gpu_type")
	rec = rec.model_copy(update={"env": None})
	p = api.predict(rec).root
	assert "isn't in the record" in p["memory.peak"].reason and str(knob_gpu) in p["memory.peak"].reason


def test_the_real_a10_grpo_record_reads_its_measured_peak_once_its_gpu_is_recorded():
	"""The video's opening lens: the run fitted at 23.18 GB reserved plus a 0.30 GB measured context on the A10."""
	rec = _grpo_a10()
	env = _grpo_l4().env.model_copy(
		update={"gpu": _grpo_l4().env.gpu.model_copy(update={"name": "A10", "memory_bytes": None})}
	)
	knobs = [k.model_copy(update={"current": "A10"}) if k.id == "gpu_type" else k for k in rec.knobs]
	p = api.predict(rec.model_copy(update={"env": env, "knobs": knobs})).root
	assert p["memory.peak"].status == "measured"
	assert p["memory.peak"].point == pytest.approx(23.50e9, rel=0.005)
	assert "isn't in the record" not in (p["memory.peak"].reason or "")
	assert p["risk.oom"].reason.startswith("tight")


def test_a_skeleton_without_probes_gets_no_unrecorded_gpu_note():
	"""Review, Nov 1: before any probe runs a record has no env, and that isn't news."""
	rec = _grpo_a10()
	sm = rec.job.script_map
	sized = sm.model_copy(update={"model": sm.model.model_copy(update={"param_count": 494_032_768})})
	bare = rec.model_copy(update={"probes": [], "job": rec.job.model_copy(update={"script_map": sized})})
	p = api.predict(bare).root
	assert p["memory.peak"].status != "abstain"  # review: so the note's own guard isn't what keeps it out
	assert "isn't in the record" not in (p["memory.peak"].reason or "")


def _lora_l4(**extras):
	rec = load_record(
		(Path(__file__).resolve().parent / "fixtures" / "b-a-smoke-l4-main-4.record.json").read_text()
	)
	if not extras:
		return rec
	probes = [
		p.model_copy(update={"extras": {**(p.extras or {}), **extras}}) if p.id == "p1" else p
		for p in rec.probes
	]
	return rec.model_copy(update={"probes": probes})


def test_a_query_is_sized_by_the_dataset_s_longest_row_when_the_recorder_found_it():
	"""C, Nov 1: the LoRA zoo record read 30.7 GiB at eval batch 4, sized at seq_len 1,024 while its rows reached 517."""
	plain = api.predict(_lora_l4(), {"eval_batch_size": 4}).root["memory.peak"]
	whole = {"dataset_max_tokens": 700, "dataset_rows_scanned": 51760, "dataset_rows_total": 51760}
	shorter = api.predict(_lora_l4(**whole), {"eval_batch_size": 4}).root
	assert shorter["memory.peak"].point < 0.8 * plain.point
	assert "longest row, 700 tokens (measured over all 51,760 training rows)" in shorter["memory.peak"].reason


def test_a_partial_scan_never_sizes_a_query_below_seq_len():
	"""Review, Nov 1: 1,000 rows of 5M scanned, longest 520, read P(OOM) 2.5% for a run that may meet 1,024."""
	plain = api.predict(_lora_l4(), {"eval_batch_size": 4}).root["memory.peak"]
	part = {"dataset_max_tokens": 520, "dataset_rows_scanned": 1000, "dataset_rows_total": 5_000_000}
	p = api.predict(_lora_l4(**part), {"eval_batch_size": 4}).root["memory.peak"]
	assert p.point == pytest.approx(plain.point)
	assert "scanned 1,000 of 5,000,000 training rows" in p.reason


def test_without_the_longest_row_the_readout_says_it_sized_for_seq_len():
	p = api.predict(_lora_l4(), {"eval_batch_size": 4}).root["memory.peak"]
	assert "sized for rows of 1,024 tokens, the seq_len" in p.reason and "reached 517" in p.reason


def test_rows_that_reach_the_probe_s_cut_keep_seq_len_and_say_so():
	whole = {"dataset_max_tokens": 1024, "dataset_rows_scanned": 51760, "dataset_rows_total": 51760}
	p = api.predict(_lora_l4(**whole), {"eval_batch_size": 4}).root["memory.peak"]
	assert p.point == pytest.approx(api.predict(_lora_l4(), {"eval_batch_size": 4}).root["memory.peak"].point)
	assert "the training set has rows that long" in p.reason


def test_the_eval_wedge_carries_the_measured_context():
	"""Nov 1: the eval moment's split took the 0.75 GiB guess while the point took the probes' 0.25 GiB."""
	at8 = api.predict(_lora_l4(), {"eval_batch_size": 8}).root
	at4 = api.predict(_lora_l4(), {"eval_batch_size": 4}).root
	assert at8["memory.context"].point == pytest.approx(at4["memory.context"].point)


def test_when_no_single_move_clears_an_oom_the_record_carries_grids_over_pairs_that_do():
	"""Chiya, Nov 1: on the LoRA zoo record every single fix still read 0.97; eval batch 4 with seq_len 512 reads
	17.5 GiB, P(OOM) 2.5%, tight. The guide reads such a combined move off a 2-D local grid."""
	from preflight.schema.encoding import decode_f16

	rec = _lora_l4()
	pairs = api.clearing_pairs(rec)
	assert ("eval_batch_size", "seq_len") in pairs and len(pairs) <= api.MAX_CLEARING_PAIRS
	grids = api.fill_predictions(rec).predictions.grids
	g = next(
		g
		for g in grids
		if g.target_id == "risk.oom" and [a.knob_id for a in g.axes] == ["eval_batch_size", "seq_len"]
	)
	ev, sq = (a.values for a in g.axes)
	cell = ev.index(4) * len(sq) + sq.index(512)
	p = decode_f16(g.point, len(ev) * len(sq))[cell] * g.scale
	assert p <= api.CLEARS_P_OOM


def test_no_pair_grids_when_the_run_already_fits():
	rec = _lora_l4()
	fixed = [
		k.model_copy(update={"current": {"eval_batch_size": 4, "seq_len": 512}[k.id]})
		if k.id in ("eval_batch_size", "seq_len")
		else k
		for k in rec.knobs
	]
	fits = rec.model_copy(update={"knobs": fixed})
	assert api.predict(fits).root["risk.oom"].point <= api.CLEARS_P_OOM
	assert api.clearing_pairs(fits) == []
	assert all(
		len(g.axes) == 1 for g in api.fill_predictions(fits).predictions.grids if g.target_id == "risk.oom"
	)


def test_a_gpu_swap_doesn_t_stop_the_pair_search():
	"""Review, Nov 1: Nov 1 records offer every GPU, and a B200 'cleared' the LoRA run, so no pair was searched."""
	rec = _lora_l4()
	many = ["L4", "A10", "L40S", "A100-SXM-40GB", "A100-SXM-80GB", "H100-SXM-80GB", "B200"]
	knobs_ = [
		k.model_copy(update={"domain": k.domain.model_copy(update={"values": many})})
		if k.id == "gpu_type"
		else k
		for k in rec.knobs
	]
	pairs = api.clearing_pairs(rec.model_copy(update={"knobs": knobs_}))
	assert ("eval_batch_size", "seq_len") in pairs
	assert all("gpu" not in a and "gpu" not in b for a, b in pairs)


def test_pair_grids_are_named_apart_from_a_record_s_own_2d_grids():
	grids = api.fill_predictions(_lora_l4()).predictions.grids
	two = [g for g in grids if len(g.axes) == 2]
	assert two and all(g.id.endswith(api.CLEARING_GRID_SUFFIX) for g in two)
	assert len({g.id for g in grids}) == len(grids)


def test_the_run_length_falls_back_on_the_probes_dataset_length():
	"""C, Nov 1: the live LoRA record's time and cost abstained ("no dataset size"), though every probe recorded
	train_dataset_len 50,724 (planned_steps 3,170 at batch 4 x grad_accum 4)."""
	rec = FlightRecord.model_validate(
		json.loads(
			(
				Path(__file__).resolve().parent / "fixtures/lora_live_nov1_record.json"
			).read_text()
		)
	)
	steps, why = api.total_steps(rec)
	assert why == "" and steps == pytest.approx(3170, abs=1)
	assert api.predict(rec).root["time.total"].status != "abstain"


def test_the_probe_dataset_length_follows_the_runner_s_rules():
	"""Review, Nov 1: the fallback took the first probe in list order. Patched probes (a patch may subsample) and an
	oom probe before its first step (its latest loader may be an eval one) don't count; ok comes first; a query that
	packs differently from the probe abstains (packing changes the row count)."""
	path = Path(__file__).resolve().parent / "fixtures/lora_live_nov1_record.json"
	rec = FlightRecord.model_validate(json.loads(path.read_text()))

	def with_probes(fn):
		return rec.model_copy(update={"probes": [fn(p) for p in rec.probes]})

	patched = with_probes(
		lambda p: p.model_copy(update={"extras": {**p.extras, "patched": 1.0, "train_dataset_len": 64.0}})
	)
	assert api.total_steps(patched)[0] is None
	eval_first = rec.model_copy(
		update={
			"probes": [
				rec.probes[0].model_copy(
					update={"steps": 0, "extras": {**rec.probes[0].extras, "train_dataset_len": 2000.0}}
				),
				*rec.probes[1:],
			]
		}
	)
	assert api.total_steps(eval_first)[0] == pytest.approx(3170, abs=1)
	steps, why = api.total_steps(rec, {"packing": True})
	assert steps is None and "packing changes the row count" in why
	# packing set in the script map, no knob: a seq_len move abstains too
	data = rec.job.script_map.data.model_copy(update={"packing": True})
	sm = rec.job.script_map.model_copy(update={"data": data})
	packed = rec.model_copy(
		update={
			"job": rec.job.model_copy(update={"script_map": sm}),
			"knobs": [k for k in rec.knobs if k.id != "packing"],
		}
	)
	assert api.total_steps(packed, {"seq_len": 512})[0] is None
	assert api.total_steps(packed)[0] == pytest.approx(3170, abs=1)


def test_a_cell_with_no_run_over_the_margin_reads_p_oom_from_the_whole_pool():
	"""Chiya, Nov 1 (option b): 19 L4 fine-tune runs can't show P(OOM) below 2.5%, so with none of them over the
	margin, the 487 held-out runs decide, and the reason says so. 7.7 GiB on a 22 GiB L4 fits; 17.5 GiB is tight."""
	path = Path(__file__).resolve().parent / "fixtures/lora_live_nov1_record.json"
	rec = FlightRecord.model_validate(json.loads(path.read_text()))
	small = api.predict(rec, {"eval_batch_size": 4, "seq_len": 128}).root["risk.oom"]
	assert small.point < 0.01 and small.reason.startswith("fits: ")
	assert "from all 487 held-out runs; none of the 19 L4 finetune runs went over" in small.reason
	close = api.predict(rec, {"eval_batch_size": 4, "seq_len": 512}).root["risk.oom"]
	assert (
		0.01 <= close.point < 0.05
		and close.reason.startswith("tight: ")
		and "fits needs below 1%" in close.reason
	)
	# a cell with a run over the margin keeps its own count
	big = api.predict(rec, {"eval_batch_size": 4, "seq_len": 1024}).root["risk.oom"]
	assert big.reason.startswith("fails: ") and "487" not in big.reason
