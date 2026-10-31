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
	"""K1's own behaviour, without the held-out cross-GPU band, whose ±45% width makes every GPU move abstain."""
	from dataclasses import replace as dc_replace

	from preflight.engine import conformal

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
		top["verified"] is False and top["ooms"] == 9 and top["n"] == 10
	)  # shown as "9 of 10 similar runs did"


def test_one_threads_quiet_grid_doesnt_silence_another():
	import threading

	seen = []
	with api._quiet_serving(True):
		t = threading.Thread(target=lambda: seen.append(api._QUIET.get()))
		t.start()
		t.join()
		assert api._QUIET.get() is True
	assert seen == [False] and api._QUIET.get() is False


def test_cross_gpu_time_takes_the_held_out_band_and_its_card_says_so(ft):
	from preflight.engine import conformal, steptime

	cell = conformal.TABLE.lookup("time.step.cross", "A100-SXM-80GB", None)
	assert cell is not None and cell.key == ("time.step.cross", "", "")
	moved = api.predict(ft, {"gpu_type": "A100-SXM-80GB"}).root["time.step"]
	assert "held-out corpus pairs" in moved.reason  # with this band the move abstains (the test below)
	(card,) = api._cards({steptime.MODEL_ID_CROSS}, ft)
	assert card.notes.startswith("calibration: held_out") and card.coverage == pytest.approx(
		2130 / 2286, abs=0.02
	)


def test_with_the_held_out_band_a_gpu_move_says_run_a_probe_there(ft):
	# Chiya, Oct 31: the honest cross-GPU band is about +-45%, past the +-50% abstention once the probe's own error
	# is added, so moving the GPU dial asks for a probe on that GPU instead of showing a range
	e = api.predict(ft, {"gpu_type": "A100-SXM-80GB"}).root["time.step"]
	assert e.status == "abstain" and "run a probe here" in e.reason and "held-out corpus pairs" in e.reason
