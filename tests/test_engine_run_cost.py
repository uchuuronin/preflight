"""engine.api.run_cost: the run's cost breakdown and its basis (contract 0.2.0, gap 21)."""

from pathlib import Path

import pytest

from preflight.engine import api, gpus, oom, prices
from preflight.schema import FlightRecord, load_record

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


def _with_preemption(rec: FlightRecord, mtti_s: float) -> FlightRecord:
	c = rec.job.constraints
	c = c.model_copy(
		update={"preemption": c.preemption.model_copy(update={"kind": "poisson", "mtti_s": mtti_s})}
	)
	return rec.model_copy(update={"job": rec.job.model_copy(update={"constraints": c})})


def test_on_demand_total_is_the_gpu_line_and_matches_cost_total(ft):
	basis, jc = api.run_cost(ft)
	assert basis.price_table == f"prices@{prices.PRICE_TABLE_VERSION}" and basis.billing == "on_demand"
	assert [x.label_key for x in jc.lines] == ["cost.line.gpu"]
	line = jc.lines[0]
	assert line.kind == "gpu" and line.quantity.unit == "gpu_hours" and line.rate_usd == pytest.approx(0.80)
	assert line.price_row == "gpu/modal/L4/us/on_demand"
	assert line.usd.point == pytest.approx(line.quantity.point * line.rate_usd)
	assert jc.total_usd.point == pytest.approx(api.predict(ft).root["cost.total"].point)
	assert jc.gpu_hours.point == pytest.approx(line.quantity.point)
	assert jc.market_equivalent_usd is None


def test_money_at_risk_sits_beside_the_total_and_spans_the_bin(ft):
	_, jc = api.run_cost(ft)
	risk, total = jc.at_risk_usd, jc.total_usd
	assert risk.status != "abstain"
	assert "first eval (200 steps in)" in risk.reason  # the eval batch peaks at the first eval, 200 steps in
	p = api.predict(ft).root["risk.oom"].point
	lo, hi, _ = next(b for b in oom.BINS if b[2] == oom.bin_label(p))
	cost_pt = risk.point / p  # the cost up to the crash
	# the range is the bin's edges times the cost band, which the point sits inside
	assert risk.lo < lo * cost_pt and risk.hi > min(hi, 1.0) * cost_pt
	# the cost band is symmetric in log space, so the two edges' factors multiply to about 1
	assert (risk.lo / (lo * cost_pt)) * (risk.hi / (min(hi, 1.0) * cost_pt)) == pytest.approx(1.0, rel=0.02)
	assert risk.lo <= risk.point <= risk.hi < total.point
	# the total never includes it
	assert total.point == pytest.approx(sum(x.usd.point for x in jc.lines))


def test_a_measured_oom_is_certain_and_still_stops_at_the_first_eval(ft):
	probes = [p.model_copy(update={"status": "oom"}) if p.id == "p2" else p for p in ft.probes]
	rec = ft.model_copy(update={"probes": probes})
	assert api.predict(rec).root["risk.oom"].status == "measured"
	_, jc = api.run_cost(rec)
	_, predicted = api.run_cost(ft)
	risk = jc.at_risk_usd
	assert "a probe ran out of memory" in risk.reason and "first eval (200 steps in)" in risk.reason
	# the whole cost up to the first eval; within 5%, since p2 now drops out of the step-time fit
	cost_to_crash = predicted.at_risk_usd.point / api.predict(ft).root["risk.oom"].point
	assert risk.point == pytest.approx(cost_to_crash, rel=0.05)
	# P is 1 exactly, so the low end isn't pulled down to the bin's 0.7: only the cost band is left
	assert risk.lo / risk.point > 0.75


def test_no_eval_interval_means_the_end_of_the_first_epoch(ft):
	tr = ft.job.script_map.training.model_copy(update={"eval_every": None})
	sm = ft.job.script_map.model_copy(update={"training": tr})
	rec = ft.model_copy(update={"job": ft.job.model_copy(update={"script_map": sm})})
	_, jc = api.run_cost(rec)
	assert "end of the first epoch" in jc.at_risk_usd.reason


def test_a_training_step_peak_crashes_at_the_first_step(pt):
	_, jc = api.run_cost(pt)
	assert "first step" in jc.at_risk_usd.reason
	assert jc.at_risk_usd.hi < 0.01 * jc.total_usd.point


def test_own_cluster_shows_gpu_hours_first_and_a_grey_market_figure(ft):
	_, jc = api.run_cost(ft, billing="own_cluster")
	assert jc.total_usd is None
	assert all(x.usd is None and x.rate_usd is None for x in jc.lines)
	_, od = api.run_cost(ft)
	assert jc.gpu_hours.point == pytest.approx(od.gpu_hours.point)
	assert jc.market_equivalent_usd.point == pytest.approx(od.total_usd.point)
	assert "GPU-hours" in jc.market_equivalent_usd.reason
	assert (
		"GPU-hours at risk" in jc.at_risk_usd.reason.split(";")[0]
	)  # hours lead; dollars are the grey figure
	assert jc.at_risk_usd.point == pytest.approx(od.at_risk_usd.point)  # same money, shown as hours first


def test_spot_adds_redone_work_when_the_interruption_rate_is_known(pt):
	_, plain = api.run_cost(pt, billing="spot")
	assert len(plain.lines) == 1 and "redone work isn't counted" in plain.total_usd.reason
	assert "no spot rate listed" in plain.lines[0].usd.reason
	rec = _with_preemption(pt, 4 * 3600.0)
	_, jc = api.run_cost(rec, billing="spot")
	assert [x.label_key for x in jc.lines] == ["cost.line.gpu", "cost.line.gpu_redo"]
	redo = jc.lines[1]
	assert redo.quantity.point > 0
	# lows and highs add line by line
	assert jc.total_usd.lo == pytest.approx(sum(x.usd.lo for x in jc.lines))
	assert jc.total_usd.hi == pytest.approx(sum(x.usd.hi for x in jc.lines))
	assert jc.gpu_hours.point == pytest.approx(sum(x.quantity.point for x in jc.lines))


def test_unknown_provider_uses_the_cheapest_listed_rate_and_says_so(ft):
	_, jc = api.run_cost(ft, provider="somecloud")
	assert jc.lines[0].rate_usd == pytest.approx(0.80)
	assert "no somecloud rate for L4" in jc.lines[0].usd.reason


def test_unpriced_gpu_abstains_with_the_reason(ft):
	_, jc = api.run_cost(ft, {"gpu_type": "V100-SXM2-32GB"})
	assert jc.total_usd.status == "abstain"
	assert jc.at_risk_usd.status == "abstain"
	# the line says it can't be priced, rather than looking unbilled like an own-cluster line
	assert all(x.usd is not None and x.usd.status == "abstain" for x in jc.lines)


def test_region_and_spot_fallbacks_say_what_they_used(ft):
	_, jc = api.run_cost(ft, region="eu", billing="spot")
	why = jc.lines[0].usd.reason
	assert "no modal (eu) spot or on-demand rate for L4" in why and "on-demand rate (modal, us)" in why


def test_more_gpus_split_the_steps_but_bill_every_gpu(pt):
	_, one = api.run_cost(pt)
	_, two = api.run_cost(pt, {"gpus": 2})
	if two.total_usd.status == "abstain":
		pytest.skip(two.total_usd.reason)
	assert "2 x" in two.lines[0].quantity.reason
	# half the steps on twice the GPUs, plus the all-reduce: never fewer GPU-hours than one GPU
	assert two.gpu_hours.point >= 0.99 * one.gpu_hours.point
	assert two.total_usd.point == pytest.approx(two.gpu_hours.point * two.lines[0].rate_usd)


def test_redone_work_counts_on_demand_too(pt):
	_, jc = api.run_cost(_with_preemption(pt, 4 * 3600.0))
	assert [x.label_key for x in jc.lines] == ["cost.line.gpu", "cost.line.gpu_redo"]
	_, own = api.run_cost(_with_preemption(pt, 4 * 3600.0), billing="own_cluster")
	assert own.gpu_hours.point == pytest.approx(jc.gpu_hours.point)


def test_agent_cost_is_a_tokens_line(ag):
	base = ag.agent_sample.tasks
	tasks = [base[i % len(base)].model_copy(update={"task_id": f"x{i}"}) for i in range(30)]
	rec = ag.model_copy(update={"agent_sample": ag.agent_sample.model_copy(update={"tasks": tasks})})
	_, jc = api.run_cost(rec)
	(line,) = jc.lines
	assert line.kind == "tokens" and line.quantity.unit == "tokens"
	pred = api.predict(rec).root
	assert line.usd.point == pytest.approx(pred["cost.total"].point)
	assert line.rate_usd == pytest.approx(pred["cost.total"].point / pred["tokens.total"].point)
	assert jc.total_usd.point == pytest.approx(line.usd.point)
	assert jc.gpu_hours is None and jc.at_risk_usd is None
	_, few = api.run_cost(ag)  # six tasks: the total honestly abstains
	assert few.total_usd.status == "abstain"


def test_fill_predictions_replaces_the_fixture_breakdown(ft):
	assert ft.cost_basis.price_table == "golden-fixture@0"
	out = api.fill_predictions(ft)
	assert out.cost_basis.price_table.startswith("prices@")
	assert out.job_cost.lines[0].price_row == "gpu/modal/L4/us/on_demand"
	assert api.MODEL_ID_RUN_COST in {m.id for m in out.models}
	FlightRecord.model_validate(out.model_dump())  # still a valid record
	again = api.fill_predictions(out)  # filling twice keeps one card per model
	ids = [m.id for m in again.models]
	assert len(ids) == len(set(ids))


def test_every_price_row_names_a_gpu_the_engine_knows():
	for row in prices.GPU_RATES:
		assert gpus.resolve(row.gpu) == row.gpu
		assert gpus.price(row.gpu, row.provider) is not None or row.kind != "on_demand"
