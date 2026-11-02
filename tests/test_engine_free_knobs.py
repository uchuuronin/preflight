"""After the freeze (Nov 2): knobs that change neither memory nor time carry the estimates across
(engine.carry_free_knobs), the grids say so (Grid.independent_of), and the range test's probe stays out of the fits."""

from contextlib import contextmanager
from pathlib import Path

import pytest

from preflight.engine import api, knobs
from preflight.schema import load_record

ROOT = Path(__file__).resolve().parents[1]
LORA = (
	ROOT / "tests/fixtures/b-a-smoke-l4-main-4.record.json"
)  # the zoo LoRA run on the L4, its eval OOM probed
RECORDS = [
	LORA,
	ROOT / "zoo/grpo_trl/record_nov1_a10.json",
	ROOT / "zoo/vision_timm/record_nov1_l4.json",
	ROOT / "zoo/gnn_pyg/record_nov1_l4.json",
	ROOT / "zoo/retrieval_st/record_nov1_l4.json",
]
TARGETS = ("memory.peak", "risk.oom", "time.step", "time.total", "cost.total")


@contextmanager
def _carry(on: bool):
	token = api._FEATURES_NOW.set({**api.FEATURES, "carry_free_knobs": on})
	try:
		yield
	finally:
		api._FEATURES_NOW.reset(token)


def _moved(k):
	v = k.current
	if isinstance(v, bool):
		return not v
	if isinstance(v, int | float):
		return v * 2 if v else 1
	others = [x for x in (k.domain.values or []) if x != v]
	return others[0] if others else None


def _read(pred, t):
	e = pred.get(t)
	return None if e is None else (e.status, e.point, e.lo, e.hi)


def test_every_free_knob_is_free_of_memory_time_and_cost_in_its_spec():
	for kid in api.FREE_KNOBS:
		assert not ({"memory", "time", "cost"} & knobs.knob_spec(kid).affects), kid
	# the ones kept out on purpose: each changes memory or a generation's length in some setting
	assert not {"kl_beta", "lora_dropout", "temperature", "loss_type"} & api.FREE_KNOBS


@pytest.mark.parametrize(
	"path", RECORDS, ids=lambda p: p.parent.name if p.parent.name != "fixtures" else "lora"
)
def test_moving_a_free_knob_carries_every_estimate_across(path):
	rec = load_record(path.read_text())
	free = [k for k in rec.knobs if k.id in api.FREE_KNOBS]
	assert free or path != LORA
	with _carry(True):
		base = api.predict(rec).root
		for k in free:
			v = _moved(k)
			if v is None:
				continue
			moved = api.predict(rec, {k.id: v}).root
			for t in TARGETS:
				assert _read(moved, t) == _read(base, t), (path.name, k.id, t)


def test_off_a_free_knob_other_than_lr_still_drops_the_probe_match():
	"""Why the flag matters: off, moving warmup_steps loses the eval OOM the probe measured (the bug it fixes)."""
	rec = load_record(LORA.read_text())
	with _carry(False):
		base, moved = api.predict(rec).root, api.predict(rec, {"warmup_steps": 100}).root
	assert base["memory.peak"].status == "abstain" and moved["memory.peak"].status == "extrapolated"
	with _carry(True):
		moved = api.predict(rec, {"warmup_steps": 100}).root
	assert moved["memory.peak"].status == "abstain" and moved["risk.oom"].point == 1.0


def test_grids_name_the_knobs_their_target_doesnt_depend_on_only_with_the_flag():
	rec = load_record(LORA.read_text())
	with _carry(False):
		assert all(g.independent_of is None for g in api.build_grids(rec, ["batch_size"]))
	with _carry(True):
		grids = {g.target_id: g for g in api.build_grids(rec, ["batch_size"])}
	present = {k.id for k in rec.knobs} & api.FREE_KNOBS
	assert present >= {"lr", "warmup_steps", "lora_alpha"}
	assert {"memory.peak", "risk.oom", "time.total", "cost.total"} <= set(grids)
	for t, g in grids.items():
		listed = t.split(".")[0] in ("memory", "time", "cost") or t == "risk.oom"
		assert set(g.independent_of or ()) == (present if listed else set()), t
		assert "lora_dropout" not in (g.independent_of or ())
	with _carry(True):
		lr_grid = {g.target_id: g for g in api.build_grids(rec, ["lr"], local=True)}
	assert "lr" not in (lr_grid["memory.peak"].independent_of or ())  # never both an axis and free


def test_a_grid_rejects_a_knob_that_is_both_an_axis_and_independent():
	rec = load_record(LORA.read_text())
	with _carry(True):
		g = api.build_grids(rec, ["batch_size"])[0]
	with pytest.raises(ValueError):
		g.model_validate({**g.model_dump(), "independent_of": ["batch_size"]})


def test_every_listed_knob_really_leaves_its_grid_unchanged():
	"""What a reader does with independent_of (hold the knob anywhere) is what predict does at a moved value."""
	for path in RECORDS:
		rec = load_record(path.read_text())
		axis = next((k.id for k in rec.knobs if k.id == "batch_size"), None)
		if axis is None:
			continue
		with _carry(True):
			for g in api.build_grids(rec, [axis]):
				for kid in g.independent_of or ():
					k = rec.knob(kid)
					moved = rec.model_copy(
						update={
							"knobs": [
								x.model_copy(update={"current": _moved(k)}) if x.id == kid else x
								for x in rec.knobs
							]
						}
					)
					(h,) = [x for x in api.build_grids(moved, [axis]) if x.target_id == g.target_id]
					assert (h.point, h.lo, h.hi, h.abstain) == (g.point, g.lo, g.hi, g.abstain), (
						path.name,
						kid,
						g.target_id,
					)


def test_the_range_tests_probe_is_left_out_of_the_fits():
	rec = load_record(LORA.read_text())
	p1 = next(p for p in rec.probes if p.id == "p1")
	ramp = p1.model_copy(
		update={
			"id": "p1-lr",
			"memory": None,
			"step_time": None,
			"throughput": None,
			"stability": None,
			"step_log": [],
			"extras": {"lr_range.suggested_lr": 7.9e-5, "lr_range.checked": 1.0},
		}
	)
	with_ramp = rec.model_copy(update={"probes": [*rec.probes, ramp]})
	assert api._fit_view(with_ramp).probes == rec.probes
	assert api._fit_view(rec) is rec
	old_key = rec.model_copy(
		update={"probes": [p1.model_copy(update={"extras": {"lr_range_divergence_lr": 0.1}})]}
	)
	assert api._fit_view(old_key) is old_key  # the golden probes' older key isn't the range test's probe
	a, b = api.predict(rec).root, api.predict(with_ramp).root
	assert {t: _read(a, t) for t in TARGETS} == {t: _read(b, t) for t in TARGETS}
	filled = api.fill_predictions(with_ramp)
	assert [p.id for p in filled.probes] == [p.id for p in with_ramp.probes]  # it stays on the record
