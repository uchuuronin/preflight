"""RL post-training models (engine/rl.py) and their targets on the RL lens (engine.api, behind FEATURES["rl"])."""

import math
from pathlib import Path

import numpy as np
import pytest

from preflight.engine import api, rl
from preflight.schema import FlightRecord, load_record
from preflight.schema.record import GenerationStats, Knob, LengthStats, RLSignals

EX = Path(__file__).resolve().parents[1] / "schema" / "examples"


# ---------------------------------------------------------------- zero-variance groups


def test_zero_spread_is_p_to_the_g_plus_its_complement():
	assert rl.zero_spread([0.5] * 40, 4, None).point == pytest.approx(2 * 0.5**4)
	assert rl.zero_spread([0.0, 1.0], 8, None).point == pytest.approx(1.0)
	mixed = rl.zero_spread([0.2, 0.8], 3, None)
	assert mixed.point == pytest.approx(0.2**3 + 0.8**3)
	assert mixed.lo <= mixed.point <= mixed.hi


def test_a_prompt_seen_passing_4_of_4_isnt_certain_to_pass_16_of_16():
	plug_in = rl.zero_spread([1.0] * 50, 16, None).point
	posterior = rl.zero_spread([1.0] * 50, 16, 4)
	assert plug_in == 1.0
	# Jeffreys Beta(4.5, 0.5): E[p^16] exactly, by the Beta-function identity
	exact = math.exp(math.lgamma(20.5) + math.lgamma(5.0) - math.lgamma(4.5) - math.lgamma(21.0))
	exact += math.exp(math.lgamma(16.5) + math.lgamma(5.0) - math.lgamma(0.5) - math.lgamma(21.0))
	assert posterior.point == pytest.approx(exact, rel=1e-9)
	assert posterior.lo < posterior.point < posterior.hi < 1.0


def test_bigger_groups_waste_fewer_prompts():
	rates = [0.1, 0.3, 0.5, 0.7, 0.9] * 8
	fr = [rl.zero_spread(rates, g, 8).point for g in (2, 4, 8, 16)]
	assert fr == sorted(fr, reverse=True)


# ---------------------------------------------------------------- rollout lengths


def test_the_batch_maximum_is_exact_for_drawing_with_replacement():
	x = np.random.default_rng(1).lognormal(5.5, 0.6, 50)
	brute = np.minimum(np.random.default_rng(2).choice(x, (200_000, 16)), 400).max(axis=1).mean()
	assert rl.expected_batch_max(x, 16, 400) == pytest.approx(brute, rel=0.005)
	assert rl.expected_batch_max(x, 1, None) == pytest.approx(x.mean())


def test_lognormal_fit_hits_the_mean_and_p95():
	mu, sigma = rl.lognormal_from(300, 700)
	assert math.exp(mu + sigma**2 / 2) == pytest.approx(300)
	assert math.exp(mu + 1.6448536269514722 * sigma) == pytest.approx(700)


def test_rollout_follows_the_longest_response_and_the_cap():
	s = rl.LengthSample(mean=300, p95=700, max=1024, n=64)
	same = rl.rollout_seconds(10.0, s, 32, 1024, 32, 1024)
	assert same.point == pytest.approx(10.0) and same.lo < 10.0 < same.hi
	more = rl.rollout_seconds(10.0, s, 32, 1024, 128, 1024)
	assert more.point > 10.0  # more sequences per call: a longer longest response
	short = rl.rollout_seconds(10.0, s, 32, 1024, 32, 256)
	assert short.point < 0.5 * 10.0  # a tight cap cuts the tail that set the pace
	assert rl.truncation_share(s, 256) > rl.truncation_share(s, 1024)


# ---------------------------------------------------------------- on the RL lens


def _rl_record() -> FlightRecord:
	base = load_record((EX / "finetune.json").read_text())
	knobs = [
		*base.knobs,
		Knob(
			id="group_size", label_key="knob.group_size", kind="int", current=8, domain={"min": 2, "max": 64}
		),
		Knob(
			id="max_completion_len",
			label_key="knob.max_completion_len",
			unit="tokens",
			kind="int",
			current=512,
			domain={"min": 64, "max": 4096},
		),
	]
	rates = [0.0, 0.125, 0.25, 0.5, 0.75, 0.875, 1.0, 1.0] * 4
	probes = []
	for p in base.probes:
		st = (
			p.step_time.model_copy(
				update={"by_phase": {"rollout": 0.6, "forward": 0.15, "backward": 0.2, "optimizer": 0.05}}
			)
			if p.step_time
			else None
		)
		probes.append(
			p.model_copy(
				update={
					"knobs": {**p.knobs, "group_size": 8, "max_completion_len": 512},
					"step_time": st,
					"generation": GenerationStats(
						calls=4,
						seconds_total=4.0,
						sequences=64,
						new_tokens_mean=220,
						new_tokens_p95=480,
						new_tokens_max=512,
						max_new_tokens=512,
						truncation_rate=0.04,
					),
					"rl": RLSignals(
						response_length=LengthStats(mean=220, p95=480, max=512, n=64),
						pass_rate_per_prompt=rates,
						group_size=8,
						zero_spread_fraction=0.3,
						truncation_rate=0.04,
					),
				}
			)
		)
	job = base.job.model_copy(update={"lens": "rl"})
	return base.model_copy(update={"job": job, "knobs": knobs, "probes": probes})


@pytest.fixture
def rl_on(monkeypatch):
	monkeypatch.setitem(api.FEATURES, "rl", True)


def test_rl_targets_stay_off_until_the_flag(monkeypatch):
	rec = _rl_record()
	monkeypatch.setitem(api.FEATURES, "rl", False)
	assert "rl.zero_spread" not in api.predict(rec).root
	assert "rl.zero_spread" not in {t.id for t in api.targets_for(rec)}


def test_rl_lens_reports_zero_spread_rollout_and_truncation(rl_on):
	rec = _rl_record()
	pred = api.predict(rec).root
	assert {"rl.zero_spread", "time.rollout", "rl.truncation"} <= set(pred)
	assert {t.id for t in api.targets_for(rec)} >= set(api.RL_TARGETS)
	z = pred["rl.zero_spread"]
	assert z.status == "predicted" and 0 < z.lo <= z.point <= z.hi < 1 and "measured 30%" in z.reason
	bigger = api.predict(rec, {"group_size": 32}).root["rl.zero_spread"]
	assert bigger.point < z.point and bigger.status == "extrapolated"
	t = pred["rl.truncation"]
	assert "measured 4%" in t.reason and 0 <= t.point <= 1


def _as_trl(rec: FlightRecord, truncation: float | None = None) -> FlightRecord:
	sm = rec.job.script_map.model_copy(update={"framework": "trl"})
	out = rec.model_copy(update={"job": rec.job.model_copy(update={"framework": "trl", "script_map": sm})})
	if truncation is None:
		return out
	probes = [
		p.model_copy(
			update={
				"rl": p.rl.model_copy(update={"truncation_rate": truncation}),
				"generation": p.generation.model_copy(update={"truncation_rate": truncation}),
			}
		)
		for p in out.probes
	]
	return out.model_copy(update={"probes": probes})


def test_the_batch_and_the_cap_move_step_time_through_the_rollout(rl_on):
	rec = _as_trl(_rl_record())
	base = api.predict(rec).root
	bigger = api.predict(rec, {"batch_size": 8}).root
	assert bigger["time.rollout"].point > base["time.rollout"].point  # more sequences: a longer longest one
	capped = api.predict(rec, {"max_completion_len": 128}).root
	assert capped["time.rollout"].point < base["time.rollout"].point
	assert capped["time.step"].point < base["time.step"].point and "rollout:" in capped["time.step"].reason
	assert capped["time.total"].point < base["time.total"].point


def test_group_size_in_trl_moves_the_prompts_not_the_rollout(rl_on):
	rec = _as_trl(_rl_record())
	base, g32 = api.predict(rec).root, api.predict(rec, {"group_size": 32}).root
	# TRL's batch counts completions: 4x the group means a quarter of the prompts, the same generation work
	assert g32["time.rollout"].point == pytest.approx(base["time.rollout"].point)
	assert g32["rl.zero_spread"].point < base["rl.zero_spread"].point


def test_group_size_abstains_when_the_trainer_doesnt_say_what_the_batch_counts(rl_on):
	rec = _rl_record()  # hf_trainer: prompts or completions per batch isn't known
	pred = api.predict(rec, {"group_size": 32}).root
	assert (
		pred["time.rollout"].status == "abstain" and "prompts or completions" in pred["time.rollout"].reason
	)
	assert (
		pred["time.step"].status == "abstain"
	)  # its biggest phase can't be carried, so neither can the step
	assert pred["rl.zero_spread"].status != "abstain"  # the closed form doesn't need the batch's meaning


def test_raising_the_cap_past_a_probe_that_hit_it_abstains(rl_on):
	# 0.2% of responses hit the cap: with 16 per call, 3% of calls did, so the longer tail barely set the pace
	ok = api.predict(_as_trl(_rl_record(), truncation=0.002), {"max_completion_len": 1024}).root
	assert ok["time.rollout"].status == "extrapolated"
	t = ok["rl.truncation"]
	assert (
		t.status == "extrapolated" and t.lo == 0.0 and t.hi == pytest.approx(0.002)
	)  # at most what hit the old cap
	# 4% of responses is 48% of calls of 16: the unseen tail sets the pace about half the time
	half = api.predict(_as_trl(_rl_record()), {"max_completion_len": 1024}).root
	assert half["time.rollout"].status == "abstain" and "4%" in half["time.rollout"].reason
	hit = api.predict(_as_trl(_rl_record(), truncation=0.41), {"max_completion_len": 4096}).root
	assert hit["time.rollout"].status == "abstain" and "41%" in hit["time.rollout"].reason
	assert hit["rl.truncation"].status == "abstain"
	same = api.predict(_as_trl(_rl_record(), truncation=0.41)).root["rl.truncation"]
	assert (
		same.status == "predicted" and same.lo <= same.point <= same.hi
	)  # at the probe's own cap it's measured


def test_the_probe_at_the_querys_settings_is_the_one_used(rl_on):
	rec = _as_trl(_rl_record())
	cur = {k.id: k.current for k in rec.knobs}
	match = next(p for p in rec.probes if all(cur.get(k) == v for k, v in p.knobs.items() if k in cur))
	q = {k: v for k, v in match.knobs.items() if k in cur}
	assert api._rl_probe(rec, {**cur, **q}) is match


def test_without_the_model_the_per_phase_line_cant_see_the_cap(rl_on, monkeypatch):
	rec = _as_trl(_rl_record())
	on = (
		api.predict(rec, {"max_completion_len": 128}).root["time.step"].point
		/ api.predict(rec).root["time.step"].point
	)
	monkeypatch.setitem(api.FEATURES, "rl", False)
	off = api.predict(rec, {"max_completion_len": 128}).root
	off_ratio = (
		off["time.step"].point / api.predict(rec).root["time.step"].point if off["time.step"].point else 1.0
	)
	assert on < off_ratio or off["time.step"].status == "abstain"


def test_missing_rl_data_abstains_with_a_reason(rl_on):
	rec = _rl_record()
	bare = rec.model_copy(
		update={"probes": [p.model_copy(update={"rl": None, "generation": None}) for p in rec.probes]}
	)
	pred = api.predict(bare).root
	assert pred["rl.zero_spread"].status == "abstain" and "pass rates" in pred["rl.zero_spread"].reason
	assert pred["time.rollout"].status == "abstain"
	assert pred["time.step"].status != "abstain"  # step time falls back to the per-phase line


def test_rl_cards_say_they_arent_checked_on_a_full_run_yet(rl_on):
	out = api.fill_predictions(_rl_record())
	cards = {m.id: m for m in out.models}
	assert "not yet checked against a full RL run" in cards[rl.MODEL_ID_ZERO].notes
	assert "not yet checked against a full RL run" in cards[rl.MODEL_ID_ROLLOUT].notes
	FlightRecord.model_validate(out.model_dump())


def test_too_few_prompts_abstain(rl_on):
	# Nov 1, GRPO zoo runs 2 and 3: from 3 prompts the band missed 17 new ones; from 8 it covered 9
	rec = _rl_record()
	few = rec.model_copy(
		update={
			"probes": [
				p.model_copy(
					update={"rl": p.rl.model_copy(update={"pass_rate_per_prompt": [0.0, 0.375, 0.125]})}
				)
				for p in rec.probes
			]
		}
	)
	e = api.predict(few, {"group_size": 8}).root["rl.zero_spread"]
	assert e.status == "abstain" and "only 3 prompts" in e.reason
	assert api.predict(rec, {"group_size": 8}).root["rl.zero_spread"].status != "abstain"


def test_rollout_band_covers_the_gap_between_two_grpo_runs():
	# fitted on A's GRPO zoo runs 2 and 3 (Nov 1): run 3's time per call from run 2's lengths and time
	s2 = rl.LengthSample(mean=131.97, p95=440.0, max=512.0, n=32)
	iv = rl.rollout_seconds(60.28 / 4, s2, 8, 512, 8, 512)
	assert iv.lo <= 257.92 / 20 <= iv.hi


def test_the_callers_flags_decide_a_fill_and_feature_falls_back_to_features(monkeypatch):
	"""D, Nov 1: engine.rl in flags.json switched nothing. The runner passes its flag reader now."""
	rec = _rl_record()
	monkeypatch.setitem(api.FEATURES, "rl", False)
	on = api.fill_predictions(rec, flags=lambda k: k == "engine.rl")
	assert "rl.zero_spread" in on.predictions.estimates
	off = api.fill_predictions(rec, flags=lambda k: False)
	assert "rl.zero_spread" not in off.predictions.estimates
	assert "rl.zero_spread" not in api.fill_predictions(rec).predictions.estimates  # FEATURES again
	assert api.feature("rl") is False  # the fill's flags don't outlive it


def test_every_engine_feature_has_its_key_in_flags_json():
	import json
	from pathlib import Path

	flags = json.loads((Path(__file__).resolve().parents[1] / "flags.json").read_text())
	assert set(api.FLAG_OF) == set(api.FEATURES)
	for name, key in api.FLAG_OF.items():
		assert key in flags, key
		assert flags[key] == api.FEATURES[name], f"{key}: flags.json and FEATURES disagree"
	engine_keys = {k for k in flags if k.startswith("engine.")}
	assert engine_keys == {
		*api.FLAG_OF.values(),
		"engine.memory",
	}  # engine.memory: the runner's gate on every fill
	prod = json.loads((Path(__file__).resolve().parents[1] / "flags.production.json").read_text())
	# review, Nov 1: a production override can only name a key the engine reads
	assert {k for k in prod if k.startswith("engine.")} <= engine_keys
