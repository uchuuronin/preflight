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


def test_group_size_and_cap_now_move_step_time(rl_on, monkeypatch):
	rec = _rl_record()
	base = api.predict(rec).root["time.step"]
	more = api.predict(rec, {"group_size": 32}).root
	assert more["time.step"].point > base.point and "rollout:" in more["time.step"].reason
	assert more["time.total"].point > api.predict(rec).root["time.total"].point
	capped = api.predict(rec, {"max_completion_len": 128}).root["time.step"]
	assert capped.point < base.point
	on_change = more["time.step"].point - base.point
	monkeypatch.setitem(api.FEATURES, "rl", False)
	off = (
		api.predict(rec, {"group_size": 32}).root["time.step"].point
		- api.predict(rec).root["time.step"].point
	)
	assert off < 0.5 * on_change  # without the model, the per-phase line barely sees a group-size change


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
