"""GRPO groups from reward calls: pass rate per prompt and zero spread (B's zero-variance model)."""

from preflight.recorder.rl_groups import reward_groups


def test_pass_rate_comes_from_the_binary_reward_and_spread_from_the_sum():
	calls = [
		{"func": 0, "values": [1, 1, 0, 0, 1, 1, 1, 1]},  # correctness: prompt A half right, B all right
		{"func": 1, "values": [0.2, 0.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]},  # format bonus
		{"func": 0, "values": [0, 0, 0, 0, 0, 1, 0, 0]},
		{"func": 1, "values": [0.0] * 8},
	]
	got = reward_groups(calls, 4)
	assert got["group_size"] == 4
	assert got["pass_rate_per_prompt"] == [0.5, 1.0, 0.0, 0.25]
	# summed rewards: A varies, B all 1.2 (no spread), C all 0 (no spread), D varies
	assert got["zero_spread_fraction"] == 0.5


def test_without_a_binary_reward_there_is_no_pass_rate_but_still_spread():
	got = reward_groups([{"func": 0, "values": [0.3, 0.3, 0.7, 0.1]}], 2)
	assert "pass_rate_per_prompt" not in got and got["zero_spread_fraction"] == 0.5


def test_nothing_without_calls_or_a_group_size():
	assert reward_groups([], 8) == {} and reward_groups([{"func": 0, "values": [1, 0]}], None) == {}
	assert reward_groups([{"func": 0, "values": [1, 0]}], 1) == {}


def test_pass_rates_are_capped():
	got = reward_groups([{"func": 0, "values": [1.0, 0.0] * 600}], 2)
	assert len(got["pass_rate_per_prompt"]) == 256
