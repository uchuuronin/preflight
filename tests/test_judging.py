"""The judging window (Nov 5 to 8, ET): the ladder starts at the A10, one worker kept warm."""

from preflight.runner.judging import in_judging, judging_gpu, warm_workers

NOV4_2359 = 1793854740.0  # 2026-11-04 23:59 EST
NOV5_0001 = 1793854860.0  # 2026-11-05 00:01 EST
NOV8_2359 = 1794200340.0  # 2026-11-08 23:59 EST
NOV9_0001 = 1794200460.0  # 2026-11-09 00:01 EST


def test_the_window_is_nov_5_to_8_in_eastern_time():
	assert [in_judging(t) for t in (NOV4_2359, NOV5_0001, NOV8_2359, NOV9_0001)] == [False, True, True, False]
	assert [warm_workers(t) for t in (NOV4_2359, NOV5_0001)] == [0, 1]


def test_during_judging_an_l4_job_starts_on_the_a10_when_allowed():
	assert judging_gpu("L4", None, NOV5_0001) == "A10"
	assert judging_gpu("L4", None, NOV4_2359) == "L4"  # not before judging
	assert judging_gpu("L4", ["L4"], NOV5_0001) == "L4"  # the visitor didn't allow the A10
	assert judging_gpu("A100-SXM-80GB", None, NOV5_0001) == "A100-SXM-80GB"  # a bigger need stays


def test_the_window_moves_with_its_env_settings(monkeypatch):
	monkeypatch.setenv("PREFLIGHT_JUDGING_FROM", "2026-11-04")
	assert in_judging(NOV4_2359)


def test_only_judgings_a10_start_may_fall_back_and_only_before_a_point_ran():
	# a job that moved to the A10 after a fallback keeps its points there: no global A10 -> L4 entry
	from preflight.runner.pipeline import GPU_FALLBACK

	assert "A10" not in GPU_FALLBACK and GPU_FALLBACK["L4"] == ("A10", "L40S")
