"""Recorder overhead from paired step ratios (ground truth): the arithmetic, the alternating mode and the bare run."""

import json
from pathlib import Path

import pytest

from preflight.recorder.overhead import hook_ratios, median_ratio, report, sync_ratios

FIX = Path(__file__).resolve().parent / "fixtures"


def _log(kinds_walls):
	return [{"kind": k, "wall_s": w} for k, w in kinds_walls]


def test_synced_steps_are_paired_with_the_clean_steps_either_side():
	log = _log(
		[("warmup", 9.0), ("timed", 1.0), ("phase", 1.1), ("timed", 1.0), ("phase", 1.2), ("timed", 1.0)]
	)
	assert sync_ratios(log) == pytest.approx([1.1, 1.2])
	# drift cancels: a run slowing down steadily shows no overhead
	drift = _log([("timed", 1.0), ("phase", 1.1), ("timed", 1.2), ("phase", 1.3), ("timed", 1.4)])
	assert sync_ratios(drift) == pytest.approx([1.0, 1.0])


def test_hook_ratios_pair_by_step_index_and_skip_warmup_and_synced_steps():
	log = _log([("warmup", 5.0), ("warmup", 5.0), ("timed", 1.05), ("phase", 1.3), ("timed", 1.02)])
	bare = [None, 4.0, 1.0, 1.0, 1.0]
	assert hook_ratios(log, bare, warmup=2) == pytest.approx([1.05, 1.02])


def test_the_interval_brackets_the_median_and_is_reproducible():
	ratios = [1.0 + 0.01 * (i % 7) for i in range(200)]
	a, b = median_ratio(ratios), median_ratio(ratios)
	assert a == b and a.low <= a.median <= a.high and a.pairs == 200
	assert (
		report(_log([("timed", 1.0), ("phase", 1.1), ("timed", 1.0)]), [], warmup=0)["sync_overhead"][
			"source"
		]
		== "measured"
	)
	with pytest.raises(ValueError):
		median_ratio([])


@pytest.mark.torch
def test_alternating_mode_times_clean_and_synced_steps_from_events(tmp_path):
	from preflight.probes.local import ProbeSpec, run_probe

	src = (FIX / "tiny_loop.py").read_text()
	r = run_probe(ProbeSpec(probe_id="p1", files={"t.py": src}, entrypoint="t.py", warmup=2, timed=8, timeout_s=120,
		alternate_sync=True, out_dir=tmp_path))  # fmt: skip
	assert r.probe.status == "ok", r.probe.error
	kinds = [s.kind for s in r.probe.step_log]
	assert kinds[:2] == ["warmup", "warmup"] and kinds[2:10] == ["timed", "phase"] * 4
	assert len(sync_ratios([{"kind": s.kind, "wall_s": s.wall_s} for s in r.probe.step_log])) == 3


@pytest.mark.torch
def test_the_bare_run_records_only_step_walls(tmp_path):
	from preflight.recorder.bare import main

	script = tmp_path / "t.py"
	script.write_text((FIX / "tiny_loop.py").read_text())
	out = tmp_path / "bare.json"
	assert main([str(script), str(out), "--steps", "6", "--seed", "1"]) == 0
	res = json.loads(out.read_text())
	assert (
		res["steps"] == 6
		and res["ended"] == "steps"
		and len(res["walls_s"]) == 6
		and res["walls_s"][0] is None
	)
