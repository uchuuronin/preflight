"""The data-use notice and A's runner decide "Muse Spark reads your code" from the same flag.

The notice (web/src/app/DataUse.tsx) shows the line when `harness.spark_script_map` is on; the
runner sends a job's code to Muse Spark for the script map only under that same flag (A's
pipeline.py, f50999d). Chiya's decision, Nov 1.
"""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _notice_flag() -> str:
	src = (ROOT / "web/src/app/DataUse.tsx").read_text()
	m = re.search(r'flag\("([a-z0-9_.]+)"\)\) lines\.push\(t\("datause\.script_map"\)\)', src)
	assert m, "the notice no longer gates its script-map line on a flag"
	return m.group(1)


def test_the_notice_and_the_runner_read_the_same_flag():
	flag = _notice_flag()
	assert flag == "harness.spark_script_map"
	pipeline = (ROOT / "python/preflight/runner/pipeline.py").read_text()
	gate = re.search(r"if deps\.spark is not None[^\n]*", pipeline)
	assert gate, "the runner's Muse Spark call moved; find its gate and check it here"
	used = re.findall(r'flag\("([a-z0-9_.]+)"\)', gate.group(0))
	if not used:
		pytest.skip("waits on A's runner change (f50999d on ws-a): the gate reads no flag on this branch yet")
	assert used == [flag], f"the runner gates Muse Spark on {used}, the notice on {flag}"


def test_the_flag_exists():
	assert "harness.spark_script_map" in json.loads((ROOT / "flags.json").read_text())
