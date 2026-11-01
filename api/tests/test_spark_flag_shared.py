"""The data-use notice and A's runner decide "Muse Spark reads your code" from the same flag.

The notice (web/src/app/DataUse.tsx) shows the line when `harness.spark_script_map` is on; the
runner may call Muse Spark to map a script only under that same flag. Chiya's decision, Nov 1.
"""

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
	spark = (ROOT / "python/preflight/harness/spark.py").read_text()
	runner = (ROOT / "python/preflight/runner/modal_app.py").read_text()
	m = re.search(r'^SCRIPT_MAP_FLAG\s*=\s*"([a-z0-9_.]+)"', spark, re.M)
	if not m:
		pytest.skip(
			"waits on A's runner change: preflight.harness.spark.SCRIPT_MAP_FLAG (docs/requests/from-c.md)"
		)
	assert m.group(1) == flag
	body = runner[runner.index("def _spark_client") :].split("\n\n\n", 1)[0]
	assert "SCRIPT_MAP_FLAG" in body or flag in body, (
		"the runner's Muse Spark client must be gated on the flag"
	)


def test_the_flag_exists_and_production_can_set_it():
	import json

	assert "harness.spark_script_map" in json.loads((ROOT / "flags.json").read_text())
