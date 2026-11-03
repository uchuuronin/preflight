"""scripts/cross_gpu_real.py: the real-model pair for the cross-GPU row reads and scores every cell."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_every_cell_is_predicted_and_scored():
	spec = importlib.util.spec_from_file_location("cgr", ROOT / "scripts/cross_gpu_real.py")
	m = importlib.util.module_from_spec(spec)
	spec.loader.exec_module(m)
	rep = m.check(json.loads((ROOT / "tests/fixtures/validation/cross_gpu_real_nov2.json").read_text()))
	assert rep["n"] == 4 and all(r["predicted_s"] and r["ratio"] for r in rep["cells"])
	assert 0 <= rep["within_20"] <= rep["n"] and 0 <= rep["inside"] <= rep["n"]
