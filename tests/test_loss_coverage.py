"""scripts/loss_coverage.py: held-out curves in, coverage of each half of the loss forecast out (Nov 2)."""

import gzip
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("loss_coverage", ROOT / "scripts" / "loss_coverage.py")
lc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lc)


def _row(key, diverged=False, n=200):
	loss = [2.0 + 4.0 * (t + 1) ** -0.5 for t in range(n)]
	return {"key": key, "gpu": "L4", "config": {"params": {"precision": "bf16"}}, "label": {"diverged": diverged},
		"series": {"loss": loss}}  # fmt: skip


def test_prepare_keeps_runs_that_didnt_diverge_and_scores_each_half(tmp_path, monkeypatch):
	monkeypatch.setattr(lc, "SEEN", (60,))
	rows = tmp_path / "rows.jsonl.gz"
	with gzip.open(rows, "wt") as f:
		for r in (_row("a"), _row("b"), _row("c", diverged=True)):
			f.write(json.dumps(r) + "\n")
	curves = tmp_path / "curves.json.gz"
	got = lc.prepare([rows], curves)
	assert got == {"curves": 2, "skipped": {"diverged": 1, "short or non-finite": 0}}
	truth = sum(2.0 + 4.0 * (t + 1) ** -0.5 for t in range(190, 200)) / 10
	# LC-PFN's half: one curve covered, one missed, so it sits at 50%
	pfn = tmp_path / "pfn.json"
	pfn.write_text(json.dumps({"rows": [
		{"key": "a", "seen": 60, "q": [truth * 0.95, truth, truth * 1.05]},
		{"key": "b", "seen": 60, "q": [truth * 1.1, truth * 1.2, truth * 1.3]},
	]}))  # fmt: skip
	rep = lc.score(curves, pfn)
	r = rep["by_seen"]["60"]
	assert (r["lcpfn"]["covered"], r["lcpfn"]["scored"]) == (1, 2) and not r["lcpfn"]["meets_90"]
	assert r["power"]["scored"] + r["power"]["abstained"] == 2
	assert rep["lcpfn_wired"] is False  # LC-PFN covered 1 of 2
	# no LC-PFN forecasts at all: never wired, whatever the power law's own coverage
	pfn.write_text(json.dumps({"rows": [{"key": "a", "seen": 60, "q": None, "error": "boom"}]}))
	assert lc.score(curves, pfn)["lcpfn_wired"] is False
