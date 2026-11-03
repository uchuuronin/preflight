"""The checks page's data (web/src/coverage/coverage.json, read by web/src/app/screens/Checks.tsx) from the final
validation table's own runs (Chiya, Nov 3): the engine against full runs, not a held-out probe.

	uv run --extra probe python scripts/checks_from_final.py docs/validation/final/main.json > web/src/coverage/coverage.json

The same scoring as the final table (`python/preflight/corpus/validate.py` `score_run`, production flags, a run that
calibrated a row left out of that row), in the shape Checks.tsx reads. Four metrics: step time on the same GPU (within
15%), peak memory (within 10%; the engine's point includes the CUDA context, the full run's reserved peak doesn't),
inside the 90% interval (memory, step time and the agent's cost), and out of memory caught (every full run that ran
out: caught at P(OOM) 50% or more). A run with nothing to score is listed under `skipped`.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from preflight.corpus import validate as V
from preflight.engine import api

ROOT = Path(__file__).resolve().parents[1]
BARS = {
	"step_time": (0.15, 0.88, "guess"),
	"peak_memory": (0.10, 0.88, "guess"),
	"interval": (None, 0.88, "cited"),
	"oom": (None, 1.0, "cited"),
}  # fmt: skip (the coverage page's bars, api/scripts/coverage.py RATE_BARS)


def wilson(k: int, n: int, z: float = 1.959963984540054) -> list[float] | None:
	if not n:
		return None
	p = k / n
	d = 1 + z * z / n
	c = (p + z * z / (2 * n)) / d
	h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
	return [round(c - h, 4), round(c + h, 4)]


def _est(e) -> dict | None:
	if e is None:
		return None
	return {k: v for k, v in {"status": e.status, "point": e.point, "lo": e.lo, "hi": e.hi, "model_id": e.model_id,
		"reason": e.reason}.items() if v is not None}  # fmt: skip


def checks_for(run: dict, base: Path) -> tuple[list[dict], dict | None]:
	if run.get("preflight_usd") is not None and not run.get("ground_truth"):
		return (
			[],
			None,
		)  # a check-in scored only for its own cost (the table's cost row), nothing for this page
	if not run.get("record") or not (base / run["record"]).exists():
		return [], {"run": run.get("script"), "why": "nothing_measured", "codes": []}
	cal = set(run.get("calibrates") or ())
	if "*" in cal:
		return [], None
	rec = V._record(base / run["record"])
	gt = (
		V._load(base / run["ground_truth"])
		if run.get("ground_truth") and (base / run["ground_truth"]).exists()
		else {}
	)
	pred = api.predict(rec, run.get("values") or None).root
	bare = gt.get("bare") or {}
	gpu = api._env_gpu(rec)
	head = {"run": run["script"], "probe": "full run", "gpu": gpu, "knobs": {}}
	out = []
	mem, peak = pred.get("memory.peak"), bare.get("peak_reserved_bytes")
	if "memory" not in cal and bare.get("ended") in V.FINISHED and peak:
		ok = V._within(mem.point if mem and mem.status != "abstain" else None, peak, 0.10)
		c = {**head, "metric": "peak_memory", "measured": peak, "predicted": _est(mem)}
		if ok is not None:
			c.update(error=round(mem.point / peak - 1, 4), hit=ok)
		out.append(c)
		inside = V._inside(mem, peak)
		if inside is not None:
			out.append({**head, "metric": "interval", "of": "memory.peak", "measured": peak, "predicted": _est(mem),
				"hit": inside})  # fmt: skip
	step, actual = pred.get("time.step"), V.steady_step_s(gt)
	run_gpu = (run.get("values") or {}).get("gpu_type") or gt.get("gpu")
	if "step" not in cal and actual and gpu and run_gpu:
		if V._same_gpu(
			gpu, run_gpu
		):  # the page's step-time metric is the same-GPU row; another GPU adds its band
			ok = V._within(step.point if step and step.status != "abstain" else None, actual, 0.15)
			c = {**head, "metric": "step_time", "measured": actual, "predicted": _est(step)}
			if ok is not None:
				c.update(error=round(step.point / actual - 1, 4), hit=ok)
			out.append(c)
		inside = V._inside(step, actual)
		if inside is not None:
			out.append({**head, "metric": "interval", "of": "time.step", "measured": actual, "predicted": _est(step),
				"hit": inside})  # fmt: skip
	if V.ran_out(gt):
		oom = pred.get("risk.oom")
		p = oom.point if oom is not None and oom.status != "abstain" else None
		verdict = "abstained" if p is None else ("caught" if p >= V.OOM_ALARM else "missed")
		out.append({**head, "metric": "oom", "measured": "oom", "predicted": _est(oom), "verdict": verdict,
			"predicted_peak": {"point": mem.point} if mem is not None and mem.point else None})  # fmt: skip
	if run.get("agent_total_usd") is not None and "agent" not in cal and "cost.total" in pred:
		e, usd = pred["cost.total"], run["agent_total_usd"]
		inside = V._inside(e, usd)
		if inside is not None:
			out.append({**head, "metric": "interval", "of": "cost.total", "measured": usd, "predicted": _est(e),
				"hit": inside})  # fmt: skip
	return out, None if out else {"run": run["script"], "why": "nothing_measured", "codes": []}


def summary(checks: list[dict]) -> dict:
	out = {}
	for m, (tol, bar, src) in BARS.items():
		cs = [c for c in checks if c["metric"] == m]
		if m == "oom":
			scored = list(cs)
			hits = sum(c["verdict"] == "caught" for c in cs)
			abst = 0
		else:
			scored = [c for c in cs if "hit" in c]
			hits = sum(c["hit"] for c in scored)
			abst = len(cs) - len(scored)
		n = len(scored)
		out[m] = {"tolerance": tol, "rate_bar": bar, "rate_bar_source": src, "n": n, "hits": hits,
			"rate": round(hits / n, 4) if n else None, "wilson95": wilson(hits, n), "abstained": abst,
			"sized_for_seq_len": 0, "runs": len({c["run"] for c in cs})}  # fmt: skip
	return out


def main(argv: list[str]) -> int:
	manifest = Path(argv[0])
	m = json.loads(manifest.read_text())
	flags = V.production_flags(ROOT)
	token = api._FEATURES_NOW.set(api.features_from(lambda k: flags.get(k, False)))
	try:
		checks, skipped = [], []
		for run in m["runs"]:
			cs, sk = checks_for(run, manifest.parent)
			checks += cs
			if sk:
				skipped.append(sk)
	finally:
		api._FEATURES_NOW.reset(token)
	commit = subprocess.run(["git", "log", "-1", "--format=%h", "--", "python/preflight/engine"], cwd=ROOT,
		capture_output=True, text=True).stdout.strip() or None  # fmt: skip
	runs = [r["script"] for r in m["runs"] if "*" not in (r.get("calibrates") or ())
		and not (r.get("preflight_usd") is not None and not r.get("ground_truth"))]  # fmt: skip
	out = {
		"generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%MZ"),
		"engine_commit": commit,
		"inputs": [str(manifest.relative_to(ROOT)) if manifest.is_absolute() else str(manifest)],
		"runs": runs,
		"runs_measured": sorted({c["run"] for c in checks}),
		"summary": summary(checks),
		"checks": checks,
		"skipped": skipped,
	}
	json.dump(out, sys.stdout, indent="\t")
	sys.stdout.write("\n")
	return 0


if __name__ == "__main__":
	sys.exit(main(sys.argv[1:]))
