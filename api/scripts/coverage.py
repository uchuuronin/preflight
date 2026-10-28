"""The coverage page's numbers: B's engine checked against real probe measurements it never saw.

For every successful probe in a real record (A's runs on Modal), the engine predicts that probe's
setting from the record with that probe removed, and the prediction is scored against what the
probe measured. Nothing is scored against data it was fitted on: the held-out probe, and any
probe that re-ran the same point (an eval OOM's `p1-eval` beside `p1`), are both left out.

Checks, matching the Project Goals' validation table:
  step_time       held-out step time within STEP_TOLERANCE of the probe's median step
  interval        the probe's measurement inside the estimate's 90% interval (target 88%)
  peak_memory     held-out peak within MEMORY_TOLERANCE, only where the probe ran the job as
                  the knobs describe it (see `memory_comparable`)
  oom             each measured out-of-memory: did the engine, without that probe, say "fails"?

Abstentions are listed, never counted as hits or misses. Each proportion carries a Wilson 95%
interval. Run from the repo root (it needs the engine, not the API):

	ENGINE_COMMIT=$(git log -1 --format=%h -- python/preflight/engine) \
		uv run python api/scripts/coverage.py <record.json> ... > web/src/coverage/coverage.json
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from preflight.engine.api import predict
from preflight.engine.memory import FAILS_AT, FITS_BELOW
from preflight.schema import load_record

# Source: cited, Project Goals "Validation targets": step time on the same GPU within 15%.
STEP_TOLERANCE = 0.15
# Source: cited, Project Goals: peak memory within 10%.
MEMORY_TOLERANCE = 0.10
# Source: cited, Project Goals: 90% intervals cover at least 88%.
COVERAGE_TARGET = 0.88
# Source: derived, the usual two-sided 95% normal quantile for the Wilson score interval.
Z95 = 1.959964
OK = ("ok", "cached")


def wilson(k: int, n: int, z: float = Z95) -> list[float] | None:
	"""Wilson score interval for k successes in n trials; None when there are no trials."""
	if n == 0:
		return None
	p = k / n
	d = 1 + z * z / n
	centre = (p + z * z / (2 * n)) / d
	half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
	return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def _siblings(pid: str, other: str) -> bool:
	"""Probes of the same point: p1 and p1-eval (A re-runs an eval OOM's point with eval off)."""
	return other == pid or other.startswith(pid + "-") or pid.startswith(other + "-")


def memory_comparable(rec, probe) -> str | None:
	"""Why a probe's peak can't be scored against a prediction for its knobs, or None if it can.

	After an eval OOM, A re-runs the job's points with eval off, and no knob records that, so those
	probes' peaks are training-only while the engine predicts the job as written, eval included."""
	eval_oom = any(q.error_code == "oom.eval" for q in rec.probes)
	if eval_oom and probe.eval is None:
		return "ran with eval off after the eval ran out of memory; no knob says so, so the engine predicts the job with eval"
	return None


def _est(e) -> dict | None:
	if e is None:
		return None
	d = e.model_dump(mode="json", exclude_none=True)
	return {
		k: d.get(k) for k in ("status", "point", "lo", "hi", "model_id", "reason") if d.get(k) is not None
	}


def check_record(name: str, raw: str) -> tuple[list[dict], list[dict]]:
	rec = load_record(raw)
	checks: list[dict] = []
	skipped: list[dict] = []
	gpu = rec.env.gpu.name if rec.env and rec.env.gpu else None
	if not any(p.status in OK for p in rec.probes) and not any(p.status == "oom" for p in rec.probes):
		codes = sorted({p.error_code or p.status for p in rec.probes})
		skipped.append({"run": name, "why": f"no probe measured anything ({', '.join(codes)})"})
		return checks, skipped
	for p in rec.probes:
		held = rec.model_copy(update={"probes": [q for q in rec.probes if not _siblings(p.id, q.id)]})
		where = {"run": name, "probe": p.id, "gpu": gpu, "knobs": dict(p.knobs)}
		pred = predict(held, dict(p.knobs)).root
		if p.status == "oom":
			risk = pred.get("risk.oom")
			e = _est(risk)
			if not e or e["status"] == "abstain":
				verdict = "abstained"
			elif e["point"] >= FAILS_AT:
				verdict = "caught"
			elif e["point"] < FITS_BELOW:
				verdict = "missed"
			else:
				verdict = "at risk"
			checks.append(
				{
					**where,
					"metric": "oom",
					"measured": p.error_code or "oom",
					"predicted": e,
					"verdict": verdict,
				}
			)
			continue
		if p.status not in OK:
			continue
		st = pred.get("time.step")
		measured = p.step_time.p50_s if p.step_time else None
		e = _est(st)
		if measured and e and e["status"] != "abstain":
			err = abs(e["point"] - measured) / measured
			checks.append(
				{
					**where,
					"metric": "step_time",
					"measured": measured,
					"predicted": e,
					"error": round(err, 4),
					"hit": err <= STEP_TOLERANCE,
				}
			)
			checks.append(
				{
					**where,
					"metric": "interval",
					"of": "time.step",
					"measured": measured,
					"predicted": e,
					"hit": e["lo"] <= measured <= e["hi"],
				}
			)
		elif measured:
			checks.append(
				{**where, "metric": "step_time", "measured": measured, "predicted": e, "verdict": "abstained"}
			)
		peak = p.memory.peak_allocated_bytes if p.memory else None
		why = memory_comparable(rec, p)
		if peak and why:
			skipped.append({**where, "metric": "peak_memory", "why": why})
		elif peak:
			m = _est(pred.get("memory.peak"))
			if m and m["status"] != "abstain":
				err = abs(m["point"] - peak) / peak
				checks.append(
					{
						**where,
						"metric": "peak_memory",
						"measured": peak,
						"predicted": m,
						"error": round(err, 4),
						"hit": err <= MEMORY_TOLERANCE,
					}
				)
				checks.append(
					{
						**where,
						"metric": "interval",
						"of": "memory.peak",
						"measured": peak,
						"predicted": m,
						"hit": m["lo"] <= peak <= m["hi"],
					}
				)
			else:
				checks.append(
					{
						**where,
						"metric": "peak_memory",
						"measured": peak,
						"predicted": m,
						"verdict": "abstained",
					}
				)
	return checks, skipped


def summarise(checks: list[dict]) -> dict:
	out: dict = {}
	targets = {
		"step_time": f"within {STEP_TOLERANCE:.0%}",
		"peak_memory": f"within {MEMORY_TOLERANCE:.0%}",
		"interval": f"at least {COVERAGE_TARGET:.0%} inside the 90% interval",
		"oom": "no missed out-of-memory",
	}
	for metric, target in targets.items():
		rows = [c for c in checks if c["metric"] == metric]
		if metric == "oom":
			scored = [c for c in rows if c["verdict"] in ("caught", "missed", "at risk")]
			k = sum(1 for c in scored if c["verdict"] == "caught")
		else:
			scored = [c for c in rows if "hit" in c]
			k = sum(1 for c in scored if c["hit"])
		n = len(scored)
		out[metric] = {
			"target": target,
			"n": n,
			"hits": k,
			"rate": round(k / n, 4) if n else None,
			"wilson95": wilson(k, n),
			"abstained": sum(1 for c in rows if c.get("verdict") == "abstained"),
		}
	return out


def build(paths: list[str]) -> dict:
	checks: list[dict] = []
	skipped: list[dict] = []
	for path in paths:
		name = Path(path).stem
		c, s = check_record(name, Path(path).read_text())
		checks += c
		skipped += s
	return {
		"generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%MZ"),
		"engine_commit": os.environ.get("ENGINE_COMMIT") or None,
		"runs": [Path(p).stem for p in paths],
		"summary": summarise(checks),
		"checks": checks,
		"skipped": skipped,
	}


if __name__ == "__main__":
	json.dump(build(sys.argv[1:]), sys.stdout, indent="\t")
	sys.stdout.write("\n")
