"""The coverage page's numbers: B's engine checked against real probe measurements it never saw.

For every successful probe in a real record (A's runs on Modal), the engine predicts that probe's
setting from the record with that probe removed, and the prediction is scored against what the
probe measured. Nothing is scored against data it was fitted on: the held-out probe, and any
probe that re-ran the same point (an eval OOM's `p1-eval` beside `p1`), are both left out.

Checks, matching the Project Goals' validation table:
  step_time       held-out step time within STEP_TOLERANCE of the probe's median step (the
                  engine fits on p50_s too)
  interval        the probe's measurement inside the estimate's 90% interval (target 88%)
  peak_memory     held-out peak within MEMORY_TOLERANCE of the probe's reserved peak: the engine
                  predicts reserved bytes plus a CUDA-context constant, which comes off, and for a
                  probe that ran no eval pass its eval term comes off too (`memory_target`)
  oom             each measured out-of-memory: without that probe, did the engine say "fails"
                  or "at risk" (both warn the visitor), or "fits" (a miss)?

Each prediction has one probe fewer than the cockpit has, so this is a harder test than the
cockpit's own numbers face; with three probes it can be close to reading a neighbour back.

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

from preflight.engine import conformal
from preflight.engine.api import (
	_MATCH_IGNORES,
	_probe_values,
	predict_as_run,
	probe_ran_eval,
)  # the engine's own idea of "same point", and its predictions as a probe ran
from preflight.engine.memory import DEFAULT_CONTEXT_BYTES, FAILS_AT, FITS_BELOW
from preflight.schema import load_record

# Source: cited, Project Goals "Validation targets": step time on the same GPU within 15%.
STEP_TOLERANCE = 0.15
# Source: cited, Project Goals: peak memory within 10%.
MEMORY_TOLERANCE = 0.10
# The rate each metric is held to, with where it comes from (the page shows both).
RATE_BARS = {
	# Source: cited, Project Goals: 90% intervals cover at least 88%.
	"interval": (0.88, "cited"),
	# Source: guess, the Project Goals give a tolerance per check but no rate; the same 88% as the
	# intervals until Chiya sets one.
	"step_time": (0.88, "guess"),
	"peak_memory": (0.88, "guess"),
	# Source: cited, Project Goals: no missed out-of-memory.
	"oom": (1.0, "cited"),
}
OK = ("ok", "cached")


def wilson(k: int, n: int) -> list[float] | None:
	"""B's Wilson 95% interval (engine/conformal.py), rounded outwards so rounding never narrows it."""
	if n == 0:
		return None
	lo, hi = conformal.wilson(k, n)
	return [max(0.0, math.floor(lo * 1e4) / 1e4), min(1.0, math.ceil(hi * 1e4) / 1e4)]


def _point(rec, probe) -> dict:
	values = _probe_values(rec, probe)
	return {k: v for k, v in values.items() if k not in _MATCH_IGNORES}


def siblings(rec, probe, other) -> bool:
	"""Probes of the same point, held out together: an eval OOM's `p1-eval` beside `p1`, a re-run
	or cache hit at the same knob values, or the same cache key."""
	if other.id == probe.id or other.id.startswith(probe.id + "-") or probe.id.startswith(other.id + "-"):
		return True
	if probe.cache_key and other.cache_key == probe.cache_key:
		return True
	return _point(rec, other) == _point(rec, probe)


def ran_without_eval(probe) -> bool:
	"""B's one rule (engine.api.probe_ran_eval): A's `extras.eval_off` marker first, then whether the
	probe recorded eval stats."""
	return not probe_ran_eval(probe)


def memory_target(probe, pred) -> tuple[dict | None, str]:
	"""The prediction to score a probe's reserved peak against, and how it was formed. `pred` is the
	engine's prediction for the probe as it ran (engine.api.predict_as_run: no eval pass when the
	probe ran none), so only the engine's CUDA-context constant comes off."""
	peak = _est(pred.get("memory.peak"))
	if not peak or peak["status"] == "abstain":
		return peak, "abstained"
	ctx = float(DEFAULT_CONTEXT_BYTES)
	m = {**peak, **{k: peak[k] - ctx for k in ("point", "lo", "hi") if k in peak}}
	return m, "peak_less_eval_and_context" if ran_without_eval(probe) else "peak_less_context"


def longest_row(probe) -> float | None:
	"""The longest padded row the probe processed (its step log), or None when it logged none."""
	rows = [s.padded_len for s in (probe.step_log or []) if s.padded_len]
	return float(max(rows)) if rows else None


def sized_for_seq_len(probe) -> tuple[float, float] | None:
	"""(longest row, seq_len) when the probe's rows were all shorter than its seq_len. The engine
	sizes memory by seq_len on purpose (a full run meets the dataset's longest row, which a few
	probe batches don't), so such a probe's measured peak is below what the engine should predict:
	its memory check is labelled and kept out of the score until A records the longest row
	(Chiya, Oct 30)."""
	longest = longest_row(probe)
	seq = (probe.knobs or {}).get("seq_len")
	if longest is None or not isinstance(seq, (int, float)) or longest >= seq:
		return None
	return longest, float(seq)


def _est(e) -> dict | None:
	if e is None:
		return None
	d = e.model_dump(mode="json", exclude_none=True)
	return {
		k: d.get(k) for k in ("status", "point", "lo", "hi", "model_id", "reason") if d.get(k) is not None
	}


def held_out(rec, probe):
	"""The record without the probe and its siblings, and without anything derived from them."""
	keep = [q for q in rec.probes if not siblings(rec, probe, q)]
	return rec.model_copy(
		update={"probes": keep, "predictions": type(rec.predictions)(), "models": [], "findings": []}
	)


def check_record(name: str, raw: str) -> tuple[list[dict], list[dict]]:
	rec = load_record(raw)
	checks: list[dict] = []
	skipped: list[dict] = []
	gpu = rec.env.gpu.name if rec.env and rec.env.gpu else None
	if not any(p.status in OK or p.status == "oom" for p in rec.probes):
		codes = sorted({p.error_code or p.status for p in rec.probes})
		skipped.append({"run": name, "why": "nothing_measured", "codes": codes})
		return checks, skipped
	for p in rec.probes:
		if p.status not in OK and p.status != "oom":
			continue
		where = {"run": name, "probe": p.id, "gpu": gpu, "knobs": dict(p.knobs)}
		pred = predict_as_run(held_out(rec, p), p).root
		if p.status == "oom":
			e = _est(pred.get("risk.oom"))
			if not e or e["status"] == "abstain":
				verdict = "abstained"
			elif e["point"] >= FAILS_AT:
				verdict = "caught"
			elif e["point"] < FITS_BELOW:
				verdict = "missed"
			else:
				verdict = "warned"
			peak = _est(pred.get("memory.peak"))
			checks.append(
				{
					**where,
					"metric": "oom",
					"measured": p.error_code or "oom",
					"predicted": e,
					"predicted_peak": peak,
					"verdict": verdict,
				}
			)
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
		reserved = p.memory.peak_reserved_bytes if p.memory else None
		short = sized_for_seq_len(p) if reserved else None
		if reserved and short:
			m, basis = memory_target(p, pred)
			checks.append(
				{
					**where,
					"metric": "peak_memory",
					"measured": reserved,
					"predicted": m,
					"basis": basis,
					"verdict": "sized_for_seq_len",
					"longest_row": short[0],
					"seq_len": short[1],
				}
			)
		elif reserved:
			m, basis = memory_target(p, pred)
			if m and m["status"] != "abstain":
				err = abs(m["point"] - reserved) / reserved
				row = {**where, "measured": reserved, "predicted": m, "basis": basis}
				checks.append(
					{**row, "metric": "peak_memory", "error": round(err, 4), "hit": err <= MEMORY_TOLERANCE}
				)
				checks.append(
					{**row, "metric": "interval", "of": "memory.peak", "hit": m["lo"] <= reserved <= m["hi"]}
				)
			else:
				checks.append(
					{
						**where,
						"metric": "peak_memory",
						"measured": reserved,
						"predicted": m,
						"verdict": "abstained",
					}
				)
	return checks, skipped


def summarise(checks: list[dict]) -> dict:
	out: dict = {}
	tolerance = {"step_time": STEP_TOLERANCE, "peak_memory": MEMORY_TOLERANCE, "interval": None, "oom": None}
	for metric, tol in tolerance.items():
		rows = [c for c in checks if c["metric"] == metric]
		if metric == "oom":
			scored = [c for c in rows if c["verdict"] in ("caught", "warned", "missed")]
			k = sum(1 for c in scored if c["verdict"] != "missed")  # a warning is not a miss
		else:
			scored = [c for c in rows if "hit" in c]
			k = sum(1 for c in scored if c["hit"])
		n = len(scored)
		bar, bar_source = RATE_BARS[metric]
		out[metric] = {
			"tolerance": tol,
			"rate_bar": bar,
			"rate_bar_source": bar_source,
			"n": n,
			"hits": k,
			"rate": round(k / n, 4) if n else None,
			"wilson95": wilson(k, n),
			"abstained": sum(1 for c in rows if c.get("verdict") == "abstained"),
			"sized_for_seq_len": sum(1 for c in rows if c.get("verdict") == "sized_for_seq_len"),
			"runs": len({c.get("run") for c in scored}),
		}
	return out


def run_name(path: str) -> str:
	return Path(path).name.removesuffix(".json").removesuffix(".record")


def build(paths: list[str]) -> dict:
	checks: list[dict] = []
	skipped: list[dict] = []
	for path in paths:
		c, s = check_record(run_name(path), Path(path).read_text())
		checks += c
		skipped += s
	measured = sorted({c["run"] for c in checks})
	return {
		"generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%MZ"),
		"engine_commit": os.environ.get("ENGINE_COMMIT") or None,
		"inputs": [Path(p).as_posix() for p in paths],
		"runs": [run_name(p) for p in paths],
		"runs_measured": measured,
		"summary": summarise(checks),
		"checks": checks,
		"skipped": skipped,
	}


if __name__ == "__main__":
	json.dump(build(sys.argv[1:]), sys.stdout, indent="\t")
	sys.stdout.write("\n")
