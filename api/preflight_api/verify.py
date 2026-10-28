"""Verify checks both sides (competitive review item 3): a fix passes only when its target got
better AND nothing the training run means changed. The same function decides the verify run's
result in the API and the pass/fail of Preflight's GitHub Action (.github/actions/verify), so a
fix can't pass in one and fail in the other.

Checks, each against the run before the fix:
  target_improved     the metric the fix was for is better by more than measurement noise
  effective_batch     samples per optimizer step unchanged (batch x accumulation x replicas)
  tokens_seen         tokens per step x steps unchanged, within rounding
  lr_schedule         learning rate, warm-up, run length and the per-step LR series unchanged
  throughput_floor    tokens per second no lower than THROUGHPUT_FLOOR of before
  fixed_seed_loss     loss at the same steps with the same seed within LOSS_TOLERANCE

A check whose inputs weren't measured is "not measured", never a pass: the verify result is then
"error" (can't verify), not "passed".

Stdlib only, so the Action runs it with a bare Python:

	python api/preflight_api/verify.py before.json after.json [--metric memory.peak]
	exit 0 passed, 1 failed, 2 couldn't verify
"""

from __future__ import annotations

import json
import math
import sys
from typing import Any

# Source: guess, two probes of the same setting differ by up to about 2% in peak memory and step
# time (allocator and kernel jitter); replaced by the spread of A's repeated zoo probes.
NOISE = 0.02
# Source: guess, a fix may cost up to a tenth of throughput and still be worth it for memory;
# below that the guide should offer it as a trade-off, not a fix. Revisited with Chiya.
THROUGHPUT_FLOOR = 0.90
# Source: guess, same seed and same effective batch differ by bf16 non-determinism only, well
# under 1% of the loss over the first steps; replaced by A's repeat-run spread on the zoo.
LOSS_TOLERANCE = 0.01
# Source: derived, tokens per step are integers; 0.5% covers padding-length rounding.
TOKENS_TOLERANCE = 0.005
LOWER_IS_BETTER = ("memory.", "time.", "cost.", "risk.")


def _probe(rec: dict, probe_id: str | None = None) -> dict:
	"""The probe to judge: the one named, else the last successful one."""
	probes = rec.get("probes") or []
	if probe_id:
		for p in probes:
			if p.get("id") == probe_id:
				return p
	ok = [p for p in probes if p.get("status") in ("ok", "cached")]
	return ok[-1] if ok else (probes[-1] if probes else {})


def _training(rec: dict) -> dict:
	return ((rec.get("job") or {}).get("script_map") or {}).get("training") or {}


def _knob(rec: dict, probe: dict, *ids: str) -> Any:
	for kid in ids:
		if kid in (probe.get("knobs") or {}):
			return probe["knobs"][kid]
		for k in rec.get("knobs") or []:
			if k.get("id") == kid:
				return k.get("current")
	return None


def effective_batch(rec: dict, probe: dict) -> float | None:
	extras = probe.get("extras") or {}
	if isinstance(extras.get("samples_per_step"), int | float):
		return float(extras["samples_per_step"])
	batch = _knob(rec, probe, "batch_size", "per_device_train_batch_size")
	accum = (
		_knob(rec, probe, "grad_accum", "gradient_accumulation_steps")
		or _training(rec).get("grad_accum")
		or 1
	)
	if isinstance(batch, int | float) and isinstance(accum, int | float):
		return float(batch) * float(accum)
	return None


def total_steps(rec: dict) -> float | None:
	tr = _training(rec)
	if isinstance(tr.get("max_steps"), int | float) and tr["max_steps"] > 0:
		return float(tr["max_steps"])
	return None


def tokens_seen(rec: dict, probe: dict) -> float | None:
	per = (probe.get("extras") or {}).get("tokens_per_step")
	steps = total_steps(rec)
	if isinstance(per, int | float) and steps:
		return float(per) * steps
	return None


def _metric(rec: dict, probe: dict, metric: str) -> float | None:
	if probe.get("status") == "oom" and metric.startswith(("memory.", "risk.")):
		return math.inf  # ran out of memory: worse than any measured peak
	mem = probe.get("memory") or {}
	if metric == "memory.peak":
		v = mem.get("peak_allocated_bytes") or mem.get("peak_reserved_bytes")
		return float(v) if v else None
	if metric.startswith("memory."):
		v = (mem.get("by_term") or {}).get(metric.split(".", 1)[1])
		return float(v) if isinstance(v, int | float) else None
	if metric in ("time.step", "time.step_mean"):
		v = (probe.get("step_time") or {}).get("mean_s")
		return float(v) if v else None
	if metric == "throughput.tokens_per_s":
		v = (probe.get("throughput") or {}).get("tokens_per_s")
		return float(v) if v else None
	est = ((rec.get("predictions") or {}).get("estimates") or {}).get(metric) or {}
	v = est.get("point")
	return float(v) if isinstance(v, int | float) else None


def _series(probe: dict, name: str) -> list[float]:
	return [
		x
		for x in ((probe.get("stability") or {}).get("series") or {}).get(name) or []
		if isinstance(x, int | float)
	]


def _check(cid: str, ok: bool | None, before: Any = None, after: Any = None, detail: str = "") -> dict:
	return {"id": cid, "ok": ok, "before": before, "after": after, "detail": detail}


def _same(a: float | None, b: float | None, rel: float) -> bool | None:
	if a is None or b is None:
		return None
	return abs(a - b) <= rel * max(abs(a), abs(b), 1e-12)


def check(
	before: dict,
	after: dict,
	*,
	metric: str = "memory.peak",
	before_probe: str | None = None,
	after_probe: str | None = None,
) -> dict:
	"""Compare a run before a fix with the verify run after it. Returns the verify result."""
	pb, pa = _probe(before, before_probe), _probe(after, after_probe)
	checks: list[dict] = []

	mb, ma = _metric(before, pb, metric), _metric(after, pa, metric)
	if mb is None or ma is None or (math.isinf(mb) and math.isinf(ma)):
		ok = None if mb is None or ma is None else False
		checks.append(
			_check(
				"target_improved",
				ok,
				mb,
				ma,
				f"{metric} not measured on both sides" if ok is None else "out of memory both times",
			)
		)
	else:
		lower = metric.startswith(LOWER_IS_BETTER)
		better = ma < mb * (1 - NOISE) if lower else ma > mb * (1 + NOISE)
		checks.append(_check("target_improved", better, mb, ma, metric))

	eb, ea = effective_batch(before, pb), effective_batch(after, pa)
	checks.append(_check("effective_batch", None if eb is None or ea is None else eb == ea, eb, ea))

	tb, ta = tokens_seen(before, pb), tokens_seen(after, pa)
	checks.append(_check("tokens_seen", _same(tb, ta, TOKENS_TOLERANCE), tb, ta))

	trb, tra = _training(before), _training(after)
	keys = ("lr", "warmup_steps", "max_steps", "epochs")
	conf_b = {k: trb.get(k) for k in keys}
	conf_a = {k: tra.get(k) for k in keys}
	lrb, lra = _series(pb, "lr"), _series(pa, "lr")
	n = min(len(lrb), len(lra))
	series_same = all(_same(x, y, 1e-6) for x, y in zip(lrb[:n], lra[:n], strict=True)) if n else None
	known = conf_b["lr"] is not None or n > 0
	lr_ok = None if not known else (conf_b == conf_a and series_same is not False)
	checks.append(_check("lr_schedule", lr_ok, conf_b, conf_a, f"{n} steps of LR compared"))

	sb = (pb.get("throughput") or {}).get("tokens_per_s")
	sa = (pa.get("throughput") or {}).get("tokens_per_s")
	floor = None if not sb or sa is None else sa >= THROUGHPUT_FLOOR * sb
	checks.append(_check("throughput_floor", floor, sb, sa, f"floor {THROUGHPUT_FLOOR:.0%} of before"))

	lb, la = _series(pb, "loss"), _series(pa, "loss")
	m = min(len(lb), len(la))
	delta = (
		max((abs(x - y) / max(abs(x), 1e-12) for x, y in zip(lb[:m], la[:m], strict=True)), default=None)
		if m
		else None
	)
	checks.append(
		_check(
			"fixed_seed_loss",
			None if delta is None else delta <= LOSS_TOLERANCE,
			lb[:m][-1:] or None,
			la[:m][-1:] or None,
			f"max relative delta over {m} steps",
		)
	)

	failed = [c for c in checks if c["ok"] is False]
	unknown = [c for c in checks if c["ok"] is None]
	status = "failed" if failed else "error" if unknown else "passed"
	return {
		"status": status,
		"metric": metric,
		"before": None if mb is None or math.isinf(mb) else mb,
		"after": None if ma is None or math.isinf(ma) else ma,
		"fixed_seed_loss_delta": delta,
		"effective_batch_unchanged": checks[1]["ok"],
		"probe_id": pa.get("id"),
		"checks": checks,
	}


def main(argv: list[str]) -> int:
	import argparse

	ap = argparse.ArgumentParser(
		description="Preflight verify: did the fix help, and did nothing else change?"
	)
	ap.add_argument("before")
	ap.add_argument("after")
	ap.add_argument("--metric", default="memory.peak")
	ap.add_argument("--json", action="store_true")
	a = ap.parse_args(argv)
	with open(a.before) as f:
		before = json.load(f)
	with open(a.after) as f:
		after = json.load(f)
	res = check(before, after, metric=a.metric)
	if a.json:
		print(json.dumps(res, indent=2, default=str))
	else:
		for c in res["checks"]:
			mark = {True: "pass", False: "FAIL", None: "not measured"}[c["ok"]]
			print(
				f"{mark:>12}  {c['id']:<17} before {c['before']!s:<14} after {c['after']!s:<14} {c['detail']}"
			)
		print(f"verify: {res['status']}")
	return {"passed": 0, "failed": 1}.get(res["status"], 2)


if __name__ == "__main__":
	sys.exit(main(sys.argv[1:]))
