"""Muse Spark checks a finished record's plots against its own numbers (plan item 31, slim).

	findings, calls, stats = diagnose_record(ModelClient.meta(), rec)

One call per finished record, with two rendered plots, the memory timeline (allocated, reserved
and device-used bytes per step) and the loss curve, and the raw probe numbers they were drawn
from. The model is asked to check one against the other: where a plot shows something the numbers
don't bear out (or the reverse), that's a finding; if they agree, one "consistent" finding. Each
names the plot region and cites one raw number by its path in the numbers it was given.

Code checks every citation before it becomes a finding: the path has to resolve in the record and
the value the model quotes has to match what's there, so a finding can't rest on a number the
record doesn't hold. Findings that fail are left out and counted in stats. Spark Open's check of
each claim against the probes (item 30) runs on what this returns. No search grounding.
Off unless harness.spark_diagnosis is on.
"""

import base64
import io
import json
import math
import re
from typing import Literal

from pydantic import BaseModel, Field

from preflight.schema import Evidence, Finding, FlightRecord

MAX_PROBES = 4  # plotted and sent; the record's first ok probes
MAX_STEPS = 200  # step_log rows per probe in the numbers
MAX_LOSS = 400  # loss points per probe in the numbers
REL_TOL = 0.01  # a quoted value matches the record's within 1%
MAX_CHECKS = 8  # findings kept from one diagnosis
GB = 1e9

SYSTEM = """You check a training run's plots against the numbers they were drawn from.
You get two plots of one record (the memory timeline: allocated, reserved and device-used memory
per step, in GB; and the loss per step) and, as JSON, the raw probe numbers behind them.
Check one against the other. A finding is a place where the plot and the numbers disagree: a
spike, plateau, trend or gap the plot shows that the numbers don't bear out, or a number the plot
should show but doesn't (a peak above the device's memory, a step the plot skips). Also flag
what both show that a user must act on only if the plot misrepresents it. Don't describe the plots.
If they agree, return one finding with verdict "consistent".
Every finding names the plot ("memory_timeline" or "loss"), the region of it (e.g. "steps 30-35,
the allocated line"), and cites exactly one number by its path in the JSON, e.g.
probes[p1].step_log[12].peak_allocated_bytes, probes[p1].stability.series.loss[7] or
probes[p1].memory.peak_allocated_bytes, with that number's value copied exactly as the JSON has it."""


class PlotCheck(BaseModel):
	verdict: Literal["disagrees", "consistent"]
	plot: Literal["memory_timeline", "loss"]
	plot_region: str = Field(description="Where on the plot, e.g. 'steps 30-35, the reserved line'")
	path: str = Field(description="The cited number's path in the JSON, e.g. probes[p1].step_log[12].wall_s")
	value: float = Field(description="That number, copied exactly as the JSON has it")
	detail: str = Field(description="One or two sentences: what the plot shows and what the number says")


class Diagnosis(BaseModel):
	checks: list[PlotCheck]


# ---------------------------------------------------------------- the numbers and the plots


def _probes(rec: FlightRecord) -> list:
	return [p for p in rec.probes if p.status == "ok" and p.step_log][:MAX_PROBES]


def _num(v):
	return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else v


def numbers(rec: FlightRecord) -> dict:
	"""The raw probe numbers the plots are drawn from, keyed as the citations name them."""
	out = {}
	for p in _probes(rec):
		mem = p.memory
		loss = (p.stability.series.get("loss") if p.stability and p.stability.series else None) or []
		out[p.id] = {
			"knobs": p.knobs,
			"steps": p.steps,
			"memory": {
				"peak_allocated_bytes": mem.peak_allocated_bytes if mem else None,
				"peak_reserved_bytes": mem.peak_reserved_bytes if mem else None,
				"device_total_bytes": mem.device_total_bytes if mem else None,
			},
			"step_log": [
				{
					k: _num(getattr(s, k, None))
					for k in (
						"index",
						"kind",
						"wall_s",
						"peak_allocated_bytes",
						"peak_reserved_bytes",
						"device_used_bytes",
					)
				}
				for s in p.step_log[:MAX_STEPS]
			],
			"stability": {"series": {"loss": [_num(x) for x in loss[:MAX_LOSS]]}},
		}
	return {"probes": out}


def _png(fig) -> bytes:
	buf = io.BytesIO()
	fig.savefig(buf, format="png", dpi=80)
	return buf.getvalue()


def render(rec: FlightRecord) -> dict[str, bytes]:
	"""The memory timeline and the loss curve as PNGs, drawn from exactly what numbers() sends."""
	import matplotlib

	matplotlib.use("Agg")
	import matplotlib.pyplot as plt

	nums = numbers(rec)["probes"]
	out = {}
	fig, ax = plt.subplots(figsize=(8, 4))
	for pid, n in nums.items():
		rows = n["step_log"]
		x = [r["index"] for r in rows]
		for key, style in (
			("peak_allocated_bytes", "-"),
			("peak_reserved_bytes", "--"),
			("device_used_bytes", ":"),
		):
			ys = [r[key] / GB if r[key] is not None else math.nan for r in rows]
			if any(not math.isnan(y) for y in ys):
				ax.plot(x, ys, style, label=f"{pid} {key.replace('_bytes', '').replace('_', ' ')}")
		total = n["memory"]["device_total_bytes"]
		if total:
			ax.axhline(total / GB, color="grey", linewidth=0.8, label=f"{pid} device total")
	ax.set_xlabel("step (step_log index)")
	ax.set_ylabel("GB")
	ax.set_title("Memory timeline")
	ax.legend(fontsize=7)
	out["memory_timeline"] = _png(fig)
	plt.close(fig)
	fig, ax = plt.subplots(figsize=(8, 4))
	for pid, n in nums.items():
		loss = n["stability"]["series"]["loss"]
		if loss:
			ax.plot(range(len(loss)), [math.nan if v is None else v for v in loss], label=pid)
	ax.set_xlabel("step (index in stability.series.loss)")
	ax.set_ylabel("loss")
	ax.set_title("Loss")
	ax.legend(fontsize=7)
	out["loss"] = _png(fig)
	plt.close(fig)
	return out


# ---------------------------------------------------------------- the code-side citation check

_PATH = re.compile(r"^probes\[([^\]]{1,64})\]((?:\.[A-Za-z_]{1,64}|\[\d{1,6}\]){1,8})$")  # bounded
_PART = re.compile(r"\.([A-Za-z_]{1,64})|\[(\d{1,6})\]")


def resolve(nums: dict, path: str):
	"""The value at a citation's path in numbers(), or None if the path doesn't resolve."""
	m = _PATH.match(path.strip())
	if not m or m.group(1) not in nums["probes"]:
		return None
	cur = nums["probes"][m.group(1)]
	for key, idx in _PART.findall(m.group(2)):
		try:
			cur = cur[key] if key else cur[int(idx)]
		except (KeyError, IndexError, TypeError, ValueError):
			return None
	return cur if isinstance(cur, int | float) and not isinstance(cur, bool) else None


def _matches(quoted: float, actual: float) -> bool:
	return math.isclose(quoted, actual, rel_tol=REL_TOL, abs_tol=1e-9)


# ---------------------------------------------------------------- the call


def diagnose_record(client, rec: FlightRecord) -> tuple[list[Finding], list, dict]:
	"""(findings, the model calls made, stats). No ok probe with steps: no call."""
	stats = {"checks": 0, "kept": 0, "unresolved": 0, "mismatched": 0}
	if not _probes(rec):
		return [], [], stats
	nums = numbers(rec)
	try:
		pngs = render(rec)
	except ImportError:  # no matplotlib where this runs: no plots, so no call
		stats["no_renderer"] = 1
		return [], [], stats
	content = [{"type": "text", "text": "Raw probe numbers:\n" + json.dumps(nums, separators=(",", ":"))}]
	for name, png in pngs.items():
		content.append({"type": "text", "text": f"Plot: {name}"})
		content.append({"type": "image_url",
			"image_url": {"url": "data:image/png;base64," + base64.b64encode(png).decode()}})  # fmt: skip
	msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]
	before = len(client.calls)
	try:
		d, _ = client.structured(msgs, Diagnosis, purpose="diagnosis", reasoning_effort="high")
	except Exception:
		return [], client.calls[before:], stats  # failed attempts stay in provenance
	findings = []
	for c in d.checks[:MAX_CHECKS]:  # bounded, so the check behind it is too
		stats["checks"] += 1
		actual = resolve(nums, c.path)
		if actual is None:
			stats["unresolved"] += 1
			continue
		if not _matches(c.value, actual):
			stats["mismatched"] += 1
			continue
		stats["kept"] += 1
		pid = _PATH.match(c.path.strip()).group(1)
		row = re.search(r"\.step_log\[(\d{1,6})\]", c.path)
		step = nums["probes"][pid]["step_log"][int(row.group(1))]["index"] if row else None
		consistent = c.verdict == "consistent"
		findings.append(Finding(
			id=f"diagnosis.{len(findings) + 1}", severity="info" if consistent else "warn",
			code=f"diagnosis.{c.verdict}",
			title=("Plots and numbers agree" if consistent else f"The {c.plot.replace('_', ' ')} and its numbers disagree"),
			detail=f"{c.detail.strip()[:400]} (plot: {c.plot}, {c.plot_region.strip()[:120]}; number: {c.path} = {actual})",
			node_id=None, source="spark_diagnosis",
			evidence=[Evidence(metric=c.path, value=float(actual), probe_id=pid, step=step)]))  # fmt: skip
	return findings, client.calls[before:], stats
