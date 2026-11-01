"""What the guide endpoint needs from a flight record, read once and cached per record.

Everything the model may name (knob ids, knob values, target ids, map nodes) comes from here,
so typed inputs and the server-built tools can only ever mention things the record holds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

KnobValue = bool | int | float | str


@dataclass(frozen=True)
class KnobView:
	id: str
	kind: str  # int | float | categorical | bool
	current: KnobValue
	values: tuple[KnobValue, ...]  # categorical choices, or grid axis values for numeric knobs
	lo: float | None
	hi: float | None
	needs_code: bool
	patch_template: str | None

	def allows(self, value: Any) -> bool:
		"""Is `value` a setting this knob can take? Booleans are never numbers here."""
		if self.kind == "bool":
			return isinstance(value, bool)
		if self.kind == "categorical":
			return not isinstance(value, bool) and value in self.values
		if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
			return False
		if self.kind == "int" and float(value) != int(value):
			return False
		if self.values and value in self.values:
			return True
		if self.lo is not None and value < self.lo:
			return False
		return not (self.hi is not None and value > self.hi)


@dataclass(frozen=True)
class RecordView:
	key: str  # record_id: the server's own id for the run, never taken from the request
	lens: str
	title: str
	knobs: dict[str, KnobView]
	targets: dict[str, str]  # id -> unit
	nodes: frozenset[str]
	budget_usd: float | None
	samples_tasks: bool
	extra: dict[str, Any] = field(default_factory=dict)
	#: target id -> the readout it drives (memory, time, cost, risk), for the number limits
	readouts: dict[str, str] = field(default_factory=dict)
	#: GPUs the record ran on, for GPU-hours when no gpus knob says otherwise
	gpu_count: int = 1

	@classmethod
	def from_record(cls, rec: dict) -> RecordView:
		grids = (rec.get("predictions") or {}).get("grids") or []
		axis_values: dict[str, set] = {}
		for g in grids:
			for ax in g.get("axes", []):
				axis_values.setdefault(ax["knob_id"], set()).update(
					v for v in ax.get("values", []) if not isinstance(v, bool)
				)
		knobs: dict[str, KnobView] = {}
		for k in rec.get("knobs") or []:
			dom = k.get("domain") or {}
			vals = list(dom.get("values") or [])
			if k.get("kind") in ("int", "float"):
				vals = sorted({*vals, *axis_values.get(k["id"], set())} - {None})
			knobs[k["id"]] = KnobView(
				id=k["id"],
				kind=k.get("kind", "categorical"),
				current=k.get("current"),
				values=tuple(vals),
				lo=dom.get("min"),
				hi=dom.get("max"),
				needs_code=bool(k.get("needs_code")),
				patch_template=k.get("patch_template"),
			)
		targets = {t["id"]: t.get("unit", "") for t in (rec.get("predictions") or {}).get("targets") or []}
		job = rec.get("job") or {}
		constraints = job.get("constraints") or {}
		return cls(
			key=str(rec.get("record_id") or rec.get("share_token") or "unknown"),
			lens=str(job.get("lens", "")),
			title=str(job.get("title") or ""),
			knobs=knobs,
			targets=targets,
			nodes=frozenset(n["id"] for n in (rec.get("graph") or {}).get("nodes") or []),
			budget_usd=constraints.get("budget_usd"),
			samples_tasks=bool(rec.get("agent_sample")) or job.get("lens") == "agent",
			readouts={
				t["id"]: t["readout"]
				for t in (rec.get("predictions") or {}).get("targets") or []
				if t.get("readout")
			},
			gpu_count=int(((rec.get("env") or {}).get("gpu") or {}).get("count") or 1),
		)

	def target_for(self, readout: str) -> str | None:
		return next((t for t, r in self.readouts.items() if r == readout), None)
