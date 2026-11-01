"""Typed inputs for POST /v1/guide/turns. Nothing here carries free text to the model except
the visitor's own message (bounded); every id must name something in the record, and every
number is a number. Validation against the record happens in `check_*` below, after the
record is resolved on the server.
"""

from __future__ import annotations

import math
import re
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .record_view import RecordView

Id = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_.:-]{0,79}$")]
Code = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_.]{0,47}$")]
Unit = Annotated[str, Field(pattern=r"^[A-Za-z/_%]{0,16}$")]
Value = bool | int | float | Annotated[str, Field(max_length=64, pattern=r"^[A-Za-z0-9_.!:+-]{1,64}$")]
MESSAGE_MAX = 500  # source: guess, a long question; anything longer is a document, not a turn


class Strict(BaseModel):
	model_config = ConfigDict(extra="forbid", frozen=True)


def _finite(v: float | None) -> float | None:
	if v is not None and not math.isfinite(v):
		raise ValueError("numbers must be finite")
	return v


class Readout(Strict):
	point: float | None = None
	lo: float | None = None
	hi: float | None = None
	unit: Unit = ""
	status: Literal["measured", "predicted", "extrapolated", "abstain"]
	reason: Code | None = None

	@field_validator("point", "lo", "hi")
	@classmethod
	def _finite_numbers(cls, v: float | None) -> float | None:
		return _finite(v)


class Objective(Strict):
	kind: Literal[
		"min_cost", "min_time", "fit_memory", "min_risk", "max_throughput", "max_quality_per_usd", "custom"
	]
	target_id: Id | None = None


Positive = Annotated[float, Field(gt=0, le=1e12)]


class Limits(Strict):
	"""The visitor's confirmed limits on the guide (web/src/guide/limits.ts), enforced here on every
	move and probe the model asks for: allowed and excluded values, knobs not to change (frozen)
	and knobs set aside after "Not this" (declined) before anything runs; the number limits
	(budget, deadline, GPU-hours) against the readouts the browser reports for a move, before and
	after it (routes/guide.py number_limit_broken), with the move put back if it breaks one."""

	#: knob -> the only values it may take (e.g. allowed GPUs)
	allowed: dict[Id, list[Value]] = Field(default_factory=dict, max_length=96)
	#: knob -> values the visitor turned down ("Not this")
	excluded: dict[Id, list[Value]] = Field(default_factory=dict, max_length=96)
	#: knobs the visitor said not to change: never moved, never unpinned by the guide
	frozen: list[Id] = Field(default_factory=list, max_length=96)
	#: knobs set aside after "Not this": the guide offers another lever
	declined: list[Id] = Field(default_factory=list, max_length=96)
	budget_usd: Positive | None = None
	deadline_s: Positive | None = None
	gpu_hours: Positive | None = None

	@field_validator("allowed", "excluded")
	@classmethod
	def _short_lists(cls, v: dict[str, list]) -> dict[str, list]:
		if any(len(x) > 64 for x in v.values()):
			raise ValueError("at most 64 values per knob")
		return v

	def stricter(self, other: Limits) -> Limits:
		"""Both sets of limits at once: allowed lists intersected, everything ruled out or frozen
		joined, the lower of each number. Used where the server's copy meets the browser's."""
		allowed = dict(self.allowed)
		for kid, vs in other.allowed.items():
			allowed[kid] = [v for v in allowed[kid] if v in vs] if kid in allowed else list(vs)
		excluded = {k: list(v) for k, v in self.excluded.items()}
		for kid, vs in other.excluded.items():
			excluded[kid] = [*excluded.get(kid, []), *(v for v in vs if v not in excluded.get(kid, []))]
			# Never cut: dropping a value would allow it again. Past 64, the allowed list is the tool.
			if len(excluded[kid]) > 64:
				raise ValueError(f"more than 64 values ruled out on {kid}")

		def low(a: float | None, b: float | None) -> float | None:
			return b if a is None else a if b is None else min(a, b)

		return Limits(
			allowed=allowed,
			excluded=excluded,
			frozen=sorted({*self.frozen, *other.frozen}),
			declined=sorted({*self.declined, *other.declined}),
			budget_usd=low(self.budget_usd, other.budget_usd),
			deadline_s=low(self.deadline_s, other.deadline_s),
			gpu_hours=low(self.gpu_hours, other.gpu_hours),
		)

	def excluding(self, knob_id: str, value: Any) -> Limits:
		"""These limits with one more value ruled out and its knob set aside ("Not this")."""
		ex = dict(self.excluded)
		ex[knob_id] = [*ex.get(knob_id, []), value] if value not in ex.get(knob_id, []) else ex[knob_id]
		if len(ex[knob_id]) > 64:
			raise ValueError(f"more than 64 values ruled out on {knob_id}")
		dec = self.declined if knob_id in self.declined else [*self.declined, knob_id]
		return self.model_copy(update={"excluded": ex, "declined": dec})

	def allows(self, knob_id: str, value: Any) -> bool:
		if knob_id in self.frozen:
			return False
		allow = self.allowed.get(knob_id)
		# A knob listed with no values allows none (two sets of limits that share none).
		if allow is not None and value not in allow:
			return False
		return value not in self.excluded.get(knob_id, [])


class Setting(Strict):
	"""The visitor's cockpit right now."""

	knobs: dict[Id, Value] = Field(max_length=96)
	pinned: list[Id] = Field(default_factory=list, max_length=96)
	objective: Objective
	limits: Limits = Field(default_factory=Limits)


# ---- tool results the browser computed (one per pending browser call)


class PredictOut(Strict):
	tool: Literal["predict"]
	call_id: Annotated[str, Field(max_length=80)]
	readouts: dict[Id, Readout] = Field(max_length=24)


class Ranked(Strict):
	knob_id: Id
	impact: float = Field(ge=0, le=1)
	direction: Literal["increase", "decrease", "change"]
	best_value: Value
	delta: float | None = None
	fit: Literal["fits", "tight"] = "fits"


class Combined(Strict):
	"""A setting that changes several knobs together where no single knob clears a failure, from B's
	pair grids (web/src/guide/combos.ts). fits: the person applies it with one click on the guide's
	card; tight (P(OOM) up to 0.05): the card offers to measure it first. set_knob can't apply it."""

	knobs: dict[Id, Value] = Field(min_length=2, max_length=4)
	risk: float = Field(ge=0, le=1)
	peak_bytes: float | None = Field(default=None, ge=0)
	fit: Literal["fits", "tight"] = "fits"


class SensitivityOut(Strict):
	tool: Literal["sensitivity"]
	call_id: Annotated[str, Field(max_length=80)]
	ranked: list[Ranked] = Field(max_length=12)
	combined: Combined | None = None


class SetKnobOut(Strict):
	tool: Literal["set_knob"]
	call_id: Annotated[str, Field(max_length=80)]
	moved: bool
	knob_id: Id
	from_value: Value | None = None
	to_value: Value | None = None
	readouts: dict[Id, Readout] = Field(default_factory=dict, max_length=24)
	#: the same readouts at the setting before the move, so the server can check number limits
	baseline: dict[Id, Readout] = Field(default_factory=dict, max_length=24)
	refused: (
		Literal[
			"pinned",
			"not_in_domain",
			"changes_results",
			"outside_limits",
			"over_budget",
			"past_deadline",
			"over_quota",
			"would_be_tight",
			"would_fail",
			"not_trusted",
			"one_per_round",
		]
		| None
	) = None


class HighlightOut(Strict):
	tool: Literal["highlight"]
	call_id: Annotated[str, Field(max_length=80)]
	nodes: list[Id] = Field(max_length=24)


class PinOut(Strict):
	tool: Literal["pin"]
	call_id: Annotated[str, Field(max_length=80)]
	knob_id: Id
	pinned: bool


ToolOut = Annotated[
	PredictOut | SensitivityOut | SetKnobOut | HighlightOut | PinOut, Field(discriminator="tool")
]


class Decision(Strict):
	"""What the visitor did with the guide's proposed move."""

	knob_id: Id
	#: kept: reverted and pinned at the visitor's value; declined: "Not this" (excluded, next lever)
	action: Literal["applied", "reverted", "pinned", "kept", "declined", "changed"]
	value: Value | None = None


class Approval(Strict):
	call_id: Annotated[str, Field(max_length=80)]
	approved: bool


class TurnIn(Strict):
	record: Annotated[str, Field(pattern=r"^[A-Za-z0-9_:-]{1,80}$")]
	setting: Setting
	state: Annotated[str, Field(max_length=200_000)] | None = None
	message: Annotated[str, Field(max_length=MESSAGE_MAX)] | None = None
	decision: Decision | None = None
	results: list[ToolOut] = Field(default_factory=list, max_length=16)
	approvals: list[Approval] = Field(default_factory=list, max_length=8)
	#: begin a tour: only valid on a fresh conversation
	start: bool = False
	language: Annotated[str, Field(pattern=r"^[a-z]{2,3}(-[A-Za-z0-9]{2,8})?$")] | None = None

	@field_validator("message")
	@classmethod
	def _clean_message(cls, v: str | None) -> str | None:
		if v is None:
			return None
		v = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", "", v).strip()
		return v or None


# ---- checks against the resolved record


class InputError(ValueError):
	pass


def check_limits(limits: Limits, rec: RecordView) -> None:
	"""Every knob and value a set of limits names is one the record has."""
	for kid in [*limits.frozen, *limits.declined]:
		if kid not in rec.knobs:
			raise InputError(f"unknown limited knob {kid}")
	for kid, values in [*limits.allowed.items(), *limits.excluded.items()]:
		k = rec.knobs.get(kid)
		if k is None:
			raise InputError(f"unknown limited knob {kid}")
		for v in values:
			if not k.allows(v) and v != k.current:
				raise InputError(f"{kid} can't be {v!r}")


def check_setting(s: Setting, rec: RecordView) -> None:
	for kid, v in s.knobs.items():
		k = rec.knobs.get(kid)
		if k is None:
			raise InputError(f"unknown knob {kid}")
		if not k.allows(v) and v != k.current:
			raise InputError(f"{kid} can't be {v!r}")
	for kid in s.pinned:
		if kid not in rec.knobs:
			raise InputError(f"unknown pinned knob {kid}")
	check_limits(s.limits, rec)
	if s.objective.target_id and s.objective.target_id not in rec.targets:
		raise InputError(f"unknown objective target {s.objective.target_id}")


def _check_readouts(r: dict[str, Readout], rec: RecordView) -> None:
	for tid in r:
		if tid not in rec.targets:
			raise InputError(f"unknown target {tid}")


def check_result(out: Any, rec: RecordView) -> None:
	if isinstance(out, PredictOut):
		_check_readouts(out.readouts, rec)
	elif isinstance(out, SensitivityOut):
		for r in out.ranked:
			k = rec.knobs.get(r.knob_id)
			if k is None or not k.allows(r.best_value):
				raise InputError(
					f"ranking names {r.knob_id}={r.best_value!r}, which the record doesn't allow"
				)
		for kid, v in (out.combined.knobs if out.combined else {}).items():
			k = rec.knobs.get(kid)
			if k is None or not k.allows(v):
				raise InputError(f"combined move names {kid}={v!r}, which the record doesn't allow")
	elif isinstance(out, SetKnobOut):
		k = rec.knobs.get(out.knob_id)
		if k is None:
			raise InputError(f"unknown knob {out.knob_id}")
		for v in (out.from_value, out.to_value):
			if v is not None and not k.allows(v) and v != k.current:
				raise InputError(f"{out.knob_id} can't be {v!r}")
		_check_readouts(out.readouts, rec)
		_check_readouts(out.baseline, rec)
	elif isinstance(out, HighlightOut):
		bad = [n for n in out.nodes if n not in rec.nodes and not n.startswith("knob:")]
		if bad:
			raise InputError(f"unknown map nodes {bad}")
	elif isinstance(out, PinOut) and out.knob_id not in rec.knobs:
		raise InputError(f"unknown knob {out.knob_id}")


def check_decision(d: Decision, rec: RecordView) -> None:
	k = rec.knobs.get(d.knob_id)
	if k is None:
		raise InputError(f"unknown knob {d.knob_id}")
	if d.value is not None and not k.allows(d.value) and d.value != k.current:
		raise InputError(f"{d.knob_id} can't be {d.value!r}")
