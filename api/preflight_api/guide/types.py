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


class Setting(Strict):
	"""The visitor's cockpit right now."""

	knobs: dict[Id, Value] = Field(max_length=96)
	pinned: list[Id] = Field(default_factory=list, max_length=96)
	objective: Objective


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


class SensitivityOut(Strict):
	tool: Literal["sensitivity"]
	call_id: Annotated[str, Field(max_length=80)]
	ranked: list[Ranked] = Field(max_length=12)


class SetKnobOut(Strict):
	tool: Literal["set_knob"]
	call_id: Annotated[str, Field(max_length=80)]
	moved: bool
	knob_id: Id
	from_value: Value | None = None
	to_value: Value | None = None
	readouts: dict[Id, Readout] = Field(default_factory=dict, max_length=24)
	refused: (
		Literal[
			"pinned",
			"not_in_domain",
			"changes_results",
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
	action: Literal["applied", "reverted", "pinned", "changed"]
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
	elif isinstance(out, SetKnobOut):
		k = rec.knobs.get(out.knob_id)
		if k is None:
			raise InputError(f"unknown knob {out.knob_id}")
		for v in (out.from_value, out.to_value):
			if v is not None and not k.allows(v) and v != k.current:
				raise InputError(f"{out.knob_id} can't be {v!r}")
		_check_readouts(out.readouts, rec)
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
