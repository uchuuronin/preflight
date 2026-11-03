"""Types for the prediction API (B's predict, build_grids, sensitivity).

The functions live in python/preflight/engine/api.py; the shapes live here so
the record, the TypeScript PredictionProvider and the guide tools share them.
"""

import math
from typing import Literal

import numpy as np
from pydantic import Field, RootModel, SerializationInfo, model_serializer, model_validator

from .base import KnobValue, Strict, Unit
from .encoding import MAX_GRID_CELLS, decode_f16, f16_len, mask_len_ok

EstimateStatus = Literal["measured", "predicted", "extrapolated", "abstain"]
Readout = Literal["memory", "time", "cost", "risk"]


class EstimateRate(Strict):
	"""0.6.0 (B, Nov 1): how a cost estimate splits into a quantity times a rate, so a client can
	re-price it at the visitor's own rate without parsing the reason."""

	quantity: str = Field(description="The target holding the quantity, e.g. cost.gpu_hours")
	usd_per_unit: float = Field(ge=0)
	unit: str = Field(description="The quantity's unit, e.g. gpu_hours")
	price_row: str | None = Field(default=None, description="The price-table row id the rate came from")


class Estimate(Strict):
	"""One number on screen: a point with an interval, or an honest abstention."""

	point: float | None
	lo: float | None
	hi: float | None
	unit: Unit
	status: EstimateStatus
	model_id: str
	source: Literal["grid", "exact"]
	reason: str | None = Field(default=None, description="Why the engine abstained or extrapolated")
	level: float = Field(default=0.9, gt=0, lt=1, description="Nominal coverage of [lo, hi]")
	rate: EstimateRate | None = Field(default=None, description="0.6.0: cost estimates only, quantity x rate")

	@model_serializer(mode="wrap")
	def _keep_nulls(self, handler, info: SerializationInfo):
		# point, lo and hi are required and nullable (an abstention is null, never absent): records are
		# stored with exclude_none, which would drop them, and then neither load_record nor C's
		# types (`point: number | null`) could read the record back. Only exclude_none is undone; an
		# explicit include or exclude is honoured, and the keys keep the fields' order.
		out = handler(self)
		if (
			not isinstance(out, dict)
			or not info.exclude_none
			or info.include is not None
			or info.exclude is not None
		):
			return out
		# (no field has an alias, so the field names are the keys)
		return {k: out.get(k) for k in type(self).model_fields if k in out or k in ("point", "lo", "hi")}

	@model_validator(mode="after")
	def _check(self):
		if self.status == "abstain":
			if not self.reason:
				raise ValueError("an abstaining estimate must say why")
			return self
		if self.point is None:
			raise ValueError(f"status {self.status} needs a point")
		lo = self.point if self.lo is None else self.lo
		hi = self.point if self.hi is None else self.hi
		if not (lo <= self.point <= hi):
			raise ValueError(f"interval [{lo}, {hi}] does not bracket point {self.point}")
		return self


class Prediction(RootModel[dict[str, Estimate]]):
	"""target_id -> Estimate, for one knob setting."""


class Target(Strict):
	id: str = Field(description="Dotted id, e.g. memory.peak, time.total, cost.total, risk.oom")
	label_key: str
	unit: Unit
	readout: Readout | None = Field(default=None, description="Which of the four readouts shows it, if any")


class GridAxis(Strict):
	knob_id: str
	values: list[KnobValue] = Field(min_length=1)
	scale: Literal["linear", "log", "categorical"] = "linear"


class Grid(Strict):
	"""Predictions precomputed over a knob grid, row-major over `axes`.

	point, lo and hi are base64 little-endian float16 holding real / scale;
	abstain is a base64 LSB-first bit mask. Grids over MAX_GRID_CELLS cells go
	to a sidecar blob. Use encoding.choose_scale() when writing one.
	"""

	id: str
	target_id: str
	unit: Unit
	axes: list[GridAxis] = Field(min_length=1, max_length=4)
	point: str = ""
	lo: str = ""
	hi: str = ""
	abstain: str = ""
	model_id: str
	scale: float = Field(default=1.0, gt=0, description="Power of two; real value = decoded float16 x scale")
	fixed: dict[str, KnobValue] = Field(
		default_factory=dict, description="Values every knob not on an axis is held at"
	)
	independent_of: list[str] | None = Field(
		default=None,
		description="0.6.4: knobs this target doesn't depend on (the learning rate, say): the grid answers at any value of "
		"them, so a reader holds them nowhere",
	)
	sidecar: str | None = Field(
		default=None, pattern=r"^blob://", description="Set when the grid is too large to inline"
	)

	@property
	def cells(self) -> int:
		return math.prod(len(a.values) for a in self.axes)

	def values(self, name: str = "point") -> np.ndarray:
		"""Decoded, rescaled values of point, lo or hi, shaped by the axes."""
		shape = [len(a.values) for a in self.axes]
		return (decode_f16(getattr(self, name), self.cells) * self.scale).reshape(shape)

	@model_validator(mode="after")
	def _check(self):
		axis_ids = [a.knob_id for a in self.axes]
		if len(set(axis_ids)) != len(axis_ids):
			raise ValueError("a knob appears on two axes")
		if set(axis_ids) & set(self.fixed):
			raise ValueError("a knob cannot be both an axis and fixed")
		if set(axis_ids) & set(self.independent_of or ()):
			raise ValueError("a knob cannot be both an axis and one the target doesn't depend on")
		if self.sidecar is not None:
			return self
		n = self.cells
		if n > MAX_GRID_CELLS:
			raise ValueError(f"grid has {n} cells; inline limit is {MAX_GRID_CELLS}, use a sidecar blob")
		for name in ("point", "lo", "hi"):
			got = f16_len(getattr(self, name))
			if got != n:
				raise ValueError(f"{name} holds {got} values, axes need {n}")
			if not np.isfinite(decode_f16(getattr(self, name))).all():
				raise ValueError(
					f"{name} has non-finite cells; values must be finite after scaling (raise scale)"
				)
		if not mask_len_ok(self.abstain, n):
			raise ValueError(f"abstain mask does not cover {n} cells")
		return self


class Predictions(Strict):
	"""Written by B."""

	targets: list[Target] = Field(default_factory=list)
	estimates: dict[str, Estimate] = Field(
		default_factory=dict, description="Estimates at the record's current knob values"
	)
	grids: list[Grid] = Field(default_factory=list)


ObjectiveKind = Literal[
	"min_time", "min_cost", "max_throughput", "fit_memory", "min_risk", "max_quality_per_usd", "custom"
]


class Objective(Strict):
	kind: ObjectiveKind = "min_time"
	target_id: str | None = Field(default=None, description="For custom objectives: the target to optimise")
	weights: dict[str, float] = Field(default_factory=dict, description="target_id -> weight, for blends")
	description: str | None = Field(default=None, description="The user's use case in their own words")


class KnobImpact(Strict):
	knob_id: str
	impact: float = Field(description="Change in the objective across the knob's range, normalised to [0, 1]")
	direction: Literal["increase", "decrease", "either"]
	best_value: KnobValue | None = None
	reason: str | None = None
