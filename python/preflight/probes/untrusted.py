"""Checks on what the probe child reports beyond the schema'd summary: the kernel table and the model shape.

The child shares a process with visitor code, so anything it writes may be
malformed or hostile (a record_function label a megabyte long, a million rows,
NaN times). These checks run in the parent. A table or shape that fails them
is dropped with a short reason; it never fails the probe, which measured the
rest fine.
"""

from pydantic import ValidationError

from preflight.recorder.kernels import KERNEL_ROWS
from preflight.recorder.shape_keys import BOOL_KEYS as _BOOL_ARCH
from preflight.recorder.shape_keys import INT_KEYS as _INT_ARCH
from preflight.recorder.shape_keys import TEXT_KEYS as _TEXT_ARCH
from preflight.schema import KernelTable, StepSample
from preflight.schema.record import KERNEL_NAME_LEN, KERNEL_TABLE_ROWS
from preflight.schema.script_map import ModelInfo

MAX_ROWS = KERNEL_TABLE_ROWS
MAX_NAME = KERNEL_NAME_LEN
assert KERNEL_ROWS + 17 <= MAX_ROWS  # top rows, a tail row per class (B has 16), the unattributed row
MAX_TEXT = 64


def kernel_table(raw) -> tuple[KernelTable | None, str | None]:
	"""(table, None) when it checks out, else (None, why)."""
	if raw is None:
		return None, None
	if not isinstance(raw, dict):
		return None, "the kernel table was not readable"
	if "error" in raw:
		return None, "the profiler could not summarise this run"
	rows = raw.get("rows")
	if isinstance(rows, list) and len(rows) <= MAX_ROWS:
		# a visitor's record_function label can be any length; keep the row, cut the name
		for r in rows:
			if isinstance(r, dict) and isinstance(r.get("name"), str):
				r["name"] = r["name"][:MAX_NAME]
	try:
		return KernelTable.model_validate(raw), None
	except ValidationError:
		return None, "the kernel table was not readable"


def model_shape(raw) -> dict | None:
	"""Known keys with sane values only; anything else is dropped field by field."""
	if not isinstance(raw, dict) or "error" in raw:
		return None
	out: dict = {}
	for k in _INT_ARCH:
		v = raw.get(k)
		if isinstance(v, int) and not isinstance(v, bool) and 0 < v <= 2**31 - 1:
			out[k] = v
	for k in _BOOL_ARCH:
		if isinstance(raw.get(k), bool):
			out[k] = raw[k]
	for k in _TEXT_ARCH:
		v = raw.get(k)
		if isinstance(v, str) and v and len(v) <= MAX_TEXT and v.isprintable():
			out[k] = v
	return out or None


_SHAPE_SOURCES = ("hf_config", "modules")


def with_shape(model: ModelInfo | None, shape: dict | None) -> ModelInfo | None:
	"""The script map's model, with the shape the running model reported filled in over it.

	What was built beats what the code suggested, field by field. Every field is
	validated as ModelInfo would validate it: a value the schema refuses (a
	"batchnorm" norm) is dropped and the code's value kept, so a record never
	carries what it can't load. shape_source is set only when the shape says
	where it came from. A model the code never named gets one named by its model
	type, so the shape isn't lost.
	"""
	if not shape:
		return model
	base = (
		model.model_dump(exclude_none=True)
		if model is not None
		else {"name": shape.get("model_type") or "model"}
	)
	fields = {k: v for k, v in shape.items() if k != "source" and k in ModelInfo.model_fields and k != "name"}
	if shape.get("source") in _SHAPE_SOURCES:
		fields["shape_source"] = shape["source"]
	merged = dict(base)
	for k, v in fields.items():
		try:
			ModelInfo.model_validate({**merged, k: v})
		except ValidationError:
			continue
		merged[k] = v
	try:
		return ModelInfo.model_validate(merged)
	except ValidationError:  # the base itself didn't validate (no model and an odd model_type)
		return model


def step_log(raw) -> list[StepSample]:
	"""Each step the child reported that checks out; a malformed step is dropped, not the probe."""
	if not isinstance(raw, list):
		return []
	out = []
	for item in raw[:256]:
		try:
			out.append(StepSample.model_validate(item))
		except ValidationError:
			continue
	return out
