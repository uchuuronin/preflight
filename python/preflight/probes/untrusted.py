"""Checks on what the probe child reports beyond the schema'd summary: the kernel table and the model shape.

The child shares a process with visitor code, so anything it writes may be
malformed or hostile (a record_function label a megabyte long, a million rows,
NaN times). These checks run in the parent. A table or shape that fails them
is dropped with a short reason; it never fails the probe, which measured the
rest fine.
"""

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from preflight.recorder.kernels import KERNEL_ROWS
from preflight.recorder.shape_keys import BOOL_KEYS as _BOOL_ARCH
from preflight.recorder.shape_keys import INT_KEYS as _INT_ARCH
from preflight.recorder.shape_keys import TEXT_KEYS as _TEXT_ARCH

MAX_ROWS = KERNEL_ROWS + 32  # top rows, one tail row per kernel class, the unattributed row
MAX_NAME = 200
MAX_TEXT = 64


class _Row(BaseModel):
	model_config = ConfigDict(extra="forbid", populate_by_name=True, allow_inf_nan=False)

	name: str = Field(max_length=MAX_NAME)
	klass: str | None = Field(default=None, alias="class", max_length=MAX_TEXT)
	calls: float = Field(ge=0)
	device_s: float = Field(ge=0, le=86_400)
	flops: float | None = Field(default=None, ge=0)
	bytes: float | None = Field(default=None, ge=0)
	dtype: str = Field(max_length=8)


class _Table(BaseModel):
	model_config = ConfigDict(extra="forbid")

	rows: list[_Row] = Field(max_length=MAX_ROWS)
	steps: int = Field(ge=1, le=1_000)
	backend: str = Field(pattern=r"^(cuda|rocm|cpu)$")
	allow_tf32: bool
	classified_by: str | None = Field(default=None, max_length=MAX_TEXT)


def kernel_table(raw) -> tuple[dict | None, str | None]:
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
		t = _Table.model_validate(raw)
	except ValidationError:
		return None, "the kernel table was not readable"
	return t.model_dump(by_alias=True), None


def model_shape(raw) -> dict | None:
	"""Known keys with sane values only; anything else is dropped field by field."""
	if not isinstance(raw, dict) or "error" in raw:
		return None
	out: dict = {}
	for k in _INT_ARCH:
		v = raw.get(k)
		if isinstance(v, int) and not isinstance(v, bool) and 0 < v <= 10_000_000:
			out[k] = v
	for k in _BOOL_ARCH:
		if isinstance(raw.get(k), bool):
			out[k] = raw[k]
	for k in _TEXT_ARCH:
		v = raw.get(k)
		if isinstance(v, str) and v and len(v) <= MAX_TEXT and v.isprintable():
			out[k] = v
	return out or None
