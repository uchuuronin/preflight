from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
	"""Base for every contract model: unknown fields are an error, not a silent drop."""

	model_config = ConfigDict(extra="forbid", populate_by_name=True, use_enum_values=True)


# A knob takes one of these. bool sits first so `true` is never read as 1.
KnobValue = bool | int | float | str

# blob://<bucket-relative key>; raw traces, logs and diffs live in the object store, never inline
BlobRef = Annotated[str, Field(pattern=r"^blob://[A-Za-z0-9._\-/]+$")]

Lens = Literal["finetune", "pretrain", "rl", "agent", "eval", "serving", "retrieval", "gnn", "vision"]
LENSES: tuple[str, ...] = Lens.__args__  # type: ignore[attr-defined]

Unit = str  # "bytes", "s", "usd", "tokens/s", "samples", "fraction", "W", "kgCO2e", ...
