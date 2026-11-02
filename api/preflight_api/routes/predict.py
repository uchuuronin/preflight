"""POST /v1/records/{ref}/predict: the engine's own prediction at settings the record's grids don't hold.

The web reads predictions off a record's grids (GridProvider), instantly. A grid holds only the knob
values it was built over, so off them the web abstained, though the engine predicts there: on the
Nov 2 LoRA record, lr 2e-4 (1.709 s a step, the same as 1e-4) and the A10 (1.236 s, extrapolated
across GPUs). This asks the engine (`preflight.engine.api.predict`) for a few settings at a time.
Pure computation on the record as stored: no model, no GPU, nothing written. Behind
`c.engine_fallback` (C, Nov 2; off until the PM turns it on).
"""

from __future__ import annotations

import asyncio
import math
import time

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from .. import edge
from ..errors import ApiError
from ..flags import flag
from ..stores import StoreBusy, resolve_record

router = APIRouter(prefix="/v1")

# Source: derived, a campaign's runs on one calendar (the planner's own cap is 512) arrive a page at a
# time; 64 settings at about 3 ms each (measured Nov 2 on the LoRA record) is about 0.2 s of CPU.
MAX_CELLS = 64
MAX_KNOBS = 32  # source: guess, records carry about 20 knobs
# Its own bucket, not the ledger's: the ledger counts guide turns, which this mustn't spend. Source:
# guess, a visitor dragging a knob lands on a new setting at most a few times a second.
BURST = 30
REFILL_PER_S = 3.0
_buckets: dict[str, tuple[float, float]] = {}  # ip key -> (tokens, at); one container's view


def _take(ip: str, now: float | None = None) -> bool:
	now = time.monotonic() if now is None else now
	tokens, at = _buckets.get(ip, (BURST, now))
	tokens = min(BURST, tokens + (now - at) * REFILL_PER_S)
	if tokens < 1:
		_buckets[ip] = (tokens, now)
		return False
	_buckets[ip] = (tokens - 1, now)
	if len(_buckets) > 10_000:  # bounded: drop the oldest half
		for k, _ in sorted(_buckets.items(), key=lambda kv: kv[1][1])[:5_000]:
			_buckets.pop(k, None)
	return True


class PredictBody(BaseModel):
	cells: list[dict[str, bool | int | float | str]] = Field(min_length=1, max_length=MAX_CELLS)


def _clean(knobs: dict, ids: set[str]) -> dict:
	"""Only the record's own knobs, finite numbers, short strings: anything else isn't a setting."""
	if len(knobs) > MAX_KNOBS:
		raise ApiError(422, "invalid_request", "That setting names more knobs than the record has.")
	out = {}
	for k, v in knobs.items():
		if k not in ids:
			raise ApiError(422, "invalid_request", f"This record has no knob {k!r}.")
		if isinstance(v, float) and not math.isfinite(v):
			raise ApiError(422, "invalid_request", f"{k} needs a finite value.")
		if isinstance(v, str) and len(v) > 64:
			raise ApiError(422, "invalid_request", f"{k}'s value is too long.")
		out[k] = v
	return out


def _predict_all(raw: dict, cells: list[dict]) -> list[dict]:
	from preflight.engine.api import predict
	from preflight.schema import FlightRecord

	rec = FlightRecord.model_validate(raw)
	ids = {k.id for k in rec.knobs}
	out = []
	for knobs in cells:
		try:
			pred = predict(rec, _clean(knobs, ids)).root
		except ApiError:
			raise
		except Exception:
			# a setting the engine can't read (a value of the wrong kind, say) answers nothing, not a 500
			out.append({})
			continue
		out.append({tid: e.model_dump(mode="json", exclude_none=True) for tid, e in pred.items()})
	return out


@router.post("/records/{ref}/predict")
async def predict_cells(ref: str, body: PredictBody, request: Request) -> dict:
	if not flag("c.engine_fallback"):
		raise ApiError(404, "not_found", "Not found.")
	try:
		raw = await asyncio.to_thread(resolve_record, ref, request.app.state.samples, request.app.state.records)
	except StoreBusy as e:
		raise ApiError(
			503, "record_store_busy", "Records are a moment behind; try again.", retryable=True, retry_after=2
		) from e
	if raw is None:
		raise ApiError(404, "record_not_found", "No flight record matches this link.")
	if not _take(edge.ip_key(edge.client_ip(request))):
		raise ApiError(
			429, "rate_limited", "Too many requests from here; try again shortly.", retryable=True, retry_after=2
		)
	try:
		cells = await asyncio.to_thread(_predict_all, raw, body.cells)
	except ApiError:
		raise
	except Exception as e:
		raise ApiError(422, "record_unreadable", "The engine can't read this record.") from e
	return {"cells": cells}
