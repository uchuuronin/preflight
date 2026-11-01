"""POST /v1/guide/limits: the visitor's limits in their own words, read by Muse Spark.

"must stay on A100, under $40, can't touch the dataset" comes back as structured limits the
visitor confirms one by one as chips: allowed values for a knob, knobs not to change, a budget,
a wall-time deadline and a GPU-hours quota. Nothing here applies a limit: the browser applies
only what the visitor confirms, and the guide's tools enforce it (web/src/guide/limits.ts,
routes/guide.py).

What can come back is bounded by the record: knob ids and values the record has, and numbers
that appear in the visitor's own text (a model can't invent a budget). Anything it couldn't
map is returned as "not understood", in the visitor's words, so nothing is silently dropped.
Spend: one model call, its worst case reserved in the shared ledger first, as for a guide turn.
"""

from __future__ import annotations

import datetime as dt
import re
from typing import Annotated

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .. import edge
from ..errors import ApiError
from ..flags import flag
from ..guide import muse_spark
from ..guide.limits_reader import (
	INSTRUCTIONS,
	MAX_OUTPUT_TOKENS,
	WORST_INPUT_TOKENS,
	_knob_list,
	_tool,
	checked,
)
from ..guide.record_view import RecordView
from ..guide.types import InputError, Limits, check_limits
from ..jobs import visitor_of
from .guide import _record, view_of

router = APIRouter(prefix="/v1")

TEXT_MAX = 300  # source: guess, a few sentences of constraints; longer is a document


class LimitsIn(BaseModel):
	model_config = ConfigDict(extra="forbid")
	record: Annotated[str, Field(pattern=r"^[A-Za-z0-9_:-]{1,80}$")]
	text: Annotated[str, Field(min_length=1, max_length=TEXT_MAX)]

	@field_validator("text")
	@classmethod
	def _clean(cls, v: str) -> str:
		v = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", v).strip()
		if not v:
			raise ValueError("empty")
		return v


@router.post("/guide/limits")
async def read_limits(request: Request) -> dict:
	if not flag("c.guide.live") or not muse_spark.configured():
		raise ApiError(
			503,
			"limits_reader_off",
			"Reading limits from text needs the live guide, which is off. Set them below.",
		)
	try:
		body = LimitsIn.model_validate(await request.json())
	except (ValidationError, ValueError) as e:
		raise ApiError(422, "invalid_request", "Write your limits in up to 300 characters.") from e
	app = request.app
	raw = await _record(body.record, app)
	if raw is None:
		raise ApiError(404, "record_not_found", "That flight record doesn't exist or has expired.")
	rec = view_of(raw)
	sid: str = request.state.session
	worst = (WORST_INPUT_TOKENS * muse_spark.PRICE_IN + MAX_OUTPUT_TOKENS * muse_spark.PRICE_OUT) / 1e6
	gate = await app.state.ledger.reserve(
		visitor=sid, run=f"{rec.key}:{sid}", ip=edge.ip_key(edge.client_ip(request)), usd=worst
	)
	if not gate.ok:
		raise ApiError(
			429,
			"rate_limited",
			"You've used the guide's allowance for now; set your limits below instead.",
			extra={"retry_after_s": gate.retry_after_s},
		)
	instructions = f"{INSTRUCTIONS}\n\nThis job's knobs:\n{_knob_list(rec)}"
	try:
		result = await muse_spark.create_turn(
			instructions=instructions,
			input_items=[{"role": "user", "content": body.text}],
			tools=[_tool(rec)],
			max_output_tokens=MAX_OUTPUT_TOKENS,
			client=app.state.http,
		)
	except muse_spark.GuideModelError as e:
		await app.state.ledger.release(gate.reservation)
		raise ApiError(503, "limits_reader_unavailable", f"{e} Set your limits below instead.") from e
	except Exception:
		await app.state.ledger.settle(gate.reservation, worst)
		raise
	await app.state.ledger.settle(gate.reservation, result.cost_usd)
	call = next((c for c in result.calls if c.name == "set_limits"), None)
	out = (
		checked(call.arguments, rec, body.text)
		if call
		else {"limits": [], "not_understood": [body.text[:80]], "dropped": []}
	)
	return {
		**out,
		"provenance": {
			"provider": "meta",
			"model": muse_spark.MODEL,
			"purpose": "guide.limits",
			"input_tokens": result.input_tokens,
			"output_tokens": result.output_tokens,
			"cost_usd": round(result.cost_usd, 6),
			"at": dt.datetime.now(dt.UTC).isoformat(),
		},
	}


class LimitsStateIn(BaseModel):
	model_config = ConfigDict(extra="forbid")
	record: Annotated[str, Field(pattern=r"^[A-Za-z0-9_:-]{1,80}$")]
	limits: Limits


async def _view(ref: str, request: Request) -> RecordView:
	raw = await _record(ref, request.app)
	if raw is None:
		raise ApiError(404, "record_not_found", "That flight record doesn't exist or has expired.")
	return view_of(raw)


@router.put("/guide/limits/state")
async def put_limits(request: Request) -> dict:
	"""The limits the visitor confirmed on this record, kept on the server (session_limits.py)."""
	try:
		body = LimitsStateIn.model_validate(await request.json())
	except (ValidationError, ValueError) as e:
		raise ApiError(422, "invalid_request", "The limits aren't in the expected shape.") from e
	rec = await _view(body.record, request)
	try:
		check_limits(body.limits, rec)
	except InputError as e:
		raise ApiError(422, "invalid_request", str(e)) from e
	await request.app.state.session_limits.put(visitor_of(request.state.session), rec.key, body.limits)
	return {"saved": True}


@router.get("/guide/limits/state")
async def get_limits(record: str, request: Request) -> dict:
	"""This session's limits on a record, so a reload keeps them."""
	if not re.fullmatch(r"[A-Za-z0-9_:-]{1,80}", record):
		raise ApiError(404, "record_not_found", "That flight record doesn't exist or has expired.")
	rec = await _view(record, request)
	got = await request.app.state.session_limits.get(visitor_of(request.state.session), rec.key)
	return {"limits": got.model_dump() if got else None}
