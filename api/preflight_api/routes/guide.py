"""POST /v1/guide/turns: proxy to Muse Spark. Holds the key, enforces the token budget.

The browser sends the conversation and the tool definitions it built this turn (with
pinned knobs already removed from set_knob). The server adds its own instructions, checks
every tool name against the guide's fixed set, calls the model and returns its text and
function calls for the browser to execute. Over any budget, it answers 503 with
fallback=rule_based and the browser switches to the rule-based tour.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any, Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from ..errors import ApiError
from ..flags import flag
from ..guide import muse_spark
from ..guide.budget import MAX_INPUT_TOKENS_PER_TURN, MAX_OUTPUT_TOKENS_PER_TURN
from ..guide.prompt import INSTRUCTIONS

router = APIRouter(prefix="/v1")

GUIDE_TOOLS = {
	"predict",
	"sensitivity",
	"set_knob",
	"pin",
	"highlight",
	"snapshot",
	"compare",
	"run_probe",
	"propose_patch",
	"run_verify",
}
# Rough bound before the call; the model's usage report is what gets billed.
CHARS_PER_TOKEN = 3.5


class ToolDef(BaseModel):
	type: Literal["function"]
	name: str
	description: str = Field(max_length=2000)
	parameters: dict[str, Any]
	strict: bool | None = None


class TurnRequest(BaseModel):
	input: list[dict[str, Any]] = Field(max_length=200)
	tools: list[ToolDef] = Field(max_length=len(GUIDE_TOOLS))
	record_ref: str | None = Field(default=None, max_length=64)
	effort: Literal["low", "medium", "high"] = "low"


def _fallback(reason: str, message: str) -> ApiError:
	return ApiError(503, "guide_unavailable", message, extra={"fallback": "rule_based", "reason": reason})


@router.post("/guide/turns")
async def guide_turn(body: TurnRequest, request: Request) -> dict:
	if not flag("c.guide.live") or not muse_spark.configured():
		raise _fallback("disabled", "The live guide is off right now, so the rule-based tour is running instead.")

	unknown = [t.name for t in body.tools if t.name not in GUIDE_TOOLS]
	if unknown:
		raise ApiError(422, "invalid_request", f"Unknown guide tools: {', '.join(unknown)}.")

	approx_tokens = len(json.dumps(body.input)) / CHARS_PER_TOKEN
	if approx_tokens > MAX_INPUT_TOKENS_PER_TURN:
		raise _fallback(
			"turn_too_large", "This conversation is too long for the live guide; the rule-based tour takes over."
		)

	visitor = request.state.session
	ledger = request.app.state.ledger
	decision = ledger.check(visitor, body.record_ref)
	if not decision.allowed:
		raise _fallback(
			decision.reason or "budget", "You've used today's live guide turns; the rule-based tour takes over."
		)

	try:
		result = await muse_spark.create_turn(
			instructions=INSTRUCTIONS,
			input_items=body.input,
			tools=[t.model_dump(exclude_none=True) for t in body.tools],
			max_output_tokens=MAX_OUTPUT_TOKENS_PER_TURN,
			reasoning_effort=body.effort,
			client=request.app.state.http,
		)
	except muse_spark.GuideModelError as e:
		raise _fallback("model_error", f"{e} The rule-based tour takes over for now.") from e

	ledger.charge(visitor, body.record_ref, result.cost_usd)
	return {
		"text": result.text,
		"calls": [{"call_id": c.call_id, "name": c.name, "arguments": c.arguments} for c in result.calls],
		"provenance": {
			"provider": "meta",
			"model": muse_spark.MODEL,
			"purpose": "guide_turn",
			"input_tokens": result.input_tokens,
			"output_tokens": result.output_tokens,
			"cached_tokens": result.cached_tokens,
			"cost_usd": round(result.cost_usd, 6),
			"request_id": result.request_id,
			"at": dt.datetime.now(dt.UTC).isoformat(),
		},
	}
