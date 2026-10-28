"""POST /v1/guide/turns: one round of the live guide (Muse Spark).

Not a model proxy. The browser sends only typed facts: which record, the cockpit's setting,
the results of the browser tools the model asked for last round, what the visitor did with
the last proposed move, which spending tools they approved, and at most one short message of
their own. The server resolves the record itself, builds the tools from it, keeps the
conversation in signed state (guide/state.py), reserves the worst-case cost of the call in
the shared ledger before calling the model, guards every call the model makes, and checks
every number in the reply against this turn's evidence before showing it.

Answers 503 with fallback=rule_based whenever the live guide can't run (flag off, budget
spent, model down); the browser switches to the rule-based tour and the visitor never sees
an error page.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import logging
import math
from functools import lru_cache
from typing import Annotated, Any, Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .. import edge
from ..errors import ApiError
from ..flags import flag
from ..guide import muse_spark, numbers
from ..guide.executors import ExecContext, execute
from ..guide.prompt import INSTRUCTIONS, language_line
from ..guide.record_view import RecordView
from ..guide.state import BadState, GuideState, Pending, open_state, seal
from ..guide.tools import BROWSER_TOOLS, SERVER_TOOLS, build_tools
from ..guide.types import (
	InputError,
	SetKnobOut,
	TurnIn,
	Value,
	check_decision,
	check_result,
	check_setting,
)
from ..stores import resolve_record

log = logging.getLogger("preflight.guide")
router = APIRouter(prefix="/v1")

MAX_OUTPUT_TOKENS = 900  # source: guess; 50 words of text plus a few tool calls. Measure (Oct 28).
MAX_INPUT_TOKENS = 40_000  # source: derived, MAX_BYTES of state / about 3 characters per token
MAX_ROUNDS_PER_TURN = 8  # source: guess, sensitivity + highlight + set_knob + slack
CHARS_PER_TOKEN_WORST = 3.0  # source: guess, conservative (English averages about 4)


def _fallback(reason: str, message: str, retry_after_s: float | None = None) -> ApiError:
	extra: dict[str, Any] = {"fallback": "rule_based", "reason": reason}
	if retry_after_s is not None:
		extra["retry_after_s"] = retry_after_s
	return ApiError(503, "guide_unavailable", message, extra=extra)


@lru_cache(maxsize=64)
def _view(key: str, raw: str) -> RecordView:
	return RecordView.from_record(json.loads(raw))


def view_of(rec: dict) -> RecordView:
	return _view(str(rec.get("record_id")), json.dumps(rec, sort_keys=True))


def _event(text: str) -> dict:
	return {"role": "user", "content": f"[event] {text}"}


def _worst_case_usd(instructions: str, items: list[dict], tools: list[dict]) -> tuple[float, int]:
	chars = len(instructions) + len(json.dumps(items)) + len(json.dumps(tools))
	tokens_in = math.ceil(chars / CHARS_PER_TOKEN_WORST)
	return (tokens_in * muse_spark.PRICE_IN + MAX_OUTPUT_TOKENS * muse_spark.PRICE_OUT) / 1e6, tokens_in


def _turn_evidence(state: GuideState) -> list[Any]:
	outs = []
	for it in state.items[state.turn_start :]:
		if it.get("type") == "function_call_output":
			with contextlib.suppress(ValueError):
				outs.append(json.loads(it.get("output", "null")))
		elif it.get("role") == "user":
			outs.append(it.get("content", ""))
	return outs


def _guard_set_knob(
	args: dict, rec: RecordView, pinned: set[str], state: GuideState, moved_now: bool
) -> str | None:
	"""Server half of the set_knob guard (the browser checks the fit rule, it has the provider)."""
	kid = args.get("knob_id")
	if state.move is not None or moved_now:
		return "one_per_round"
	if not isinstance(kid, str) or kid not in rec.knobs:
		return "unknown_knob"
	if kid in pinned:
		return "pinned"
	if not rec.knobs[kid].allows(args.get("value")):
		return "not_in_domain"
	return None


@router.post("/guide/turns")
async def guide_turn(request: Request) -> dict:
	if not flag("c.guide.live") or not muse_spark.configured():
		raise _fallback(
			"disabled", "The live guide is off right now, so the rule-based tour is running instead."
		)
	try:
		body = TurnIn.model_validate(await request.json())
	except (ValidationError, ValueError) as e:
		raise ApiError(422, "invalid_request", "The guide request isn't in the expected shape.") from e

	app = request.app
	rec_raw = resolve_record(body.record, app.state.samples, app.state.records)
	if rec_raw is None:
		raise ApiError(404, "record_not_found", "That flight record doesn't exist or has expired.")
	rec = view_of(rec_raw)
	sid: str = request.state.session
	pinned = set(body.setting.pinned)
	try:
		check_setting(body.setting, rec)
		state = open_state(body.state, sid=sid, record_key=rec.key)
	except InputError as e:
		raise ApiError(422, "invalid_request", str(e)) from e
	except BadState as e:
		raise ApiError(
			409, "guide_state_invalid", "The guide's conversation can't continue; it starts again."
		) from e

	# ---- what happened since last round
	pending = {p.call_id: p for p in state.pending}
	answered = False
	for out in body.results:
		p = pending.get(out.call_id)
		if p is None or p.name != out.tool or p.needs_approval:
			raise ApiError(422, "invalid_request", f"No pending {out.tool} call {out.call_id}.")
		try:
			check_result(out, rec)
		except InputError as e:
			raise ApiError(422, "invalid_request", str(e)) from e
		if isinstance(out, SetKnobOut) and out.moved:
			state.move = out.knob_id
		state.add(
			{
				"type": "function_call_output",
				"call_id": out.call_id,
				"output": out.model_dump_json(exclude={"tool", "call_id"}),
			}
		)
		del pending[out.call_id]
		answered = True

	ctx_knobs = dict(body.setting.knobs)
	for a in body.approvals:
		p = pending.get(a.call_id)
		if p is None or not p.needs_approval:
			raise ApiError(422, "invalid_request", f"No call {a.call_id} is waiting for approval.")
		del pending[a.call_id]
		if a.approved:
			result = await execute(
				p.name,
				p.arguments,
				ExecContext(
					record=rec, record_ref=body.record, visitor=sid, call_id=p.call_id, knobs=ctx_knobs
				),
				app.state.jobs,
			)
			state.add(
				_event(f"The person approved {p.name} (call {p.call_id}). Result: {json.dumps(result)}")
			)
			numbers._flatten(result, state.evidence)
		else:
			state.add(_event(f"The person declined {p.name} (call {p.call_id}); don't ask again this round."))
		answered = True

	unanswered = [p for p in pending.values() if not p.needs_approval]
	if unanswered:
		raise ApiError(
			422, "invalid_request", f"Results missing for {', '.join(p.call_id for p in unanswered)}."
		)
	state.pending = list(pending.values())

	new_turn = False
	if body.decision:
		try:
			check_decision(body.decision, rec)
		except InputError as e:
			raise ApiError(422, "invalid_request", str(e)) from e
		d = body.decision
		value = f" ({json.dumps(d.value)})" if d.value is not None else ""
		state.move = None
		new_turn = True
		state.turn_start = len(state.items)
		state.evidence = []
		state.add(_event(f"The person {d.action} the move of {d.knob_id}{value}. Plan the next round."))
	if body.message:
		new_turn = True
		if not body.decision:
			state.turn_start = len(state.items)
			state.evidence = []
		state.add({"role": "user", "content": body.message})
	if body.start and not state.items:
		new_turn = True
		aim = body.setting.objective.kind
		state.add(
			_event(
				f"The person opened the cockpit on {rec.title or rec.lens}; their aim is {aim}. Start the tour."
			)
		)
	if not (answered or new_turn):
		raise ApiError(422, "invalid_request", "Nothing new for the guide this round.")
	if new_turn:
		state.rounds = 0
	if state.rounds >= MAX_ROUNDS_PER_TURN:
		raise _fallback(
			"too_many_rounds", "The live guide went round in circles; the rule-based tour takes over."
		)

	setting_note = _event(
		"Current setting: "
		+ json.dumps(
			{
				"knobs": body.setting.knobs,
				"pinned": sorted(pinned),
				"objective": body.setting.objective.model_dump(exclude_none=True),
			}
		)
	)

	# ---- reserve, call, settle
	tools = build_tools(rec, pinned, move_pending=state.move is not None)
	offered = {t["name"] for t in tools}
	instructions = INSTRUCTIONS + language_line(body.language)
	items = [*state.items, setting_note]
	worst, tokens_in = _worst_case_usd(instructions, items, tools)
	if tokens_in > MAX_INPUT_TOKENS:
		raise _fallback(
			"turn_too_large",
			"This conversation is too long for the live guide; the rule-based tour takes over.",
		)
	ledger = app.state.ledger
	decision = await ledger.reserve(
		visitor=sid, run=f"{rec.key}:{sid}", ip=edge.ip_key(edge.client_ip(request)), usd=worst
	)
	if not decision.ok:
		raise _fallback(
			decision.reason or "budget",
			"You've used the live guide's allowance for now; the rule-based tour takes over.",
			decision.retry_after_s,
		)
	try:
		result = await muse_spark.create_turn(
			instructions=instructions,
			input_items=items,
			tools=tools,
			max_output_tokens=MAX_OUTPUT_TOKENS,
			client=app.state.http,
		)
	except muse_spark.GuideModelError as e:
		await ledger.release(decision.reservation)
		raise _fallback("model_error", f"{e} The rule-based tour takes over for now.") from e
	except Exception:
		await ledger.settle(decision.reservation, worst)  # unknown outcome: keep the worst case
		raise
	await ledger.settle(decision.reservation, result.cost_usd)
	state.rounds += 1

	# ---- guard what the model did
	calls_out: list[dict] = []
	moved_now = False
	for c in result.calls:
		state.add(
			{
				"type": "function_call",
				"call_id": c.call_id,
				"name": c.name,
				"arguments": json.dumps(c.arguments),
			}
		)
		refusal = None
		if c.name not in offered:
			refusal = "tool_not_offered"
		elif c.name == "set_knob":
			refusal = _guard_set_knob(c.arguments, rec, pinned, state, moved_now)
			moved_now = moved_now or refusal is None
		if refusal:
			state.add(
				{
					"type": "function_call_output",
					"call_id": c.call_id,
					"output": json.dumps({"refused": refusal}),
				}
			)
			continue
		if c.name in SERVER_TOOLS:
			state.pending.append(Pending(c.call_id, c.name, c.arguments, needs_approval=True))
			state.add(
				{
					"type": "function_call_output",
					"call_id": c.call_id,
					"output": json.dumps({"status": "awaiting_person_approval"}),
				}
			)
			calls_out.append(
				{"call_id": c.call_id, "name": c.name, "arguments": c.arguments, "runs_in": "approval"}
			)
		elif c.name in BROWSER_TOOLS:
			state.pending.append(Pending(c.call_id, c.name, c.arguments))
			calls_out.append(
				{"call_id": c.call_id, "name": c.name, "arguments": c.arguments, "runs_in": "browser"}
			)

	text, withheld = result.text.strip(), None
	if text:
		evidence = numbers.evidence_from(
			_turn_evidence(state),
			state.evidence,
			body.setting.knobs,
			numbers.record_numbers(rec_raw),
			body.message or "",
		)
		bad = numbers.unsupported(text, evidence, numbers.record_vocabulary(rec_raw))
		state.add({"role": "assistant", "content": text})
		if bad:
			withheld = bad
			state.add(
				{
					"role": "user",
					"content": f"[check] Your last reply quoted {', '.join(bad)}, which no tool returned this round, so it wasn't shown. Quote only tool results.",
				}
			)
			text = ""

	awaiting = (
		"results"
		if any(c["runs_in"] == "browser" for c in calls_out)
		else "decision"
		if state.move is not None or moved_now
		else "approval"
		if any(p.needs_approval for p in state.pending)
		else "visitor"
	)
	try:
		sealed = seal(state, sid=sid, record_key=rec.key)
	except BadState as e:
		raise _fallback(
			"turn_too_large",
			"This conversation is too long for the live guide; the rule-based tour takes over.",
		) from e
	return {
		"text": text,
		"withheld": withheld,
		"calls": calls_out,
		"awaiting": awaiting,
		"state": sealed,
		"provenance": {
			"provider": "meta",
			"model": muse_spark.MODEL,
			"purpose": "guide",
			"input_tokens": result.input_tokens,
			"output_tokens": result.output_tokens,
			"cached_tokens": result.cached_tokens,
			"cost_usd": round(result.cost_usd, 6),
			"request_id": result.request_id,
			"at": dt.datetime.now(dt.UTC).isoformat(),
		},
	}


class ToolIn(BaseModel):
	"""A spending tool the visitor approved from the rule-based tour (no model involved)."""

	model_config = ConfigDict(extra="forbid")
	record: Annotated[str, Field(pattern=r"^[A-Za-z0-9_:-]{1,80}$")]
	tool: Literal["run_probe", "run_verify", "sample_more", "propose_patch"]
	setting: dict[Annotated[str, Field(pattern=r"^[a-z][a-z0-9_.:-]{0,79}$")], Value] = Field(max_length=96)
	knobs: dict[Annotated[str, Field(pattern=r"^[a-z][a-z0-9_.:-]{0,79}$")], Value] = Field(
		default_factory=dict, max_length=96
	)
	knob_id: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_.:-]{0,79}$")] | None = None
	reason: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_.]{0,47}$")] | None = None


@router.post("/guide/tools")
async def guide_tool(request: Request) -> dict:
	"""Run one approved spending tool. Each click is its own idempotency key (the request id)."""
	try:
		body = ToolIn.model_validate(await request.json())
	except (ValidationError, ValueError) as e:
		raise ApiError(422, "invalid_request", "The tool request isn't in the expected shape.") from e
	app = request.app
	rec_raw = resolve_record(body.record, app.state.samples, app.state.records)
	if rec_raw is None:
		raise ApiError(404, "record_not_found", "That flight record doesn't exist or has expired.")
	rec = view_of(rec_raw)
	for kid, v in {**body.setting, **body.knobs}.items():
		k = rec.knobs.get(kid)
		if k is None or (not k.allows(v) and v != k.current):
			raise ApiError(422, "invalid_request", f"{kid} can't be {v!r}.")
	# The job store caps jobs per visitor; the ledger's IP bucket stops a visitor minting sessions
	# to get round that (a zero-dollar reservation spends a turn and an IP token).
	sid = request.state.session
	gate = await app.state.ledger.reserve(
		visitor=sid, run=f"{rec.key}:{sid}", ip=edge.ip_key(edge.client_ip(request)), usd=0.0
	)
	if not gate.ok:
		raise ApiError(
			429,
			"rate_limited",
			"Too many requests from here; try again shortly.",
			extra={"retry_after_s": gate.retry_after_s},
		)
	await app.state.ledger.settle(gate.reservation, 0.0)
	key = request.headers.get("Idempotency-Key") or request.state.request_id
	args = {"knobs": body.knobs, "knob_id": body.knob_id, "reason": body.reason}
	ctx = ExecContext(
		record=rec,
		record_ref=body.record,
		visitor=request.state.session,
		call_id=key,
		knobs=dict(body.setting),
	)
	return await execute(body.tool, args, ctx, app.state.jobs)
