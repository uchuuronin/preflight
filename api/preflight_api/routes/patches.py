"""/v1/patches: the fix loop's race. Muse Code writes up to three candidate fixes at once; each
comes back as a diff, and verify (A's job store, mode "verify" with the patch) picks the winner.

POST /v1/patches                  {record, finding?, candidates: [{template_id, value?}]} -> 202
GET  /v1/patches/{patch_id}       the race as the visitor may see it (service.public)
POST /v1/patches/{patch_id}/cancel

Who may race: anyone on a sample with a script (A's SAMPLE_REFS) or a record of public code
(a repo); on uploaded code, only the visitor whose own job made the record (owners.py), and never
a record derived from another (a probe on a shared link would otherwise hand its code out).

Spend: the race's whole cap (run.stop_usd) is reserved in the ledger before anything starts,
against the guide's day cap and a fix budget per address, kept open as long as a race can run,
and settled with what Muse Code reported once the race ends (the cap when nothing was).
Behind `c.fix_loop`.
"""

from __future__ import annotations

import re
from typing import Annotated, Any

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .. import edge
from ..errors import ApiError
from ..fix import run
from ..fix.service import Race, new_patch_id, now, public
from ..fix.templates import TEMPLATES, task_for
from ..flags import flag
from ..guide.record_view import RecordView
from ..jobs import visitor_of
from .guide import _record

router = APIRouter(prefix="/v1")

PATCH_ID = re.compile(r"^fx_[A-Za-z0-9_-]{16,40}$")
PUBLIC_SOURCES = {"repo", "sample", "template"}
# Source: derived, the fix Function's own limit (api/fix_app.py RACE_TIMEOUT_S, 12 min) plus a
# minute: past it the race is lost and its reservation settles at the cap.
RACE_LOST_S = 13 * 60
NOT_OPEN = "Code fixes aren't open yet."
# Source: guess, about five three-candidate races a day from one address (a judging room or a
# campus behind one address), a fifth of the shared US$10 day (ledger.Limits.usd_per_day).
FIX_USD_PER_ADDRESS_DAY = 2.00

Value = bool | int | float | str


class CandidateIn(BaseModel):
	model_config = ConfigDict(extra="forbid")
	template_id: Annotated[str, Field(pattern=r"^[a-z]+\.[a-z0-9_]{1,40}$")]
	value: Value | None = None


class PatchIn(BaseModel):
	model_config = ConfigDict(extra="forbid")
	record: Annotated[str, Field(pattern=r"^[A-Za-z0-9_:-]{1,80}$")]
	finding: Annotated[str, Field(pattern=r"^[A-Za-z0-9_.:-]{1,80}$")] | None = None
	candidates: list[CandidateIn] = Field(min_length=1, max_length=3)


def _fixer(request: Request):
	fixer = getattr(request.app.state, "fixer", None)
	if fixer is None or not flag("c.fix_loop"):
		raise ApiError(503, "fixes_not_open", NOT_OPEN)
	return fixer


def _a_ref(ref: str, rec: dict, samples) -> str:
	"""The record_ref A's resolve() reads: `sample:<id>` for a bundled sample, else the token."""
	from preflight.runner.record_jobs import SAMPLE_REFS

	if samples.get(ref) is not None:
		if ref not in SAMPLE_REFS:
			raise ApiError(422, "no_script", "This sample has no script for Muse Code to fix.")
		return f"sample:{ref}"
	return ref


async def _may_race(request: Request, ref: str, rec: dict) -> None:
	src = (rec.get("job") or {}).get("source") or {}
	if src.get("kind") in PUBLIC_SOURCES or request.app.state.samples.get(ref) is not None:
		return
	owners = getattr(request.app.state, "owners", None)
	mine = (
		owners is not None
		and rec.get("parent_id") is None
		and await owners.owner(ref) == visitor_of(request.state.session)
	)
	if not mine:
		raise ApiError(
			403,
			"not_your_code",
			"Muse Code fixes uploaded code only for whoever checked it in, from the link they got.",
		)


def _checked(body: PatchIn, view: RecordView) -> list[dict]:
	seen, out = set(), []
	for c in body.candidates:
		t = TEMPLATES.get(c.template_id)
		if t is None or view.lens not in t.lenses:
			raise ApiError(422, "invalid_request", f"{c.template_id} isn't a fix for a {view.lens} run.")
		k = view.knobs.get(t.knob) if t.knob else None
		if k is not None and c.value is not None and not k.allows(c.value):
			raise ApiError(422, "invalid_request", f"{t.knob} can't be {c.value!r}.")
		try:
			task_for(c.template_id, c.value)
		except (KeyError, ValueError) as e:
			raise ApiError(422, "invalid_request", f"{c.template_id}: {e}") from e
		pair = (c.template_id, repr(c.value))
		if pair in seen:
			raise ApiError(422, "invalid_request", "Each candidate is a different fix.")
		seen.add(pair)
		out.append({"template_id": c.template_id, "value": c.value})
	return out


def _measured(rec: dict, finding_id: str | None) -> str | None:
	"""Preflight's own sentence about the problem: a rule finding's title, never a model's words."""
	if finding_id is None:
		return None
	f = next((x for x in rec.get("findings") or [] if x.get("id") == finding_id), None)
	if f is None:
		raise ApiError(422, "invalid_request", "That finding isn't on this record.")
	title = f.get("title") or ""
	if f.get("source") != "rule" or not title or len(title) > 200 or "\n" in title:
		return None
	return title


@router.post("/patches", status_code=202)
async def start_race(request: Request) -> dict:
	fixer = _fixer(request)
	try:
		body = PatchIn.model_validate(await request.json())
	except (ValidationError, ValueError) as e:
		raise ApiError(422, "invalid_request", "The fix request isn't in the expected shape.") from e
	app = request.app
	rec = await _record(body.record, app)
	if rec is None:
		raise ApiError(404, "record_not_found", "That flight record doesn't exist or has expired.")
	view = RecordView.from_record(rec)
	cands = _checked(body, view)
	measured = _measured(rec, body.finding)
	ref = _a_ref(body.record, rec, app.state.samples)
	await _may_race(request, body.record, rec)

	sid: str = request.state.session
	cap = run.stop_usd(len(cands))
	ip = edge.ip_key(edge.client_ip(request))
	# The fix budget is per address, not per session (sessions are free to mint), and is big
	# enough for a room of visitors behind one address; the reservation stays open as long as a
	# race can run, so what Muse Code reports is what gets counted.
	gate = await app.state.ledger.reserve(
		visitor=sid, run=f"fix:ip:{ip}", ip=ip, usd=cap, run_cap=FIX_USD_PER_ADDRESS_DAY, ttl_s=RACE_LOST_S
	)
	if not gate.ok:
		today = gate.reason in ("daily_cap", "run_cap")
		raise ApiError(
			429,
			"fix_limit" if today else "rate_limited",
			"Muse Code has done as many fixes as it can here today; try again tomorrow."
			if today
			else "Too many requests from here; try again shortly.",
			extra={"reason": gate.reason, "retry_after_s": gate.retry_after_s},
		)
	try:
		call_id = await fixer.start(record_ref=ref, candidates=cands, measured=measured, stop_usd=cap)
	except Exception as e:
		await app.state.ledger.release(gate.reservation)
		raise ApiError(
			503, "fixes_unavailable", "Muse Code couldn't start; nothing was spent.", retryable=True
		) from e
	race = Race(
		patch_id=new_patch_id(),
		visitor=visitor_of(sid),
		record_ref=body.record,
		call_id=call_id,
		reservation=gate.reservation,
		worst_usd=cap,
		started=now(),
	)
	await app.state.races.put(race)
	return {**public(race), "candidates_asked": cands, "cap_usd": cap}


async def _race(request: Request, patch_id: str) -> Race:
	race = await request.app.state.races.get(patch_id) if PATCH_ID.match(patch_id) else None
	if race is None or race.visitor != visitor_of(request.state.session):
		raise ApiError(404, "patch_not_found", "No fix matches that id.")
	return race


async def _settle(request: Request, race: Race, result: dict[str, Any]) -> None:
	race.result, race.settled = result, True
	spent = result.get("usd_from_reported_tokens")
	# What Muse Code reported, even past the cap (calls already in flight when the box stops are
	# still billed); the cap only when nothing was reported.
	usd = race.worst_usd if not isinstance(spent, (int, float)) else max(0.0, spent)
	if race.reservation:
		await request.app.state.ledger.settle(race.reservation, usd)
	await request.app.state.races.put(race)


@router.get("/patches/{patch_id}")
async def get_race(patch_id: str, request: Request) -> dict:
	fixer = _fixer(request)
	race = await _race(request, patch_id)
	if not race.settled:
		try:
			res = await fixer.poll(race.call_id)
		except Exception:  # a fixer that can't be asked is the same as no answer yet
			res = None
		if res is None and now() - race.started > RACE_LOST_S:
			res = {"error": "lost", "candidates": []}
		if res is not None:
			await _settle(request, race, res)
	return public(race)


@router.post("/patches/{patch_id}/cancel")
async def cancel_race(patch_id: str, request: Request) -> dict:
	fixer = _fixer(request)
	race = await _race(request, patch_id)
	if not race.settled:
		try:
			await fixer.cancel(race.call_id)
		finally:
			# What it spent before the stop isn't known here: the reservation settles at the cap.
			await _settle(request, race, {"error": "cancelled", "candidates": []})
	return public(race)
