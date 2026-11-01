"""The guide's tools, built on the server from the record for every round. The browser never
supplies a tool definition, so the endpoint can't be used to call the model with tools of
someone else's choosing.

Browser tools run in the visitor's tab (predict, sensitivity, set_knob, highlight, pin): they
read the PredictionProvider the screen uses, so the guide quotes the same numbers the dials
show. Server tools spend money (run_probe, propose_patch, run_verify, sample_more): the model
may only ask; the call waits for the visitor's click and then runs here (executors.py).
"""

from __future__ import annotations

from .record_view import RecordView
from .types import Limits

BROWSER_TOOLS = frozenset({"predict", "sensitivity", "set_knob", "highlight", "pin"})
SERVER_TOOLS = frozenset({"run_probe", "propose_patch", "run_verify", "sample_more"})
ALL_TOOLS = BROWSER_TOOLS | SERVER_TOOLS
SAMPLE_MORE_TASKS = 20  # source: contract (sample_more samples 20 more agent tasks)

OBJECTIVES = ["min_cost", "min_time", "fit_memory", "min_risk", "max_throughput"]
_VALUE = {"anyOf": [{"type": "number"}, {"type": "string"}, {"type": "boolean"}]}


def _fn(name: str, description: str, properties: dict, required: list[str]) -> dict:
	return {
		"type": "function",
		"name": name,
		"description": description,
		"parameters": {
			"type": "object",
			"properties": properties,
			"required": required,
			"additionalProperties": False,
		},
	}


def _left(rec: RecordView, kid: str, limits: Limits) -> bool:
	"""A knob still has a value to move to inside the limits (a categorical one is checked value by
	value; a numeric one has a range, so it stays unless its allowed list leaves only where it is)."""
	k = rec.knobs[kid]
	allow = limits.allowed.get(kid)
	if allow:
		return any(v != k.current and limits.allows(kid, v) for v in allow)
	if k.kind in ("categorical", "bool"):
		values = k.values if k.kind == "categorical" else (True, False)
		return any(v != k.current and limits.allows(kid, v) for v in values)
	return True


def build_tools(
	rec: RecordView, pinned: set[str], *, move_pending: bool, limits: Limits | None = None
) -> list[dict]:
	limits = limits or Limits()
	knob_ids = sorted(rec.knobs)
	# Pinned knobs, and knobs the visitor's limits leave nowhere to go, aren't offered to set_knob.
	movable = [k for k in knob_ids if k not in pinned and k not in limits.declined and _left(rec, k, limits)]
	code_knobs = sorted(k for k, v in rec.knobs.items() if v.needs_code and v.patch_template)
	nodes = sorted(rec.nodes)
	tools = [
		_fn(
			"sensitivity",
			"Unpinned knobs ranked by how much their best single move improves the aim. Only moves that "
			"fit in memory are ranked first; a 'tight' entry needs measuring before anyone relies on it.",
			{"objective": {"type": "string", "enum": OBJECTIVES}},
			["objective"],
		),
		_fn(
			"predict",
			"Readouts (point, interval, status) at a setting; knobs not given keep their current value. "
			"Quote only numbers a tool returned this turn.",
			{
				"knobs": {
					"type": "object",
					"propertyNames": {"enum": knob_ids},
					"additionalProperties": _VALUE,
				}
			},
			["knobs"],
		),
		_fn(
			"highlight",
			"Light map nodes while you talk about them.",
			{"nodes": {"type": "array", "items": {"type": "string", "enum": nodes}, "maxItems": 8}},
			["nodes"],
		),
		_fn(
			"pin",
			"Pin or unpin a knob, only when the person asks you to.",
			{"knob_id": {"type": "string", "enum": knob_ids}, "pinned": {"type": "boolean"}},
			["knob_id", "pinned"],
		),
		_fn(
			"run_probe",
			"Measure a setting on a real GPU. Costs money; the person must click to approve, then it runs.",
			{
				"knobs": {
					"type": "object",
					"propertyNames": {"enum": knob_ids},
					"additionalProperties": _VALUE,
				},
				"reason": {"type": "string", "maxLength": 200},
			},
			["knobs", "reason"],
		),
		_fn(
			"run_verify",
			"One short real run of the current plan plus a fixed-seed loss check. Costs money; needs the person's click.",
			{},
			[],
		),
	]
	if code_knobs:
		tools.append(
			_fn(
				"propose_patch",
				"Ask Muse Code to write the code change a code knob needs. Costs money; needs the person's click.",
				{"knob_id": {"type": "string", "enum": code_knobs}},
				["knob_id"],
			)
		)
	if rec.samples_tasks:
		tools.append(
			_fn(
				"sample_more",
				f"Run {SAMPLE_MORE_TASKS} more of the run's tasks for real to narrow the cost range. Only when "
				"that range straddles the budget. Costs API money; needs the person's click.",
				{"reason": {"type": "string", "maxLength": 200}},
				["reason"],
			)
		)
	# One move per round. While a move waits for the person, no browser tool is offered at all, so
	# the guide explains the move and stops (Oct 27: with highlight still offered, Muse Spark kept
	# highlighting instead of stopping; docs/decisions/guide-limits.md).
	if move_pending:
		return [t for t in tools if t["name"] in SERVER_TOOLS]
	if movable:
		tools.insert(
			0,
			_fn(
				"set_knob",
				"Move one unpinned knob on the live map; the person applies, reverts or pins it. Only to a "
				"setting that fits and stays inside the person's limits (the setting lists them: allowed "
				"values, excluded values, budget, deadline, GPU-hours); anything else is refused. One move "
				"per round: after it, explain in one or two sentences and stop.",
				{
					"knob_id": {"type": "string", "enum": movable},
					"value": _VALUE,
					"why": {"type": "string", "maxLength": 200},
				},
				["knob_id", "value", "why"],
			),
		)
	return tools
