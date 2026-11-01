"""Spark Open checks each Muse Spark diagnosis claim against the probe numbers (plan item 30, slim).

	checked, calls = check_diagnosis(ModelClient.baseten_spark_open(), rec, findings)

Muse Spark diagnoses (harness/diagnosis.py); Spark Open checks it. One conversation per claim:
Spark Open gets the claim and how the record's numbers are addressed, not the numbers, and must
fetch the ones the claim rests on through a tool call, `probe_number(path)`, which code answers
from the record. Then it says whether the fetched numbers support the claim, citing one path.

Code decides, not the model: a claim stands only if Spark Open says it's supported AND the path it
cites is one it fetched through the tool in that conversation AND that path resolves to a number.
Anything else (unsupported, no fetch, a path that doesn't resolve, a failed or malformed call)
becomes "abstain, or run a probe". Every call, tool rounds included, is returned for provenance.
"""

import json
import re
import time

from preflight.harness.diagnosis import numbers, resolve
from preflight.schema import Evidence, Finding, FlightRecord

MAX_ROUNDS = 3  # tool rounds per claim before it abstains
MAX_FETCHES = 12  # tool calls answered per claim
MAX_CLAIMS = 6  # claims checked per record; the rest abstain unchecked
DEADLINE_S = 120.0  # one budget for the whole check, every claim's calls included

TOOL = {
	"type": "function",
	"function": {
		"name": "probe_number",
		"description": "One raw number from the record's probes, by path. Returns {path, value}; value is null "
		"when the path doesn't resolve to a number.",
		"parameters": {
			"type": "object",
			"properties": {
				"path": {"type": "string", "description": "e.g. probes[p1].step_log[12].peak_allocated_bytes"}
			},
			"required": ["path"],
			"additionalProperties": False,
		},
	},
}

SYSTEM = """You check one claim about a training run against the run's raw probe numbers.
You can't see the numbers: fetch each one the claim rests on with the probe_number tool, by path.
When you have what you need, answer with JSON only:
{"verdict": "supported" or "unsupported", "path": "<the fetched path that supports it, or null>",
"reason": "<one sentence>"}
"supported" means a number you fetched bears the claim out. If no fetched number does, say
"unsupported". Don't guess a number you didn't fetch."""


def _paths_guide(nums: dict) -> str:
	lines = []
	for pid, p in nums["probes"].items():
		n_steps, n_loss = len(p["step_log"]), len(p["stability"]["series"]["loss"])
		lines.append(
			f"probes[{pid}].memory.{{peak_allocated_bytes, peak_reserved_bytes, device_total_bytes}}; "
			f"probes[{pid}].steps; "
			f"probes[{pid}].step_log[0..{n_steps - 1}].{{index, wall_s, peak_allocated_bytes, peak_reserved_bytes, "
			f"device_used_bytes}}; probes[{pid}].stability.series.loss[0..{n_loss - 1}]"
		)
	return "\n".join(lines)


def _text(content) -> str:
	"""A message's text, whether the provider sends a string or a list of parts."""
	if isinstance(content, list):
		return "".join(
			p.get("text", "") for p in content if isinstance(p, dict) and isinstance(p.get("text"), str)
		)
	return content if isinstance(content, str) else ""


def _claim(f: Finding) -> str:
	"""What Spark Open is asked to check: Muse Spark's claim and the plot region, without the path
	and value it cited, so the check finds its own number rather than echoing the diagnosis's."""
	detail = re.sub(r"; number: [^;()]*\)$", ")", (f.detail or "").strip())
	# Muse Spark's prose may quote the value too: numbers there are masked, the plot region's kept
	prose, sep, region = detail.partition(" (plot: ")
	prose = re.sub(r"\d[\d,._]*(?:[eE][+-]?\d+)?", "#", prose)
	return f"{f.title}. {prose}{sep}{region}"


def _answer(text: str) -> dict | None:
	m = re.search(r"\{.*\}", text or "", re.S)
	if not m:
		return None
	try:
		d = json.loads(m.group(0))
	except ValueError:
		return None
	return d if isinstance(d, dict) else None


def _worst_call_usd(client, msgs: list[dict]) -> float:
	"""What one more call could cost at most: its input as estimated, the client's full output cap."""
	from preflight.harness.spark import estimate_tokens

	p_in, _, p_out = client.prices.get(client.model, (0.0, 0.0, 0.0))
	return (estimate_tokens(msgs) * p_in + client.max_output_tokens * p_out) / 1e6


def check_claim(
	client, nums: dict, finding: Finding, deadline: float | None = None, room=None
) -> tuple[str, str | None, str, list]:
	"""(verdict "supported" or "abstain", the supporting path, the reason, the calls made)."""
	msgs = [
		{"role": "system", "content": SYSTEM},
		{
			"role": "user",
			"content": f"Claim: {_claim(finding)}\n\nPaths:\n{_paths_guide(nums)}",
		},
	]
	fetched: dict[str, float | None] = {}
	before = len(client.calls)
	try:
		for _ in range(MAX_ROUNDS):
			if deadline is not None and time.monotonic() >= deadline:
				return "abstain", None, "the check ran out of time", client.calls[before:]
			if room is not None and _worst_call_usd(client, msgs) > room():
				return "abstain", None, "the check's dollar cap was reached", client.calls[before:]
			msg, _call = client.chat_message(
				msgs, purpose="diagnosis_check", reasoning_effort="low", tools=[TOOL]
			)
			tool_calls = msg.get("tool_calls") or []
			if not tool_calls:
				break
			msgs.append(
				{"role": "assistant", "content": _text(msg.get("content")) or None, "tool_calls": tool_calls}
			)
			for tc in tool_calls:
				try:
					path = str(json.loads(tc["function"]["arguments"]).get("path", ""))[:200]
				except (KeyError, TypeError, ValueError):
					path = ""
				value = resolve(nums, path) if path and len(fetched) < MAX_FETCHES else None
				if path and len(fetched) < MAX_FETCHES:
					fetched[path] = value
				msgs.append({"role": "tool", "tool_call_id": str(tc.get("id") or ""),
					"content": json.dumps({"path": path, "value": value})})  # fmt: skip
		else:
			return "abstain", None, "no answer within the tool rounds", client.calls[before:]
	except Exception as exc:  # a failed check is no support; the attempts stay in provenance
		return "abstain", None, f"the check didn't complete ({type(exc).__name__})", client.calls[before:]
	ans = _answer(_text(msg.get("content")))
	calls = client.calls[before:]
	if ans is None:
		return "abstain", None, "no readable answer", calls
	path = ans.get("path")
	reason = str(ans.get("reason") or "")[:200]
	if ans.get("verdict") != "supported":
		return "abstain", None, reason or "no fetched number supports it", calls
	if not isinstance(path, str) or path not in fetched:
		return "abstain", None, "the supporting number wasn't fetched through the tool", calls
	if fetched[path] is None:
		return "abstain", None, "the supporting path doesn't resolve to a number", calls
	return "supported", path, reason, calls


def check_diagnosis(
	client, rec: FlightRecord, findings: list[Finding], budget_usd: float | None = None
) -> tuple[list[Finding], list]:
	"""The diagnosis findings with Spark Open's check applied, and every call made. With budget_usd,
	no call starts whose worst case would take the check's spend past it."""
	nums = numbers(rec)
	out, calls = [], []
	deadline = time.monotonic() + DEADLINE_S
	start = len(client.calls)

	def room() -> float:
		return budget_usd - sum(c.cost_usd for c in client.calls[start:])

	for i, f in enumerate(findings):
		if i < MAX_CLAIMS:
			verdict, path, reason, cs = check_claim(
				client, nums, f, deadline, room if budget_usd is not None else None
			)
		else:
			verdict, path, reason, cs = "abstain", None, f"not checked (only the first {MAX_CLAIMS} are)", []
		calls += cs
		if verdict == "supported":
			out.append(f.model_copy(update={
				"detail": f"{f.detail} Spark Open checked it against {path} = {resolve(nums, path)}.",
				"evidence": [*f.evidence, Evidence(metric="spark_open_check", value=path, probe_id=f.evidence[0].probe_id if f.evidence else None)]}))  # fmt: skip
		else:
			out.append(f.model_copy(update={
				"severity": "info", "code": "diagnosis.abstain",
				"title": f"Not backed by a probe number: abstain, or run a probe ({f.title[:80]})",
				"detail": f"{f.detail} Spark Open's check: {reason}."}))  # fmt: skip
	return out, calls
