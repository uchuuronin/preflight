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

from preflight.harness.diagnosis import numbers, resolve
from preflight.schema import Evidence, Finding, FlightRecord

MAX_ROUNDS = 4  # tool rounds per claim before it abstains
MAX_FETCHES = 12  # tool calls answered per claim

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


def _answer(text: str) -> dict | None:
	m = re.search(r"\{.*\}", text or "", re.S)
	if not m:
		return None
	try:
		d = json.loads(m.group(0))
	except ValueError:
		return None
	return d if isinstance(d, dict) else None


def check_claim(client, nums: dict, finding: Finding) -> tuple[str, str | None, str, list]:
	"""(verdict "supported" or "abstain", the supporting path, the reason, the calls made)."""
	msgs = [
		{"role": "system", "content": SYSTEM},
		{
			"role": "user",
			"content": f"Claim: {finding.title}. {finding.detail or ''}\n\nPaths:\n{_paths_guide(nums)}",
		},
	]
	fetched: dict[str, float | None] = {}
	before = len(client.calls)
	try:
		for _ in range(MAX_ROUNDS):
			msg, _call = client.chat_message(
				msgs, purpose="diagnosis_check", reasoning_effort="low", tools=[TOOL]
			)
			tool_calls = msg.get("tool_calls") or []
			if not tool_calls:
				break
			msgs.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tool_calls})
			for tc in tool_calls:
				try:
					path = str(json.loads(tc["function"]["arguments"]).get("path", ""))[:200]
				except (KeyError, TypeError, ValueError):
					path = ""
				value = resolve(nums, path) if path and len(fetched) < MAX_FETCHES else None
				if path and len(fetched) < MAX_FETCHES:
					fetched[path] = value
				msgs.append({"role": "tool", "tool_call_id": tc.get("id", ""),
					"content": json.dumps({"path": path, "value": value})})  # fmt: skip
		else:
			return "abstain", None, "no answer within the tool rounds", client.calls[before:]
	except Exception as exc:  # a failed check is no support; the attempts stay in provenance
		return "abstain", None, f"the check didn't complete ({type(exc).__name__})", client.calls[before:]
	ans = _answer(msg.get("content") or "")
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


def check_diagnosis(client, rec: FlightRecord, findings: list[Finding]) -> tuple[list[Finding], list]:
	"""The diagnosis findings with Spark Open's check applied, and every call made."""
	nums = numbers(rec)
	out, calls = [], []
	for f in findings:
		verdict, path, reason, cs = check_claim(client, nums, f)
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
