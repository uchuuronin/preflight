"""One Spark diagnosis of a stored record and Spark Open's check of it, outside the pipeline: the
real check before harness.spark_diagnosis goes near production (Chiya, Nov 1). The record isn't
changed; what comes back is the findings, every call and the spend, which never passes max_usd.
"""

import json

from preflight.harness.diagnosis import SYSTEM, diagnose_record, numbers
from preflight.harness.diagnosis_check import check_diagnosis
from preflight.harness.spark import IMAGE_TOKENS
from preflight.schema import FlightRecord

DIAGNOSIS_OUTPUT_TOKENS = 4_000  # the diagnosis call's output cap here: eight findings fit in far less
MARGIN = 1.5  # [guess] characters per token is an estimate, not a ceiling
REASK_EXTRA_TOKENS = (
	1_500  # the re-ask carries up to 4,000 characters of the first answer and a 2,000-character error
)
CHECK_OUTPUT_TOKENS = 1_500  # each Spark Open check call's output cap


def _diagnosis_worst_usd(client, rec: FlightRecord) -> float:
	"""The diagnosis at its worst: two calls (the first and a re-ask), each the numbers, two plots
	and the system prompt in, the output cap out."""
	if client.model not in client.prices:
		return float("inf")  # an unpriced model has no bound
	p_in, _, p_out = client.prices[client.model]
	inp = (len(json.dumps(numbers(rec), separators=(",", ":"))) + len(SYSTEM)) // 4 + 2 * IMAGE_TOKENS
	one = inp * p_in + client.max_output_tokens * p_out
	return MARGIN * (2 * one + REASK_EXTRA_TOKENS * p_in) / 1e6


def run_once(rec: FlightRecord, spark, spark_open, max_usd: float = 0.5) -> dict:
	"""Muse Spark's diagnosis if its worst case fits under max_usd, then Spark Open's check of each
	claim with what's left (no check call starts whose worst case would pass it)."""
	spark.max_output_tokens = min(spark.max_output_tokens, DIAGNOSIS_OUTPUT_TOKENS)
	worst = _diagnosis_worst_usd(spark, rec)
	out: dict = {"record_id": rec.record_id, "diagnosis_worst_usd": round(worst, 4)}
	if worst > max_usd:
		return {**out, "skipped": f"the diagnosis could cost up to ${worst:.2f}, over the ${max_usd:.2f} cap",
			"stats": {}, "diagnosis": [], "checked": None, "calls": [], "usd": 0.0}  # fmt: skip
	found, calls, stats = diagnose_record(spark, rec)
	# a call that failed may still have been billed (a read timeout records no usage): count it at its worst
	spent = sum(c.cost_usd for c in calls) + sum(worst / 2 for c in calls if not c.ok and not c.cost_usd)
	out.update(stats=stats, diagnosis=[f.model_dump(mode="json") for f in found])
	if not found:
		out["checked"] = None
		out["check_skipped"] = (
			"the diagnosis call failed" if calls and not any(c.ok for c in calls) else "no findings"
		)
	elif spark_open is None:
		out["checked"], out["check_skipped"] = None, "no Spark Open client"
	else:
		spark_open.max_output_tokens = min(spark_open.max_output_tokens, CHECK_OUTPUT_TOKENS)
		checked, checks = check_diagnosis(spark_open, rec, found, budget_usd=max(0.0, max_usd - spent))
		calls = [*calls, *checks]
		out["checked"] = [f.model_dump(mode="json") for f in checked]
	out["calls"] = [c.model_dump(mode="json") for c in calls]
	out["usd"] = round(sum(c.cost_usd for c in calls), 6)
	return out
