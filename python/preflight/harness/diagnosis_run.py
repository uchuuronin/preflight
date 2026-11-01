"""One Spark diagnosis of a stored record and Spark Open's check of it, outside the pipeline: the
real check before harness.spark_diagnosis goes near production (Chiya, Nov 1). The record isn't
changed; what comes back is the findings, every call and the spend.
"""

from preflight.harness.diagnosis import diagnose_record
from preflight.harness.diagnosis_check import check_diagnosis
from preflight.schema import FlightRecord

CHECK_RESERVE_USD = 0.15  # [derived] Spark Open's check at its worst: 6 claims x 3 rounds at about $0.007


def run_once(rec: FlightRecord, spark, spark_open, max_usd: float = 0.5) -> dict:
	"""Muse Spark's diagnosis, then Spark Open's check unless the diagnosis left less than the
	check's worst case under max_usd (or there's no Spark Open client)."""
	found, calls, stats = diagnose_record(spark, rec)
	spent = sum(c.cost_usd for c in calls)
	out: dict = {
		"record_id": rec.record_id,
		"stats": stats,
		"diagnosis": [f.model_dump(mode="json") for f in found],
	}
	if found and spark_open is not None and spent + CHECK_RESERVE_USD <= max_usd:
		checked, checks = check_diagnosis(spark_open, rec, found)
		calls = [*calls, *checks]
		out["checked"] = [f.model_dump(mode="json") for f in checked]
	else:
		out["checked"] = None
		out["check_skipped"] = (
			"no findings"
			if not found
			else "no Spark Open client"
			if spark_open is None
			else "not enough of the cap left"
		)
	out["calls"] = [c.model_dump(mode="json") for c in calls]
	out["usd"] = round(sum(c.cost_usd for c in calls), 6)
	return out
