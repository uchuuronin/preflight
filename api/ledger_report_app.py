"""Print the spend ledger's totals for the day, split by what spent them (PM, Nov 1: "send the
guide ledger's Muse Spark total so far"). Reads only the snapshot the API's SpendLedger saves to
the Modal Dict `preflight-ledger` after every change (api/app.py); writes nothing, starts no
container of the API's, no GPU, no model call.

The ledger keeps one day (US Eastern) and starts again at midnight, so this is the total since
midnight ET. Run keys (api/preflight_api/routes): `<record key>:<session>` for the guide and the
limits box (both Muse Spark), `jobs:<session>` for check-ins (always $0, a rate gate), and
`fix:ip:<address>` for Muse Code fixes. No key, session, address or record is printed (the log
goes to the remote-runs branch): only sums and counts.

	modal run --env main api/ledger_report_app.py::report
"""

from __future__ import annotations

import json

import modal

app = modal.App("preflight-ledger-report")


def summarise(snap: dict) -> dict:
	day = snap.get("day") or {}
	runs: dict[str, float] = {k: float(v) for k, v in (day.get("run_usd") or {}).items()}
	spark = {k: v for k, v in runs.items() if not k.startswith(("jobs:", "fix:"))}
	fix = {k: v for k, v in runs.items() if k.startswith("fix:")}
	opened = snap.get("open") or []
	open_spark = [r for r in opened if not str(r.get("run", "")).startswith(("jobs:", "fix:"))]
	return {
		"day_et": day.get("day"),
		"settled_usd_all": round(float(day.get("usd") or 0.0), 4),
		"muse_spark_usd": round(sum(spark.values()), 4),
		"muse_spark_sessions_on_records": len(spark),
		"muse_spark_most_on_one_run_usd": round(max(spark.values(), default=0.0), 4),
		"muse_code_fix_usd": round(sum(fix.values()), 4),
		"open_reservations": len(opened),
		"open_muse_spark_worst_case_usd": round(sum(float(r.get("usd") or 0) for r in open_spark), 4),
		"turns_today": sum(int(v) for v in (day.get("turns_visitor") or {}).values()),
		"visitors_today": len(day.get("turns_visitor") or {}),
	}


@app.local_entrypoint()
def report() -> None:
	store = modal.Dict.from_name("preflight-ledger", environment_name="main")
	snap = store.get("snapshot")
	if not snap:
		raise SystemExit("no ledger snapshot")
	print(json.dumps(summarise(snap), indent=1), flush=True)
