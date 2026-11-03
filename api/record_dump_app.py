"""One production record, by its record id, read back from the records volume for a sample refill
(PM, Nov 3: the LoRA sample from A's fresh check-in), printed between two marker lines without its
share token, so the log never holds a capability. Records are found by their record id, scanned on a
Modal worker with the volume mounted.

    modal run --env main api/record_dump_app.py::dump --rid 01M6HT93WFJWV96WVAMS8FZS9G
"""

from __future__ import annotations

import json

import modal

app = modal.App("preflight-record-dump")
vol = modal.Volume.from_name("preflight-records", environment_name="main")


@app.function(volumes={"/records": vol}, timeout=600)
def find(rid: str) -> dict | None:
	import pathlib

	for path in pathlib.Path("/records").glob("*.json"):
		try:
			rec = json.loads(path.read_text())
		except Exception:  # noqa: S112  a half-written or foreign file: not one of ours
			continue
		if rec.get("record_id") == rid:
			rec.pop("share_token", None)
			return rec
	return None


@app.local_entrypoint()
def dump(rid: str) -> None:
	rec = find.remote(rid)
	print(json.dumps({"found": rec is not None}))
	if rec is None:
		return
	print("---RECORD---")
	print(json.dumps(rec))
	print("---END---")
