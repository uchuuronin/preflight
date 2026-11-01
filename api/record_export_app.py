"""Copy one record from the production records volume into the repo's run outputs, its share
token stripped, so other agents can read it as a file (B's before and after on the live LoRA
record, Nov 1). Reads only: nothing is written to the volume, no function runs remotely, no GPU.

The volume names each record `<share_token>.json`, so the record is found by its record id; no
file name, share token or parent token is ever printed (the log goes to the remote-runs branch).

	modal run --env main api/record_export_app.py::export --record-id 01M6C75GZ9BR4250GCX0BYY5X8 --name lora-live-nov1
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import modal

app = modal.App("preflight-record-export")
ULID = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")
NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,40}$")
OUT = Path("out")


@app.local_entrypoint()
def export(record_id: str, name: str) -> None:
	if not ULID.match(record_id) or not NAME.match(name):
		raise SystemExit("record id must be a ULID and name lowercase letters, digits and dashes")
	vol = modal.Volume.from_name("preflight-records", environment_name="main")
	found = None
	scanned = 0
	for entry in vol.listdir("/"):
		path = entry.path.lstrip("/")
		if "/" in path or not path.endswith(".json") or path.startswith(".tmp-"):
			continue
		scanned += 1
		raw = b"".join(vol.read_file(path))
		if f'"{record_id}"'.encode() not in raw:
			continue
		rec = json.loads(raw)
		if rec.get("record_id") == record_id:
			found = (rec, path[: -len(".json")])
			break
	print(f"scanned {scanned} records", flush=True)
	if found is None:
		raise SystemExit("no record with that id")
	rec, token = found
	text = json.dumps(rec, indent=1)
	# The record's own token, and any other link it carries, become placeholders.
	text = text.replace(token, f"{name}-share-token-00")
	rec = json.loads(text)
	for key in ("parent_share_token",):
		rec.pop(key, None)
	OUT.mkdir(exist_ok=True)
	(OUT / f"{name}.record.json").write_text(json.dumps(rec, indent=1) + "\n")
	print(f"wrote out/{name}.record.json ({len(text)} bytes), share token replaced", flush=True)
