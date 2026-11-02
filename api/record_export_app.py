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
	if not ULID.fullmatch(record_id) or not NAME.fullmatch(name):
		raise SystemExit("record id must be a ULID and name lowercase letters, digits and dashes")
	vol = modal.Volume.from_name("preflight-records", environment_name="main")
	found = None
	tokens: list[str] = []
	scanned = 0
	for entry in vol.listdir("/"):
		path = entry.path.lstrip("/")
		if "/" in path or not path.endswith(".json") or path.startswith(".tmp-"):
			continue
		tokens.append(path[: -len(".json")])
	for token in tokens:
		if found is not None:
			break
		scanned += 1
		try:
			raw = b"".join(vol.read_file(f"{token}.json"))
		except Exception as e:  # never print the error: its message can name the file, a token
			print(f"skipped one record ({type(e).__name__})", flush=True)
			continue
		if f'"{record_id}"'.encode() not in raw:
			continue
		rec = json.loads(raw)
		if rec.get("record_id") == record_id:
			found = (rec, token)
	print(f"scanned {scanned} of {len(tokens)} records", flush=True)
	if found is None:
		raise SystemExit("no record with that id")
	rec, token = found
	src = (rec.get("job") or {}).get("source") or {}
	ref = src.get("record_ref")
	if src.get("kind") == "record" and not (isinstance(ref, str) and ref.startswith("sample:")):
		raise SystemExit("this record continues another record by its link; not exported")
	text = json.dumps(rec, indent=1).replace(token, f"{name}-share-token-00")
	# Fail closed: no other record's token may appear anywhere in it.
	if any(t in text for t in tokens if t != token and len(t) >= 8):
		raise SystemExit("another record's link appears in this record; not exported")
	OUT.mkdir(exist_ok=True)
	(OUT / f"{name}.record.json").write_text(text + "\n")
	print(f"wrote out/{name}.record.json ({len(text)} bytes), share token replaced", flush=True)
