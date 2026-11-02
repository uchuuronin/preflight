"""Copy one stored flight record out of A's records volume as a repo fixture, by its record id,
with its share token replaced (PM, Nov 1: refresh PF102 from the new live LoRA record). Reads
only; writes nothing to the volume, starts no GPU, calls no model.

Records are kept as `<share token>.json` (api/preflight_api/stores.py), so the file is found by
the `record_id` inside it and neither its name nor its token is ever printed: the log and the
output go to the remote-runs branch. The output fails closed if the token appears anywhere else
in the record.

	modal run --env main api/record_export_app.py::export --record-id 01M6D5N3VSJZ4EFEHW7MDDK6Z3 --placeholder lora-live-nov1-share-token-00

The record lands in `out/record.json` in the run's folder.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import modal

app = modal.App("preflight-record-export")
RECORDS_DIR = "/records"
records_vol = modal.Volume.from_name("preflight-records", environment_name="main")
ULID = re.compile(r"^[0-9A-HJKMNP-TV-Z]{26}$")
PLACEHOLDER = re.compile(r"^[a-z0-9-]{8,64}$")


def strip_token(record: dict, placeholder: str) -> str:
	"""The record as JSON with its share token replaced; raises if the token is anywhere else."""
	token = record.get("share_token")
	if not isinstance(token, str) or len(token) < 8:
		raise ValueError("record has no share token")
	out = dict(record, share_token=placeholder)
	text = json.dumps(out, indent=1, ensure_ascii=False) + "\n"  # as the fixture files are
	if token in text:
		raise ValueError("the share token appears elsewhere in the record")
	return text


def _read_only(vol):
	try:  # the SDK's current spelling, as runner/modal_app.py does
		return vol.with_mount_options(read_only=True)
	except AttributeError:  # older SDKs
		return vol.read_only()


@app.function(volumes={RECORDS_DIR: _read_only(records_vol)}, timeout=300)
def find(record_id: str) -> dict | None:
	for path in Path(RECORDS_DIR).glob("*.json"):
		try:
			rec = json.loads(path.read_text())
		except (OSError, ValueError):
			continue
		if isinstance(rec, dict) and rec.get("record_id") == record_id:
			return rec
	return None


@app.local_entrypoint()
def export(record_id: str, placeholder: str) -> None:
	if not ULID.match(record_id):
		raise SystemExit("record id: a 26-character ULID")
	if not PLACEHOLDER.match(placeholder):
		raise SystemExit("placeholder: lower-case letters, digits and hyphens")
	rec = find.remote(record_id)
	if rec is None:
		raise SystemExit("no record with that id")
	text = strip_token(rec, placeholder)
	Path("out").mkdir(exist_ok=True)
	Path("out/record.json").write_text(text)
	grids = (rec.get("predictions") or {}).get("grids") or []
	print(
		json.dumps(
			{
				"record_id": record_id,
				"created_at": rec.get("created_at"),
				"bytes": len(text.encode()),
				"grids": len(grids),
				"clears_grids": sum(1 for g in grids if str(g.get("id", "")).endswith(".clears")),
				"probes": [[p.get("id"), p.get("status")] for p in rec.get("probes") or []],
			},
			indent=1,
		),
		flush=True,
	)
