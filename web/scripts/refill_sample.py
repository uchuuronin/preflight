"""Refill a measured sample's predictions on today's engine, with production's flags (B, Nov 3: the
Pretrain and GNN samples after the held-anchor fix and the GNN's epochs fix). The probes, map and
everything measured stay as they are; only the predictions (grids, estimates, models) are B's
engine's again. Share tokens never go in the repo.

	.venv/bin/python web/scripts/refill_sample.py web/src/workspace/fixtures/gnn_pyg-a10-nov2.record.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from preflight.engine.api import fill_predictions
from preflight.schema.record import FlightRecord

ROOT = Path(__file__).resolve().parents[2]


def flags():
	prod = json.loads((ROOT / "flags.production.json").read_text())
	base = json.loads((ROOT / "flags.json").read_text())

	def flag(name: str) -> bool:
		v = prod.get(name, base.get(name))
		if isinstance(v, dict):
			v = v.get("on", v.get("default"))
		return bool(v)

	return flag


def main(path: str) -> None:
	raw = json.loads(Path(path).read_text())
	rec = fill_predictions(FlightRecord.model_validate(raw), flags=flags())
	out = rec.model_dump(mode="json", exclude_none=True)
	# the record's own identity stays; the model's default would make a new share token
	for k in ("record_id", "created_at", "job"):
		if k in raw:
			out[k] = raw[k]
	out.pop("share_token", None)
	with open(path, "w") as f:
		json.dump(out, f, indent="\t", ensure_ascii=False)
		f.write("\n")
	print(f"refilled {path}: {len(out['predictions'].get('grids', []))} grids")


if __name__ == "__main__":
	for p in sys.argv[1:]:
		main(p)
