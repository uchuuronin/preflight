"""The campaign sample (PF420), from measured cells (PM, Nov 3: every sample measured).

The LoRA zoo script's live record (PF102, measured on an L4) plus the four campaign cells measured on
an A10 on Nov 2 (C's D17: batch 2 and 4 x lr 1e-4 and 2e-4, GitHub run 41493588241; their probes read
back from the records volume by `api/campaign_cells_app.py`, run 41657481934, printed between two
marker lines in its log). The cells' probes join the record, B's engine refills its predictions (so a
cell's numbers are its probe's where one ran), and B's planner plans the grid on the A10 for a small
team. Every number is a measurement or B's.

	git show origin/remote-runs:runs/20261103-144930-41657481934-c-campaign-cells/log.txt \\
		| sed -n '/---CELLS---/,/---END---/p' | sed '1d;$d' > /tmp/cells.json
	.venv/bin/python web/scripts/make_campaign_sample.py /tmp/cells.json \\
		web/src/workspace/fixtures/lora-live-nov1.record.json web/src/workspace/fixtures/campaign-grid.record.json
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from preflight.campaign import planner
from preflight.engine.api import fill_predictions
from preflight.schema.record import FlightRecord

ROOT = Path(__file__).resolve().parents[2]
# The measured grid: the cells probed on the A10.
GRID = {"batch_size": [2, 4], "lr": [0.0001, 0.0002], "gpu_type": ["A10"]}
# Three people's lanes on a shared cluster, a 4 h job limit (Slurm --time) and an interruption about
# once a day: an illustrative lab's setup (guesses), the same as the old sample's.
CLUSTER = planner.Cluster(lanes={"chiya": 2, "ana": 1, "ben": 1}, wall_time_cap_s=4 * 3600, mtti_s=24 * 3600)
# A Monday after judging opens, so the dates read as ahead of a visitor on Nov 4 to 8.
START = datetime(2026, 11, 9, 9, 0, tzinfo=UTC)


def flags() -> callable:
	prod = json.loads((ROOT / "flags.production.json").read_text())
	base = json.loads((ROOT / "flags.json").read_text())

	def flag(name: str) -> bool:
		v = prod.get(name, base.get(name))
		if isinstance(v, dict):
			v = v.get("on", v.get("default"))
		return bool(v)

	return flag


def main(cells_path: str, base_path: str, out_path: str) -> None:
	cells = json.loads(Path(cells_path).read_text())
	raw = json.loads(Path(base_path).read_text())
	ids = {p["id"] for p in raw.get("probes") or []}
	for cell in cells.values():
		c = cell["cell"]
		tag = f"a10-bs{c['batch_size']}-lr{c['lr']:g}"
		for p in cell["probes"]:
			p = dict(p)
			p["id"] = f"{tag}-eval" if p["id"].endswith("-eval") else tag
			assert p["id"] not in ids, p["id"]
			ids.add(p["id"])
			raw["probes"].append(p)
	rec = fill_predictions(FlightRecord.model_validate(raw), flags=flags())
	plan = planner.plan(rec, GRID, CLUSTER, start=START)
	out = rec.model_dump(mode="json", exclude_none=True)
	out["campaign"] = plan.to_campaign().model_dump(mode="json", exclude_none=True)
	card = plan.card.model_dump(mode="json", exclude_none=True)
	out["models"] = [m for m in out.get("models", []) if m.get("id") != card["id"]] + [card]
	out["record_id"] = "01M6ENNQ0CAMPA10GR1D000000"
	out["job"]["title"] = "LoRA research grid: batch size and learning rate, measured on an A10"
	out.pop("share_token", None)  # share tokens never go in the repo (the model's default makes one)
	with open(out_path, "w") as f:
		json.dump(out, f, indent="\t", ensure_ascii=False)
		f.write("\n")
	print(
		f"wrote {out_path}: {len(plan.cells)} runs, {len(plan.cut_list)} cut, finish {plan.finish_s.status}"
	)


if __name__ == "__main__":
	main(*sys.argv[1:4])
