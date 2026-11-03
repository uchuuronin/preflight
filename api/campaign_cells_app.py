"""The four measured campaign cells (D17, Nov 2: the LoRA zoo script on an A10, batch 2 and 4 x lr 1e-4
and 2e-4, GitHub run 41493588241), read back from the production records volume so the campaign
sample (PF420) can be rebuilt from them (PM, Nov 3: every sample measured).

Each cell's job made a follow-up record; this prints, per cell, the probes that job planned at the
cell's setting (the ones naming the A10 and the cell's knobs: `campaign_probe_app.measured`'s rule),
as JSON between two marker lines. Records are found by their record id, scanned on a Modal worker
with the volume mounted. Printed: probes only (knobs, timings, memory, step logs, error codes), never a
share token or anything else of the record.

    modal run --env main api/campaign_cells_app.py::cells
"""

from __future__ import annotations

import json

import modal

app = modal.App("preflight-campaign-cells")
vol = modal.Volume.from_name("preflight-records", environment_name="main")

# Source: the run's log (remote-runs, runs/20261102-174157-41493588241-c-campaign-probes-nov2-a10).
CELLS = {
	"01M6ENNPTKQZ4SF26EYJGGHKFD": {"batch_size": 2, "lr": 0.0001},
	"01M6ENNQDWNRZT02YD583B34PR": {"batch_size": 2, "lr": 0.0002},
	"01M6ENNQ83SM005EHXE6PRAQ0M": {"batch_size": 4, "lr": 0.0001},
	"01M6ENNRSJ6X1N4WDWH77WY9KS": {"batch_size": 4, "lr": 0.0002},
}
GPU = "A10"


@app.function(volumes={"/records": vol}, timeout=600)
def scan() -> dict:
	import pathlib

	out: dict[str, list] = {}
	for path in pathlib.Path("/records").glob("*.json"):
		try:
			rec = json.loads(path.read_text())
		except Exception:  # noqa: S112  a half-written or foreign file: not one of ours
			continue
		rid = rec.get("record_id")
		if rid not in CELLS:
			continue
		cell = CELLS[rid]
		probes = [
			p
			for p in rec.get("probes") or []
			if (k := p.get("knobs") or {}).get("gpu_type") == GPU
			and all(k.get(n) == v for n, v in cell.items())
		]
		out[rid] = probes
		if len(out) == len(CELLS):
			break
	return out


@app.local_entrypoint()
def cells() -> None:
	found = scan.remote()
	print(json.dumps({"found": sorted(found), "missing": sorted(set(CELLS) - set(found))}))
	print("---CELLS---")
	print(json.dumps({rid: {"cell": CELLS[rid], "probes": ps} for rid, ps in found.items()}))
	print("---END---")
