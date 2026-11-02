"""Campaign grid probes (PM, Nov 2 item 8): one short probe job per cell of a grid, through the
production job store, so the campaign calendar can use measured numbers where they exist.

The validation grid: 2 batch sizes x 2 learning rates on the LoRA zoo script (`sample:lora-qwen`,
zoo/finetune_lora_hf, the Nov 2 live record measured on an L4), each cell probed on --gpu (L4 by
default: the record's own GPU, where the cockpit has predictions to compare; on an A10 it abstains
for every cell, "off grid", so there the probes are the only numbers). A cell is one `mode: probe` job at the cell's full setting: one point probe
(A's pipeline._followup_plan), with its eval and checkpoint pulled into the window as for any probe.

    modal run --env main api/campaign_probe_app.py::grid_probes --gpu L4

Spend: every cell is submitted at once, each under A's per-job cap (US$0.60), so the worst case is
the cells times that cap; it must be under the cap here before anything is submitted. A job still
running when the script stops is cancelled. Each cell has its own visitor (the job store runs one
job per visitor at a time, 3 a day). What's printed: per cell, the probe's status, its measured
step time and training peak, the job's record id and cost. Nothing that opens a record is printed.
The prediction each cell is compared with is the web's (web/scripts/campaign_predicted.test.ts), made
from the Nov 2 record before these probes ran.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent
app = modal.App("preflight-campaign-probes")

# Source: the PM's order, Nov 2 11:16 (item 8: "L4 or A10, cap $3").
CAP_USD = 3.0
POLL_S = 15
# Source: guess, as verify_race_app.py: an A10 schedules in seconds; past this the job is cancelled.
JOB_TIMEOUT_S = 40 * 60
# Source: the PM's order (2 batch sizes x 2 learning rates); 2 and 4 are the record's own probed batch
# sizes on the L4, 1e-4 its learning rate and 2e-4 the next the LoRA knob offers.
GRID = [{"batch_size": b, "lr": lr} for b in (2, 4) for lr in (0.0001, 0.0002)]


def cell_id(cell: dict) -> str:
	return f"bs{cell['batch_size']}-lr{cell['lr']:g}"


def measured(rec: dict, cell: dict, gpu: str) -> dict:
	"""The cell's probe, as measured: the newest probe the job planned at the cell's setting. A
	follow-up record also carries the probes it was made from (the sample's own, on the L4); those
	name neither `gpu_type` nor `lr` in their knobs, while a probe planned from a full setting
	(probes/plan.py point_probe) names every knob it was given, so a probe counts only when it names
	the cell's GPU and every knob the cell sets."""

	def at_cell(p: dict) -> bool:
		k = p.get("knobs") or {}
		return "gpu_type" in k and k["gpu_type"] == gpu and all(n in k and k[n] == v for n, v in cell.items())

	probes = [p for p in rec.get("probes") or [] if at_cell(p)]
	p = probes[-1] if probes else None
	if p is None:
		return {"probe": None}
	mem = p.get("memory") or {}
	st = p.get("step_time") or {}
	extras = p.get("extras") or {}
	train_peak = extras.get("train_peak_before_eval_bytes") or mem.get("peak_allocated_bytes")
	return {
		"at_cell": [(q.get("id"), q.get("status")) for q in probes],  # an eval-off re-run follows an eval OOM
		"probe": p.get("id"),
		"status": p.get("status"),
		"steps": p.get("steps"),
		"step_s_mean": st.get("mean_s"),
		"step_s_p50": st.get("p50_s"),
		"train_peak_bytes": train_peak,
		"peak_allocated_bytes": mem.get("peak_allocated_bytes"),
		"knobs": {k: (p.get("knobs") or {}).get(k) for k in ("batch_size", "lr", "gpu_type")},
	}


@app.local_entrypoint()
def grid_probes(attempt: str = "a", gpu: str = "L4") -> None:
	sys.path.insert(0, str(ROOT / "api"))  # this runs on the caller's machine, from the repo
	import verify_race_app as vr

	if gpu not in ("L4", "A10"):
		raise SystemExit("gpu: L4 or A10")  # the PM's order
	if len(GRID) * vr.JOB_CAP_USD > CAP_USD:
		raise SystemExit(f"worst case US${len(GRID) * vr.JOB_CAP_USD:.2f} is over the cap")
	svc = modal.Cls.from_name("preflight-runner", "JobStoreService", environment_name="main")()
	vol = modal.Volume.from_name("preflight-records", environment_name="main")
	jobs: dict[str, str] = {}
	out: dict[str, dict] = {}
	t0 = time.monotonic()
	try:
		for cell in GRID:
			cid = cell_id(cell)
			req = {
				"source": {"kind": "record", "record_ref": "sample:lora-qwen"},
				"mode": "probe",
				"knobs": {**vr.KNOBS, **cell, "gpu_type": gpu},
				"gpu": gpu,
				"constraints": {"gpu_types": [gpu]},
				"purpose": "c.campaign_probe",
			}
			try:
				job = svc.submit.remote(
					req, f"c.campaign.nov2.{attempt}.{gpu}.{cid}", f"c-campaign-{cid}-{gpu}-nov2-{attempt}"
				)
			except Exception as e:
				out[cid] = {"status": "refused", "why": f"{type(e).__name__}: {str(e)[:300]}"}
				continue
			jobs[cid] = job.get("job_id")
			print(f"{cid}: submitted", flush=True)
		pending = dict(jobs)
		while pending and time.monotonic() - t0 < JOB_TIMEOUT_S:
			for cid, jid in list(pending.items()):
				st = svc.status.remote(jid)
				if st.get("status") in ("done", "failed", "cancelled"):
					pending.pop(cid)
					out[cid] = {"job_status": st.get("status"), "record_id": st.get("record_id"),
						"error_code": st.get("error_code"), "minutes": round((time.monotonic() - t0) / 60, 1),
						"_token": st.get("share_token")}  # fmt: skip
					print(f"{cid}: {st.get('status')}", flush=True)
			if pending:
				time.sleep(POLL_S)
	finally:
		for cid, jid in jobs.items():
			if cid not in out:  # still running: a timeout, an error polling, or the run being stopped
				try:
					svc.cancel.remote(jid)
				except Exception as e:
					print(f"{cid}: cancel failed ({type(e).__name__}); A's job caps still apply", flush=True)
				out[cid] = {"job_status": "timed_out"}
	spent = 0.0
	cells = {cell_id(c): c for c in GRID}
	for cid, o in out.items():
		token = o.pop("_token", None)
		rec = vr._read_record(vol, token) if token else None
		final = rec is not None and o.get("job_status") == "done"
		cost = vr._record_cost(rec) if final else (vr.JOB_CAP_USD if o.get("job_status") else 0.0)
		spent += cost
		o["usd"] = round(cost, 4) if final else f"unknown, counted at {cost}"
		if rec is not None:
			o["measured"] = measured(rec, cells[cid], gpu)
			o["gpu"] = ((rec.get("env") or {}).get("gpu") or {}).get("name")
	print(json.dumps({"spent_usd": round(spent, 4), "cap_usd": CAP_USD, "gpu": gpu, "cells": out}, indent=1))
	raise SystemExit(0 if all((o.get("measured") or {}).get("status") for o in out.values()) else 1)
