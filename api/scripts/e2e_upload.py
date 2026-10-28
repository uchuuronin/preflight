"""End to end on a real GPU: a bug-zoo script checked in as an upload through the real API, run by
A's deployed job store, and read back through the API from the records volume.

What runs where:
- This Modal function (CPU, no secrets) holds the real FastAPI app in process, with
  `runner.queue` on, the job store pointed at A's deployed "preflight-runner" app, and A's
  "preflight-records" volume mounted read-only, exactly as api/app.py mounts it.
- POST /v1/jobs with the zoo script's files as a `kind: upload` source (what the check-in desk
  sends); A's job store runs it on its own workers and GPU sandboxes, under the job's own probe
  budget (A's pipeline). The GPU is A's choice from the script's size, held at or below L4 here.
- It polls GET /v1/jobs/{id}, then reads GET /v1/jobs/{id}/events and
  GET /v1/records/{share_token}, timing each step.

Every run ends in Modal before the workflow kills the job: the function has its own timeout and
startup timeout, the runner cancels the call at a deadline under the workflow's `timeout`
input, and the function cancels the job in A's store (POST /v1/jobs/{id}/cancel) before its own
deadline, so no GPU keeps running for a job nobody is following.

Through GitHub Actions (the API holds no keys here; A's workers hold their own):

	modal run --env main api/scripts/e2e_upload.py --sample finetune_lora_hf

Writes out/e2e.json (timings and checks) and out/e2e_record.json (the record as the API served
it), which web/e2e/record_map.mjs then opens in the cockpit.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import modal

_T0 = time.monotonic()  # on the runner: before the image build, which also eats the workflow's time
# Read only while building the image, on the machine that runs this. Inside the container Modal
# loads this file on its own as /root/e2e_upload.py, where the repo isn't.
_HERE = Path(__file__).resolve()
ROOT = (
	_HERE.parents[2]
	if len(_HERE.parents) > 2 and (_HERE.parents[2] / "flags.json").is_file()
	else Path("/repo-not-here")
)
RECORDS_DIR = "/records"

# Source: guess, A's check-in copy says about 4 minutes on one GPU; a cold start, the model
# download and a queue wait can triple that. The job is cancelled past this.
JOB_DEADLINE_S = 20 * 60
# Source: derived, JOB_DEADLINE_S plus two minutes to cancel, read the events and the record.
FUNCTION_TIMEOUT_S = JOB_DEADLINE_S + 120
# Source: guess, a healthy container start is seconds; two minutes means a crash on import fails
# here, not in a retry loop (the Oct 27 guide-tour lesson).
STARTUP_TIMEOUT_S = 120
# Source: guess, time the workflow needs after the run to publish out/ to the remote-runs branch.
PUBLISH_MARGIN_S = 120
# Source: guess, a stage lasts tens of seconds; a poll every 3 s times each to within 3 s.
POLL_S = 3.0

image = (
	modal.Image.debian_slim(python_version="3.11")
	.pip_install("fastapi>=0.115", "pydantic>=2.8,<3", "httpx>=0.27", "numpy>=1.26")
	.env(
		{
			"PREFLIGHT_ROOT": "/root/repo",
			"PREFLIGHT_FLAGS": json.dumps({"runner.queue": True}),
			"PREFLIGHT_JOBSTORE_APP": "preflight-runner",
			"PREFLIGHT_JOBSTORE_CLS": "JobStoreService",
			"PREFLIGHT_RECORDS_DIR": RECORDS_DIR,
		}
	)
	.add_local_file(ROOT / "flags.json", "/root/repo/flags.json")
	.add_local_dir(ROOT / "python" / "preflight", "/root/repo/python/preflight")
	.add_local_dir(ROOT / "api" / "preflight_api", "/root/repo/api/preflight_api")
	.add_local_dir(ROOT / "zoo", "/root/repo/zoo")
	.add_local_file(_HERE.parent / "e2e_flow.py", "/root/repo/api/scripts/e2e_flow.py")
)
for rel in (
	"web/src/workspace/fixtures/dev-finetune.record.json",
	"schema/examples/finetune.json",
	"schema/examples/pretrain.json",
	"schema/examples/agent.json",
):
	image = image.add_local_file(ROOT / rel, f"/root/repo/{rel}")

app = modal.App("preflight-e2e", image=image)
records_vol = modal.Volume.from_name("preflight-records")  # A's; never created here


def _read_only(vol):
	try:
		return vol.with_mount_options(read_only=True)
	except AttributeError:  # older SDKs
		return vol.read_only()


@app.function(
	volumes={RECORDS_DIR: _read_only(records_vol)},
	timeout=FUNCTION_TIMEOUT_S,
	startup_timeout=STARTUP_TIMEOUT_S,
	retries=0,
	cpu=1.0,
)
async def fly(sample: str, gpu: str, job_deadline_s: float) -> dict:
	sys.path[:0] = ["/root/repo/api/scripts", "/root/repo/api", "/root/repo/python"]
	import httpx
	from e2e_flow import flow

	from preflight_api.app import create_app
	from preflight_api.ledger import MemoryLedger
	from preflight_api.stores import LocalRecordStore

	files = {
		p.name: p.read_text() for p in sorted((Path("/root/repo/zoo") / sample).iterdir()) if p.is_file()
	}
	api = create_app(
		records=LocalRecordStore(Path(RECORDS_DIR), reload=records_vol.reload), ledger=MemoryLedger()
	)
	async with httpx.AsyncClient(
		transport=httpx.ASGITransport(app=api), base_url="https://testserver", timeout=60
	) as c:
		out = await flow(
			c,
			files,
			entrypoint="train.py",
			gpu=gpu,
			job_deadline_s=job_deadline_s,
			poll_s=POLL_S,
			label=f"zoo/{sample}",
		)
	out["sample"] = sample
	try:
		from preflight.schema import load_record

		if out.get("record"):
			load_record(out["record"])
		out["record_loads_on_this_branch"] = bool(out.get("record"))
	except Exception as e:  # A's schema may be newer than this branch's copy
		out["record_loads_on_this_branch"] = f"{type(e).__name__}: {str(e)[:300]}"
	return out


@app.local_entrypoint()
def main(sample: str = "finetune_lora_hf", gpu: str = "L4") -> None:
	budget = int(os.environ.get("RUN_TIMEOUT") or 30) * 60  # the workflow's limit, in seconds
	deadline = min(
		FUNCTION_TIMEOUT_S + STARTUP_TIMEOUT_S, budget - PUBLISH_MARGIN_S - (time.monotonic() - _T0)
	)
	job_deadline = min(JOB_DEADLINE_S, deadline - STARTUP_TIMEOUT_S - 120)
	if job_deadline < 5 * 60:
		raise SystemExit(f"e2e: {budget // 60} min isn't enough for a probe job; ask for more")
	call = fly.spawn(sample, gpu, job_deadline)
	try:
		res = call.get(timeout=deadline)
	except (TimeoutError, modal.exception.TimeoutError):
		call.cancel()
		raise SystemExit(f"e2e: no result after {deadline:.0f} s; cancelled in Modal") from None
	Path("out").mkdir(exist_ok=True)
	record = res.pop("record", None)
	if record is not None:
		Path("out/e2e_record.json").write_text(json.dumps(record, indent="\t") + "\n")
	Path("out/e2e.json").write_text(json.dumps(res, indent="\t", default=str) + "\n")
	print(
		json.dumps(
			{
				k: res.get(k)
				for k in ("submit", "marks", "checks", "record_read_s", "record_loads_on_this_branch")
			},
			indent=2,
			default=str,
		)
	)
	sys.path.insert(0, str(_HERE.parent))
	from e2e_flow import passed

	if not passed(res.get("checks") or {}):
		sys.exit(1)
