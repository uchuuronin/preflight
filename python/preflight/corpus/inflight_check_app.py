"""In-flight mode, the real test (PM, Nov 2, item 4): `preflight watch` alongside a real zoo run on an L4 (or an A10).

	modal run --env main python/preflight/corpus/inflight_check_app.py::check

The pretrain sample (schema/examples/pretrain.json, nanoGPT 124M) runs for real under the collector
(recorder/inflight.py, what `preflight watch` runs), twice, a few minutes each, then is stopped with SIGTERM the way
Slurm preempts a job:

- A, on plan: the record re-predicted for the GPU it runs on. Its bands should hold; any alert here is a false alarm
  (or a miss of the engine's, reported as it is).
- B, off plan: the record as published, predicted for an A100-SXM-40GB, and the script edited to batch 16 where the
  record says 12. Slower than the A100 band (step time, two windows in a row), more memory than predicted (peak, at
  once), and a total time past its band: all three alerts should fire.

For each run it reports the alerts with their window, how many windows the live jsonl holds, how often the live
record was rewritten and whether its last copy parses as a FlightRecord with the `live` probe. Writes
out/inflight_check.json. Cap $1 (L4 at $0.80/h; two 4-minute runs plus start-up).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

# review, Nov 2: in the container this file sits at /root/<name>.py, with no repo above it; the local paths below are
# only read where the app is launched
_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[3] if len(_HERE.parents) > 3 else Path("/root")
RUN_S = 240  # guess: long enough for several 10-step windows at about a second a step
WINDOW = 10  # steps per window: the collector's default is 20; 10 gives more windows in 4 minutes
BATCH_B = (
	16  # the record says 12; 16 needs about a third more activation memory and still fits an L4 [estimate]
)

app = modal.App("preflight-inflight-check")


def _warm():
	"""TinyStories and the GPT-2 tokenizer into the image, so the GPU isn't billed for the download."""
	from datasets import load_dataset
	from transformers import AutoTokenizer

	load_dataset("roneneldan/TinyStories", split="train")
	AutoTokenizer.from_pretrained("gpt2")


image = (
	modal.Image.debian_slim(python_version="3.11")
	.pip_install(
		"torch==2.5.1",
		"numpy<2.3",
		"pydantic>=2.8,<3",
		"httpx>=0.27",
		"datasets==3.1.0",
		"transformers==4.46.3",
	)
	.run_function(_warm)
	.add_local_file(str(ROOT / "schema/examples/pretrain.json"), "/root/pretrain.json")
	.add_local_file(str(ROOT / "zoo/pretrain_nanogpt/train.py"), "/root/nanogpt/train.py")
	.add_local_python_source("preflight")
)


def _gpu_name(device: str) -> str:
	return "L4" if "L4" in device else "A10" if "A10" in device else device


def _last_step(out: Path) -> int:
	"""The step of the live jsonl's last window, 0 before the first."""
	for f in out.glob("*.live.jsonl"):
		lines = f.read_text().splitlines()
		if lines:
			try:
				return int(json.loads(lines[-1])["step"])
			except (ValueError, KeyError):
				return 0
	return 0


def _run(
	parent: Path,
	script: Path,
	out: Path,
	run_s: float = RUN_S,
	total_steps: int = 60000,
	stop_step: int | None = None,
) -> dict:
	import signal
	import subprocess
	import sys

	cmd = [sys.executable, "-m", "preflight.recorder.inflight", str(parent), str(out), str(script),
		"--total-steps", str(total_steps), "--window", str(WINDOW)]  # fmt: skip
	t0 = time.monotonic()
	import os

	env = {
		**os.environ,
		"PYTHONPATH": "/root",
	}  # the package sits in /root; the script's own directory is cwd
	p = subprocess.Popen(
		cmd, cwd=script.parent, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
	)
	rewrites, last_mtime = 0, None
	while time.monotonic() - t0 < run_s and p.poll() is None:
		time.sleep(1)
		for f in out.glob("*.json"):
			m = f.stat().st_mtime
			if m != last_mtime:
				rewrites, last_mtime = rewrites + 1, m
		if stop_step is not None and _last_step(out) >= stop_step:
			break  # review, Nov 2: a fast load could reach the planted eval OOM inside the time limit
	if p.poll() is None:
		p.send_signal(signal.SIGTERM)
	try:
		_so, se = p.communicate(timeout=120)
	except subprocess.TimeoutExpired:
		p.kill()
		_so, se = p.communicate()
	from preflight.schema import load_record

	recs = sorted(out.glob("*.json"))
	logs = sorted(out.glob("*.live.jsonl"))
	windows = [json.loads(x) for x in logs[0].read_text().splitlines()] if logs else []
	row: dict = {
		"exit_code": p.returncode,
		"wall_s": round(time.monotonic() - t0, 1),
		"alerts": [x for x in se.splitlines() if x.startswith("[preflight]")],
		"windows": len(windows),
		"window_alerts": [{"step": w["step"], "alerts": w["alerts"]} for w in windows if w.get("alerts")],
		"step_s": [round(w["step_s"], 4) for w in windows],
		"peak_reserved_gb": [round(w["peak_bytes"] / 1e9, 2) for w in windows if "peak_bytes" in w],
		"record_rewrites": rewrites,
		"stderr_tail": se[-1500:],
	}
	try:
		rec = load_record(recs[0].read_text())
		live = next(p for p in rec.probes if p.id == "live")
		row["live_record"] = {"parses": True, "source_kind": rec.job.source.kind, "parent_id": rec.parent_id,
			"status": live.status, "steps": live.steps}  # fmt: skip
	except Exception as exc:  # reported, not raised: the run's other numbers still stand
		row["live_record"] = {"parses": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
	return row


@app.function(image=image, gpu=["L4", "A10"], timeout=30 * 60)
def check_remote() -> dict:
	import torch

	from preflight.engine import api
	from preflight.recorder.inflight import bands_from
	from preflight.schema import load_record

	device = torch.cuda.get_device_name(0)
	gpu = _gpu_name(device)
	published = load_record(Path("/root/pretrain.json").read_text())
	knobs = [k.model_copy(update={"current": gpu}) if k.id == "gpu_type" else k for k in published.knobs]
	on_plan = api.fill_predictions(published.model_copy(update={"knobs": knobs}))
	a_parent, b_parent = Path("/tmp/on_plan.json"), Path("/tmp/published.json")
	a_parent.write_text(on_plan.model_dump_json())
	b_parent.write_text(published.model_dump_json())
	src = Path("/root/nanogpt/train.py").read_text()
	Path("/tmp/a").mkdir()
	Path("/tmp/b").mkdir()
	Path("/tmp/a/train.py").write_text(src)
	Path("/tmp/b/train.py").write_text(src.replace("batch_size = 12\n", f"batch_size = {BATCH_B}\n", 1))
	out = {"device": device, "gpu": gpu}
	for name, parent, script in (
		("A_on_plan", a_parent, Path("/tmp/a/train.py")),
		("B_off_plan", b_parent, Path("/tmp/b/train.py")),
	):
		rec = load_record(parent.read_text())
		b = bands_from(rec, 60000)
		out[name] = {
			"predicted_for": rec.knob("gpu_type").current,
			"bands": {
				"step_s": b.step_s,
				"peak_gb": b.peak_bytes and [round(x / 1e9, 2) for x in b.peak_bytes],
				"total_h": b.total_s and [round(x / 3600, 2) for x in b.total_s],
			},
			**_run(parent, script, Path(f"/tmp/out_{name}")),
		}
	return out


@app.local_entrypoint()
def check():
	t0 = time.monotonic()
	got = check_remote.remote()
	got["wall_s"] = round(time.monotonic() - t0, 1)
	Path("out").mkdir(exist_ok=True)
	Path("out/inflight_check.json").write_text(json.dumps(got, indent=1))
	print(
		json.dumps(
			{
				k: (
					{kk: vv for kk, vv in v.items() if kk not in ("step_s", "stderr_tail")}
					if isinstance(v, dict)
					else v
				)
				for k, v in got.items()
			},
			indent=1,
		)
	)


# --- The cleaner check (Nov 2, after run 9): a record whose probes ran on the same GPU -------------------------------
#
# 	modal run --env main python/preflight/corpus/inflight_check_app.py::check_a10
#
# The pretrain sample's bands came from hand-made probes, so all three alerts fired on plan. This one's parent is a
# production record whose probes ran on an A10 (A's a-triage-lora-prod-2, the zoo LoRA sample, committed as
# api/tests/fixtures/a-triage-lora-prod-2.record.json): its step band (1.25 to 1.57 s) and total band (72.6 to 82.9
# min, 3,170 steps) come from the engine on those probes. The same script runs on an A10 under the collector, as the
# record says, and is stopped before its first eval (step 200), where it is planted to run out of memory. On plan,
# so any alert is a false alarm. Its peak memory band abstains (the eval OOM), so no memory alert can fire; the run's
# peak is reported beside the probe's. Cap $1 (A10 at $1.10/h, about 8 minutes).

LORA_RUN_S = 270  # guess: about a minute of loading, then about 145 steps at 1.45 s
LORA_STOP_STEP = 180  # and stopped here at the latest: the eval at step 200 is planted to run out of memory

LORA_FIXTURE = "api/tests/fixtures/a-triage-lora-prod-2.record.json"


def _warm_lora():
	from datasets import load_dataset
	from huggingface_hub import snapshot_download

	snapshot_download("Qwen/Qwen2.5-1.5B")
	load_dataset("yahma/alpaca-cleaned", split="train")


lora_image = (
	modal.Image.debian_slim(python_version="3.11")
	.pip_install(
		"torch==2.5.1",
		"numpy<2.3",
		"pydantic>=2.8,<3",
		"httpx>=0.27",
		"datasets==3.1.0",
		"transformers==4.46.3",
		"peft==0.13.2",
		"accelerate==1.1.1",
	)
	.run_function(_warm_lora)
	.add_local_file(str(ROOT / LORA_FIXTURE), "/root/lora_parent.json")
	.add_local_file(str(ROOT / "zoo/finetune_lora_hf/train.py"), "/root/lora/train.py")
	.add_local_python_source("preflight")
)


@app.function(image=lora_image, gpu="A10", timeout=30 * 60)
def check_a10_remote() -> dict:
	import torch

	from preflight.recorder.inflight import bands_from
	from preflight.schema import load_record

	parent = Path("/root/lora_parent.json")
	rec = load_record(parent.read_text())
	b = bands_from(rec)
	probe = next(p for p in rec.probes if p.id == "p1")
	parent_row = {
		"record_id": rec.record_id,
		"gpu": rec.knob("gpu_type").current,
		"probe_p50_s": probe.step_time.p50_s,
		"probe_peak_reserved_gb": round(probe.memory.peak_reserved_bytes / 1e9, 2),
	}
	bands = {
		"step_s": b.step_s,
		"peak_gb": b.peak_bytes,
		"total_min": b.total_s and [round(x / 60, 1) for x in b.total_s],
		"total_steps": b.total_steps,
	}
	run = _run(
		parent,
		Path("/root/lora/train.py"),
		Path("/tmp/out_lora"),
		LORA_RUN_S,
		b.total_steps or 3170,
		LORA_STOP_STEP,
	)
	return {"device": torch.cuda.get_device_name(0), "parent": parent_row, "bands": bands, **run}


@app.local_entrypoint()
def check_a10():
	t0 = time.monotonic()
	got = check_a10_remote.remote()
	got["wall_s"] = round(time.monotonic() - t0, 1)
	Path("out").mkdir(exist_ok=True)
	Path("out/inflight_check_a10.json").write_text(json.dumps(got, indent=1))
	print(json.dumps({k: v for k, v in got.items() if k not in ("stderr_tail",)}, indent=1))
