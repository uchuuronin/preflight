"""Where a zoo script's memory peak happens (Chiya, Nov 2: gnn's and retrieval's full-run peaks are invisible to the probes).

	modal run --env main python/preflight/corpus/memtrace_app.py::trace --scripts gnn_pyg,retrieval_st

Each script runs to its end on an A10 (or an L4), unchanged, with a sampler beside it: a sitecustomize that, once the
script has touched CUDA, reads the allocator's allocated and reserved bytes every 0.1 s, and the line of the script's
own code the main thread is on at that moment. So the timeline names the phase each peak falls in (a training step,
an eval, an encode pass, a `.to(device)`). Writes out/memtrace.json: per script the peak's time and line, the
largest allocated and reserved per phase (the script's function and line), and the timeline thinned to 2,000 points.
Cap $1 (A10 at $1.10/h; gnn about a minute, retrieval about 8).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

_HERE = Path(__file__).resolve()
ROOT = _HERE.parents[3] if len(_HERE.parents) > 3 else Path("/root")

SITECUSTOMIZE = r"""
import atexit, json, os, sys, threading, time
_OUT = os.environ["MEMTRACE_OUT"]
_SCRIPT = os.environ["MEMTRACE_SCRIPT"]
_samples = []
_main = threading.main_thread()
# a child Python inherits PYTHONPATH and this environment: only the first process, the script's own, writes the file
_OWNER = os.environ.setdefault("MEMTRACE_OWNER", str(os.getpid())) == str(os.getpid())

def _where():
	f = sys._current_frames().get(_main.ident)
	while f is not None:
		if f.f_code.co_filename.endswith(_SCRIPT):
			return f"{f.f_code.co_name}:{f.f_lineno}"
		f = f.f_back
	return None

def _run():
	t0 = time.monotonic()
	while True:
		time.sleep(0.1)
		torch = sys.modules.get("torch")
		try:
			if torch is None or not torch.cuda.is_initialized():
				continue
			# the running max too: a fast step frees its activations between two 0.1 s reads
			_samples.append((round(time.monotonic() - t0, 2), torch.cuda.memory_allocated(),
				torch.cuda.memory_reserved(), _where(), torch.cuda.max_memory_allocated()))
		except Exception:
			continue
		if len(_samples) % 100 == 0:
			_dump()  # every 10 s, so a run killed at its time limit keeps what it had

def _dump():
	if not _OWNER:
		return
	tmp = _OUT + ".tmp"
	with open(tmp, "w") as f:
		json.dump(list(_samples), f)
	os.replace(tmp, _OUT)

if _OWNER:
	threading.Thread(target=_run, daemon=True).start()
	atexit.register(_dump)
"""

image = (
	modal.Image.debian_slim(python_version="3.11")
	.pip_install(
		"torch==2.5.1",
		"numpy<2.3",
		"datasets==3.1.0",
		"transformers==4.46.3",
		"torch-geometric==2.6.1",
		"sentence-transformers==3.3.1",
		"accelerate==1.1.1",
	)
	.add_local_dir(str(ROOT / "zoo"), "/root/zoo")
)

app = modal.App("preflight-memtrace")
TIMEOUT_S = 25 * 60


def _summary(samples: list, wall_s: float) -> dict:
	if not samples:
		return {"samples": 0}
	peak_a = max(samples, key=lambda s: s[1])
	peak_r = max(samples, key=lambda s: s[2])
	# the tick where the allocator's running max reached its final value: the peak fell in the 0.1 s before it
	top_max = samples[-1][4]
	peak_m = next(s for s in samples if s[4] == top_max)
	phases: dict = {}
	for t, a, r, where, _m in samples:
		p = phases.setdefault(
			where or "outside the script", {"first_s": t, "last_s": t, "alloc": 0, "reserved": 0, "n": 0}
		)
		p["last_s"], p["n"] = t, p["n"] + 1
		p["alloc"], p["reserved"] = max(p["alloc"], a), max(p["reserved"], r)
	top = sorted(phases.items(), key=lambda kv: -kv[1]["reserved"])[:15]
	step = max(1, len(samples) // 2000)
	return {
		"samples": len(samples),
		"wall_s": wall_s,
		"peak_allocated": {"t_s": peak_a[0], "bytes": peak_a[1], "where": peak_a[3]},
		"peak_reserved": {"t_s": peak_r[0], "bytes": peak_r[2], "where": peak_r[3]},
		"max_allocated": {
			"t_s": peak_m[0],
			"bytes": peak_m[4],
			"where": peak_m[3],
			"note": "peak within 0.1 s before",
		},
		"phases": [{"where": k, **v} for k, v in top],
		"timeline": samples[::step],
	}


@app.function(image=image, gpu=["A10", "L4"], timeout=TIMEOUT_S + 120)
def trace_one(script: str) -> dict:
	import os
	import shutil
	import subprocess
	import sys

	import torch

	src = Path("/root/zoo") / script
	work = Path("/tmp/work") / script
	shutil.copytree(src, work)
	entry = next(n for n in ("train.py", "run.py", "serve.py", "main.py") if (work / n).exists())
	site = Path("/tmp/site")
	site.mkdir(exist_ok=True)
	(site / "sitecustomize.py").write_text(SITECUSTOMIZE)
	out = Path("/tmp") / f"{script}.memtrace.json"
	env = {**os.environ, "PYTHONPATH": str(site), "MEMTRACE_OUT": str(out), "MEMTRACE_SCRIPT": f"/{entry}"}
	t0 = time.monotonic()
	try:
		p = subprocess.run(
			[sys.executable, entry], cwd=work, env=env, capture_output=True, text=True, timeout=TIMEOUT_S
		)
		code, so, se = p.returncode, p.stdout, p.stderr
	except subprocess.TimeoutExpired as e:  # the sampler wrote every 10 s: keep what it had
		code, so, se = "timeout", _text(e.stdout), _text(e.stderr)
	wall = round(time.monotonic() - t0, 1)
	samples = json.loads(out.read_text()) if out.exists() else []
	return {"script": script, "device": torch.cuda.get_device_name(0), "exit": code,
		"stdout_tail": so[-2500:], "stderr_tail": se[-1500:], **_summary(samples, wall)}  # fmt: skip


def _text(b) -> str:
	return b.decode(errors="replace") if isinstance(b, bytes) else (b or "")


@app.local_entrypoint()
def trace(scripts: str = "gnn_pyg,retrieval_st"):
	names = [s for s in scripts.split(",") if s]
	got = [
		g if isinstance(g, dict) else {"error": repr(g)} for g in trace_one.map(names, return_exceptions=True)
	]
	Path("out").mkdir(exist_ok=True)
	Path("out/memtrace.json").write_text(json.dumps(got, indent=1))
	for g in got:
		print(
			json.dumps(
				{k: v for k, v in g.items() if k not in ("timeline", "stdout_tail", "stderr_tail")}, indent=1
			)
		)
