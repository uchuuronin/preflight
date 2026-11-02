"""The video's in-flight shot (9b) on a real GPU: its own steps on an A10 (D, Nov 2; Chiya, 17:33).

	modal run --env main python/preflight/runner/watch_gpu_app.py::check

On a laptop CPU, `preflight watch` shows a memory range but no step-time range (the probes ran
without a known GPU), so the step alert can't fire on camera. This runs the shot's steps on an A10
(B's in-flight test worked there) to say whether filming on a GPU would show both alerts:

  B. "film": zoo/notebook_tiny/tiny_train.py with max_steps raised; `preflight run`, then
     `preflight watch` on its record with the same script (no change: no alert expected)
  C. "alert": B's record watched against the shot's step 5 change (hidden 256 -> 8192, batch 64 ->
     4096), stopped with SIGTERM after WATCH_S if it hasn't finished

Writes out/watch_gpu.json: the record's GPU and gpu_type knob, its time.step / time.total /
memory.peak estimates or why they abstained (the first row); per watch, steps a second, the last 40
`[preflight]` lines and every alert.
Its own app ("preflight-runner-d-watch"), an A10 only, the runner's probe image with this checkout's
code, and the repository's flags.json as PREFLIGHT_FLAGS (as a clone's `preflight run` reads it; the
image ships no flag file). Each row is printed as it's made, so a timeout keeps what ran. Cap: the
container's timeout (18 min) at the A10's US$1.10 an hour is US$0.33 [derived].
"""

import json
from pathlib import Path

import modal

from preflight.runner.modal_app import APP_NAME, probe_image

app = modal.App(APP_NAME + "-d-watch")
TIMEOUT_S = 18 * 60
WATCH_S = 150  # guess: long enough for a few windows and both alerts; the shot films about a minute
FILM_STEPS = 200_000  # guess: the video plan's number for about a minute on a GPU; measured here
USD_PER_H = 1.10  # cited: probes/gpu_choice.py's A10 list price


@app.function(image=probe_image, gpu="A10", timeout=TIMEOUT_S)
def film_and_alert(src: str, flags_json: str = "{}") -> dict:
	import os
	import re
	import signal
	import subprocess
	import sys
	import tempfile
	import time

	t0 = time.monotonic()
	env = {k: v for k, v in os.environ.items() if not k.startswith(("MODAL_", "META_", "BASETEN_"))
		and not any(w in k.upper() for w in ("TOKEN", "SECRET", "KEY", "PASSWORD"))} | {"PYTHONUNBUFFERED": "1",
		"PREFLIGHT_FLAGS": flags_json}  # fmt: skip

	def cli(*args: str, limit: float) -> dict:
		started = time.monotonic()
		p = subprocess.Popen([sys.executable, "-m", "preflight.cli.main", *args], stdout=subprocess.PIPE,
			stderr=subprocess.PIPE, text=True, env=env)  # fmt: skip
		stopped = False
		try:
			out, err = p.communicate(timeout=limit)
		except subprocess.TimeoutExpired:
			p.send_signal(signal.SIGTERM)  # the watch writes its live record as cancelled
			stopped = True
			try:
				out, err = p.communicate(timeout=60)
			except subprocess.TimeoutExpired:
				p.kill()
				out, err = p.communicate()
		return {"exit": p.returncode, "seconds": round(time.monotonic() - started, 1), "stopped": stopped,
			"stdout": out[-20000:], "stderr": err[-20000:]}  # fmt: skip

	def estimates(path: Path) -> dict:
		from preflight.schema import load_record

		rec = load_record(path.read_text())
		out: dict = {"gpu": rec.env.gpu.name if rec.env and rec.env.gpu else None,
			"backend": rec.env.backend if rec.env else None}  # fmt: skip
		try:
			out["gpu_type_knob"] = rec.knob("gpu_type").current
		except KeyError:
			out["gpu_type_knob"] = None
		for key in ("time.step", "time.total", "memory.peak"):
			e = rec.predictions.estimates.get(key) if rec.predictions else None
			out[key] = None if e is None else {"status": e.status, "lo": e.lo, "hi": e.hi, "reason": e.reason}
		return out

	def live(records: Path, parent: str) -> dict:
		from preflight.schema import load_record

		recs = [p for p in records.glob("*.json") if p.stem != parent]
		if not recs:
			return {"live_record": None}
		newest = max(recs, key=lambda p: p.stat().st_mtime)
		p = load_record(newest.read_text()).probes[-1]
		jsonl = records / f"{newest.stem}.live.jsonl"
		lines = jsonl.read_text().splitlines() if jsonl.exists() else []
		return {"live_status": p.status, "live_steps": p.steps, "live_duration_s": p.duration_s,
			"steps_per_s": (p.steps / p.duration_s) if p.steps and p.duration_s else None, "windows": len(lines)}  # fmt: skip

	rows: list = []
	with tempfile.TemporaryDirectory() as tmp:
		tmp = Path(tmp)
		film_src = src.replace("max_steps = 200", f"max_steps = {FILM_STEPS}")
		alert_src = film_src.replace("batch_size = 64", "batch_size = 4096")
		alert_src = re.sub(r"nn\.Linear\(32, 256\), nn\.ReLU\(\), nn\.Linear\(256, 1\)",
			"nn.Linear(32, 8192), nn.ReLU(), nn.Linear(8192, 1)", alert_src)  # fmt: skip
		changed = {"max_steps": film_src != src, "batch": "batch_size = 4096" in alert_src,
			"hidden": "nn.Linear(32, 8192)" in alert_src}  # fmt: skip
		if not all(changed.values()):
			return {"failed": f"the script's text didn't take the shot's changes: {changed}"}
		film, alert = tmp / "film" / "tiny_train.py", tmp / "alert" / "tiny_train.py"
		for f, text in ((film, film_src), (alert, alert_src)):
			f.parent.mkdir()
			f.write_text(text)
		records = tmp / "records"
		pre = cli("run", str(film), "--out", str(records), "--require-gpu", limit=420)
		row: dict = {
			"case": "A-run",
			"changed": changed,
			"run": {k: pre[k] for k in ("exit", "seconds")},
			"run_stderr": pre["stderr"][-1500:],
		}
		try:
			at = pre["stdout"].rfind("\n{\n") + 1
			parent = Path(json.loads(pre["stdout"][at:])["record"])
			row["record"] = estimates(parent)
		except (ValueError, KeyError, OSError) as exc:
			row["record_error"] = f"{type(exc).__name__}: {exc}"[:300]
			rows.append(row)
			return {"rows": rows, "container_s": round(time.monotonic() - t0, 1)}
		rows.append(row)
		print(json.dumps(row, default=str)[:4000], flush=True)
		if pre["exit"] != 0:
			return {"rows": rows, "container_s": round(time.monotonic() - t0, 1)}
		for case, script in (("B-film", film), ("C-alert", alert)):
			out = records / case
			w = cli("watch", str(parent), str(script), "--out", str(out), "--window", "200", limit=WATCH_S)
			lines = [x for x in w["stderr"].splitlines() if x.startswith("[preflight]")]
			r = {"case": case, "watch": {k: w[k] for k in ("exit", "seconds", "stopped")},
				"preflight_lines": lines[-40:], "alerts": [x for x in lines if re.search(r"step \d+: ", x)],
				"stderr_tail": w["stderr"][-1200:]}  # fmt: skip
			try:
				r |= live(out, parent.stem)
			except Exception as exc:
				r["live_error"] = f"{type(exc).__name__}: {exc}"[:300]
			rows.append(r)
			print(
				json.dumps({k: v for k, v in r.items() if k != "preflight_lines"}, default=str)[:4000],
				flush=True,
			)
	s = round(time.monotonic() - t0, 1)
	return {"rows": rows, "container_s": s, "container_usd": round(s * USD_PER_H / 3600, 4)}


@app.local_entrypoint()
def check():
	root = Path(__file__).resolve().parents[3]
	src = (root / "zoo/notebook_tiny/tiny_train.py").read_text()
	Path("out").mkdir(exist_ok=True)
	print("[watch gpu] waiting for an A10", flush=True)
	try:
		res = film_and_alert.remote(src, (root / "flags.json").read_text())
	except Exception as exc:  # recorded
		res = {"failed": f"{type(exc).__name__}: {str(exc)[:500]}"}
	Path("out/watch_gpu.json").write_text(json.dumps(res, indent=1, default=str))
	for r in res.get("rows", []):
		print(json.dumps({k: r.get(k) for k in ("case", "changed", "record", "watch", "alerts", "steps_per_s",
			"live_status", "live_steps")}, default=str)[:3000], flush=True)  # fmt: skip
	print(json.dumps({k: res.get(k) for k in ("container_s", "container_usd", "failed")}), flush=True)
