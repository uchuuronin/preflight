"""Does `preflight watch` work on a CPU run, and does an alert fire? (D, Nov 2; Chiya's list item 2)

Run through remote-run.yml (a CPU runner; no GPU, no model call, no spend):
	python scripts/d_watch_cpu_check.py
Installs CPU torch into the run's venv if it's missing (the CI torch job's command), then, on
zoo/notebook_tiny/tiny_train.py and two copies of it:

  A. as it is: `preflight run`, then `preflight watch` on its record
  B. "film": max_steps 200 -> 6000, so the run lasts long enough to watch in the cockpit
  C. "alert": B's record, watched with batch_size 64 -> 1024, a real change the bands didn't see

Writes out/watch_cpu.json (per step: exit code, seconds, the record's time.step / time.total /
memory.peak estimates or why they abstained, the live record's probe, every `[preflight]` line)
after every case. The scripts' copies are only in the runner's temp folder. The subprocesses get
the environment without the run's secrets (no MODAL_, META_, BASETEN_ or token-like names), so
nothing saved can echo one.
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TINY = ROOT / "zoo/notebook_tiny/tiny_train.py"
OUT = Path("out")
FILM_STEPS = 50000  # guess: about a minute on a laptop CPU at a millisecond a step; the check measures it
SECRETISH = ("MODAL_", "META_", "BASETEN_", "GITHUB_", "ACTIONS_")


def clean_env() -> dict:
	return {k: v for k, v in os.environ.items() if not k.startswith(SECRETISH)
		and not any(w in k.upper() for w in ("TOKEN", "SECRET", "KEY", "PASSWORD"))} | {"PYTHONUNBUFFERED": "1"}  # fmt: skip


def sh(args: list[str], timeout: int = 900) -> dict:
	t0 = time.monotonic()
	r = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env=clean_env())
	return {"cmd": " ".join(args[-6:]), "exit": r.returncode, "seconds": round(time.monotonic() - t0, 1),
		"stdout": r.stdout[-20000:], "stderr": r.stderr[-6000:]}  # fmt: skip


def estimates(path: Path) -> dict:
	from preflight.schema import load_record

	rec = load_record(path.read_text())
	out = {}
	for k in ("time.step", "time.total", "memory.peak"):
		e = rec.predictions.estimates.get(k)
		out[k] = None if e is None else {
			"status": e.status, "lo": e.lo, "hi": e.hi, "value": getattr(e, "value", None),
			"why": getattr(e, "abstain_reason", None) or getattr(e, "reason", None)}  # fmt: skip
	out["backend"] = rec.env.backend if rec.env else None
	out["probes"] = [{"id": p.id, "status": p.status, "steps": p.steps, "planned": p.extras.get("planned_steps"),
		"error": (p.error or "")[:200]} for p in rec.probes]  # fmt: skip
	return out


def live(out_dir: Path, parent_token: str) -> dict:
	from preflight.schema import load_record

	recs = [p for p in out_dir.glob("*.json") if p.stem != parent_token]
	if not recs:
		return {"live_record": None}
	newest = max(recs, key=lambda p: p.stat().st_mtime)
	rec = load_record(newest.read_text())
	p = rec.probes[-1]
	jsonl = out_dir / f"{newest.stem}.live.jsonl"
	lines = jsonl.read_text().splitlines() if jsonl.exists() else []
	return {"live_record": newest.name, "probe": {"id": p.id, "status": p.status, "steps": p.steps,
		"duration_s": p.duration_s}, "windows": len(lines), "first_window": json.loads(lines[0]) if lines else None}  # fmt: skip


def run_and_watch(name: str, script: Path, watch_script: Path, rows: list, records: Path) -> None:
	pre = sh([sys.executable, "-m", "preflight.cli.main", "run", str(script), "--out", str(records / name)])
	row: dict = {
		"case": name,
		"run": {k: pre[k] for k in ("exit", "seconds")},
		"run_stderr": pre["stderr"][-1500:],
	}
	try:
		at = pre["stdout"].rfind("\n{\n") + 1  # the CLI's summary is the last JSON object it prints
		parent = Path(json.loads(pre["stdout"][at:])["record"])
	except (ValueError, KeyError):
		parent = None
	if pre["exit"] != 0 or parent is None or not parent.exists():
		row["run_stdout"] = pre["stdout"]
		rows.append(row)
		return
	row["parent"] = parent.name
	row["record"] = estimates(parent)
	w = sh([sys.executable, "-m", "preflight.cli.main", "watch", str(parent), str(watch_script), "--out",
		str(records / name)])  # fmt: skip
	row["watch"] = {k: w[k] for k in ("exit", "seconds")}
	row["preflight_lines"] = [x for x in w["stderr"].splitlines() if x.startswith("[preflight]")]
	row["alerts"] = [x for x in row["preflight_lines"] if re.search(r"step \d+: ", x)]
	row["watch_stderr_tail"] = w["stderr"][-1500:]
	row |= live(records / name, parent.stem)
	rows.append(row)


def main() -> None:
	OUT.mkdir(exist_ok=True)
	setup = {}
	try:
		import torch  # noqa: F401
	except ImportError:
		r = sh(["uv", "pip", "install", "--python", sys.executable, "torch>=2.4", "psutil", "--index-url",
			"https://download.pytorch.org/whl/cpu", "--extra-index-url", "https://pypi.org/simple"])  # fmt: skip
		setup["install"] = {k: r[k] for k in ("exit", "seconds")} | {"tail": r["stderr"][-800:]}
	v = sh(
		[
			sys.executable,
			"-c",
			"import torch, os; print(torch.__version__, torch.cuda.is_available(), os.cpu_count())",
		]
	)
	setup["torch"] = v["stdout"].strip() or v["stderr"][-300:]
	rows: list = []

	def save():
		(OUT / "watch_cpu.json").write_text(json.dumps({"setup": setup, "rows": rows}, indent=1, default=str))

	def case(name, fn):
		try:
			fn()
		except Exception as exc:  # recorded; the next case still runs and the file is still written
			rows.append({"case": name, "failed": f"{type(exc).__name__}: {str(exc)[:500]}"})
		save()

	with tempfile.TemporaryDirectory() as tmp:
		tmp = Path(tmp)
		src = TINY.read_text()
		long_src = src.replace("max_steps = 200", f"max_steps = {FILM_STEPS}")
		film = tmp / "film" / "tiny_train.py"
		film.parent.mkdir()
		film.write_text(long_src)
		alert = tmp / "alert" / "tiny_train.py"
		alert.parent.mkdir()
		alert.write_text(long_src.replace("batch_size = 64", "batch_size = 1024"))
		assert film.read_text() != src and alert.read_text() != film.read_text()
		records = tmp / "records"
		case("A-as-is", lambda: run_and_watch("A-as-is", TINY, TINY, rows, records))
		case("B-film", lambda: run_and_watch("B-film", film, film, rows, records))

		def c_alert():
			# C: B's prediction, the changed script watched against it
			b = next((r for r in rows if r.get("case") == "B-film" and r.get("parent")), None)
			if b is None:
				rows.append({"case": "C-alert", "failed": "no B record"})
				return
			parent = records / "B-film" / b["parent"]
			w = sh([sys.executable, "-m", "preflight.cli.main", "watch", str(parent), str(alert), "--out",
				str(records / "C-alert")])  # fmt: skip
			row = {"case": "C-alert", "parent": parent.name, "watch": {k: w[k] for k in ("exit", "seconds")}}
			row["preflight_lines"] = [x for x in w["stderr"].splitlines() if x.startswith("[preflight]")]
			row["alerts"] = [x for x in row["preflight_lines"] if re.search(r"step \d+: ", x)]
			row["watch_stderr_tail"] = w["stderr"][-1500:]
			row |= live(records / "C-alert", parent.stem)
			rows.append(row)

		case("C-alert", c_alert)
	for r in rows:
		print(
			json.dumps({k: r.get(k) for k in ("case", "run", "watch", "alerts", "failed")}, default=str),
			flush=True,
		)


if __name__ == "__main__":
	main()
