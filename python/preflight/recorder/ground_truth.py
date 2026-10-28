"""Ground truth for one zoo script on this machine's GPU: a bare run, then a recorder run.

	python -m preflight.recorder.ground_truth zoo/finetune_lora_hf out/gt.json --bare-steps 210
	python -m preflight.recorder.ground_truth zoo/pretrain_nanogpt out/gt.json --bare-steps 1000 \\
		--recorder-steps 200 --install-requirements

The bare run (recorder/bare.py) has a CUDA event at each optimizer step and nothing else, so it
shows how the script really runs and where it really fails (the LoRA script's eval OOM included).
The recorder run is a probe that alternates clean and synced steps (Recorder alternate_sync), and
recorder/overhead.py pairs the two into the recorder's overhead. The same code runs on Modal's L4
(runner/ops_app.py::ground_truth) and on the AMD droplet (amd-run.yml), so the two are comparable.

Our own zoo scripts on our own GPUs, never visitor code. As root (a cloud box) the bare run gets
its own root-owned caches, and the probe drops to its unprivileged uid with its own, as the CLI's
probes do; nothing either writes is read by the other.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ZOO_SCRIPTS = {"finetune_lora_hf", "pretrain_nanogpt"}
BARE_TIMEOUT_S = 2400


def _bare(work: Path, bare_steps: int, *, install_requirements: bool, caches: Path, env: dict) -> dict:
	env = {**os.environ, **env}
	if install_requirements:
		from preflight.recorder.requirements import install

		# into this run's own prefix and pip cache, never the probes' (owned by the probe uid on a
		# root box): install() reads both from the environment
		saved = {
			k: os.environ.pop(k) for k in ("PREFLIGHT_DEPS_DIR", "PREFLIGHT_PIP_CACHE") if k in os.environ
		}
		try:
			err, dirs = install(work, caches / "deps", log=lambda m: print(m, flush=True))
		finally:
			os.environ.update(saved)
		if err:
			return {"failed": f"requirements: {err}"}
		env["PYTHONPATH"] = os.pathsep.join([*map(str, dirs), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
	out = work / "bare.json"
	try:
		t = subprocess.run([sys.executable, "-m", "preflight.recorder.bare", str(work / "train.py"), str(out),
			"--steps", str(bare_steps)], capture_output=True, text=True, timeout=BARE_TIMEOUT_S, env=env, cwd=work)  # fmt: skip
	except subprocess.TimeoutExpired:
		# keep going: the recorder run is still worth having from a paid session
		return {"failed": f"the bare run took longer than {BARE_TIMEOUT_S} s"}
	if out.exists():
		return json.loads(out.read_text())
	return {"failed": (t.stdout + t.stderr)[-3000:]}


def measure(
	script_dir: Path,
	*,
	bare_steps: int,
	recorder_steps: int = 60,
	warmup: int = 5,
	install_requirements: bool = False,
	timeout_s: float = 2400,
	bare_env: dict | None = None,
) -> dict:
	"""Both runs and the paired overhead, as one dict (what out/ground_truth_<script>.json holds).
	bare_env: environment overrides for the bare run only (its own caches on a root box)."""
	from preflight.probes.local import ProbeSpec, run_probe
	from preflight.recorder.overhead import report
	from preflight.runner.sources import _read_tree

	files = _read_tree(script_dir)
	if "train.py" not in files:
		raise ValueError(f"{script_dir} has no train.py")
	caches = Path(tempfile.mkdtemp(prefix="gt-caches-"))
	work = Path(tempfile.mkdtemp(prefix="gt-"))
	try:
		for rel, text in files.items():
			(work / rel).parent.mkdir(parents=True, exist_ok=True)
			(work / rel).write_text(text)
		t0 = time.monotonic()
		bare = _bare(work, bare_steps, install_requirements=install_requirements, caches=caches,
			env=bare_env or {})  # fmt: skip
		bare_wall = time.monotonic() - t0
		r = run_probe(ProbeSpec(probe_id="gt", files=files, entrypoint="train.py", warmup=warmup,
			timed=recorder_steps, timeout_s=timeout_s, alternate_sync=True,
			install_requirements=install_requirements))  # fmt: skip
	finally:
		shutil.rmtree(work, ignore_errors=True)
		shutil.rmtree(caches, ignore_errors=True)
	log = [s.model_dump() for s in r.probe.step_log]
	over = report(log, bare.get("walls_s") or [], warmup=warmup) if r.probe.status == "ok" else None
	return {"script": script_dir.name, "gpu": r.env.gpu.name if r.env and r.env.gpu else None,
		"backend": r.env.backend if r.env else None, "bare": {**bare, "wall_s": round(bare_wall, 1)},
		"recorder": {"status": r.probe.status, "error": r.probe.error, "error_code": r.probe.error_code,
			"steps": r.probe.steps, "p50_s": r.probe.step_time.p50_s if r.probe.step_time else None,
			"peak_allocated_bytes": r.probe.memory.peak_allocated_bytes if r.probe.memory else None,
			"seconds": r.probe.cost.gpu_seconds if r.probe.cost else None,
			"step_log": log, "log_tail": r.log_tail[-1500:]},
		"overhead": over}  # fmt: skip


def brief(res: dict) -> dict:
	"""The result without the per-step lists, for a log line."""
	out = {k: v for k, v in res.items() if k not in ("bare", "recorder")}
	out["bare"] = {k: v for k, v in res["bare"].items() if k != "walls_s"}
	out["recorder"] = {k: v for k, v in res["recorder"].items() if k not in ("step_log", "log_tail")}
	return out


def main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser(prog="python -m preflight.recorder.ground_truth")
	ap.add_argument("script_dir", type=Path)
	ap.add_argument("out", type=Path)
	ap.add_argument("--bare-steps", type=int, required=True)
	ap.add_argument("--recorder-steps", type=int, default=60)
	ap.add_argument("--warmup", type=int, default=5)
	ap.add_argument("--install-requirements", action="store_true")
	ns = ap.parse_args(argv)
	if ns.script_dir.name not in ZOO_SCRIPTS:
		print(f"ground truth is for the zoo scripts {sorted(ZOO_SCRIPTS)}", file=sys.stderr)
		return 2
	from preflight.cli.main import _run_caches_when_root

	# as root: the bare run (root) and the probe (its own uid) each get fresh caches of their own
	bare_env, own = {}, None
	if os.geteuid() == 0:
		own = Path(tempfile.mkdtemp(prefix="gt-root-", dir="/var/tmp"))
		bare_env = {"HF_HOME": str(own / "hf")}
	run = _run_caches_when_root()
	try:
		res = measure(ns.script_dir, bare_steps=ns.bare_steps, recorder_steps=ns.recorder_steps,
			warmup=ns.warmup, install_requirements=ns.install_requirements, bare_env=bare_env)  # fmt: skip
	finally:
		for d in (own, run):
			if d is not None:
				shutil.rmtree(d, ignore_errors=True)
	ns.out.parent.mkdir(parents=True, exist_ok=True)
	ns.out.write_text(json.dumps(res, indent=2))
	print(json.dumps(brief(res), indent=2), flush=True)
	return 0 if res["recorder"]["status"] == "ok" else 1


if __name__ == "__main__":
	sys.exit(main())
