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
import collections
import contextlib
import json
import os
import select
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ZOO_SCRIPTS = {"finetune_lora_hf", "pretrain_nanogpt"}
# derived: the whole measurement must end, and write its result, inside the run's own time limit
# (remote-run.yml's timeout); Oct 28's first try ran into that limit with nothing written
BUDGET_S = 40 * 60
BARE_SHARE = 0.5  # of the budget, at most, for the bare run; the recorder run gets the rest
DRAIN_S = 5.0  # guess: output still in the pipe when the command exits arrives within this


def _say(msg: str) -> None:
	print(f"[ground truth {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _stream(cmd: list[str], *, env: dict, cwd: Path, timeout_s: float) -> tuple[int | None, str]:
	"""Run cmd with its output passed through as it comes (so a run's log shows where it is), and
	return (exit code, or None when it ran past timeout_s and was killed; the output's tail).
	The command runs in its own process group: a dataloader or compile worker that inherited the
	pipe neither keeps us reading after the command exits nor outlives the kill."""
	tail = collections.deque(maxlen=64)
	deadline = time.monotonic() + timeout_s
	proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=cwd,
		start_new_session=True)  # fmt: skip
	fd = proc.stdout.fileno()
	exited_at = None
	code: int | None = None
	try:
		while True:
			now = time.monotonic()
			if proc.poll() is not None and exited_at is None:
				exited_at = now
			if exited_at is not None and now - exited_at > DRAIN_S:
				break  # exited; whatever still holds the pipe is a leftover worker
			if exited_at is None and now >= deadline:
				break
			ready, _, _ = select.select([fd], [], [], 1.0)
			if not ready:
				continue
			chunk = os.read(fd, 65536)  # whatever is there: a progress bar's \r line never blocks us
			if not chunk:
				if exited_at is None:
					proc.wait(timeout=max(0.1, deadline - time.monotonic()))
				break
			tail.append(chunk)
			sys.stdout.write(chunk.decode("utf-8", "replace"))
			sys.stdout.flush()
		code = proc.poll()
	except subprocess.TimeoutExpired:
		code = None
	finally:
		with contextlib.suppress(ProcessLookupError, PermissionError):
			os.killpg(proc.pid, signal.SIGKILL)  # the command if still running, and any leftover worker
		proc.wait()
		proc.stdout.close()
	return code, b"".join(tail).decode("utf-8", "replace")


def _bare(
	work: Path, bare_steps: int, *, install_requirements: bool, caches: Path, env: dict, timeout_s: float
) -> dict:
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
	cmd = [
		sys.executable,
		"-m",
		"preflight.recorder.bare",
		str(work / "train.py"),
		str(out),
		"--steps",
		str(bare_steps),
	]
	code, tail = _stream(cmd, env=env, cwd=work, timeout_s=timeout_s)
	if out.exists():  # bare.py writes its result last, so a file here is a whole result
		return json.loads(out.read_text())
	if code is None:
		# keep going: the recorder run is still worth having from a paid session
		return {"failed": f"the bare run took longer than {timeout_s:.0f} s", "tail": tail[-3000:]}
	if out.exists():
		return json.loads(out.read_text())
	return {"failed": tail[-3000:]}


def measure(
	script_dir: Path,
	*,
	bare_steps: int,
	recorder_steps: int = 60,
	warmup: int = 5,
	install_requirements: bool = False,
	budget_s: float = BUDGET_S,
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
		_say(
			f"bare run of {script_dir.name}: {bare_steps} optimizer steps, at most {budget_s * BARE_SHARE:.0f} s"
		)
		bare = _bare(work, bare_steps, install_requirements=install_requirements, caches=caches,
			env=bare_env or {}, timeout_s=budget_s * BARE_SHARE)  # fmt: skip
		bare_wall = time.monotonic() - t0
		_say(
			f"bare run ended ({bare.get('ended') or 'failed'}) after {bare_wall:.0f} s, {bare.get('steps')} steps"
		)
		left = max(120.0, budget_s - bare_wall - 60)  # a minute kept for writing the result
		_say(f"recorder run: {warmup} warmup + {recorder_steps} timed steps, at most {left:.0f} s")
		r = run_probe(ProbeSpec(probe_id="gt", files=files, entrypoint="train.py", warmup=warmup,
			timed=recorder_steps, timeout_s=left, alternate_sync=True,
			install_requirements=install_requirements))  # fmt: skip
		_say(f"recorder run ended ({r.probe.status}) after {r.probe.steps} steps")
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
	ap.add_argument(
		"--budget-min", type=float, default=BUDGET_S / 60, help="the whole measurement's time limit"
	)
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
			warmup=ns.warmup, install_requirements=ns.install_requirements, bare_env=bare_env,
			budget_s=ns.budget_min * 60)  # fmt: skip
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
