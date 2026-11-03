"""Run one corpus config in its own process, with a deadline.

Each config gets a fresh process, so a fresh CUDA/ROCm context: a kernel fault
(cuda_error) can't poison the configs after it, and memory left over from an
OOM can't inflate the next config's peak and fake an OOM label. A config that
hangs is killed at its deadline and recorded as a `timeout` row carrying its
billed seconds (the dead-letter record), so it is never retried silently and
its spend is never lost. A child that dies without a result is classified with
A's exit classifier, so corpus rows and probes share one error vocabulary.

	python -m preflight.corpus.isolate --child cfg.json out.json --device cuda [--data FILE]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

from preflight.corpus.configs import CorpusConfig

KILL_GRACE_S = 30.0  # guess: time for a killed child to exit before the parent stops waiting for it
DEFAULT_TIMEOUT_S = {
	"kernels": 600.0,
	"divergence": 1500.0,
	"divergence_real": 1500.0,
	"hf": 900.0,  # guess: model download plus five points; replaced by measured p99 after the first hf wave
}  # guess: per-config deadline; replaced by 3x the measured p99 config time after the first wave


def run_isolated(
	cfg: CorpusConfig,
	device: str = "cuda",
	data_path: str | None = None,
	timeout_s: float | None = None,
	python: str = sys.executable,
) -> dict[str, Any]:
	"""The worker's row for one config, run in a child process; never raises for a config's failure."""
	from preflight.recorder.errors import classify_exit

	deadline = timeout_s if timeout_s is not None else DEFAULT_TIMEOUT_S.get(cfg.kind, 900.0)
	with tempfile.TemporaryDirectory(prefix="pf-corpus-") as tmp:
		cin, cout = Path(tmp) / "cfg.json", Path(tmp) / "out.json"
		cin.write_text(json.dumps(cfg.to_dict()))
		cmd = [python, "-m", "preflight.corpus.isolate", "--child", str(cin), str(cout), "--device", device]
		if data_path:
			cmd += ["--data", data_path]
		env = {
			**os.environ,
			"PYTHONPATH": os.pathsep.join(
				p for p in (str(Path(__file__).resolve().parents[2]), os.environ.get("PYTHONPATH", "")) if p
			),
		}
		t0 = time.monotonic()
		proc = subprocess.Popen(
			cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True
		)
		timed_out = False
		try:
			_, err = proc.communicate(timeout=deadline)
		except subprocess.TimeoutExpired:
			timed_out = True
			_kill_group(proc)
			try:  # a child stuck in the GPU driver can ignore SIGKILL for a while; never wait on it forever
				_, err = proc.communicate(timeout=KILL_GRACE_S)
			except subprocess.TimeoutExpired:
				err = b"child did not exit after SIGKILL; abandoned"
		elapsed = time.monotonic() - t0
		result: dict[str, Any] | None = None
		if not timed_out and cout.exists():
			try:
				result = json.loads(cout.read_text())
			except json.JSONDecodeError:
				result = None
		if result is None:
			c = classify_exit(
				returncode=proc.returncode,
				timed_out=timed_out,
				steps=0,
				stage="corpus config",
				timeout_s=deadline,
			)
			tail = (err or b"").decode(errors="replace")[-1500:]
			result = {"status": c.status, "error_code": c.code, "error": c.message, "stderr_tail": tail}
		result["gpu_seconds"] = elapsed  # billed wall time, including the child's start-up
		result["isolated"] = True
		return result


def _kill_group(proc: subprocess.Popen) -> None:
	with contextlib.suppress(ProcessLookupError):
		os.killpg(proc.pid, signal.SIGKILL)


def _child(cfg_path: str, out_path: str, device: str, data: str | None) -> int:
	cfg = CorpusConfig.from_dict(json.loads(Path(cfg_path).read_text()))
	try:
		from preflight.corpus import worker

		result = worker.run_config(cfg, device=device, data_path=data)
	except Exception as exc:  # worker re-raises only bugs in corpus code (user_error)
		result = {
			"status": "error",
			"error_code": "user_error",
			"error": f"{type(exc).__name__}: {str(exc)[:500]}",
			"traceback": traceback.format_exc(limit=8),
		}
	Path(out_path).write_text(json.dumps(result))
	return 0


def _main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser(prog="preflight.corpus.isolate")
	ap.add_argument("--child", nargs=2, metavar=("CFG_JSON", "OUT_JSON"), required=True)
	ap.add_argument("--device", default="cuda")
	ap.add_argument("--data")
	args = ap.parse_args(argv)
	return _child(args.child[0], args.child[1], args.device, args.data)


if __name__ == "__main__":
	sys.exit(_main())
