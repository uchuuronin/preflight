"""Subprocess entry point for one probe:  python -m preflight.recorder.run SPEC.json

Runs the user's (already overridden) script under the recorder, in its own
process so every probe gets a clean device context and a hard wall-clock cap
the parent can enforce. Writes a result JSON; never raises.
"""

import json
import os
import runpy
import sys
import traceback
from pathlib import Path


def _user_frame(tb, workdir: str) -> str | None:
	for fr in reversed(traceback.extract_tb(tb)):
		path = os.path.abspath(fr.filename)
		if path.startswith(workdir + os.sep) and "site-packages" not in path:
			return f"{os.path.relpath(path, workdir)}:{fr.lineno}"
	return None


def _write(out_path: str, result: dict) -> None:
	tmp = out_path + ".tmp"
	Path(tmp).write_text(json.dumps(result, default=str))
	os.replace(tmp, out_path)


# Limits on the visitor's code, set in this process before it runs (inherited by all it starts).
# derived: no core files (they would hold the process's memory); a file-size cap well above a
# 70B fp32 checkpoint shard so a runaway writer can't fill the disk; enough processes and threads
# for DataLoader workers times BLAS threads; descriptors for big datasets.
RLIMITS = {
	"RLIMIT_CORE": 0,
	"RLIMIT_FSIZE": 64 * 2**30,
	"RLIMIT_NPROC": 8192,
	"RLIMIT_NOFILE": 65536,
}


def set_limits() -> dict:
	"""Lower each limit to ours (never raise one); returns what was applied, for the log."""
	import resource

	applied = {}
	for name, want in RLIMITS.items():
		lim = getattr(resource, name, None)
		if lim is None:
			continue
		try:
			soft, hard = resource.getrlimit(lim)
			new_hard = want if hard == resource.RLIM_INFINITY else min(hard, want)
			resource.setrlimit(
				lim, (min(new_hard, soft if soft != resource.RLIM_INFINITY else new_hard), new_hard)
			)
			applied[name] = new_hard
		except (ValueError, OSError):
			continue
	return applied


def _seed(seed: int) -> None:
	"""Seed every generator a training script draws from by default, for fixed-seed comparisons.

	Scripts that seed themselves (HF Trainer's args.seed, set_seed) keep their own seed; the verify
	run compares two settings of the same script, so either way both runs draw the same numbers.
	"""
	import random

	random.seed(seed)
	os.environ.setdefault("PYTHONHASHSEED", str(seed))  # for subprocesses (data loader workers)
	try:
		import numpy as np

		np.random.seed(seed % 2**32)
	except ImportError:
		pass
	import torch

	torch.manual_seed(seed)  # all devices, CUDA included


def main(spec_path: str) -> int:
	spec = json.loads(Path(spec_path).read_text())
	set_limits()
	workdir = os.path.abspath(spec["workdir"])
	out_path = spec["out"]
	try:
		import torch  # noqa: F401  (fails here, inside the try, when torch is missing)

		from .errors import classify
		from .runtime import Recorder, StopProbe, detect_env, mfu
	except Exception as exc:  # torch missing or broken in the image
		_write(
			out_path,
			{
				"status": "error",
				"error_code": "recorder",
				"error": f"recorder could not start: {exc}",
				"summary": {},
			},
		)
		return 0

	os.chdir(workdir)
	sys.path.insert(0, workdir)
	if spec.get("install_requirements") and spec.get("deps"):
		from .requirements import install

		if spec.get("progress"):
			_write(spec["progress"], {"stage": "installing requirements", "steps": 0})
		err, dirs = install(Path(workdir), Path(spec["deps"]))
		if err:
			_write(out_path, {"status": "error", "error_code": "missing_package.install",
				"error": f"requirements.txt didn't install: {err}", "summary": {}})  # fmt: skip
			return 0
		for i, d in enumerate(dirs, 1):
			sys.path.insert(i, str(d))  # after the script's own files, before the image
		parts = [*(str(d) for d in dirs), os.environ.get("PYTHONPATH") or ""]
		os.environ["PYTHONPATH"] = os.pathsep.join(p for p in parts if p)
	sys.argv = [spec["entrypoint"], *spec.get("args", [])]
	if spec.get("seed") is not None:
		_seed(int(spec["seed"]))
	rec = Recorder(
		warmup=spec["warmup"],
		timed=spec["timed"],
		profile=spec.get("profile", False),
		trace_path=spec.get("trace"),
		progress_path=spec.get("progress"),
		eval_batches=spec.get("eval_batches", 2),
	).install()

	status, code, error = "ok", None, None
	too_few = "script finished after {n} optimizer steps; a probe needs at least 2"
	try:
		runpy.run_path(os.path.join(workdir, spec["entrypoint"]), run_name="__main__")
		if len(rec.steps) < 2:
			status, code, error = "error", "too_few_steps", too_few.format(n=len(rec.steps))
	except StopProbe:
		pass
	except SystemExit as exc:
		if exc.code not in (0, None):
			status, code, error = (
				"error",
				"script_exit",
				f"script exited with code {exc.code} after step {len(rec.steps)}",
			)
		elif len(rec.steps) < 2:
			status, code, error = "error", "too_few_steps", too_few.format(n=len(rec.steps))
	except BaseException as exc:
		n = len(rec.steps)
		c = classify(exc, steps=n, in_eval=rec.eval_active)
		where = _user_frame(exc.__traceback__, workdir)
		if isinstance(exc, SyntaxError) and exc.filename:
			where = f"{os.path.relpath(os.path.abspath(exc.filename), workdir)}:{exc.lineno}"
		when = f"after step {n}" if n else f"before the first optimizer step (stage: {rec.stage})"
		detail = f"{type(exc).__name__}: {str(exc)[:500]}"
		status, code = c.status, c.code
		error = f"{c.message}. {detail}" + (f" at {where}" if where else "") + f", {when}"
	finally:
		rec.uninstall()

	summary = rec.summary()
	env = detect_env()
	flops, util = mfu(summary, env)
	summary["flops_per_step"] = flops
	summary["mfu"] = util
	_write(out_path, {"status": status, "error_code": code, "error": error, "summary": summary, "env": env})
	return 0


if __name__ == "__main__":
	sys.exit(main(sys.argv[1]))
