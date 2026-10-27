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


def main(spec_path: str) -> int:
	spec = json.loads(Path(spec_path).read_text())
	workdir = os.path.abspath(spec["workdir"])
	out_path = spec["out"]
	try:
		import torch

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
	sys.argv = [spec["entrypoint"], *spec.get("args", [])]
	if spec.get("seed") is not None:
		torch.manual_seed(int(spec["seed"]))
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
