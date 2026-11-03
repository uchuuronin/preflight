"""Subprocess entry point for one probe:  python -m preflight.recorder.run SPEC.json

Runs the user's (already overridden) script under the recorder, in its own
process so every probe gets a clean device context and a hard wall-clock cap
the parent can enforce. Writes a result JSON; never raises.
"""

import json
import os
import runpy
import sys
import time
import traceback
from pathlib import Path


def _user_frame(tb, workdir: str) -> str | None:
	for fr in reversed(traceback.extract_tb(tb)):
		path = os.path.abspath(fr.filename)
		if path.startswith(workdir + os.sep) and "site-packages" not in path:
			return f"{os.path.relpath(path, workdir)}:{fr.lineno}"
	return None


def _what_installed(spec: dict) -> str:
	"""What a failed install was installing, for its message."""
	has_req = Path("requirements.txt").is_file()
	what = "requirements.txt" if has_req or not spec.get("install_repo") else "the repo"
	if spec.get("extra_packages"):
		what = (
			f"{what} with the extra packages" if has_req or spec.get("install_repo") else "the extra packages"
		)
	return what


def _local_missing(exc: BaseException, workdir: str, spec: dict) -> str | None:
	"""runner.entrypoint_path: when the module Python couldn't import is the repo's own (a folder or file the source
	has, anywhere in it), say that, not "not installed"; None otherwise."""
	name = getattr(exc, "name", None) or ""
	top = name.split(".")[0]
	if not top or not top.isidentifier():
		return None
	root = Path(workdir)
	entry_dir = (root / spec.get("entrypoint", "")).parent
	for base in dict.fromkeys([entry_dir, root, root / "src"]):
		if (base / top).is_dir() or (base / f"{top}.py").is_file():
			return (f"The script imports {name}, the repo's own module, which Python couldn't find on its path "
				f"(it's under {os.path.relpath(base, root) if base != root else 'the top of the repo'}/)")  # fmt: skip
	try:  # elsewhere in the source: not read, or not on the path the script runs with
		hits = [p for p in root.rglob(top) if p.is_dir()][:1] or [p for p in root.rglob(f"{top}.py")][:1]
	except OSError:
		hits = []
	if hits:
		where = os.path.relpath(hits[0].parent, root)
		return (f"The script imports {name}, the repo's own module under {where}/, which isn't on the path "
			"the script runs with")  # fmt: skip
	return None


def _write(out_path: str, result: dict) -> None:
	tmp = out_path + ".tmp"
	Path(tmp).write_text(json.dumps(result, default=str), encoding="utf-8")
	for i in range(
		40
	):  # Windows refuses a replace while the parent is reading the file (D, Nov 3): wait it out
		try:
			os.replace(tmp, out_path)
			return
		except PermissionError:
			if i == 39:
				raise
			time.sleep(0.05 * min(i + 1, 5))


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
	"""Lower each limit to ours (never raise one); returns what was applied, for the log. None on Windows,
	which has no `resource` module (D, Nov 3): the probe runs without them there."""
	try:
		import resource
	except ImportError:
		return {}

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


def _seed(seed: int, *, force: bool = False) -> None:
	"""Seed every generator a training script draws from by default, for fixed-seed comparisons.

	force: also make the script's own seeding (random.seed, numpy's, torch.manual_seed, so HF's
	set_seed too) use this seed, so two verify runs at two seeds really differ and their spread is
	the seed's. Explicit torch.Generator objects the script seeds itself keep their own seed.
	"""
	import random

	if force:
		_force_seed(seed)

	random.seed(seed)
	os.environ.setdefault("PYTHONHASHSEED", str(seed))  # for subprocesses (data loader workers)
	try:
		import numpy as np

		np.random.seed(seed % 2**32)
	except ImportError:
		pass
	import torch

	torch.manual_seed(seed)  # all devices, CUDA included


def _force_seed(seed: int) -> None:
	import random

	import torch

	def pinned(fn):
		def call(*_args, **_kw):
			return fn(seed % 2**32)

		return call

	random.seed = pinned(random.seed)
	torch.manual_seed = pinned(torch.manual_seed)
	torch.random.manual_seed = torch.manual_seed
	if hasattr(torch, "cuda"):
		torch.cuda.manual_seed = pinned(torch.cuda.manual_seed)
		torch.cuda.manual_seed_all = pinned(torch.cuda.manual_seed_all)
	try:
		import numpy as np

		np.random.seed = pinned(np.random.seed)
	except ImportError:
		pass


def _rank() -> int:
	try:
		return int(os.environ.get("RANK") or 0)
	except ValueError:
		return 0


def main(spec_path: str) -> int:
	spec = json.loads(Path(spec_path).read_text())
	set_limits()
	workdir = os.path.abspath(spec["workdir"])
	out_path = spec["out"]
	rank = _rank()
	if rank > 0:
		# torchrun (spec nproc > 1): every rank runs under its own recorder; rank 0 writes the
		# result and progress, the others a result of their own beside it (memory per rank)
		out_path = f"{out_path}.rank{rank}"
		# profiling stays on so every rank runs the same steps (a rank that stops early leaves rank 0's
		# next all-reduce without a partner); its trace goes beside rank 0's and is never read
		trace = f"{spec['trace']}.rank{rank}" if spec.get("trace") else None
		spec = {**spec, "progress": None, "trace": trace}
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
		done = Path(spec["deps"]).with_name("deps.done.json")
		if rank == 0:
			extra = [x for x in spec.get("extra_packages") or [] if isinstance(x, str)]
			err, dirs = install(Path(workdir), Path(spec["deps"]), repo=bool(spec.get("install_repo")),
				entrypoint=spec.get("entrypoint"), extra=extra, uv=bool(spec.get("install_uv")),
				pyg=bool(spec.get("install_pyg")), build=bool(spec.get("install_exec")),
				torch_note=bool(spec.get("torch_note")),
				sdists=tuple(x for x in spec.get("install_sdists") or () if isinstance(x, str)))  # fmt: skip
			_write(str(done), {"error": err, "dirs": [str(d) for d in dirs]})
		else:
			# torchrun: rank 0 installs once into the shared prefix; the other ranks wait for it and
			# put the same directories on their path (never a second pip into files rank 0 runs from)
			while not done.exists():
				time.sleep(0.5)
			got = json.loads(done.read_text())
			err, dirs = got.get("error"), [Path(d) for d in got.get("dirs", [])]
		if err:
			_write(out_path, {"status": "error", "error_code": "missing_package.install",
				"error": f"{_what_installed(spec)} didn't install: {err}", "summary": {}})  # fmt: skip
			return 0
		for i, d in enumerate(dirs, 1):
			sys.path.insert(i, str(d))  # after the script's own files, before the image
		parts = [*(str(d) for d in dirs), os.environ.get("PYTHONPATH") or ""]
		os.environ["PYTHONPATH"] = os.pathsep.join(p for p in parts if p)
	if spec.get("install_repo") or spec.get("entry_path"):
		# runner.entrypoint_path (C, Nov 2: `import image_classification` beside a scope-entrypoint main.py) and
		# probes.repo_install: as `python path/to/script.py` does, the script's own folder comes first, so it
		# imports the modules beside it (OGB arxiv's gnn.py imports its logger.py; the full run did, the probe
		# didn't: a-all3-8, Nov 2); after the installs, so the deps keep their place behind the repo's own files
		entry_dir = os.path.normpath(os.path.dirname(os.path.join(workdir, spec["entrypoint"])))
		if entry_dir != os.path.normpath(workdir):
			sys.path.insert(0, entry_dir)
	sys.argv = [spec["entrypoint"], *spec.get("args", [])]
	if spec.get("seed") is not None:
		_seed(int(spec["seed"]), force=bool(spec.get("seed_force")))
	rec = Recorder(
		warmup=spec["warmup"],
		timed=spec["timed"],
		profile=spec.get("profile", False),
		trace_path=spec.get("trace"),
		progress_path=spec.get("progress"),
		eval_batches=spec.get("eval_batches", 2),
		alternate_sync=bool(spec.get("alternate_sync")),
		generate_sample=int(spec.get("generate_sample") or 0),
		generate_calls=int(spec.get("generate_calls") or 0),
		lr_range=spec.get("lr_range") or None,  # D's range test, behind probes.lr_range
		eval_units=int(spec.get("eval_units") or 0),  # probes.eval_units
		step_clock=bool(spec.get("step_clock")),  # probes.step_clock
	).install()
	if spec.get(
		"dataset_cache"
	):  # probes.dataset_cache: torchvision's datasets from the volume, when mounted
		from .dataset_cache import CACHE_DIR, patch

		if os.path.isdir(CACHE_DIR):
			rec._importer.on("torchvision.datasets.utils", lambda mod: patch(mod, CACHE_DIR))

	status, code, error = "ok", None, None
	too_few = "script finished after {n} optimizer steps; a probe needs at least 2"

	def measured() -> bool:
		"""Enough to keep: two optimizer steps, or (probes.eval_units) an eval that finished on its own
		after at least one measured generate call or scoring batch (gsm8k at --limit 16: two calls, then done)."""
		if len(rec.steps) >= 2:
			return True
		# an eval only: a script that took an optimizer step is a training run, judged by its steps
		return bool(rec.eval_units) and not rec.steps and (rec._gen_calls_done >= 1 or bool(rec._fwd))

	try:
		runpy.run_path(os.path.join(workdir, spec["entrypoint"]), run_name="__main__")
		if not measured():
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
		elif not measured():
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
		msg = c.message
		local = (
			_local_missing(exc, workdir, spec)
			if code == "missing_package" and spec.get("entry_path")
			else None
		)
		if local:  # the repo's own module, not a package to install (C, Nov 2: Boarding said "add it to requirements.txt")
			code, msg = "missing_module.local", local
		error = f"{msg}. {detail}" + (f" at {where}" if where else "") + f", {when}"
	finally:
		rec.uninstall()

	summary = rec.summary()
	if spec.get("lr_range"):
		summary["lr_range_error"] = rec.lr_range_error
		# per step: was its loss not finite (the series holds None for both that and no loss read)
		summary["lr_range_nonfinite"] = [int(bool(st.get("nonfinite"))) for st in rec.steps]
		# optimizers that stepped other weights (the ramp sets one), the step a skipping scaler stalled at, and
		# whether the script stepped another optimizer over the ramp's weights instead (it rebuilt its own)
		summary["lr_range_other_optimizers"] = rec.range_other_optimizers()
		summary["lr_range_stalled"] = rec.lr_range_stalled
		summary["lr_range_replaced"] = int(rec.lr_range_replaced)
		# each step's update size, the sampled weights' summed change (probes/lr_range.py follows_rate)
		summary["lr_range_update"] = [st.get("range_upd") for st in rec.steps]
	# the training set's longest row (B, Oct 30), read after the hooks are gone so nothing is timed
	try:
		from preflight.recorder.dataset_len import measure_bounded

		if status == "ok" and rec._train_dataset is not None:  # a failed probe writes its result now
			summary.update(measure_bounded(rec._train_dataset))
	except Exception as exc:
		print(f"[preflight] dataset lengths not read: {type(exc).__name__}", file=sys.stderr, flush=True)
	env = detect_env()
	flops, util = mfu(summary, env)
	summary["flops_per_step"] = flops
	summary["mfu"] = util
	_write(out_path, {"status": status, "error_code": code, "error": error, "summary": summary, "env": env})
	return 0


if __name__ == "__main__":
	sys.exit(main(sys.argv[1]))
