"""preflight: map a script, or measure it on this machine and write its flight record.

	preflight map train.py                      # the script map, as JSON
	preflight run train.py -- --batch_size 16   # probes on this machine's GPU (or CPU)
	preflight run path/to/project --entrypoint train.py --out records/

`run` uses the same pipeline as the hosted app, with probes on the local device
and no GPU bill; the record lands in --out as <share_token>.json.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from preflight.harness.static import map_static
from preflight.probes.local import InvalidSubmission
from preflight.runner.pipeline import Deps, _Ctx, local_runner, run_pipeline
from preflight.runner.records import LocalRecordWriter
from preflight.runner.sources import _read_tree


def _files(path: Path, entrypoint: str | None) -> tuple[dict[str, str], str]:
	if path.is_file():
		return {path.name: path.read_text(encoding="utf-8")}, path.name
	if path.is_dir():
		files = _read_tree(path)
		entry = entrypoint or next((p for p in ("train.py", "main.py") if p in files), None)
		if entry is None:
			raise InvalidSubmission("say which file to run with --entrypoint")
		return files, entry
	raise InvalidSubmission(f"{path} does not exist")


def main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser(prog="preflight", description=__doc__.split("\n\n")[0])
	sub = ap.add_subparsers(dest="cmd", required=True)
	m = sub.add_parser("map", help="print the script map")
	m.add_argument("path", type=Path)
	m.add_argument("--entrypoint")
	r = sub.add_parser("run", help="probe the script here and write its flight record")
	r.add_argument("path", type=Path)
	r.add_argument("--entrypoint")
	r.add_argument("--out", type=Path, default=Path("records"))
	r.add_argument(
		"--install-requirements",
		action="store_true",
		help="pip-install the script's requirements.txt in each probe (packages this Python lacks only)",
	)
	r.add_argument(
		"--require-gpu", action="store_true", help="fail if the probes ran on the CPU (no GPU was visible)"
	)
	r.epilog = "Arguments for the script go after --, e.g. preflight run train.py -- --batch_size 16"
	argv = list(sys.argv[1:] if argv is None else argv)
	script_args = argv[argv.index("--") + 1 :] if "--" in argv else []
	ns = ap.parse_args(argv[: argv.index("--")] if "--" in argv else argv)
	try:
		files, entry = _files(ns.path, ns.entrypoint)
		if ns.cmd == "map":
			print(map_static(files, entrypoint=entry).model_dump_json(indent=2, exclude_none=True))
			return 0
		source = {"kind": "cli", "files": files, "entrypoint": entry, "args": script_args}
		run_dir = _run_caches_when_root()
		ctx = _Ctx()
		store = LocalRecordWriter(ns.out)
		d = Deps(
			records=store, runner=local_runner, reader=store, install_requirements=ns.install_requirements
		)
		try:
			rec = run_pipeline({"source": source}, ctx, d)
		finally:
			if run_dir is not None:
				import shutil

				shutil.rmtree(run_dir, ignore_errors=True)  # fd-based on Linux: never follows a planted link
		store.put(rec)
	except InvalidSubmission as exc:
		print(f"preflight: {exc}", file=sys.stderr)
		return 2
	summary = [
		{"probe": p.id, "status": p.status, "knobs": p.knobs, "step_s": p.step_time.p50_s if p.step_time else None,
			"peak_bytes": p.memory.peak_allocated_bytes if p.memory else None, "error": p.error}
		for p in rec.probes
	]  # fmt: skip
	backend = rec.env.backend if rec.env is not None else None
	gpu = rec.env.gpu.name if rec.env is not None and rec.env.gpu is not None else None
	print(json.dumps({"record": str(ns.out / f"{rec.share_token}.json"), "lens": rec.job.lens,
		"backend": backend, "gpu": gpu, "probes": summary}, indent=2))  # fmt: skip
	if ns.require_gpu and backend not in ("cuda", "rocm"):
		print(f"preflight: the probes ran on {backend or 'nothing'}, not a GPU", file=sys.stderr)
		return 3
	return 0


def _run_caches_when_root() -> Path | None:
	"""Run as root (a cloud box), probes drop to an unprivileged uid with a throwaway home, so each
	would download the model and the requirements again. Give this run one fresh probe uid and one
	fresh, empty set of caches it owns: nothing is shared with earlier runs (whose probes could
	have planted links or poisoned files), and root never walks or chowns anything a probe wrote.
	Returns the run's directory, which the caller removes when the run ends."""
	if os.geteuid() != 0:
		return None
	import secrets
	import tempfile

	uid = 61_000 + secrets.randbelow(4_000)
	run = Path(tempfile.mkdtemp(prefix="preflight-run-", dir="/var/tmp"))  # root's, 0700
	os.chmod(run, 0o711)  # the probe uid passes through to its own dirs, can't list or rename
	for var, name in (("HF_HOME", "hf"), ("PREFLIGHT_PIP_CACHE", "pip"), ("PREFLIGHT_DEPS_DIR", "deps")):
		d = run / name
		d.mkdir(mode=0o700)
		os.chown(d, uid, uid)  # an empty dir root just made: nothing to follow
		os.environ[var] = str(d)
	os.environ["PREFLIGHT_PROBE_UID"] = str(uid)
	return run


if __name__ == "__main__":
	raise SystemExit(main())
