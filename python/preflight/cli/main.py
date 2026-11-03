"""preflight: map a script, or measure it on this machine and write its flight record.

	preflight map train.py                      # the script map, as JSON
	preflight run train.py -- --batch_size 16   # probes on this machine's GPU (or CPU)
	preflight run path/to/project --entrypoint train.py --out records/
	OPENAI_API_KEY=... preflight run agent_eval.py --agent-max-usd 5   # 30 of its tasks, your key
	preflight watch records/<share>.json train.py -- --epochs 3   # the real run, against its bands
	preflight demo                              # a 60-second first run: a tiny script, probed here

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


# `preflight demo`: a small real training loop with knobs worth turning (batch size, learning rate,
# loader workers), quick on a CPU and on any GPU, so a fresh install has something to probe at once
DEMO_SCRIPT = """\"\"\"Preflight's demo: a small MLP on synthetic data. Probe it with `preflight run train.py`.\"\"\"
import argparse

import torch
from torch.utils.data import DataLoader, TensorDataset

p = argparse.ArgumentParser()
p.add_argument("--batch_size", type=int, default=64)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--epochs", type=int, default=3)
args = p.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
x = torch.randn(20_000, 256)
y = (x[:, :8].sum(dim=1, keepdim=True) > 0).float()
loader = DataLoader(TensorDataset(x, y), batch_size=args.batch_size, shuffle=True, num_workers=0)
model = torch.nn.Sequential(
	torch.nn.Linear(256, 1024), torch.nn.GELU(), torch.nn.Linear(1024, 1024), torch.nn.GELU(), torch.nn.Linear(1024, 1)
).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
loss_fn = torch.nn.BCEWithLogitsLoss()
for epoch in range(args.epochs):
	for xb, yb in loader:
		xb, yb = xb.to(device), yb.to(device)
		loss = loss_fn(model(xb), yb)
		opt.zero_grad()
		loss.backward()
		opt.step()
	print(f"epoch {epoch}: loss {loss.item():.4f}")
"""


def _write_demo(where: Path) -> Path:
	"""The demo script in its own folder (made if missing); an existing train.py there is kept."""
	where.mkdir(parents=True, exist_ok=True)
	script = where / "train.py"
	if script.is_symlink():
		raise InvalidSubmission(f"{script} is a link; the demo writes only a plain file")
	if not script.exists():
		script.write_text(DEMO_SCRIPT, encoding="utf-8")
	return script


def _predict(rec, ctx):
	"""The record with the engine's predictions, as the hosted runner fills them (runner/pipeline.py run_job), when the
	engine imports here (numpy installed). D found `preflight run` never filled them (Nov 2), so `preflight watch` had no
	bands. A failed or slow fill keeps the measured record and says so; a prediction never costs the measurements."""
	try:
		import numpy  # noqa: F401  (the engine's one requirement beyond the schema)
	except ImportError:
		print("preflight: predictions skipped: the engine needs numpy (pip install numpy)", file=sys.stderr)
		return rec
	from preflight.runner.pipeline import with_predictions

	out, failed = with_predictions(rec, ctx, enabled=True)
	if failed:
		print("preflight: predictions unavailable; the measurements stand", file=sys.stderr)
	return out


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
	r.add_argument(
		"--agent-max-usd",
		type=float,
		help="agent and API runs: sample 30 tasks through a local proxy holding your OPENAI_API_KEY, "
		"spending at most this many dollars of it",
	)
	r.add_argument(
		"--no-predictions",
		action="store_true",
		help="write only the measurements (by default the engine fills the predictions, so `preflight watch` has "
		"bands to check the real run against)",
	)
	r.epilog = "Arguments for the script go after --, e.g. preflight run train.py -- --batch_size 16"
	w = sub.add_parser(
		"watch", help="run the script for real, streaming a live record against its predictions"
	)
	w.add_argument("record", type=Path, help="the flight record whose predictions the run is checked against")
	w.add_argument("path", type=Path, help="the script to run")
	w.add_argument("--out", type=Path, default=Path("records"))
	w.add_argument(
		"--total-steps", type=int, help="optimizer steps in the whole run, for the time.total check"
	)
	w.add_argument("--window", type=int, help="steps per update (default 20)")
	dm = sub.add_parser(
		"demo", help="write a small training script and probe it here: a first run in about a minute"
	)
	dm.add_argument("--dir", type=Path, default=Path("preflight-demo"), help="where the script and record go")
	argv = list(sys.argv[1:] if argv is None else argv)
	script_args = argv[argv.index("--") + 1 :] if "--" in argv else []
	ns = ap.parse_args(argv[: argv.index("--")] if "--" in argv else argv)
	if ns.cmd == "demo":
		try:
			import torch  # noqa: F401
		except ImportError:
			# Not "preflight[probe]" from PyPI: that name there is an unrelated package.
			print("preflight: the demo trains with PyTorch: pip install torch (or "
				'"preflight[probe] @ git+https://github.com/uchuuronin/preflight"), then `preflight demo` again',
				file=sys.stderr)  # fmt: skip
			return 2
		fresh = not (ns.dir / "train.py").exists()
		try:
			script = _write_demo(ns.dir)
		except InvalidSubmission as exc:
			print(f"preflight: {exc}", file=sys.stderr)
			return 2
		what = "the demo script" if fresh else "the train.py already there"
		print(
			f"preflight: probing {what}, {script}, on this machine (about a minute on a CPU)", file=sys.stderr
		)
		ns = argparse.Namespace(cmd="run", path=script, entrypoint=None, out=ns.dir / "records",
			install_requirements=False, require_gpu=False, agent_max_usd=None, demo=True, no_predictions=False)  # fmt: skip
	if ns.cmd == "watch":
		from preflight.recorder import inflight

		extra = [*(["--total-steps", str(ns.total_steps)] if ns.total_steps else []),
			*(["--window", str(ns.window)] if ns.window else [])]  # fmt: skip
		return inflight.main([str(ns.record), str(ns.out), str(ns.path), *extra, "--", *script_args])
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
			records=store,
			runner=local_runner,
			reader=store,
			install_requirements=ns.install_requirements,
			agent=_agent_sampler(ns.agent_max_usd),
		)
		try:
			rec = run_pipeline({"source": source}, ctx, d)
			if not getattr(ns, "no_predictions", False):
				rec = _predict(rec, ctx)
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
	out = {"record": str(ns.out / f"{rec.share_token}.json"), "lens": rec.job.lens, "backend": backend,
		"gpu": gpu, "probes": summary}  # fmt: skip
	if rec.agent_sample is not None:
		out["agent_tasks"] = len(rec.agent_sample.tasks)
		out["agent_sample_usd"] = round(sum(t.cost_usd or 0 for t in rec.agent_sample.tasks), 4)
	print(json.dumps(out, indent=2))
	if getattr(ns, "demo", False):
		# The web app opens records by share link only, so point at the hosted check-in, not this file.
		print(f"preflight: the flight record is {out['record']}: the probes' measurements and findings. Probe "
			"your own script with `preflight run train.py`, or check it in at https://preflight-cqn.pages.dev "
			"to fly the whole run.", file=sys.stderr)  # fmt: skip
	if rec.job.lens == "agent":
		if rec.agent_sample is None:
			print("preflight: an agent run; add --agent-max-usd 5 (and OPENAI_API_KEY) to sample 30 of its "
				"tasks with your key", file=sys.stderr)  # fmt: skip
			return 2
		return 0  # no GPU probes in an agent run
	if ns.require_gpu and backend not in ("cuda", "rocm"):
		print(f"preflight: the probes ran on {backend or 'nothing'}, not a GPU", file=sys.stderr)
		return 3
	return 0


def _agent_sampler(max_usd: float | None):
	"""Agent and API scripts: 30 tasks through a local proxy that holds your key (the script gets a
	dummy one), only when you name a spending cap. The provider is PREFLIGHT_AGENT_UPSTREAM, else
	your OPENAI_BASE_URL, else OpenAI."""
	key = os.environ.get("OPENAI_API_KEY")
	if not key or not max_usd or max_usd <= 0:
		return None
	from preflight.runner.agent_jobs import local_sampler

	upstream = (os.environ.get("PREFLIGHT_AGENT_UPSTREAM") or os.environ.get("OPENAI_BASE_URL")
		or "https://api.openai.com/v1")  # fmt: skip
	print(f"preflight: an agent run samples tasks through {upstream} with your key, at most ${max_usd:.2f}",
		file=sys.stderr)  # fmt: skip
	return local_sampler(upstream, key, max_usd_total=max_usd)


def _run_caches_when_root() -> Path | None:
	"""Run as root (a cloud box), probes drop to an unprivileged uid with a throwaway home, so each
	would download the model and the requirements again. Give this run one fresh probe uid and one
	fresh, empty set of caches it owns: nothing is shared with earlier runs (whose probes could
	have planted links or poisoned files), and root never walks or chowns anything a probe wrote.
	Returns the run's directory, which the caller removes when the run ends."""
	if getattr(os, "geteuid", lambda: -1)() != 0:  # no euid on Windows: never root there (D, Nov 3)
		return None
	from preflight.probes.local import uid_can_run_this_python

	if not uid_can_run_this_python():
		print(f"preflight: running probes as root: an unprivileged uid can't reach {sys.executable} and its "
			"packages (install Preflight where other users can read it to probe as one)", file=sys.stderr)  # fmt: skip
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
