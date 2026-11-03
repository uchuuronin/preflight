"""Ground truth for one zoo script on this machine's GPU: a bare run, then a recorder run.

	python -m preflight.recorder.ground_truth zoo/finetune_lora_hf out/gt.json --bare-steps 210
	python -m preflight.recorder.ground_truth zoo/pretrain_nanogpt out/gt.json --bare-steps 1000 \\
		--recorder-steps 200 --install-requirements
	python -m preflight.recorder.ground_truth unseen/s3 out/gt.json --bare-steps 500 --entrypoint main.py \\
		--install-requirements -- --epochs 1

The bare run (recorder/bare.py) has a CUDA event at each optimizer step and nothing else, so it
shows how the script really runs and where it really fails (the LoRA script's eval OOM included).
The recorder run is a probe that alternates clean and synced steps (Recorder alternate_sync), and
recorder/overhead.py pairs the two into the recorder's overhead. The same code runs on Modal's L4
(runner/ops_app.py::ground_truth) and on the AMD droplet (amd-run.yml), so the two are comparable.

The zoo scripts, and Monday's unseen validation scripts (third-party code, run as ops on our own
GPUs, never through the visitor path; on Modal the weights volume is mounted read-only for them).
As root (a cloud box) the bare run gets
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

# every zoo script that runs on a GPU (agent_eval_spark is an API run: its full run is the ops
# agent_sample with a larger cap); any other script comes by directory with its entrypoint
ZOO_SCRIPTS = {"finetune_lora_hf", "pretrain_nanogpt", "grpo_trl", "eval_sweep_reasoning", "gnn_pyg",
	"retrieval_st", "serving_vllm", "vision_timm", "torchtitan_debug"}  # fmt: skip
ENTRYPOINTS = ("train.py", "main.py", "serve.py", "run.py")  # what a script dir runs, first found


def plan(files: dict[str, str], entrypoint: str | None = None) -> dict:
	"""What a full run of these files needs: its entrypoint, lens, sandbox image ("probe", "serving"
	for vLLM, "titan" for torchtitan's torch) and a serving probe's sample size. An agent run has no
	GPU ground truth (its full run is the ops agent_sample)."""
	import re

	from preflight.harness.static import map_static
	from preflight.runner.pipeline import EVAL_PROBE_CALLS, SERVING_PROBE_PROMPTS, _uses_vllm

	entry = entrypoint_of(files, entrypoint)
	sm = map_static(files, entrypoint=entry)
	titan_re = re.compile(r"^\s*(from|import)\s+torchtitan\b|run_module\(\s*[\"']torchtitan\b", re.M)
	titan = any(titan_re.search(t) for k, t in files.items() if k.endswith(".py"))
	image = "serving" if _uses_vllm(sm, files) else "titan" if titan else "probe"
	return {"entrypoint": entry, "lens": sm.lens, "image": image,
		"generate_sample": SERVING_PROBE_PROMPTS if sm.lens == "serving" else 0,
		"generate_calls": EVAL_PROBE_CALLS if sm.lens == "eval" else 0}  # fmt: skip


def entrypoint_of(files: dict[str, str], wanted: str | None = None) -> str:
	"""The file a ground truth runs: the one named, else the first of ENTRYPOINTS present."""
	if wanted:
		if wanted not in files:
			raise ValueError(f"the entrypoint {wanted!r} is not among the script's files")
		return wanted
	hit = next((e for e in ENTRYPOINTS if e in files), None)
	if hit is None:
		raise ValueError(f"no {', '.join(ENTRYPOINTS)} among the script's files: name the entrypoint")
	return hit


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
	work: Path, bare_steps: int, *, install_requirements: bool, caches: Path, env: dict, timeout_s: float,
	entrypoint: str = "train.py", args: tuple[str, ...] = (), install_repo: bool = False,
	install_opts: dict | None = None,
) -> dict:  # fmt: skip
	env = {**os.environ, **env}
	if install_requirements:
		from preflight.recorder.requirements import install

		# into this run's own prefix and pip cache, never the probes' (owned by the probe uid on a
		# root box): install() reads both from the environment
		saved = {
			k: os.environ.pop(k) for k in ("PREFLIGHT_DEPS_DIR", "PREFLIGHT_PIP_CACHE") if k in os.environ
		}
		try:
			repo_kw = {"repo": True, "entrypoint": entrypoint} if install_repo else {}
			how = {
				k: v
				for k, v in (install_opts or {}).items()
				if k in ("uv", "pyg", "sdists", "build", "torch_stop")
			}  # the installer's own
			err, dirs = install(work, caches / "deps", log=lambda m: print(m, flush=True), **repo_kw, **how)  # fmt: skip
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
		str(work / entrypoint),
		str(out),
		"--steps",
		str(bare_steps),
		*(["--", *args] if args else []),
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


def _bare_hf_home(caches: Path) -> dict:
	"""The bare run's own HF_HOME, with the probes' prefetched weights linked in. On a root box the
	bare run is root and the recorder's probe is an unprivileged uid: sharing HF_HOME left root's
	dataset locks there, and the probe failed on them (Oct 31, a-gt-lora-l4-4: "Permission denied:
	/tmp/hf/datasets/..._tmp_hf_datasets_yahma___alpaca-cleaned....lock")."""
	home = os.environ.get("HF_HOME")
	if not home:
		return {}
	own = caches / "hf"
	(own / "hub").mkdir(parents=True, exist_ok=True)
	hub = Path(home) / "hub"
	if hub.is_dir():
		for entry in hub.iterdir():
			if entry.name.startswith(("models--", "datasets--")) and not (own / "hub" / entry.name).exists():
				(own / "hub" / entry.name).symlink_to(entry.resolve(), target_is_directory=True)
	return {"HF_HOME": str(own)}


def measure(
	script_dir: Path,
	*,
	bare_steps: int,
	recorder_steps: int = 60,
	warmup: int = 5,
	install_requirements: bool = False,
	budget_s: float = BUDGET_S,
	bare_env: dict | None = None,
	files: dict[str, str] | None = None,
	entrypoint: str | None = None,
	args: tuple[str, ...] = (),
	generate_sample: int = 0,
	generate_calls: int = 0,
	install_repo: bool = False,
	install_opts: dict | None = None,
) -> dict:
	"""Both runs and the paired overhead, as one dict (what out/ground_truth_<script>.json holds).
	bare_env: environment overrides for the bare run only (its own caches on a root box).
	files: the script's files, when the machine that runs it doesn't have script_dir (Modal's probe
	image has no zoo: the ops app's local entrypoint reads them from the checkout and sends them)."""
	from preflight.probes.local import ProbeSpec, run_probe
	from preflight.recorder.overhead import report
	from preflight.runner.sources import _read_tree

	files = _read_tree(script_dir) if files is None else files
	entrypoint = entrypoint_of(files, entrypoint)
	args = tuple(args)
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
			env={**_bare_hf_home(caches), **(bare_env or {})}, timeout_s=budget_s * BARE_SHARE,
			entrypoint=entrypoint, args=args, install_repo=install_repo, install_opts=install_opts)  # fmt: skip
		bare_wall = time.monotonic() - t0
		_say(
			f"bare run ended ({bare.get('ended') or 'failed'}) after {bare_wall:.0f} s, {bare.get('steps')} steps"
		)
		left = max(120.0, budget_s - bare_wall - 60)  # a minute kept for writing the result
		_say(f"recorder run: {warmup} warmup + {recorder_steps} timed steps, at most {left:.0f} s")
		# the recorder run is the probe the pipeline would run: a serving script's sampled generate call
		r = run_probe(ProbeSpec(probe_id="gt", files=files, entrypoint=entrypoint, args=list(args), warmup=warmup,
			timed=recorder_steps, timeout_s=left, alternate_sync=not (generate_sample or generate_calls),
			install_requirements=install_requirements, install_repo=install_repo, generate_sample=generate_sample,
			install_uv=bool((install_opts or {}).get("uv")), install_pyg=bool((install_opts or {}).get("pyg")),
			install_exec=bool((install_opts or {}).get("build")), torch_stop=bool((install_opts or {}).get("torch_stop")),
			dataset_cache=bool((install_opts or {}).get("dataset_cache")),
			install_sdists=tuple((install_opts or {}).get("sdists", ())),
			eval_units=int((install_opts or {}).get("eval_units") or 0),
			generate_calls=generate_calls))  # fmt: skip
		_say(f"recorder run ended ({r.probe.status}) after {r.probe.steps} steps")
	finally:
		shutil.rmtree(work, ignore_errors=True)
		shutil.rmtree(caches, ignore_errors=True)
	log = [s.model_dump() for s in r.probe.step_log]
	over = report(log, bare.get("walls_s") or [], warmup=warmup) if r.probe.status == "ok" else None
	return {"script": script_dir.name, "entrypoint": entrypoint, "args": list(args),
		"gpu": r.env.gpu.name if r.env and r.env.gpu else None,
		"backend": r.env.backend if r.env else None, "bare": {**bare, "wall_s": round(bare_wall, 1)},
		"recorder": {"status": r.probe.status, "error": r.probe.error, "error_code": r.probe.error_code,
			"steps": r.probe.steps, "p50_s": r.probe.step_time.p50_s if r.probe.step_time else None,
			"peak_allocated_bytes": r.probe.memory.peak_allocated_bytes if r.probe.memory else None,
			"seconds": r.probe.cost.gpu_seconds if r.probe.cost else None,
			"by_phase": r.probe.step_time.by_phase if r.probe.step_time else {},
			"rl": r.probe.rl.model_dump(mode="json") if r.probe.rl else None,
			"generation": r.probe.generation.model_dump(mode="json") if r.probe.generation else None,
			"extras": r.probe.extras, "step_log": log, "log_tail": r.log_tail[-1500:],
			# the whole probe, so a record can be built from the run (runner/gt_record.py)
			"probe": r.probe.model_dump(mode="json", exclude={"kernels", "trace"}),
			"env": r.env.model_dump(mode="json", exclude_none=True) if r.env else None},
		"overhead": over}  # fmt: skip


def brief(res: dict) -> dict:
	"""The result without the per-step lists, for a log line."""
	out = {k: v for k, v in res.items() if k not in ("bare", "recorder")}
	out["bare"] = {k: v for k, v in res["bare"].items() if k != "walls_s"}
	out["recorder"] = {
		k: v for k, v in res["recorder"].items() if k not in ("step_log", "log_tail", "rl", "probe")
	}
	rl = res["recorder"].get("rl")
	if rl:  # the lists stay in the json; the log line gets their summary
		rates = rl.get("pass_rate_per_prompt") or []
		out["recorder"]["rl"] = {"group_size": rl.get("group_size"), "prompts": len(rates),
			"truncation_rate": rl.get("truncation_rate"), "zero_spread_fraction": rl.get("zero_spread_fraction")}  # fmt: skip
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
	ap.add_argument("--entrypoint", help=f"the file to run (default: the first of {', '.join(ENTRYPOINTS)})")
	ap.add_argument(
		"--generate-sample", type=int, default=0, help="a serving script: prompts its probe keeps"
	)
	argv = list(sys.argv[1:] if argv is None else argv)
	script_args = argv[argv.index("--") + 1 :] if "--" in argv else []
	ns = ap.parse_args(argv[: argv.index("--")] if "--" in argv else argv)
	if not ns.script_dir.is_dir():
		print(f"{ns.script_dir} is not a directory", file=sys.stderr)
		return 2
	from preflight.runner.sources import _read_tree

	try:
		p = plan(_read_tree(ns.script_dir), ns.entrypoint)
		if not ns.generate_sample:  # a serving script's probe samples, as on Modal
			ns.generate_sample = p["generate_sample"]
		generate_calls = p["generate_calls"]
	except ValueError as exc:
		print(f"{ns.script_dir}: {exc}", file=sys.stderr)
		return 2
	from preflight.cli.main import _run_caches_when_root

	# as root the probe gets fresh caches of its own uid; measure() gives the bare run (root) its own
	bare_env, own = {}, None
	run = _run_caches_when_root()
	try:
		res = measure(ns.script_dir, bare_steps=ns.bare_steps, recorder_steps=ns.recorder_steps,
			warmup=ns.warmup, install_requirements=ns.install_requirements, bare_env=bare_env,
			budget_s=ns.budget_min * 60, entrypoint=ns.entrypoint, args=tuple(script_args),
			generate_sample=ns.generate_sample, generate_calls=generate_calls)  # fmt: skip
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
