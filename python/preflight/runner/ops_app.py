"""Operations on the runner, kept out of the deployed app: startup check, prefetch, smoke, all `modal run`.

	modal run --env main python/preflight/runner/ops_app.py::startup_check   # before any deploy
	modal run --env main python/preflight/runner/ops_app.py::prefetch
	modal run --env main python/preflight/runner/ops_app.py::smoke --gpus L4

They live in their own app ("preflight-runner-ops") because `modal run` starts a temporary copy of
whatever app the file defines. A temporary copy of runner/modal_app.py includes the job store
(min_containers=1), which claims the shared job state and locks the deployed store out (Oct 28).
This app has no job store and no schedule, so running it never touches the deployed one; it
reuses the runner's images, so what it checks is what the runner runs.
"""

import contextlib
import json
import os
import uuid
from pathlib import Path

import modal

from preflight.runner.modal_app import (
	APP_NAME,
	REMOTE_PKG,
	SERVICE_NAME,
	WEIGHTS_DIR,
	ZOO_MODELS,
	_read_only,
	image_tag,
	probe_image,
	records_vol,
	serving_image,
	titan_image,
	weights_vol,
	worker_image,
)
from preflight.runner.probe_user import PROBE_USER_IMPORTS
from preflight.runner.probe_user import probe_user_ok as _probe_user_ok

ops = modal.App(APP_NAME + "-ops")


@ops.function(image=probe_image, volumes={WEIGHTS_DIR: weights_vol}, timeout=3600)
def prefetch(models: str = "") -> list[str]:
	"""Download weights and tokenizers into the read-only weights volume (zoo models by default).

	models: comma-separated Hub ids (a plain string: `modal run` can't parse list arguments)."""
	from huggingface_hub import snapshot_download

	done = []
	for repo in [m.strip() for m in models.split(",") if m.strip()] or ZOO_MODELS:
		if repo.startswith("dataset:"):  # a dataset repo: all its files (the zoo's are small)
			snapshot_download(
				repo.removeprefix("dataset:"), repo_type="dataset", cache_dir=f"{WEIGHTS_DIR}/hf/hub"
			)
		else:
			snapshot_download(
				repo,
				cache_dir=f"{WEIGHTS_DIR}/hf/hub",
				allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "tokenizer*", "*.tiktoken"],
			)
		done.append(repo)
	weights_vol.commit()
	return done


# ---------------------------------------------------------------- startup check (no GPU)

_CHECK_LOOP = """
import torch
m = torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.GELU(), torch.nn.Linear(32, 1))
opt = torch.optim.SGD(m.parameters(), lr=0.05)
for _ in range(12):
	loss = m(torch.randn(8, 16)).pow(2).mean()
	opt.zero_grad()
	loss.backward()
	opt.step()
"""


def _load_standalone(src: Path | None = None) -> str:
	"""Load runner/modal_app.py the way Modal loads a deployed file: on its own, from a shallow path
	(/root/modal_app.py there), as module `modal_app`. Repo-path lookups at import fail here."""
	import importlib.util
	import shutil
	import tempfile

	import preflight.runner as runner

	src = src or Path(runner.__file__).with_name("modal_app.py")
	shallow = Path(tempfile.mkdtemp(prefix="mc-", dir="/"))  # one level under /, like /root
	dst = shallow / "modal_app.py"
	shutil.copy(src, dst)
	try:
		spec = importlib.util.spec_from_file_location("modal_app", dst)
		mod = importlib.util.module_from_spec(spec)
		spec.loader.exec_module(mod)
		zoo = getattr(mod, "ZOO_SRC", mod.ROOT / "zoo")
		missing = [str(d) for d in (mod.PKG, zoo) if not Path(d).is_dir()]
		flags = getattr(mod, "FLAGS_SRC", None)
		if flags is not None and not Path(flags).is_file():
			missing.append(str(flags))
		if missing:  # an image resolved here (a sandbox's) would fail to build: say so now
			raise FileNotFoundError(f"image source folders missing in the container: {missing}")
		return f"{dst} (module modal_app, tag {mod.image_tag()}, PKG {mod.PKG}, zoo {zoo})"
	finally:
		shutil.rmtree(shallow, ignore_errors=True)


@ops.function(image=worker_image, timeout=300)
def startup_check_worker() -> dict:
	"""What a job-store or worker container does when it starts, without touching the shared state:
	import this module as Modal loads it, build a job store and round-trip its snapshot, take a
	job, and read, map and skeleton the LoRA zoo sample."""
	import importlib
	import tempfile

	from preflight.harness.skeleton import skeleton_record
	from preflight.harness.static import map_static
	from preflight.runner.modal_app import _container_environment, deployed_here
	from preflight.runner.queue import InMemoryJobStore
	from preflight.runner.remote_store import DirBlobs, StoreMethods
	from preflight.runner.sources import ZOO, fetch

	# everything run_worker imports when it starts a job
	worker_path = ["preflight.runner.pipeline", "preflight.runner.records", "preflight.runner.sandbox_runner",
		"preflight.runner.worker", "preflight.runner.wire", "preflight.runner.record_jobs",
		"preflight.harness.spark", "preflight.harness.mapper", "preflight.probes.gpu_choice", "preflight.probes.plan"]  # fmt: skip
	imported = [importlib.import_module(n).__name__ for n in worker_path]
	# the job store's guard, seen from this ops container: it must find the deployed runner (or say
	# it isn't deployed) and must answer "not the deployed app" here
	guard_ok, guard_why = deployed_here()
	from preflight.runner.flags import _root, flag

	flags_seen = {
		"root": str(_root()),
		"env": os.environ.get("PREFLIGHT_ENV"),
		"engine.memory": flag("engine.memory"),
	}
	standalone = _load_standalone()
	saved: list[dict] = []
	store = InMemoryJobStore(on_change=saved.append)
	m = StoreMethods(store, blobs=DirBlobs(tempfile.mkdtemp()))
	sub = m.submit({"source": {"kind": "sample", "uri": "sample://zoo/finetune_lora_hf"}}, "check", "check-1")
	restored = InMemoryJobStore.restore(store.snapshot())
	lease = restored.lease("check-worker")
	f = fetch(lease.request["source"])
	rec = skeleton_record(map_static(f.files, entrypoint=f.entrypoint), f.source)
	return {"module": __name__, "file": __file__, "runner_loaded_as": standalone, "zoo": str(ZOO), "zoo_samples": sorted(p.name for p in ZOO.iterdir()),
		"submitted": sub, "saves": len(saved), "lens": rec.job.lens, "knobs": len(rec.knobs),
		"imported": imported, "container_environment": _container_environment(), "flags": flags_seen,
		"job_store_would_claim_here": guard_ok, "guard": guard_why,
		"image_tag": image_tag()}  # fmt: skip


@ops.function(image=probe_image, timeout=600)  # CPU on purpose: proves the probe path without a GPU
def startup_check_probe() -> dict:
	"""What a probe sandbox does, on the CPU: set up the HF cache and the shared deps dir, then run
	a tiny training loop through run_probe as the unprivileged probe uid."""
	import os

	import preflight
	from preflight.probes import remote
	from preflight.probes.local import ProbeSpec, _others_can_reach, run_probe

	pkg = Path(preflight.__file__).resolve().parents[1]
	remote.link_weights()
	remote.deps_dir()
	r = run_probe(
		ProbeSpec(probe_id="check", files={"train.py": _CHECK_LOOP}, entrypoint="train.py", timeout_s=240)
	)
	return {"pkg_root": str(pkg), "reachable_by_probe_uid": _others_can_reach(pkg), "euid": os.geteuid(),
		"probe_uid": os.environ.get("PREFLIGHT_PROBE_UID"), "status": r.probe.status, "error": r.probe.error,
		"error_code": r.probe.error_code, "steps": r.probe.steps,
		"backend": r.env.backend if r.env else None, "log_tail": r.log_tail[-600:]}  # fmt: skip


@ops.function(image=probe_image, timeout=600)
def probe_user_check_probe_image() -> dict:
	from preflight.probes.local import probe_user_check

	return probe_user_check(PROBE_USER_IMPORTS["probe"])


@ops.function(image=serving_image, timeout=600)
def probe_user_check_serving_image() -> dict:
	from preflight.probes.local import probe_user_check

	return probe_user_check(PROBE_USER_IMPORTS["serving"])


@ops.function(image=titan_image, timeout=600)
def probe_user_check_titan_image() -> dict:
	from preflight.probes.local import probe_user_check

	return probe_user_check(PROBE_USER_IMPORTS["titan"])


@ops.function(image=worker_image, timeout=600)
def startup_check_sandbox() -> dict:
	"""What run_worker does for every probe, on a CPU sandbox: the runner's own sandbox factory
	(same image, allowlist, read-only weights, timeouts), then the probe entry run through
	SandboxRunner with a tiny spec. Reports the exception text if any step fails."""
	import traceback

	from preflight.probes.local import ProbeSpec
	from preflight.runner.modal_app import _sandbox_factory
	from preflight.runner.sandbox_runner import ENTRY, SandboxRunner

	owner = modal.App._get_container_app()
	out: dict = {"entry": list(ENTRY)}
	create = _sandbox_factory(owner)
	sb = None
	try:
		sb = create("cpu")
		out["created"] = True
		cmd = "import preflight, torch; print('IMPORT', preflight.__file__, torch.__version__)"
		p = sb.exec("python", "-c", cmd, timeout=120)
		out["import_stdout"] = p.stdout.read()[-500:]
		out["import_stderr"] = p.stderr.read()[-1500:]
		p.wait()
		out["import_exit"] = p.returncode
	except Exception as exc:
		out["created"] = out.get("created", False)
		out["failed"] = f"{type(exc).__name__}: {exc}"[:2000]
		out["traceback"] = traceback.format_exc()[-3000:]
		return out
	finally:
		if sb is not None:  # never leave it billing until its idle timeout
			with contextlib.suppress(Exception):
				sb.terminate()
	with SandboxRunner(create) as runner:
		r = runner(ProbeSpec(probe_id="check", files={"train.py": _CHECK_LOOP}, entrypoint="train.py", timeout_s=240),
			"cpu", should_stop=lambda: False)  # fmt: skip
	out.update(probe_status=r.probe.status, probe_steps=r.probe.steps, probe_error=r.probe.error,
		probe_error_code=r.probe.error_code)  # fmt: skip
	return out


@ops.local_entrypoint()
def startup_check():
	"""Before any deploy: import and start both images' code as Modal runs it, with no GPU.

	modal run --env main python/preflight/runner/ops_app.py::startup_check
	Writes out/startup_check.json; exits non-zero if either side fails."""
	import json
	import sys

	result: dict = {}
	checks = (
		("worker", startup_check_worker),
		("probe", startup_check_probe),
		("sandbox", startup_check_sandbox),
		("probe_user", probe_user_check_probe_image),
		("serving_user", probe_user_check_serving_image),
	)
	for name, fn in checks:
		try:
			result[name] = fn.remote()
		except Exception as exc:  # the remote traceback is in the log above
			result[name] = {"failed": f"{type(exc).__name__}: {exc}"[:2000]}
	w, p, sb = result["worker"], result["probe"], result["sandbox"]
	result["probe_user_ok"] = _probe_user_ok(result["probe_user"], "probe")
	result["serving_user_ok"] = _probe_user_ok(result["serving_user"], "serving")
	result["ok"] = bool(
		"failed" not in w and w.get("lens") == "finetune" and w.get("knobs", 0) > 0 and w.get("saves", 0) >= 1
		and w.get("job_store_would_claim_here") is False and "runner_loaded_as" in w
		and "failed" not in p and p.get("status") == "ok" and p.get("steps", 0) > 0 and p.get("reachable_by_probe_uid")
		and "failed" not in sb and sb.get("import_exit") == 0 and sb.get("probe_status") == "ok"
		and result["probe_user_ok"] and result["serving_user_ok"]
	)  # fmt: skip
	Path("out").mkdir(exist_ok=True)
	Path("out/startup_check.json").write_text(json.dumps(result, indent=2))
	print(json.dumps(result, indent=2), flush=True)
	if not result["ok"]:
		sys.exit(1)


@ops.local_entrypoint()
def limit_check(modal_env: str = ""):
	"""Confirm the demo limit on the deployed runner: one visitor's 4th probe job of the ET day is
	refused with limit.visitor_daily. The three it accepts name a sample that doesn't exist, so the
	worker fails them at fetch with no GPU spent. Writes out/limit_check.json."""
	import json
	import time

	svc = modal.Cls.from_name(APP_NAME, SERVICE_NAME, environment_name=modal_env or None)()
	visitor = f"ops-limit-check-{uuid.uuid4().hex[:8]}"
	request = {"source": {"kind": "sample", "uri": "sample://zoo/limit-check-no-such-script"}}
	out: dict = {"visitor": visitor, "accepted": [], "refused": None}
	for i in range(4):
		try:
			sub = svc.submit.remote(request, visitor, f"limit-check-{i}")
			out["accepted"].append(sub["job_id"])
			print(f"[limit] job {i + 1} accepted: {sub['job_id']}", flush=True)
		except Exception as exc:  # QueueFull, raised in the service and re-raised here
			out["refused"] = {"job": i + 1, "type": type(exc).__name__, "code": getattr(exc, "code", None),
				"retry_after_s": getattr(exc, "retry_after_s", None), "message": str(exc)[:200]}  # fmt: skip
			print(f"[limit] job {i + 1} refused: {out['refused']}", flush=True)
			break
	deadline = time.monotonic() + 300
	ended = {}
	while time.monotonic() < deadline and len(ended) < len(out["accepted"]):  # the accepted ones end
		for j in out["accepted"]:
			st = svc.status.remote(j)
			if st["status"] in ("done", "failed", "cancelled"):
				ended[j] = {"status": st["status"], "error_code": st.get("error_code")}
		time.sleep(5)
	out["accepted_ended"] = ended
	Path("out").mkdir(exist_ok=True)
	Path("out/limit_check.json").write_text(json.dumps(out, indent=2))
	ok = (
		out["refused"] is not None
		and out["refused"]["job"] == 4
		and out["refused"]["code"] == "limit.visitor_daily"
	)
	print(f"[limit] {'PASS' if ok else 'FAIL'}: {json.dumps(out)}", flush=True)
	if not ok:
		raise SystemExit(1)


@ops.local_entrypoint()
def smoke(
	sample: str = "sample://zoo/finetune_lora_hf",
	gpus: str = "L4",
	max_minutes: int = 25,
	modal_env: str = "",
	visitor: str = "smoke",
	notebook: str = "",
	dir: str = "",
	entrypoint: str = "",
	args: str = "",
	repo: str = "",
	commit: str = "",
):
	"""Submit one zoo script and follow it to the record: the G1 critical path in one command.

	Probes stay on `gpus` (comma-separated): L4 by default, since team runs use T4 or L4 unless
	Chiya approves a bigger GPU, and without a constraint an OOM would step up the ladder (to an
	A100). --visitor: the queue's visitor (3 probe jobs a day each). --notebook path: check in an
	.ipynb from this checkout as kind notebook, as D's tab does (the name and outputs as they are).
	--dir path [--entrypoint main.py] [--args=--epochs=1,--lr=3e-4]: any directory in this checkout as an upload.
	--repo https://github.com/<owner>/<repo> --commit <40-hex> [--entrypoint ...] [--args=...]: a public
	repo at a commit, sent as kind repo, as a visitor's GitHub link is (Monday's unseen scripts). The job
	is cancelled after `max_minutes`, or if this run fails partway. The job store is looked up in
	the environment `modal run --env` names (modal_env overrides it). Writes out/smoke.json (every
	event and the final status, or the error) and out/record.json (the flight record), which
	remote-run.yml brings back with the log.
	"""
	import json
	import time

	svc = modal.Cls.from_name(APP_NAME, SERVICE_NAME, environment_name=modal_env or None)()
	source: dict = {"kind": "sample", "uri": sample}
	if notebook:  # a notebook check-in as D's tab sends it (kind notebook, one .ipynb file)
		nb = (Path(__file__).resolve().parents[3] / notebook).resolve()
		if nb.suffix != ".ipynb" or Path(__file__).resolve().parents[3] not in nb.parents:
			raise SystemExit(f"{notebook!r} isn't an .ipynb file in this checkout")
		source = {"kind": "notebook", "files": {nb.name: nb.read_text()}}
	if (entrypoint or args) and not (dir or repo):
		raise SystemExit(
			"--entrypoint and --args go with --dir or --repo (a sample or notebook names its own)"
		)
	if repo:  # the runner clones it (runner/sources._clone), exactly as for a visitor's link
		source = {"kind": "repo", "uri": repo, "commit": commit or None,
			**({"entrypoint": entrypoint} if entrypoint else {}), **({"args": _split_args(args)} if args else {})}  # fmt: skip
	if dir:  # any directory in this checkout, checked in as a visitor's upload (Monday's unseen scripts);
		# _read_tree reads the text files the runner reads (data files such as .csv aren't sent)
		from preflight.runner.sources import _read_tree

		root = Path(__file__).resolve().parents[3]
		src = (root / dir).resolve()
		if root not in src.parents or not src.is_dir():
			raise SystemExit(f"{dir!r} isn't a directory inside this checkout")
		source = {"kind": "upload", "files": _read_tree(src), **({"entrypoint": entrypoint} if entrypoint else {}),
			**({"args": _split_args(args)} if args else {})}  # fmt: skip
	request = {
		"source": source,
		"constraints": {"gpu_types": [g.strip() for g in gpus.split(",") if g.strip()]},
	}
	out = Path("out")
	out.mkdir(exist_ok=True)
	sub: dict = {}
	info: dict = {}
	events: list[dict] = []
	error = None
	try:
		sub = svc.submit.remote(request, visitor, uuid.uuid4().hex)  # one visitor: 3 jobs a day
		print("submitted", sub, flush=True)
		seen, deadline = -1, time.monotonic() + max_minutes * 60
		while True:
			for ev in svc.events.remote(sub["job_id"], seen):
				seen = ev["seq"]
				events.append(ev)
				print(json.dumps(ev), flush=True)
			info = svc.status.remote(sub["job_id"])
			if info["status"] in ("done", "failed", "cancelled"):
				break
			if time.monotonic() > deadline:
				print(f"cancelling: past {max_minutes} min", flush=True)
				svc.cancel.remote(sub["job_id"])
				deadline = float("inf")
			time.sleep(5)
	except BaseException as exc:
		error = f"{type(exc).__name__}: {exc}"[:2000]
		if sub.get("job_id") and info.get("status") not in ("done", "failed", "cancelled"):
			with contextlib.suppress(Exception):  # never leave a job running on the GPU behind us
				svc.cancel.remote(sub["job_id"])
		raise
	finally:
		print(json.dumps(info, indent=2), flush=True)
		(out / "smoke.json").write_text(
			json.dumps({"submitted": sub, "info": info, "events": events, "error": error}, indent=2)
		)
	if info.get("share_token"):
		try:
			data = b"".join(records_vol.read_file(f"{info['share_token']}.json"))
			(out / "record.json").write_bytes(data)
		except Exception as exc:  # the record is on the volume either way; the log says why it's not here
			print(f"could not copy the record: {exc}", flush=True)


# ---------------------------------------------------------------- ground truth (L4, paid: ask first)


# Modal's own limit, not just the workflow's: on Oct 28 remote-run.yml's timeout killed `modal run`
# but the L4 function ran on until stop_app stopped it. The measurement ends inside its budget
# (40 min by default); this ends the container even if that fails.
GROUND_TRUTH_TIMEOUT_S = 50 * 60


# Modal tries these in order (L4 queues of up to 59 min seen, Nov 1); no L40S: B's reading needs a
# 24 GB card (Chiya, Nov 1). The result's "gpu" says which one ran
GROUND_TRUTH_GPUS = ["L4", "A10"]


def _ground_truth(script: str, bare_steps: int, recorder_steps: int, warmup: int, budget_s: float, files: dict,
	entrypoint: str | None = None, args: tuple = (), install_requirements: bool = False,
	generate_sample: int = 0, generate_calls: int = 0) -> dict:  # fmt: skip
	"""recorder/ground_truth.py on this function's GPU: a bare run (the full run), then a recorder
	run alternating clean and synced steps, and the paired overhead. The AMD droplet runs the same."""
	from preflight.probes import remote
	from preflight.recorder.ground_truth import measure

	print(f"[ground truth] container up on the {_gpu_name()}; {len(files)} files of {script}", flush=True)
	remote.link_weights()
	remote.deps_dir()
	# the probe image has no zoo (Oct 31: "has no train.py"): the files come from the checkout
	return measure(Path("/tmp") / script, bare_steps=bare_steps, recorder_steps=recorder_steps, warmup=warmup,
		budget_s=budget_s, files=files, entrypoint=entrypoint, args=tuple(args),
		install_requirements=install_requirements, generate_sample=generate_sample,
		generate_calls=generate_calls)  # fmt: skip


# weights read-only: an unseen third-party script runs as root in the bare run, and must not be
# able to change the shared weights production probes read (review, Nov 1)
_GT = {"image": None, "volumes": {WEIGHTS_DIR: _read_only(weights_vol)}, "timeout": GROUND_TRUTH_TIMEOUT_S}


@ops.function(**{**_GT, "image": probe_image}, gpu=GROUND_TRUTH_GPUS)
def ground_truth_on_l4(*a, **kw) -> dict:
	return _ground_truth(*a, **kw)


@ops.function(**{**_GT, "image": probe_image}, gpu=GROUND_TRUTH_GPUS[::-1])  # A10 first: the L4 queue
def ground_truth_on_a10(*a, **kw) -> dict:
	return _ground_truth(*a, **kw)


@ops.function(**{**_GT, "image": serving_image}, gpu=GROUND_TRUTH_GPUS)  # vLLM: its own image
def ground_truth_serving_on_l4(*a, **kw) -> dict:
	return _ground_truth(*a, **kw)


@ops.function(**{**_GT, "image": serving_image}, gpu=GROUND_TRUTH_GPUS[::-1])
def ground_truth_serving_on_a10(*a, **kw) -> dict:
	return _ground_truth(*a, **kw)


@ops.function(**{**_GT, "image": titan_image}, gpu=GROUND_TRUTH_GPUS)  # torchtitan: torch 2.7
def ground_truth_titan_on_l4(*a, **kw) -> dict:
	return _ground_truth(*a, **kw)


@ops.function(**{**_GT, "image": titan_image}, gpu=GROUND_TRUTH_GPUS[::-1])
def ground_truth_titan_on_a10(*a, **kw) -> dict:
	return _ground_truth(*a, **kw)


# A100-80GB only where Chiya approves it (Nov 2: the unseen list's 7B LoRA and its GRPO run); never asked
# for by default, and the image is the same probe or serving image the L4 runs use
@ops.function(**{**_GT, "image": probe_image}, gpu="A100-80GB")
def ground_truth_on_a100(*a, **kw) -> dict:
	return _ground_truth(*a, **kw)


@ops.function(**{**_GT, "image": serving_image}, gpu="A100-80GB")
def ground_truth_serving_on_a100(*a, **kw) -> dict:
	return _ground_truth(*a, **kw)


GT_FUNCTIONS = {
	("probe", "A100"): ground_truth_on_a100, ("serving", "A100"): ground_truth_serving_on_a100,
	("probe", "L4"): ground_truth_on_l4, ("probe", "A10"): ground_truth_on_a10,
	("serving", "L4"): ground_truth_serving_on_l4, ("serving", "A10"): ground_truth_serving_on_a10,
	("titan", "L4"): ground_truth_titan_on_l4, ("titan", "A10"): ground_truth_titan_on_a10,
}  # fmt: skip


def _split_args(args: str) -> list[str]:
	"""A script's arguments from one flag value. remote-run.yml's command allows no quotes, so
	`--args=--epochs=1,--lr=3e-4` (commas, no spaces) is the form that gets through it, and `,,` a
	comma inside a value; a value with spaces is split as a shell would."""
	import shlex

	if "," in args and " " not in args:
		# ",," is a comma inside a value (--layers=1,,2,,3 gives --layers=1,2,3): a single comma separates
		return [a.replace("\0", ",") for a in args.replace(",,", "\0").split(",") if a]
	return shlex.split(args)


@ops.local_entrypoint()
def ground_truth(
	script: str = "",
	bare_steps: int = 200,
	recorder_steps: int = 60,
	warmup: int = 5,
	budget_min: float = 40,
	dir: str = "",
	entrypoint: str = "",
	args: str = "",
	first_gpu: str = "L4",
	install_requirements: bool = False,
	repo: str = "",
	commit: str = "",
):
	"""A full run of any script, with the same budget and stop rules (Monday's cost table, B):

	modal run --env main python/preflight/runner/ops_app.py::ground_truth --script vision_timm --bare-steps 300
	modal run ... ::ground_truth --dir validation/unseen/s3 --entrypoint main.py --args=--epochs=1 \\
		--install-requirements --first-gpu A10

	--script: a zoo script; --dir: any directory in this checkout (Monday's unseen scripts, from the
	validation-freeze tag). The entrypoint is the one named, else train.py, main.py, serve.py or
	run.py. The image follows the script: vLLM scripts the serving image, torchtitan its own.
	--repo URL --commit SHA: a public GitHub repo at a commit, read as the runner reads a visitor's link
	(runner/sources._clone: text files only, the same size caps), here on the machine that runs this.
	--first-gpu A10 tries the A10 before the L4 (the L4 queued 19 to 59 min on Nov 1); A100 runs on an
	A100-80GB only (approved runs only); never an L40S. budget_min: the whole measurement ends inside it; keep it under the run's own time limit.
	Writes out/ground_truth_<name>.json."""
	import json

	from preflight.recorder.ground_truth import ZOO_SCRIPTS, brief, plan
	from preflight.runner.sources import _read_tree

	root = Path(__file__).resolve().parents[3]
	if sum(map(bool, (script, dir, repo))) != 1:
		raise SystemExit(
			"name one of a zoo script (--script), a directory in this checkout (--dir) or a repo (--repo)"
		)
	if script:
		if script not in ZOO_SCRIPTS:
			raise SystemExit(
				f"{script!r} isn't a GPU zoo script: {sorted(ZOO_SCRIPTS)} (agent_eval_spark: agent_sample)"
			)
		src = root / "zoo" / script
	elif dir:
		src = (root / dir).resolve()
		if root not in src.parents or not src.is_dir():
			raise SystemExit(f"{dir!r} isn't a directory inside this checkout")
	if repo:
		from preflight.runner.sources import _clone

		files, sha = _clone(repo, commit or None)
		src = Path(repo.rstrip("/").removesuffix(".git").rsplit("/", 1)[-1])
		print(f"[ground truth] {repo} at {sha}: {len(files)} files", flush=True)
	else:
		files = _read_tree(src)
	p = plan(files, entrypoint or None)
	if p["lens"] == "agent":
		raise SystemExit("an agent run has no GPU ground truth: its full run is ops_app.py::agent_sample")
	entry, image, sample = p["entrypoint"], p["image"], p["generate_sample"]
	first = first_gpu.upper() if first_gpu.upper() in ("A10", "A100") else "L4"
	if (image, first) not in GT_FUNCTIONS:
		raise SystemExit(f"no {first} ground-truth function for the {image} image")
	fn = GT_FUNCTIONS[(image, first)]
	name = script or src.name
	print(f"[ground truth] {name}: {entry}, lens {p['lens']}, {image} image, {len(files)} files; "
		f"waiting for an {first}" + ("" if first == "A100" else f", else the {'L4' if first == 'A10' else 'A10'}")
		+ " (Modal queues it)", flush=True)  # fmt: skip
	res = fn.remote(name, bare_steps, recorder_steps, warmup, budget_min * 60, files, entry, tuple(_split_args(args)),
		install_requirements, sample, p["generate_calls"])  # fmt: skip
	res["lens"], res["image"] = p["lens"], image
	Path("out").mkdir(exist_ok=True)
	Path(f"out/ground_truth_{name}.json").write_text(json.dumps(res, indent=2))
	print(json.dumps(brief(res), indent=2), flush=True)


# ---------------------------------------------------------------- multi-GPU (2 x L4, paid: ask first)

MULTI_GPU_TIMEOUT_S = 20 * 60  # the container's own end, whatever the probes do


@ops.function(image=probe_image, gpu=["L4:2", "A10:2"], timeout=MULTI_GPU_TIMEOUT_S)  # L4s, else A10s
def multi_gpu_on_l4x2(files: dict, warmup: int = 3, timed: int = 10) -> dict:
	"""DDP and FSDP probes on one node of two L4s, one recorder per rank under torchrun."""
	from preflight.probes.local import ProbeSpec, run_probe

	print("[multi-gpu] container up on 2 x L4", flush=True)
	out = {}
	for name in ("ddp", "fsdp"):
		r = run_probe(ProbeSpec(probe_id=name, files={"train.py": files[f"{name}.py"]}, entrypoint="train.py",
			warmup=warmup, timed=timed, nproc=2, timeout_s=360))  # fmt: skip
		p = r.probe
		out[name] = {"status": p.status, "error": p.error, "steps": p.steps,
			"p50_s": p.step_time.p50_s if p.step_time else None,
			"peak_allocated_bytes": p.memory.peak_allocated_bytes if p.memory else None,
			"extras": p.extras, "gpu": r.env.gpu.name if r.env and r.env.gpu else None,
			"log_tail": r.log_tail[-1500:] if p.status != "ok" else ""}  # fmt: skip
		print(f"[multi-gpu] {name}: {p.status}, {p.steps} steps, extras {p.extras}", flush=True)
	return out


@ops.local_entrypoint()
def multi_gpu():
	"""modal run --env main python/preflight/runner/ops_app.py::multi_gpu
	Writes out/multi_gpu.json: the DDP and FSDP probes on 2 x L4 (tests/fixtures/multi_gpu)."""
	import json

	root = Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "multi_gpu"
	files = {f.name: f.read_text() for f in root.glob("*.py")}
	print("[multi-gpu] waiting for 2 x L4, else 2 x A10 (Modal queues it)", flush=True)
	res = multi_gpu_on_l4x2.remote(files)
	Path("out").mkdir(exist_ok=True)
	Path("out/multi_gpu.json").write_text(json.dumps(res, indent=2))
	print(json.dumps({k: {kk: v for kk, v in r.items() if kk != "log_tail"} for k, r in res.items()}, indent=2),
		flush=True)  # fmt: skip


# ---------------------------------------------------------------- torchtitan adapter check (L4, paid: ask first)

TITAN_TIMEOUT_S = 15 * 60  # the container's own end, whatever the probe does


def _titan_files() -> dict:
	from preflight.runner.sources import _read_tree

	return _read_tree(Path(__file__).resolve().parents[3] / "zoo" / "torchtitan_debug")


@ops.function(image=titan_image, timeout=600)
def startup_check_titan_image(files: dict) -> dict:
	"""CPU, before the L4: the titan image imports torch, torchtitan's trainer and the recorder, and
	torchtitan parses the zoo config. A failure here costs no GPU."""
	import json
	import os
	import subprocess
	import sys
	import tempfile

	work = Path(tempfile.mkdtemp())
	for rel, text in files.items():
		(work / rel).parent.mkdir(parents=True, exist_ok=True)
		(work / rel).write_text(text)
	cmd = ("import json, torch, torchtitan.train, preflight.recorder.runtime; "
		"from torchtitan.config_manager import ConfigManager; "
		"c = ConfigManager().parse_args(['--job.config_file', 'debug_model.toml']); "
		"from torchtitan.datasets.tokenizer.tiktoken import TikTokenizer; "
		"tok = TikTokenizer(c.model.tokenizer_path); "
		"from datasets import load_dataset; ds = load_dataset('tests/assets/c4_test', split='train'); "
		"print(json.dumps({'torch': torch.__version__, 'flavor': c.model.flavor, 'steps': c.training.steps, "
		"'vocab': tok.n_words, 'docs': len(ds)}))")  # fmt: skip
	p = subprocess.run([sys.executable, "-c", cmd], cwd=work, capture_output=True, text=True, timeout=300,
		env={**os.environ, "PYTHONPATH": REMOTE_PKG})  # fmt: skip
	out = {"exit": p.returncode, "stderr_tail": p.stderr[-1500:] if p.returncode else ""}
	if p.returncode == 0:
		out.update(json.loads(p.stdout.strip().splitlines()[-1]))
	return out


def _gpu_name() -> str:
	"""The GPU Modal gave this function (from a list of several), as the device names it. NVML, not
	torch.cuda: a CUDA context in this process would sit in the memory the probe measures."""
	try:
		import pynvml

		pynvml.nvmlInit()
		name = pynvml.nvmlDeviceGetName(pynvml.nvmlDeviceGetHandleByIndex(0))
		return (name.decode() if isinstance(name, bytes) else name).removeprefix("NVIDIA ")
	except Exception:
		return "GPU"


@ops.function(image=titan_image, gpu=["L4", "A10", "L40S"], timeout=TITAN_TIMEOUT_S)  # the first to schedule
def titan_check_on_l4(files: dict, warmup: int = 3, timed: int = 10) -> dict:
	"""torchtitan's trainer on its debug model under the recorder, one L4."""
	from preflight.probes.local import ProbeSpec, run_probe

	print(f"[titan] container up on the {_gpu_name()}", flush=True)
	r = run_probe(ProbeSpec(probe_id="titan", files=files, entrypoint="train.py", warmup=warmup, timed=timed,
		timeout_s=600))  # fmt: skip
	p = r.probe
	return {"status": p.status, "error": p.error, "error_code": p.error_code, "steps": p.steps,
		"p50_s": p.step_time.p50_s if p.step_time else None,
		"by_phase": p.step_time.by_phase if p.step_time else None,
		"peak_allocated_bytes": p.memory.peak_allocated_bytes if p.memory else None,
		"extras": p.extras, "gpu": r.env.gpu.name if r.env and r.env.gpu else None,
		"seconds": p.duration_s, "log_tail": r.log_tail[-1500:] if p.status != "ok" else ""}  # fmt: skip


@ops.local_entrypoint()
def torchtitan_check():
	"""modal run --env main python/preflight/runner/ops_app.py::torchtitan_check
	The CPU startup check first; the L4 only if it passes. Writes out/torchtitan_check.json."""
	import json
	import sys

	files = _titan_files()
	start = startup_check_titan_image.remote(files)
	start["probe_user"] = probe_user_check_titan_image.remote()
	if start.get("exit") == 0 and not _probe_user_ok(start["probe_user"], "titan"):
		start["exit"] = "probe_user"  # the probe uid can't run here: no GPU for it
	print(f"[titan] startup check: {json.dumps(start)}", flush=True)
	res = {"startup": start}
	if start.get("exit") == 0:
		print("[titan] waiting for an L4, else an A10 or an L40S (Modal queues it)", flush=True)
		res["probe"] = titan_check_on_l4.remote(files)
		print(json.dumps({k: v for k, v in res["probe"].items() if k != "log_tail"}, indent=2), flush=True)
	Path("out").mkdir(exist_ok=True)
	Path("out/torchtitan_check.json").write_text(json.dumps(res, indent=2))
	if start.get("exit") != 0 or res["probe"].get("status") != "ok":
		sys.exit(1)


# ---------------------------------------------------------------- agent sample (Meta API, paid: ask first)

AGENT_SAMPLE_TIMEOUT_S = 45 * 60  # the container's own end: the sample's 30 min plus installs and tasks
AGENT_MAX_USD = 3.0  # proposed to Chiya for one sample (the proxy stops at it); her yes sets it


def _mbpp_row_keys(n: int) -> dict[str, int]:
	"""zoo/agent_eval_spark's first n MBPP rows by their opening key (the proxy's hash of the first user
	message, as run.py writes it), so a sweep can say which sampled task is which row, planted rows
	included, without reading any content back. From the datasets-server's rows API."""
	import httpx

	from preflight.recorder.agent import _h, opening_key

	keys = {}
	for offset in range(0, n, 100):  # the rows API gives at most 100 a request
		r = httpx.get("https://datasets-server.huggingface.co/rows", params={"dataset": "google-research-datasets/mbpp",
			"config": "full", "split": "test", "offset": offset, "length": min(100, n - offset)}, timeout=60)  # fmt: skip
		r.raise_for_status()
		for item in r.json()["rows"]:
			row = item["row"]
			content = f"{row['text']}\nYour code should pass these tests:\n" + "\n".join(row["test_list"])
			keys[opening_key([_h({"role": "user", "content": content})])] = item["row_idx"]
	return keys


def _agent_sample(
	files: dict,
	entrypoint: str,
	k: int,
	max_usd: float,
	skip_keys: list | None = None,
	provider: str = "meta",
	label_mbpp: int = 0,
	timeout_s: float | None = None,
) -> dict:
	"""The hosted agent path: the recorder's proxy here with the Meta key, behind a tunnel, and the
	script in a CPU sandbox with no secrets. Returns the AgentSample and what it cost."""
	from preflight.harness.static import map_static
	from preflight.runner.agent_jobs import sandbox_sampler
	from preflight.runner.modal_app import _sandbox_factory
	from preflight.runner.sources import Fetched

	owner = modal.App._get_container_app()
	create = _sandbox_factory(owner)
	extra: dict = {}
	if (
		provider == "baseten"
	):  # the Pareto sweep: the same tasks on Spark Open, whatever model the script names
		upstream = os.environ.get("BASETEN_BASE_URL") or "https://inference.baseten.co/v1"
		key, override = (
			os.environ["BASETEN_API_KEY"],
			os.environ.get("SPARK_OPEN_MODEL", "meta-models/muse-spark-open-1026"),
		)
	elif provider == "glimmer":  # the third arm: Muse Glimmer on Preflight's own Modal host (B's app)
		from preflight.harness.spark import GLIMMER_USD_PER_S

		base = os.environ["GLIMMER_BASE_URL"].rstrip("/")
		upstream = base if base.endswith("/v1") else base + "/v1"
		key, override = "glimmer", os.environ.get("GLIMMER_MODEL", "muse-glimmer")
		extra = {"upstream_headers": {"Modal-Key": os.environ["GLIMMER_MODAL_KEY"],
			"Modal-Secret": os.environ["GLIMMER_MODAL_SECRET"]}, "usd_per_second": GLIMMER_USD_PER_S,
			"follow_redirects": 3, "timeout_s": 330.0}  # fmt: skip
	else:
		upstream = os.environ.get("META_API_BASE") or "https://api.meta.ai/v1"
		key, override = os.environ["META_API_KEY"], None
	sample = sandbox_sampler(lambda extra_egress, timeout_s: create("cpu", extra_egress, timeout_s=timeout_s), upstream,
		key, max_usd_total=max_usd, model_override=override, **extra)  # fmt: skip
	sm = map_static(files, entrypoint=entrypoint)
	print(f"[agent] {entrypoint}: lens {sm.lens}, population {getattr(sm.agent, 'n_tasks', None)}; sampling {k}",
		flush=True)  # fmt: skip
	# tasks an earlier sample already has are refused before any model call, so k are new ones
	ends: dict = {}
	s = sample(
		Fetched(files=files, entrypoint=entrypoint),
		sm,
		k=k,
		skip_keys=tuple(skip_keys or ()),
		run_id="s2" if skip_keys else "s1",
		inspect=ends.update,  # how each task ended, for scoring the sweep (no content)
		timeout_s=timeout_s,  # None: the sampler's own 30 min; the full benchmark passes its own
	)
	usd = round(sum(t.cost_usd or 0 for t in s.tasks), 4)
	print(f"[agent] {len(s.tasks)} tasks, ${usd} of model calls; {s.selection}", flush=True)
	rows = {}
	if label_mbpp:
		from preflight.recorder.agent import task_key

		try:
			by_key = _mbpp_row_keys(label_mbpp)
			rows = {t.task_id: by_key.get(task_key(t.task_id)) for t in s.tasks}
		except Exception as exc:  # the labels are for scoring; the sample stands without them
			print(f"[agent] couldn't label the rows: {type(exc).__name__}: {exc}", flush=True)
	return {
		"sample": s.model_dump(mode="json"),
		"usd": usd,
		"task_ends": ends,
		"task_rows": rows,
		"provider": provider,
	}


_AGENT_SECRETS = [
	modal.Secret.from_name("preflight-meta"),
	modal.Secret.from_name("preflight-baseten"),
	modal.Secret.from_name("preflight-glimmer"),
]
# the full benchmark runs in halves of 250 tasks (about 55 min each [estimate] at the script's 4
# workers), since remote-run.yml stops a run at 120 min: a 2-hour container, the sampler's own 95
AGENT_FULL_TIMEOUT_S = 2 * 3600
AGENT_FULL_SAMPLE_S = 95 * 60


@ops.function(image=worker_image, secrets=_AGENT_SECRETS, timeout=AGENT_SAMPLE_TIMEOUT_S)
def agent_sample_hosted(*a, **kw) -> dict:
	return _agent_sample(*a, **kw)


@ops.function(image=worker_image, secrets=_AGENT_SECRETS, timeout=AGENT_FULL_TIMEOUT_S)
def agent_sample_hosted_full(*a, **kw) -> dict:
	"""A half of the full benchmark: k tasks inside the sampler's own 95 min."""
	return _agent_sample(*a, timeout_s=AGENT_FULL_SAMPLE_S, **kw)


@ops.local_entrypoint()
def agent_sample(
	script: str = "agent_eval_spark",
	k: int = 30,
	max_usd: float = AGENT_MAX_USD,
	skip_from: str = "",
	provider: str = "meta",
	label_mbpp: int = 0,
	full: bool = False,
):
	"""modal run --env main python/preflight/runner/ops_app.py::agent_sample --script agent_eval_spark
	--full: a half of the full benchmark (--k 250, then --k 250 --skip-from the first half's output
	committed under zoo/), on a 2-hour container with the sampler's limit at 95 min.
	Writes out/agent_sample_<script>.json: the sampled tasks (counts and timings, no content).
	--skip-from zoo/<script>/<record>.json: sample tasks that record's agent sample doesn't hold."""
	import json

	from preflight.recorder.agent import task_key
	from preflight.runner.sources import _read_tree

	files = _read_tree(Path(__file__).resolve().parents[3] / "zoo" / script)
	entry = next((e for e in ("run.py", "train.py") if e in files), None)
	if entry is None:
		raise SystemExit(f"zoo/{script} has no run.py or train.py in this checkout")
	skip = []
	if skip_from:
		prior = json.loads((Path(__file__).resolve().parents[3] / skip_from).read_text())
		# a record's agent_sample, or an earlier agent_sample run's own output (its "sample")
		tasks = (prior.get("agent_sample") or prior.get("sample") or {}).get("tasks", [])
		# a list, not a set: a key twice skips two tasks opened the same way (open_run counts them)
		skip = [key for t in tasks if (key := task_key(t["task_id"]))]
		if len(skip) < len(tasks):
			print(f"[agent] {len(tasks) - len(skip)} task ids in {skip_from} have no key and may be sampled again",
				flush=True)  # fmt: skip
		print(f"[agent] skipping the {len(skip)} tasks {skip_from} already holds", flush=True)
	fn = agent_sample_hosted_full if full else agent_sample_hosted
	res = fn.remote(files, entry, k, max_usd, skip, provider, label_mbpp)
	Path("out").mkdir(exist_ok=True)
	Path(f"out/agent_sample_{script}{'' if provider == 'meta' else '_' + provider}.json").write_text(
		json.dumps(res, indent=2)
	)


# ---------------------------------------------------------------- Spark diagnosis, one real check (paid: ask first)


@ops.function(
	image=worker_image,
	secrets=[modal.Secret.from_name("preflight-meta"), modal.Secret.from_name("preflight-baseten")],
	timeout=15 * 60,
)
def diagnose_hosted(record_json: str, max_usd: float = 0.5) -> dict:
	"""harness/diagnosis_run.py's one diagnosis and check, with the Meta and Baseten keys."""
	from preflight.harness.diagnosis_run import run_once
	from preflight.harness.spark import ModelClient
	from preflight.schema import load_record

	spark_open = ModelClient.baseten_spark_open() if os.environ.get("BASETEN_API_KEY") else None
	# 180 s: up to 8,192 output tokens, reasoning included, outlast the client's default 60 s
	return run_once(load_record(record_json), ModelClient.meta(timeout_s=180.0), spark_open, max_usd)


@ops.local_entrypoint()
def diagnose(share: str, max_usd: float = 0.5):
	"""modal run --env main python/preflight/runner/ops_app.py::diagnose --share <token> [--max-usd 0.5]
	Reads the stored record, runs the diagnosis and its check once, writes out/diagnose.json."""
	import re
	import sys

	if not re.fullmatch(r"[A-Za-z0-9_-]{16,64}", share):
		print(f"[diagnose] {share!r} isn't a share token", flush=True)
		sys.exit(2)
	data = b"".join(records_vol.read_file(f"{share}.json")).decode()
	res = diagnose_hosted.remote(data, max_usd)
	Path("out").mkdir(exist_ok=True)
	Path("out/diagnose.json").write_text(json.dumps(res, indent=2))
	for f in res["checked"] or res["diagnosis"]:
		print(f"[diagnose] {f['code']}: {f['title']} | {f['detail']}", flush=True)
	print(f"[diagnose] stats {res['stats']}; {len(res['calls'])} calls, ${res['usd']:.4f}", flush=True)


# ---------------------------------------------------------------- triage of a stored probe (paid: ask first)


def _ours(rec) -> bool:
	"""A stored record holding only our own zoo code's output: a sample, not derived from one (a probe
	or verify may carry a visitor's patch), with no visitor arguments (review, Nov 1)."""
	src = rec.job.source
	return src.kind == "sample" and rec.parent_id is None and not getattr(src, "args", None)


@ops.function(
	image=worker_image,
	secrets=[modal.Secret.from_name("preflight-glimmer"), modal.Secret.from_name("preflight-baseten")],
	timeout=10 * 60,  # Glimmer scales to zero: a cold start measured 286 s (B, Nov 1)
)
def triage_hosted(record_json: str, probe_id: str, provider: str = "glimmer") -> dict:
	"""harness/triage.py on one probe of a stored record, as the pipeline would triage it: the
	probe's recorded error (a stored record keeps no log tail). provider: glimmer or spark-open."""
	import time

	from preflight.harness.spark import ModelClient
	from preflight.harness.triage import triage_probe
	from preflight.runner.flags import flag
	from preflight.schema import load_record

	if not flag("harness.spark_open_triage"):  # the pipeline's own gate (and the data-use notice's)
		return {"error": "harness.spark_open_triage is off here"}
	rec = load_record(record_json)
	# our own zoo scripts only: a visitor's record holds their error text, which would land in the
	# run log and out/ on the remote-runs branch, and a private one must never reach Baseten (review)
	if not _ours(rec):
		return {"error": "only a sample (zoo) record, not derived and without arguments, is triaged here"}
	probe = next((p for p in rec.probes if p.id == probe_id), None)
	if probe is None:
		return {"error": f"no probe {probe_id!r}: {[p.id for p in rec.probes]}"}
	if probe.status in ("ok", "cached") or not probe.error or probe.extras.get("patched"):
		return {"error": f"probe {probe_id} finished, has no recorded error or ran patched code: not triaged"}
	if provider == "glimmer":
		if not ModelClient.glimmer_configured():
			return {"error": "the preflight-glimmer secret has no GLIMMER_BASE_URL"}
		client = ModelClient.glimmer()
	else:
		client = ModelClient.baseten_spark_open()
	t0 = time.monotonic()
	f, calls = triage_probe(client, probe, "")
	return {"record_id": rec.record_id, "probe": probe_id, "status": probe.status, "provider": client.provider,
		"model": client.model, "wall_s": round(time.monotonic() - t0, 1),
		"finding": f.model_dump(mode="json") if f else None, "last_error": client.last_error,
		"calls": [c.model_dump(mode="json") for c in calls], "usd": round(sum(c.cost_usd for c in calls), 6)}  # fmt: skip


@ops.local_entrypoint()
def triage(share: str, probe: str = "p1-eval", provider: str = "glimmer"):
	"""modal run --env main python/preflight/runner/ops_app.py::triage --share <token> [--probe p1-eval]
	[--provider glimmer|spark-open]. One triage call on a stored record's probe; writes out/triage.json."""
	import re
	import sys

	if not re.fullmatch(r"[A-Za-z0-9_-]{16,64}", share) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,40}", probe):
		print("[triage] a share token and a probe id, please", flush=True)
		sys.exit(2)
	data = b"".join(records_vol.read_file(f"{share}.json")).decode()
	res = triage_hosted.remote(data, probe, provider)
	Path("out").mkdir(exist_ok=True)
	Path("out/triage.json").write_text(json.dumps(res, indent=2))
	f = res.get("finding")
	print(
		f"[triage] {res.get('error') or (f and f['title'] + ' | ' + f['detail']) or 'no finding: ' + str(res.get('last_error'))}",
		flush=True,
	)
	print(
		f"[triage] {len(res.get('calls', []))} calls, ${res.get('usd', 0):.4f}, {res.get('wall_s')} s",
		flush=True,
	)


# ---------------------------------------------------------------- the judging warm pool, without changing it


@ops.local_entrypoint()
def warm_pool_dry_run(modal_env: str = "main"):
	"""modal run --env main python/preflight/runner/ops_app.py::warm_pool_dry_run
	What the tick will do for judging (item 45), changing nothing: each hour from Nov 4 22:00 to
	Nov 9 02:00 ET, the warm workers it wants and the probe GPU an L4 job gets; the deployed
	run_worker's handle (that it can be scaled) and what the tick last set. Writes out/warm_pool.json."""
	import datetime as dt
	import time
	import zoneinfo

	from preflight.runner.judging import judging_gpu, warm_workers, window

	et = zoneinfo.ZoneInfo("America/New_York")
	t = dt.datetime(2026, 11, 4, 22, tzinfo=et)
	rows, prev = [], None
	while t <= dt.datetime(2026, 11, 9, 2, tzinfo=et):
		ts = t.timestamp()
		state = (warm_workers(ts), judging_gpu("L4", None, ts), judging_gpu("L4", ["L4"], ts))
		if state != prev:  # the hours where something changes
			rows.append({"et": t.isoformat(), "warm_workers": state[0], "l4_job_probes_on": state[1],
				"l4_only_job_probes_on": state[2]})  # fmt: skip
			prev = state
		t += dt.timedelta(hours=1)
	fn = modal.Function.from_name(APP_NAME, "run_worker", environment_name=modal_env)
	try:  # a lazy handle has every method: hydrate it to know the deployed function exists (review)
		fn.hydrate()
		scalable = any(hasattr(fn, a) for a in ("update_autoscaler", "autoscaler"))
	except Exception as exc:
		scalable = f"not found: {type(exc).__name__}"
	try:
		last = modal.Dict.from_name("preflight-jobs", environment_name=modal_env).get("warm_workers")
	except Exception as exc:
		last = f"unread: {type(exc).__name__}"
	out = {"window": window(), "changes": rows, "run_worker_scalable": scalable, "tick_last_set": last,
		"now_wants": warm_workers(time.time())}  # fmt: skip
	Path("out").mkdir(exist_ok=True)
	Path("out/warm_pool.json").write_text(json.dumps(out, indent=2))
	print(f"[warm] {json.dumps(out)}", flush=True)


# ---------------------------------------------------------------- a private job's triage, as the worker runs it


@ops.function(
	image=worker_image,
	secrets=[
		modal.Secret.from_name("preflight-meta"),
		modal.Secret.from_name("preflight-baseten"),
		modal.Secret.from_name("preflight-glimmer"),
	],
	timeout=12 * 60,
)
def private_triage_check_hosted(record_json: str) -> dict:
	"""The worker's own private-job triage (pipeline._triage, private=True) on a zoo record's failed
	probes, with the worker's clients (modal_app._glimmer_client, _triage_client) and the same
	secrets as run_worker, and harness.glimmer_private on in this process only (production keeps it
	off). Passes when Glimmer made every call and Baseten none."""
	import json as _json

	from preflight.probes.local import ProbeResult
	from preflight.runner import flags
	from preflight.runner.modal_app import _glimmer_client, _triage_client
	from preflight.runner.pipeline import Deps, _triage
	from preflight.schema import Env, load_record

	rec = load_record(record_json)
	if not _ours(rec):
		return {"error": "only a sample (zoo) record, not derived and without arguments"}
	os.environ["PREFLIGHT_FLAGS"] = _json.dumps(
		{"harness.glimmer_private": True, "harness.spark_open_triage": True}
	)
	flags.reload()
	failed = [
		p for p in rec.probes if p.status not in ("ok", "cached") and p.error and not p.extras.get("patched")
	]
	results = [ProbeResult(probe=p, env=Env(backend="cpu", provider="modal"), summary={}, kernels=None, arch=None,
		trace_file=None, log_tail="", workdir=Path("/tmp")) for p in failed[:1]]  # fmt: skip
	glimmer, spark = _glimmer_client(), _triage_client()
	deps = Deps(records=None, runner=None, glimmer=glimmer, triage=spark)
	out, findings = _triage(rec, results, deps, private=True)
	made = out.provenance[len(rec.provenance) :]
	return {"glimmer_configured": glimmer is not None, "probe": failed[0].id if failed else None,
		"findings": [{"code": f.code, "title": f.title, "detail": f.detail} for f in findings],
		"calls": [{"provider": c.provider, "ok": c.ok, "latency_s": c.latency_s, "cost_usd": c.cost_usd} for c in made],
		"baseten_calls": len(spark.calls) if spark else 0,
		"pass": bool(findings) and all(c.provider == "modal-glimmer" for c in made) and not (spark and spark.calls)}  # fmt: skip


@ops.local_entrypoint()
def private_triage_check(share: str):
	"""modal run --env main python/preflight/runner/ops_app.py::private_triage_check --share <token>"""
	import re
	import sys

	if not re.fullmatch(r"[A-Za-z0-9_-]{16,64}", share):
		print("[private] a share token, please", flush=True)
		sys.exit(2)
	data = b"".join(records_vol.read_file(f"{share}.json")).decode()
	res = private_triage_check_hosted.remote(data)
	Path("out").mkdir(exist_ok=True)
	Path("out/private_triage_check.json").write_text(json.dumps(res, indent=2))
	print(f"[private] {'PASS' if res.get('pass') else 'FAIL'}: {json.dumps(res)[:1500]}", flush=True)
	if not res.get("pass"):
		raise SystemExit(1)


# ---------------------------------------------------------------- Glimmer vs Spark Open agreement (paid: ask first)


@ops.function(
	image=worker_image,
	secrets=[modal.Secret.from_name("preflight-glimmer"), modal.Secret.from_name("preflight-baseten")],
	volumes={"/records": _read_only(records_vol)},
	timeout=40 * 60,
)
def triage_agreement_hosted(max_usd: float = 1.0, limit: int = 40) -> dict:
	"""For harness.glimmer_private (Chiya's decision): every distinct failed probe in the stored
	sample (zoo) records, triaged by Glimmer and by Spark Open on the same input (the probe's
	recorded error; a stored record keeps no log tail), and against the record's own Spark Open
	finding where it has one. Distinct: by script and error text, so a failure seen on many records
	counts once. No visitor record is read past its source kind. Stops before max_usd."""
	import hashlib
	import time

	from preflight.harness.spark import ModelClient
	from preflight.harness.triage import triage_probe
	from preflight.runner.flags import flag
	from preflight.schema import load_record

	if not flag("harness.spark_open_triage"):
		return {"error": "harness.spark_open_triage is off here"}
	import re

	cases, seen, unreadable = [], set(), 0
	for path in sorted(Path("/records").glob("*.json")):
		try:
			rec = load_record(path.read_text())
		except Exception:
			unreadable += 1
			continue
		if not _ours(rec):
			continue
		# the record's own Spark Open triage (Glimmer findings share the source; their title names it)
		stored = {f.id.split(".", 1)[1]: f.code for f in rec.findings
			if f.source == "spark_open_triage" and f.title.startswith("Spark Open")}  # fmt: skip
		for p in rec.probes:
			if p.status in ("ok", "cached") or not p.error or p.extras.get("patched"):
				continue
			# one failure seen on many records counts once: the numbers in its message (bytes, steps) aside
			shape = re.sub(r"\d[\d,.]*", "#", p.error)
			key = hashlib.sha256(f"{rec.job.source.uri}|{p.error_code}|{shape}".encode()).hexdigest()
			if key in seen:
				continue
			seen.add(key)
			cases.append(
				{
					"uri": rec.job.source.uri,
					"record_id": rec.record_id,
					"probe": p,
					"stored": stored.get(p.id),
				}
			)
	cases = cases[:limit]
	glim, spark = ModelClient.glimmer(), ModelClient.baseten_spark_open()
	spent = lambda: sum(c.cost_usd for c in glim.calls + spark.calls)  # noqa: E731
	worst = (
		glim.deadline_s * glim.usd_per_second
	) + 0.01  # one Glimmer call at its longest, plus Spark Open's
	rows, t0 = [], time.monotonic()
	for case in cases:
		p, stored_code = case["probe"], case["stored"]
		if spent() + worst > max_usd:
			break
		g, _ = triage_probe(glim, p, "")
		o, _ = triage_probe(spark, p, "")
		rows.append({"script": case["uri"], "record_id": case["record_id"], "probe": p.id, "status": p.status,
			"error_code": p.error_code, "glimmer": g.code if g else None, "spark_open": o.code if o else None,
			"stored_spark_open": stored_code, "glimmer_quoted": bool(g and "no log line quoted" not in g.detail),
			"spark_open_quoted": bool(o and "no log line quoted" not in o.detail)})  # fmt: skip
		print(f"[agree] {rows[-1]}", flush=True)
	both = [r for r in rows if r["glimmer"] and r["spark_open"]]
	stored_rows = [r for r in rows if r["stored_spark_open"]]
	return {"cases_found": len(cases), "unreadable_records": unreadable, "rows": rows, "both_answered": len(both),
		"no_answer": {"glimmer": sum(not r["glimmer"] for r in rows), "spark_open": sum(not r["spark_open"] for r in rows)},
		# Spark Open on the error alone against its own stored finding (log tail and error): what dropping the tail costs
		"spark_open_vs_stored": [sum(r["spark_open"] == r["stored_spark_open"] for r in stored_rows if r["spark_open"]),
			sum(1 for r in stored_rows if r["spark_open"])],
		"agree_same_input": sum(r["glimmer"] == r["spark_open"] for r in both),
		"with_stored": sum(1 for r in rows if r["glimmer"] and r["stored_spark_open"]),
		"agree_stored": sum(r["glimmer"] == r["stored_spark_open"] for r in rows if r["glimmer"] and r["stored_spark_open"]),
		"usd": {"glimmer": round(sum(c.cost_usd for c in glim.calls), 6), "spark_open": round(sum(c.cost_usd for c in spark.calls), 6)},
		"calls": {"glimmer": len(glim.calls), "spark_open": len(spark.calls)}, "wall_s": round(time.monotonic() - t0, 1)}  # fmt: skip


@ops.local_entrypoint()
def triage_agreement(max_usd: float = 1.0, limit: int = 40):
	"""modal run --env main python/preflight/runner/ops_app.py::triage_agreement [--max-usd 1.0]
	Glimmer's triage against Spark Open's on the zoo's failed probes; writes out/triage_agreement.json."""
	import math

	res = triage_agreement_hosted.remote(max_usd, limit)
	Path("out").mkdir(exist_ok=True)
	Path("out/triage_agreement.json").write_text(json.dumps(res, indent=2))

	def wilson(k: int, n: int) -> str:
		if not n:
			return "n/a"
		z, ph = 1.96, k / n
		d = 1 + z * z / n
		c, h = (ph + z * z / (2 * n)) / d, z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
		return f"{max(0, c - h):.0%} to {min(1, c + h):.0%}"

	if "error" in res:
		print(f"[agree] {res['error']}", flush=True)
		return
	n, k = res["both_answered"], res["agree_same_input"]
	print(f"[agree] same input: {k} of {n} agree (Wilson 95% {wilson(k, n)}); "
		f"against the stored finding: {res['agree_stored']} of {res['with_stored']} ({wilson(res['agree_stored'], res['with_stored'])}); "
		f"{res['cases_found']} distinct failed probes, no answer {res['no_answer']}; Spark Open on the error alone vs its "
		f"stored finding {res['spark_open_vs_stored']}; {res['unreadable_records']} records unreadable; spent {res['usd']}",
		flush=True)  # fmt: skip


# ---------------------------------------------------------------- serving image (CPU, before any serving probe)


@ops.function(image=serving_image, timeout=600)
def startup_check_serving_image() -> dict:
	"""The serving image starts: vLLM, its torch and numpy, and the recorder import, without a GPU."""
	import subprocess
	import sys

	cmd = ("import json, numpy, torch, vllm, preflight.probes.remote, preflight.recorder.runtime as r; "
		"print(json.dumps({'vllm': vllm.__version__,"
		" 'torch': torch.__version__, 'numpy': numpy.__version__, 'recorder': r.__file__}))")  # fmt: skip
	p = subprocess.run([sys.executable, "-c", cmd], capture_output=True, text=True, timeout=300)
	out = {"exit": p.returncode, "stderr": p.stderr[-1500:] if p.returncode else ""}
	if p.returncode == 0:
		out.update(json.loads(p.stdout.strip().splitlines()[-1]))
	return out


@ops.local_entrypoint()
def startup_check_serving():
	"""modal run --env main python/preflight/runner/ops_app.py::startup_check_serving
	Writes out/startup_check_serving.json; exits non-zero unless vLLM 0.6.4.post1 on torch 2.5.1 imports."""
	import sys

	res = startup_check_serving_image.remote()
	res["probe_user"] = probe_user_check_serving_image.remote()
	res["ok"] = (
		res.get("exit") == 0
		and res.get("vllm") == "0.6.4.post1"
		and str(res.get("torch", "")).startswith("2.5.1")
		and _probe_user_ok(res["probe_user"], "serving")
	)
	Path("out").mkdir(exist_ok=True)
	Path("out/startup_check_serving.json").write_text(json.dumps(res, indent=2))
	print(json.dumps(res, indent=2), flush=True)
	if not res["ok"]:
		sys.exit(1)
