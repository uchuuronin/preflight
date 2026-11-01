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
	image_tag,
	probe_image,
	records_vol,
	serving_image,
	titan_image,
	weights_vol,
	worker_image,
)

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
	)
	for name, fn in checks:
		try:
			result[name] = fn.remote()
		except Exception as exc:  # the remote traceback is in the log above
			result[name] = {"failed": f"{type(exc).__name__}: {exc}"[:2000]}
	w, p, sb = result["worker"], result["probe"], result["sandbox"]
	result["ok"] = bool(
		"failed" not in w and w.get("lens") == "finetune" and w.get("knobs", 0) > 0 and w.get("saves", 0) >= 1
		and w.get("job_store_would_claim_here") is False and "runner_loaded_as" in w
		and "failed" not in p and p.get("status") == "ok" and p.get("steps", 0) > 0 and p.get("reachable_by_probe_uid")
		and "failed" not in sb and sb.get("import_exit") == 0 and sb.get("probe_status") == "ok"
	)  # fmt: skip
	Path("out").mkdir(exist_ok=True)
	Path("out/startup_check.json").write_text(json.dumps(result, indent=2))
	print(json.dumps(result, indent=2), flush=True)
	if not result["ok"]:
		sys.exit(1)


@ops.local_entrypoint()
def smoke(
	sample: str = "sample://zoo/finetune_lora_hf",
	gpus: str = "L4",
	max_minutes: int = 25,
	modal_env: str = "",
):
	"""Submit one zoo script and follow it to the record: the G1 critical path in one command.

	Probes stay on `gpus` (comma-separated): L4 by default, since team runs use T4 or L4 unless
	Chiya approves a bigger GPU, and without a constraint an OOM would step up the ladder. The job
	is cancelled after `max_minutes`, or if this run fails partway. The job store is looked up in
	the environment `modal run --env` names (modal_env overrides it). Writes out/smoke.json (every
	event and the final status, or the error) and out/record.json (the flight record), which
	remote-run.yml brings back with the log.
	"""
	import json
	import time

	svc = modal.Cls.from_name(APP_NAME, SERVICE_NAME, environment_name=modal_env or None)()
	request = {
		"source": {"kind": "sample", "uri": sample},
		"constraints": {"gpu_types": [g.strip() for g in gpus.split(",") if g.strip()]},
	}
	out = Path("out")
	out.mkdir(exist_ok=True)
	sub: dict = {}
	info: dict = {}
	events: list[dict] = []
	error = None
	try:
		sub = svc.submit.remote(request, "smoke", uuid.uuid4().hex)
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


@ops.function(
	image=probe_image,
	gpu=GROUND_TRUTH_GPUS,
	volumes={WEIGHTS_DIR: weights_vol},
	timeout=GROUND_TRUTH_TIMEOUT_S,
)
def ground_truth_on_l4(
	script: str, bare_steps: int, recorder_steps: int, warmup: int, budget_s: float, files: dict
) -> dict:
	"""recorder/ground_truth.py on an L4: a bare run, then a recorder run alternating clean and
	synced steps, and the paired overhead. The AMD droplet runs the same module."""
	from preflight.probes import remote
	from preflight.recorder.ground_truth import ZOO_SCRIPTS, measure

	if script not in ZOO_SCRIPTS:
		raise ValueError(f"unknown zoo script {script!r}")
	print(f"[ground truth] container up on the GPU; {len(files)} files of {script}", flush=True)
	remote.link_weights()
	remote.deps_dir()
	# the probe image has no zoo (Oct 31: "has no train.py"): the files come from the checkout
	return measure(Path("/tmp") / script, bare_steps=bare_steps, recorder_steps=recorder_steps, warmup=warmup,
		budget_s=budget_s, files=files)  # fmt: skip


@ops.local_entrypoint()
def ground_truth(
	script: str, bare_steps: int, recorder_steps: int = 60, warmup: int = 5, budget_min: float = 40
):
	"""modal run --env main python/preflight/runner/ops_app.py::ground_truth --script finetune_lora_hf --bare-steps 210
	Writes out/ground_truth_<script>.json (step walls, peak memory, how the bare run ended, overhead).
	budget_min: the whole measurement ends inside it; keep it under the run's own time limit."""
	import json

	from preflight.recorder.ground_truth import ZOO_SCRIPTS, brief
	from preflight.runner.sources import _read_tree

	if script not in ZOO_SCRIPTS:
		raise SystemExit(f"unknown zoo script {script!r}")
	files = _read_tree(Path(__file__).resolve().parents[3] / "zoo" / script)
	if "train.py" not in files:
		raise SystemExit(f"zoo/{script} has no train.py in this checkout")
	print(
		f"[ground truth] {script}: {len(files)} files sent; waiting for an L4, else an A10 (Modal queues it)",
		flush=True,
	)
	res = ground_truth_on_l4.remote(script, bare_steps, recorder_steps, warmup, budget_min * 60, files)
	Path("out").mkdir(exist_ok=True)
	Path(f"out/ground_truth_{script}.json").write_text(json.dumps(res, indent=2))
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

	r = httpx.get("https://datasets-server.huggingface.co/rows", params={"dataset": "google-research-datasets/mbpp",
		"config": "full", "split": "test", "offset": 0, "length": n}, timeout=60)  # fmt: skip
	r.raise_for_status()
	keys = {}
	for item in r.json()["rows"]:
		row = item["row"]
		content = f"{row['text']}\nYour code should pass these tests:\n" + "\n".join(row["test_list"])
		keys[opening_key([_h({"role": "user", "content": content})])] = item["row_idx"]
	return keys


@ops.function(
	image=worker_image,
	secrets=[modal.Secret.from_name("preflight-meta"), modal.Secret.from_name("preflight-baseten")],
	timeout=AGENT_SAMPLE_TIMEOUT_S,
)
def agent_sample_hosted(
	files: dict,
	entrypoint: str,
	k: int,
	max_usd: float,
	skip_keys: list | None = None,
	provider: str = "meta",
	label_mbpp: int = 0,
) -> dict:
	"""The hosted agent path: the recorder's proxy here with the Meta key, behind a tunnel, and the
	script in a CPU sandbox with no secrets. Returns the AgentSample and what it cost."""
	from preflight.harness.static import map_static
	from preflight.runner.agent_jobs import sandbox_sampler
	from preflight.runner.modal_app import _sandbox_factory
	from preflight.runner.sources import Fetched

	owner = modal.App._get_container_app()
	create = _sandbox_factory(owner)
	if (
		provider == "baseten"
	):  # the Pareto sweep: the same tasks on Spark Open, whatever model the script names
		upstream = os.environ.get("BASETEN_BASE_URL") or "https://inference.baseten.co/v1"
		key, override = (
			os.environ["BASETEN_API_KEY"],
			os.environ.get("SPARK_OPEN_MODEL", "meta-models/muse-spark-open-1026"),
		)
	else:
		upstream = os.environ.get("META_API_BASE") or "https://api.meta.ai/v1"
		key, override = os.environ["META_API_KEY"], None
	sample = sandbox_sampler(lambda extra_egress, timeout_s: create("cpu", extra_egress, timeout_s=timeout_s), upstream,
		key, max_usd_total=max_usd, model_override=override)  # fmt: skip
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


@ops.local_entrypoint()
def agent_sample(
	script: str = "agent_eval_spark",
	k: int = 30,
	max_usd: float = AGENT_MAX_USD,
	skip_from: str = "",
	provider: str = "meta",
	label_mbpp: int = 0,
):
	"""modal run --env main python/preflight/runner/ops_app.py::agent_sample --script agent_eval_spark
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
		tasks = (prior.get("agent_sample") or {}).get("tasks", [])
		# a list, not a set: a key twice skips two tasks opened the same way (open_run counts them)
		skip = [key for t in tasks if (key := task_key(t["task_id"]))]
		if len(skip) < len(tasks):
			print(f"[agent] {len(tasks) - len(skip)} task ids in {skip_from} have no key and may be sampled again",
				flush=True)  # fmt: skip
		print(f"[agent] skipping the {len(skip)} tasks {skip_from} already holds", flush=True)
	res = agent_sample_hosted.remote(files, entry, k, max_usd, skip, provider, label_mbpp)
	Path("out").mkdir(exist_ok=True)
	Path(f"out/agent_sample_{script}{'' if provider == 'meta' else '_' + provider}.json").write_text(
		json.dumps(res, indent=2)
	)


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
	res["ok"] = (
		res.get("exit") == 0
		and res.get("vllm") == "0.6.4.post1"
		and str(res.get("torch", "")).startswith("2.5.1")
	)
	Path("out").mkdir(exist_ok=True)
	Path("out/startup_check_serving.json").write_text(json.dumps(res, indent=2))
	print(json.dumps(res, indent=2), flush=True)
	if not res["ok"]:
		sys.exit(1)
