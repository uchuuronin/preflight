"""A's execution plane on Modal: the job store, the workers, and a GPU sandbox per job.

	modal run --env main python/preflight/runner/ops_app.py::startup_check   # first, every time
	modal deploy --env main python/preflight/runner/modal_app.py            # app "preflight-runner"
	modal run --env main python/preflight/runner/ops_app.py::prefetch        # zoo weights into the volume
	modal run --env main python/preflight/runner/ops_app.py::smoke --gpus L4 # one zoo script to its record
	python -m preflight.runner.stop_app --env main                          # stop it

What runs where:
  JobStoreService  one container, max 1, the serialised source of truth for jobs (gap 2);
                   state saved to the Modal Dict "preflight-jobs" on every change.
                   C's API calls submit/status/position/events/cancel with .remote().
  run_worker       CPU containers, at most MAX_RUNNING: lease a job, run the pipeline,
                   write the record to the "preflight-records" volume. Holds the Muse
                   Spark key (secret "preflight-meta"); never runs visitor code.
  probe sandbox    one modal.Sandbox per job (and per GPU if the job steps up): GPU,
                   no secrets, egress only to Hugging Face, PyPI and GitHub, weights
                   volume mounted read-only. Visitor code runs only here, under
                   run_probe's own isolation (unprivileged uid, scrubbed env, limits).
  tick             every minute: reap expired leases, start workers for queued jobs.

C mounts "preflight-records" read-only and serves <share_token>.json until R2 lands.
"""

# Deploy-only. `modal run`, `modal serve` or `modal shell` of this file starts a temporary copy of
# the app whose job store (min_containers=1) would claim the shared job state and lock the deployed
# store out (Oct 28). JobStoreService.load therefore claims only in the deployed app's own
# containers (deployed_here); a temporary copy refuses every call. Every `modal run` entry
# (startup check, prefetch, smoke) lives in runner/ops_app.py.

import contextlib
import hashlib
import os
import uuid
from pathlib import Path

import modal

from preflight.runner.images import PROBE_NEW_PACKAGES  # modal-free: pipeline.cache_key reads it too
from preflight.runner.ownership import decide_ownership
from preflight.runner.queue import MAX_RUNNING

APP_NAME = "preflight-runner"
SERVICE_NAME = "JobStoreService"
# Deployed or run from the repo, this file is <repo>/python/preflight/runner/modal_app.py and the
# image definitions below read the repo through ROOT. Inside a container Modal loads this file on
# its own, as /root/modal_app.py (module `modal_app`, not `preflight.runner.modal_app`), so no
# lookup here may assume the repo layout. Images are also built inside a container: a sandbox's
# image (probe_image) is resolved where Sandbox.create runs, in a worker, and every local path it
# names must exist there (Oct 28: it pointed into the repo, and every probe failed in 0.01 s). So
# each path comes from the repo when it is here and from where the image put it otherwise; the
# contents are the same files, so the image and mount hashes don't change.
_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[3] if len(_HERE.parents) > 3 and (_HERE.parents[3] / "zoo").is_dir() else None
ROOT = _REPO if _REPO is not None else Path("/repo-not-here")


def _installed_package() -> Path:
	import preflight

	return Path(preflight.__file__).resolve().parent


PKG = _REPO / "python" / "preflight" if _REPO is not None else _installed_package()
# bytecode differs between the deploy machine and a container: keep it out so the mounts match
_NOT_SOURCE = ["**/__pycache__", "**/__pycache__/**", "**/*.pyc"]
ZOO_SRC = _REPO / "zoo" if _REPO is not None else Path(os.environ.get("PREFLIGHT_ZOO", "/root/zoo"))
# the feature flags the worker reads (runner/flags.py looks beside the package)
FLAGS_SRC = _REPO / "flags.json" if _REPO is not None else Path("/opt/preflight/flags.json")
REMOTE_PKG = "/opt/preflight"  # not under /root (0700): probes import it as an unprivileged uid
RECORDS_DIR = "/records"
UPLOADS_DIR = "/uploads"
WEIGHTS_DIR = "/weights"
HF_HOME = "/tmp/hf"

# what a probe may reach: models and tokenizers, packages a script installs, code it clones
EGRESS = [
	"huggingface.co",
	"*.huggingface.co",
	"*.hf.co",
	"pypi.org",
	"files.pythonhosted.org",
	"github.com",
	"codeload.github.com",
	"*.githubusercontent.com",
]
# probes.extra_hosts (off): PyG's wheel index (torch-sparse and the rest, which PyPI has only as source) and
# torchvision's dataset host for CIFAR (Nov 2: unseen scripts 8 and 9 stopped there)
# torchvision 0.20's CIFAR comes from cave.cs.toronto.edu (a-all3-9, Nov 2), not www.cs.toronto.edu
EXTRA_EGRESS = ["data.pyg.org", "www.cs.toronto.edu", "cave.cs.toronto.edu"]
PYG_INDEX = "https://data.pyg.org/whl/torch-2.5.0+cu124.html"  # PyPI's torch 2.5.1 is built for CUDA 12.4
# pinned so a probe measures the same stack every time; part of the probe-cache key
PROBE_PACKAGES = [
	"torch==2.5.1",
	# Nov 2, after freeze: torchvision stopped two of the nine unseen scripts at import; 0.20.1 is torch 2.5.1's
	"torchvision==0.20.1",
	"uv==0.5.11",  # probes.uv_install (off): its resolver before pip's
	"transformers==4.46.3",
	"accelerate==1.1.1",
	"peft==0.13.2",
	"trl==0.14.0",  # GRPOTrainer arrived in 0.14 (zoo/grpo_trl)
	"datasets==3.1.0",
	"safetensors==0.4.5",
	"psutil==6.1.0",
	"nvidia-ml-py==12.560.30",
	"pydantic>=2.8,<3",
	"numpy>=1.26,<2.2",
	"httpx>=0.27",
]
# "dataset:" marks a dataset repo. D's eval, retrieval and vision scripts' weights and data, so a
# probe's time limit isn't spent downloading them (the Nov 1 eval sweep probe ran out inside it)
ZOO_MODELS = ["Qwen/Qwen2.5-1.5B", "gpt2", "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
	"sentence-transformers/all-MiniLM-L6-v2", "timm/resnet50.a1_in1k", "dataset:openai/gsm8k",
	"dataset:sentence-transformers/natural-questions", "dataset:AI-Lab-Makerere/beans"]  # fmt: skip
WORKER_TIMEOUT_S = 3600

app = modal.App(APP_NAME)

# The public demo probes on GPUs up to the L40S (Chiya, Nov 1; A100-80GB before, never H200); a visitor's OOM may
# still step up within the job's dollar budget (pipeline.JOB_BUDGET_USD).
GPU_CEILING = "L40S"  # Chiya, Nov 1 9:47 PM: was A100-SXM-80GB, the web's default request; 48 GB fits a 7B LoRA, A100 numbers come from the cross-GPU model
# derived: Modal's sandbox rates (3x list: $0.0000394 per core-s, $0.00000667 per GiB-s, modal.com/pricing,
# Oct 27) for ~4 cores and ~16 GiB while a 1.5B LoRA probe loads and trains = $0.95/h. guess on the
# usage figures: the first smoke run's Modal bill replaces them
SANDBOX_OVERHEAD_USD_PER_H = 0.95
_base = modal.Image.debian_slim(python_version="3.11").env(
	{"PYTHONPATH": REMOTE_PKG, "PREFLIGHT_ZOO": "/root/zoo", "PREFLIGHT_GPU_CEILING": GPU_CEILING}
)
# The worker reads flags as C's API does (runner/flags.py) and ships the same flag files as the
# API's image (api/app.py), with PREFLIGHT_ENV=production: flags.json, then flags.production.json
# (Chiya, Oct 30) when the repo has it. Both are baked in, so changing a flag takes a deploy of
# the runner as well as the API; tests/test_pipeline.py holds the two images to the same files.
FLAG_ENV = "production"  # api/app.py's PREFLIGHT_ENV
FLAG_OVERRIDE = FLAGS_SRC.parent / f"flags.{FLAG_ENV}.json"
worker_image = (
	_base.apt_install("git")
	.pip_install("pydantic>=2.8,<3", "numpy>=1.26,<2.2", "httpx>=0.27")
	.pip_install("matplotlib==3.9.2")  # harness/diagnosis.py renders the plots Muse Spark reads (item 31)
	.env({"PREFLIGHT_ENV": FLAG_ENV})
	.add_local_dir(ZOO_SRC, "/root/zoo", ignore=_NOT_SOURCE)
	.add_local_dir(PKG, f"{REMOTE_PKG}/preflight", ignore=_NOT_SOURCE)
	.add_local_file(FLAGS_SRC, f"{REMOTE_PKG}/flags.json")
)
if FLAG_OVERRIDE.is_file():
	worker_image = worker_image.add_local_file(FLAG_OVERRIDE, f"{REMOTE_PKG}/flags.{FLAG_ENV}.json")
# a C compiler in the sandbox images: Triton builds its launcher and inductor's CPU backend builds its
# kernels with one at run time, so a visitor's torch.compile needs it; a layer after the packages
COMPILER_PACKAGES = ("gcc", "g++")
# the probe uid's /etc/passwd entry (probes/local.py PROBE_USER), for code that calls pwd.getpwuid
# directly; getpass.getuser() reads the USER the probe sets. A layer after the packages, so they stay cached
PROBE_USER_CMD = (  # the group too: the probe runs as gid 61000 (probes/local.py, group=uid)
	"groupadd --gid 61000 preflight-probe && useradd --uid 61000 --gid 61000 --no-create-home"
	" --home-dir /tmp --shell /usr/sbin/nologin preflight-probe"
)

probe_image = (
	_base.apt_install("git")
	.pip_install(*PROBE_PACKAGES)
	.env(
		{
			"HF_HOME": HF_HOME,
			"PREFLIGHT_WEIGHTS_HUB": f"{WEIGHTS_DIR}/hf/hub",
			"PREFLIGHT_SANDBOX_DIR": "/tmp",
			# one unprivileged uid for all of a job's probes (the sandbox is per job): they share the HF cache
			"PREFLIGHT_PROBE_UID": "61000",
			"PREFLIGHT_PYG_INDEX": PYG_INDEX,  # read only with probes.extra_hosts (recorder/requirements.py)
		}
	)
	.apt_install(*COMPILER_PACKAGES)
	.run_commands(PROBE_USER_CMD)
	.add_local_dir(PKG, f"{REMOTE_PKG}/preflight", ignore=_NOT_SOURCE)
)

# serving probes only (zoo/serving_vllm): vLLM 0.6.4.post1 pins torch 2.5.1 and needs numpy below 2,
# and its pyairports dependency has no wheel, so it can't share the probe image (numpy 2, wheels
# only). Its own pins; the probe image above is unchanged.
SERVING_PACKAGES = [
	"vllm==0.6.4.post1",
	"torch==2.5.1",
	"transformers==4.46.3",
	"numpy<2",
	"datasets==3.1.0",
	# datasets 3.1 takes any pyarrow 15 or later, and the newest needs numpy 2; 17.0 runs on numpy 1
	# (the serving smoke failed on it, Nov 1)
	"pyarrow==17.0.0",
	"psutil==6.1.0",
	# what the recorder imports in the sandbox, pinned as in the probe image
	"nvidia-ml-py==12.560.30",
	"pydantic>=2.8,<3",
	"httpx>=0.27",
]
serving_image = (
	_base.apt_install("git")
	.pip_install(*SERVING_PACKAGES)
	.env(
		{
			"HF_HOME": HF_HOME,
			"PREFLIGHT_WEIGHTS_HUB": f"{WEIGHTS_DIR}/hf/hub",
			"PREFLIGHT_SANDBOX_DIR": "/tmp",
			"PREFLIGHT_PROBE_UID": "61000",
			"PREFLIGHT_PYG_INDEX": PYG_INDEX,  # read only with probes.extra_hosts (recorder/requirements.py)
		}
	)
	.apt_install(*COMPILER_PACKAGES)
	.run_commands(PROBE_USER_CMD)
	.add_local_dir(PKG, f"{REMOTE_PKG}/preflight", ignore=_NOT_SOURCE)
)
# probes.torch_new (off; the PM, Nov 3): code whose own requirements ask for torch 2.6 or later (LitGPT's
# torch>=2.7, unseen 2) probes here; the stack moves with torch, and the image's name keys its probes' cache
# (pipeline.cache_key), so the default image's cache stays as it is
PYG_INDEX_NEW = "https://data.pyg.org/whl/torch-2.7.0+cu126.html"  # PyPI's torch 2.7.1 is built for CUDA 12.6
probe_new_image = (
	_base.apt_install("git")
	.pip_install(*PROBE_NEW_PACKAGES)
	.env(
		{
			"HF_HOME": HF_HOME,
			"PREFLIGHT_WEIGHTS_HUB": f"{WEIGHTS_DIR}/hf/hub",
			"PREFLIGHT_SANDBOX_DIR": "/tmp",
			"PREFLIGHT_PROBE_UID": "61000",
			"PREFLIGHT_PYG_INDEX": PYG_INDEX_NEW,
		}
	)
	.apt_install(*COMPILER_PACKAGES)
	.run_commands(PROBE_USER_CMD)
	.add_local_dir(PKG, f"{REMOTE_PKG}/preflight", ignore=_NOT_SOURCE)
)
SANDBOX_IMAGES = {"probe": probe_image, "serving": serving_image, "torch27": probe_new_image}

# the torchtitan adapter check only (zoo/torchtitan_debug, an ops run): torchtitan 0.1.0 targets
# torch 2.7, which the probe image's 2.5.1 isn't, so its own pins; no visitor job uses this image
TITAN_PACKAGES = [
	"torch==2.7.1",
	"torchtitan==0.1.0",
	"torchdata==0.11.0",
	"datasets==3.6.0",
	"tiktoken==0.9.0",
	"blobfile==3.0.0",
	"tomli==2.2.1",
	"tyro==0.9.24",
	"tensorboard==2.19.0",
	"psutil==6.1.0",
	"nvidia-ml-py==12.560.30",
	"pydantic>=2.8,<3",
	"numpy>=1.26,<2.3",
	"httpx>=0.27",
]
titan_image = (
	_base.pip_install(*TITAN_PACKAGES)
	.env({"HF_HOME": HF_HOME, "PREFLIGHT_SANDBOX_DIR": "/tmp", "PREFLIGHT_PROBE_UID": "61000"})
	.apt_install(*COMPILER_PACKAGES)
	.run_commands(PROBE_USER_CMD)
	.add_local_dir(PKG, f"{REMOTE_PKG}/preflight", ignore=_NOT_SOURCE)
)

jobs_dict = modal.Dict.from_name("preflight-jobs", create_if_missing=True)
cache_dict = modal.Dict.from_name("preflight-probe-cache", create_if_missing=True)
records_vol = modal.Volume.from_name("preflight-records", create_if_missing=True)
weights_vol = modal.Volume.from_name("preflight-weights", create_if_missing=True)
# uploaded files, by content hash: kept out of the job state, which is saved on every change
uploads_vol = modal.Volume.from_name("preflight-uploads", create_if_missing=True)


def _read_only(vol):
	try:
		return vol.with_mount_options(read_only=True)
	except AttributeError:  # older SDKs
		return vol.read_only()


def image_tag() -> str:
	"""Pinned packages plus the recorder's own source: a change to either invalidates cached probes."""
	import preflight

	pkg = Path(preflight.__file__).resolve().parent  # the installed package, here or in the container
	h = hashlib.sha256("\n".join(PROBE_PACKAGES).encode())
	h.update("\n".join(SERVING_PACKAGES).encode())  # the serving image's pins key its probes too
	files = [p for sub in ("recorder", "probes") for p in sorted((pkg / sub).glob("*.py"))]
	if not files:  # never key the cache on nothing: a wrong path would make every probe a hit
		raise RuntimeError(f"no recorder or probe sources under {pkg}")
	for p in files:
		h.update(p.relative_to(pkg).as_posix().encode())
		h.update(p.read_bytes())
	return "probe-" + h.hexdigest()[:16]


class DictCache:
	"""Probe cache over a Modal Dict: content-hash key, the probe and when it was cached (entries
	expire at 30 days, runner/retention.py)."""

	def __init__(self, d):
		self._d = d

	def get(self, key: str) -> dict | None:
		from preflight.runner.retention import cached_probe

		try:
			return cached_probe(self._d.get(key))
		except Exception:
			return None  # a cache outage costs a probe run, never a job

	def put(self, key: str, probe: dict) -> None:
		from preflight.runner.retention import cache_entry

		with contextlib.suppress(Exception):
			self._d[key] = cache_entry(probe)


# ---------------------------------------------------------------- the job store


def _container_environment() -> str | None:
	"""The Modal environment this container runs in. Modal 1.6.3 has no public accessor; it keeps
	it on the container IO manager (container_args.environment_name)."""
	try:
		from modal._runtime.container_io_manager import _ContainerIOManager

		env = getattr(getattr(_ContainerIOManager, "_singleton", None), "_environment_name", None)
	except Exception:
		env = None
	return env or os.environ.get("MODAL_ENVIRONMENT") or None


def deployed_here(app_name: str = APP_NAME) -> tuple[bool | None, str]:
	"""Whether this container belongs to the deployed `app_name` in its own environment, and why.

	True: it does. False: the deployed app was found and this container is another app (a
	temporary copy from modal run/serve/shell). None: it couldn't tell after retries (a lookup that
	keeps failing); the job store then fails to start, so Modal restarts it, rather than serving
	refusals for the container's whole life (decide_ownership has the rules)."""
	import time

	try:
		mine = modal.App._get_container_app()
		mine_id = mine.app_id if mine is not None else None
	except Exception as exc:
		return None, f"no container app ({type(exc).__name__})"
	env = _container_environment()

	def lookup():
		return modal.App.lookup(app_name, environment_name=env).app_id

	return decide_ownership(mine_id, lookup, where=f"{app_name} in {env or 'the default environment'}",
		env_known=bool(env), sleep=time.sleep)  # fmt: skip


class _Refusing:
	"""The job store's methods in a container that isn't the deployed app's: every call refuses."""

	def __init__(self, why: str):
		self.why = why

	def __getattr__(self, name):
		from preflight.runner.queue import StoreUnavailable

		def refuse(*_a, **_k):
			raise StoreUnavailable(f"not the deployed job store: {self.why}")

		return refuse


@app.cls(
	image=worker_image,
	volumes={UPLOADS_DIR: uploads_vol},
	max_containers=1,
	min_containers=1,
	timeout=24 * 3600,
	scaledown_window=1200,
)
@modal.concurrent(max_inputs=64)
class JobStoreService:
	@modal.enter()
	def load(self):
		from preflight.runner.queue import InMemoryJobStore
		from preflight.runner.remote_store import DirBlobs, StoreMethods

		# fencing for the saved state: during a redeploy an old and a new container can overlap.
		# The new one claims ownership before it reads, so the old one's saves fail from then on
		# (StoreUnavailable, rolled back) and nothing it accepts after the read is lost silently.
		# Modal Dicts have no compare-and-set, so a save already in flight can still land: deploys
		# of this service happen with the queue drained.
		self.generation = uuid.uuid4().hex
		self._spawns: list[float] = []
		ok, why = deployed_here()
		env = _container_environment()
		print(
			f"job store (environment {env!r}): {'claiming' if ok else 'NOT claiming'} the job state: {why}",
			flush=True,
		)
		if ok is None:
			# couldn't tell: fail the start so Modal restarts the container and asks again
			raise RuntimeError(f"job store can't confirm it is the deployed app: {why}")
		if not ok:
			# a temporary copy (modal run/serve/shell of this file): never touch the shared state
			self.m = _Refusing(why)
			return
		jobs_dict["owner"] = self.generation
		snap = jobs_dict.get("state")
		kw = {"on_change": self._save}
		self.store = InMemoryJobStore.restore(snap, **kw) if snap else InMemoryJobStore(**kw)
		self.m = StoreMethods(self.store, blobs=DirBlobs(UPLOADS_DIR, commit=uploads_vol.commit))

	def _save(self, snap: dict) -> None:
		if jobs_dict.get("owner") != self.generation:
			# only a newer deployed container can have claimed it (deployed_here): this one is
			# being replaced, so stop taking calls and let Modal route them to the new one
			with contextlib.suppress(Exception):
				import modal.experimental

				modal.experimental.stop_fetching_inputs()
			raise RuntimeError("a newer job-store container owns the state")
		jobs_dict["state"] = snap

	def _kick(self) -> None:
		"""Start a worker unless enough are running or starting: a burst of submits must not
		queue hundreds of worker starts behind the GPU cap."""
		import time

		now = time.monotonic()
		self._spawns = [t for t in self._spawns if now - t < 60]
		if self.m.demand()["running"] + len(self._spawns) >= MAX_RUNNING:
			return  # running workers loop until the queue is empty; tick covers the rest
		self._spawns.append(now)
		with contextlib.suppress(Exception):  # if it fails, tick starts a worker within a minute
			run_worker.spawn()

	@modal.method()
	def submit(self, request: dict, visitor: str, idempotency_key: str) -> dict:
		out = self.m.submit(request, visitor, idempotency_key)
		if out["created"]:
			self._kick()
		return out

	@modal.method()
	def status(self, job_id: str) -> dict:
		return self.m.status(job_id)

	@modal.method()
	def position(self, job_id: str) -> int | None:
		return self.m.position(job_id)

	@modal.method()
	def events(self, job_id: str, after: int = -1) -> list[dict]:
		return self.m.events(job_id, after)

	@modal.method()
	def cancel(self, job_id: str) -> None:
		self.m.cancel(job_id)

	@modal.method()
	def lease(self, worker: str) -> dict | None:
		return self.m.lease(worker)

	@modal.method()
	def heartbeat(self, lease: dict) -> float:
		return self.m.heartbeat(lease)

	@modal.method()
	def complete(self, lease: dict, record_id: str, spent_usd: float | None = None) -> None:
		self.m.complete(lease, record_id, spent_usd)

	@modal.method()
	def fail(
		self,
		lease: dict,
		error: str,
		retryable: bool,
		error_code: str | None = None,
		spent_usd: float | None = None,
	) -> None:
		self.m.fail(lease, error, retryable, error_code, spent_usd)

	@modal.method()
	def emit(self, job_id: str, stage: str, fields: dict, token: int | None = None) -> dict:
		return self.m.emit(job_id, stage, fields, token)

	@modal.method()
	def reap(self) -> list[str]:
		return self.m.reap()

	@modal.method()
	def expire(self, older_than_s: float) -> int:
		return self.m.expire(older_than_s)

	@modal.method()
	def demand(self) -> dict:
		return self.m.demand()


# ---------------------------------------------------------------- workers


def _sandbox_factory(owner=None):
	"""The probe sandbox, created inside `owner` (this app by default; the startup check passes its
	own running app). gpu "cpu" makes a GPU-less sandbox with every other setting the same."""

	def create(
		gpu: str, extra_egress: list[str] | tuple = (), image: str = "probe", timeout_s: int | None = None
	):
		"""extra_egress: hosts beyond EGRESS, only the agent proxy's tunnel (runner/agent_jobs.py).
		image: "probe", or "serving" for vLLM probes (SANDBOX_IMAGES). timeout_s: the sandbox's
		lifetime, 30 min unless the caller's run needs longer (an agent sample)."""
		from preflight.probes.gpu_choice import modal_gpu
		from preflight.runner.flags import flag

		# probes.extra_hosts: a probe's sandbox only; an agent sample's (the one caller with extra_egress) keeps
		# EGRESS and its tunnel
		data = flag("probes.extra_hosts") and not extra_egress
		hosts = [*EGRESS, *(EXTRA_EGRESS if data else []), *(h for h in extra_egress if h)]
		return modal.Sandbox.create(
			app=owner if owner is not None else app,
			image=SANDBOX_IMAGES[image],
			gpu=None if gpu == "cpu" else modal_gpu(gpu),
			timeout=timeout_s or 30 * 60,
			idle_timeout=10 * 60,  # a sandbox orphaned by a dead worker stops billing within ten minutes
			outbound_domain_allowlist=hosts,
			volumes={WEIGHTS_DIR: _read_only(weights_vol)},
			workdir="/tmp",
		)

	return create


def _spark_client():
	from preflight.harness.spark import ModelClient

	return ModelClient.meta() if ModelClient.meta_configured() else None


def _glimmer_client():
	"""Muse Glimmer on B's host for private-mode triage, when its URL and proxy keys are in the
	environment (the preflight-glimmer secret, mounted once it exists)."""
	from preflight.harness.spark import ModelClient

	return ModelClient.glimmer() if ModelClient.glimmer_configured() else None


def _triage_client():
	"""Spark Open on Baseten for probe-log triage, when the preflight-baseten secret is mounted."""
	from preflight.harness.spark import ModelClient

	return ModelClient.baseten_spark_open() if os.environ.get("BASETEN_API_KEY") else None


@app.function(
	image=worker_image,
	# preflight-glimmer: a private job's triage (harness.glimmer_private), key from the secret store
	# only; without it a private check-in got no triage at all (C, Nov 1)
	secrets=[
		modal.Secret.from_name("preflight-meta"),
		modal.Secret.from_name("preflight-baseten"),
		modal.Secret.from_name("preflight-glimmer"),
	],
	volumes={RECORDS_DIR: records_vol, UPLOADS_DIR: _read_only(uploads_vol)},
	timeout=WORKER_TIMEOUT_S,
	max_containers=MAX_RUNNING,
)
def run_worker() -> int:
	"""Run jobs until the queue is empty. Returns how many it ran."""
	import time

	from preflight.runner.pipeline import JOB_PROBE_WALL_S, Deps, run_job
	from preflight.runner.records import LocalRecordWriter
	from preflight.runner.remote_store import DirBlobs, RemoteJobStore
	from preflight.runner.sandbox_runner import WAIT_CAP_S, SandboxRunner
	from preflight.runner.worker import run_one

	svc = JobStoreService()
	store = RemoteJobStore(lambda m, *a, **k: getattr(svc, m).remote(*a, **k))
	writer = LocalRecordWriter(RECORDS_DIR, on_write=records_vol.commit, on_read=records_vol.reload)
	blobs = DirBlobs(UPLOADS_DIR, reload=uploads_vol.reload)
	name = f"worker-{uuid.uuid4().hex[:8]}"
	tag = image_tag()
	started = time.monotonic()
	ran = 0
	while True:
		# never lease a job this container can't finish before its own timeout: a killed worker
		# leaves the job to the reaper and a retry, and its sandbox billing until idle_timeout
		# a job's worst case: its GPU wait (capped), its probe wall, and the rest of the job
		if time.monotonic() - started > WORKER_TIMEOUT_S - (WAIT_CAP_S + JOB_PROBE_WALL_S + 15 * 60):
			return ran
		from preflight.probes.gpu_choice import usd_per_hour

		# one sandbox per job, ended with the job; an L4 that won't schedule falls back to A10, L40S
		with SandboxRunner(_sandbox_factory(), price=usd_per_hour) as runner:
			deps = Deps(records=writer, runner=runner, spark=_spark_client(), triage=_triage_client(), glimmer=_glimmer_client(),
				cache=DictCache(cache_dict),
				image=tag, blobs=blobs, reader=writer, install_requirements=True,
				overhead_usd_per_h=SANDBOX_OVERHEAD_USD_PER_H)  # fmt: skip
			if not run_one(store, name, lambda lease, ctx, deps=deps: run_job(lease, ctx, deps=deps)):
				return ran
		ran += 1


PRUNE_EVERY_S = 86_400  # PM decision (Oct 28): records go after 30 days, checked once a day


def _prune_records_daily() -> dict:
	"""Once a day (the last run's time is kept in the jobs Dict), delete everything about a job
	older than 30 days: its record, its uploaded files, its cached probe results and its entry in
	the job index (runner/retention.py). Each part runs even if another fails."""
	import itertools
	import json
	import time

	from preflight.runner.records import prune
	from preflight.runner.retention import TTL_S, prune_cache, prune_uploads, records_text, upload_refs

	last = jobs_dict.get("records_pruned_at") or 0
	if time.time() - last < PRUNE_EVERY_S:
		return {}
	jobs_dict["records_pruned_at"] = time.time()  # first, so a failure doesn't retry every minute
	out: dict = {}

	def part(name, fn):
		try:
			out[name] = fn()
		except Exception as exc:
			out[name] = f"failed: {type(exc).__name__}"
			print(f"cleanup of {name} failed: {type(exc).__name__}: {str(exc)[:300]}", flush=True)

	def records():
		records_vol.reload()
		removed = prune(RECORDS_DIR)
		if removed:
			records_vol.commit()
		return len(removed)

	def uploads():
		uploads_vol.reload()
		records_vol.reload()
		state = jobs_dict.get("state") or {}
		live = [json.dumps(j.get("request")) for j in state.get("jobs", []) if isinstance(j, dict)]
		keep = upload_refs(itertools.chain(records_text(RECORDS_DIR), live))  # one record at a time
		removed = prune_uploads(UPLOADS_DIR, keep=keep)
		if removed:
			uploads_vol.commit()
		return len(removed)

	def cache():
		removed = 0
		for key in list(cache_dict.keys()):  # one entry at a time: the values can be large
			removed += len(prune_cache([(key, cache_dict.get(key))], cache_dict.pop))
		return removed

	def jobs():
		return JobStoreService().expire.remote(TTL_S)

	for name, fn in (("records", records), ("uploads", uploads), ("probe_cache", cache), ("job_index", jobs)):
		part(name, fn)
	print(f"daily cleanup (older than 30 days): {out}", flush=True)
	return out


CLEANUP_TIMEOUT_S = 1800  # guess: a first pass over a month of records, uploads and cache entries


@app.function(
	image=worker_image,
	timeout=CLEANUP_TIMEOUT_S,
	volumes={RECORDS_DIR: records_vol, UPLOADS_DIR: uploads_vol},
)
def cleanup() -> dict:
	"""The daily 30-day cleanup, in its own container so it never holds up the minute's tick."""
	if deployed_here()[0] is not True:  # a temporary copy never deletes shared data
		return {}
	return _prune_records_daily()


@app.function(image=worker_image, schedule=modal.Period(minutes=1), timeout=120)
def tick() -> dict:
	"""Free leases whose workers died; start workers for jobs waiting in the queue; once a day,
	start the cleanup of everything about a job older than 30 days."""
	import time

	cleaning = False
	try:  # a failed start never holds up the queue, but it says so
		now = time.time()
		due = now - (jobs_dict.get("records_pruned_at") or 0) >= PRUNE_EVERY_S
		# one cleanup at a time: it marks itself done only once it starts, which can take a minute
		busy = now - (jobs_dict.get("cleanup_started_at") or 0) < CLEANUP_TIMEOUT_S
		if due and not busy and deployed_here()[0] is True:  # a temporary copy never deletes shared data
			jobs_dict["cleanup_started_at"] = now
			cleanup.spawn()
			cleaning = True
	except Exception as exc:
		print(f"cleanup start failed: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
	warm = None
	try:  # judging's warm worker (runner/judging.py, option (a)): set once when the wanted count changes
		from preflight.runner.judging import warm_workers

		want = warm_workers(time.time())
		# keyed on the deploy too: a redeploy resets the worker to its decorator's settings (Modal)
		key = f"{want}:{image_tag()}"
		if jobs_dict.get("warm_workers") != key and deployed_here()[0] is True:
			scaler = getattr(run_worker, "autoscaler", None)
			if scaler is not None and hasattr(scaler, "update"):
				scaler.update(min_containers=want)
			else:
				run_worker.update_autoscaler(min_containers=want)
			jobs_dict["warm_workers"] = key
			warm = want
	except Exception as exc:
		print(f"warm-pool update failed: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
	svc = JobStoreService()
	reaped = svc.reap.remote()
	d = svc.demand.remote()
	for _ in range(min(d["queued"], max(0, MAX_RUNNING - d["running"]))):
		run_worker.spawn()
	if d["oldest_queued_s"] > 600:
		print(f"ALERT backlog: oldest queued job has waited {d['oldest_queued_s']:.0f} s", flush=True)
	return {"reaped": reaped, "cleanup_started": cleaning, "warm_workers_set": warm, **d}
