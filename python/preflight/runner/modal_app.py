"""A's execution plane on Modal: the job store, the workers, and a GPU sandbox per job.

	modal deploy python/preflight/runner/modal_app.py          # app "preflight-runner"
	modal run python/preflight/runner/modal_app.py::prefetch   # put the zoo's weights in the volume
	modal run python/preflight/runner/modal_app.py::smoke      # one zoo script, upload to record

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

import contextlib
import hashlib
import os
import uuid
from pathlib import Path

import modal

from preflight.runner.queue import MAX_RUNNING

APP_NAME = "preflight-runner"
SERVICE_NAME = "JobStoreService"
ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT / "python" / "preflight"
REMOTE_PKG = "/root/pf"
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
# pinned so a probe measures the same stack every time; part of the probe-cache key
PROBE_PACKAGES = [
	"torch==2.5.1",
	"transformers==4.46.3",
	"accelerate==1.1.1",
	"peft==0.13.2",
	"trl==0.12.1",
	"datasets==3.1.0",
	"safetensors==0.4.5",
	"psutil==6.1.0",
	"nvidia-ml-py==12.560.30",
	"pydantic>=2.8,<3",
	"numpy>=1.26,<2.2",
	"httpx>=0.27",
]
ZOO_MODELS = ["Qwen/Qwen2.5-1.5B", "gpt2"]
WORKER_TIMEOUT_S = 3600

app = modal.App(APP_NAME)

_base = modal.Image.debian_slim(python_version="3.11").env(
	{"PYTHONPATH": REMOTE_PKG, "PREFLIGHT_ZOO": "/root/zoo"}
)
worker_image = (
	_base.apt_install("git")
	.pip_install("pydantic>=2.8,<3", "numpy>=1.26,<2.2", "httpx>=0.27")
	.add_local_dir(ROOT / "zoo", "/root/zoo")
	.add_local_dir(PKG, f"{REMOTE_PKG}/preflight")
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
		}
	)
	.add_local_dir(PKG, f"{REMOTE_PKG}/preflight")
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
	h = hashlib.sha256("\n".join(PROBE_PACKAGES).encode())
	for sub in ("recorder", "probes"):
		for p in sorted((Path(__file__).resolve().parents[1] / sub).glob("*.py")):
			h.update(p.read_bytes())
	return "probe-" + h.hexdigest()[:16]


class DictCache:
	"""Probe cache over a Modal Dict: content-hash key, probe JSON value."""

	def __init__(self, d):
		self._d = d

	def get(self, key: str) -> dict | None:
		try:
			return self._d.get(key)
		except Exception:
			return None  # a cache outage costs a probe run, never a job

	def put(self, key: str, probe: dict) -> None:
		with contextlib.suppress(Exception):
			self._d[key] = probe


# ---------------------------------------------------------------- the job store


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
		jobs_dict["owner"] = self.generation
		snap = jobs_dict.get("state")
		kw = {"on_change": self._save}
		self.store = InMemoryJobStore.restore(snap, **kw) if snap else InMemoryJobStore(**kw)
		self.m = StoreMethods(self.store, blobs=DirBlobs(UPLOADS_DIR, commit=uploads_vol.commit))
		self._spawns: list[float] = []

	def _save(self, snap: dict) -> None:
		if jobs_dict.get("owner") != self.generation:
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
	def complete(self, lease: dict, record_id: str) -> None:
		self.m.complete(lease, record_id)

	@modal.method()
	def fail(self, lease: dict, error: str, retryable: bool) -> None:
		self.m.fail(lease, error, retryable)

	@modal.method()
	def emit(self, job_id: str, stage: str, fields: dict, token: int | None = None) -> dict:
		return self.m.emit(job_id, stage, fields, token)

	@modal.method()
	def reap(self) -> list[str]:
		return self.m.reap()

	@modal.method()
	def demand(self) -> dict:
		return self.m.demand()


# ---------------------------------------------------------------- workers


def _sandbox_factory():
	def create(gpu: str):
		from preflight.probes.gpu_choice import modal_gpu

		return modal.Sandbox.create(
			app=app,
			image=probe_image,
			gpu=modal_gpu(gpu),
			timeout=30 * 60,
			idle_timeout=10 * 60,  # a sandbox orphaned by a dead worker stops billing within ten minutes
			outbound_domain_allowlist=EGRESS,
			volumes={WEIGHTS_DIR: _read_only(weights_vol)},
			workdir="/tmp",
		)

	return create


def _spark_client():
	from preflight.harness.spark import ModelClient

	key = os.environ.get("META_API_KEY") or os.environ.get("MODEL_API_KEY")
	if not key:
		return None
	base = os.environ.get("META_API_BASE") or os.environ.get("META_BASE_URL")
	return ModelClient(api_key=key, **({"base_url": base} if base else {}))


@app.function(
	image=worker_image,
	secrets=[modal.Secret.from_name("preflight-meta")],
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
	from preflight.runner.sandbox_runner import SandboxRunner
	from preflight.runner.worker import run_one

	svc = JobStoreService()
	store = RemoteJobStore(lambda m, *a, **k: getattr(svc, m).remote(*a, **k))
	writer = LocalRecordWriter(RECORDS_DIR, on_write=records_vol.commit)
	blobs = DirBlobs(UPLOADS_DIR, reload=uploads_vol.reload)
	name = f"worker-{uuid.uuid4().hex[:8]}"
	tag = image_tag()
	started = time.monotonic()
	ran = 0
	while True:
		# never lease a job this container can't finish before its own timeout: a killed worker
		# leaves the job to the reaper and a retry, and its sandbox billing until idle_timeout
		if time.monotonic() - started > WORKER_TIMEOUT_S - (JOB_PROBE_WALL_S + 15 * 60):
			return ran
		with SandboxRunner(_sandbox_factory()) as runner:  # one sandbox per job, ended with the job
			deps = Deps(records=writer, runner=runner, spark=_spark_client(), cache=DictCache(cache_dict),
				image=tag, blobs=blobs)  # fmt: skip
			if not run_one(store, name, lambda lease, ctx, deps=deps: run_job(lease, ctx, deps=deps)):
				return ran
		ran += 1


@app.function(image=worker_image, schedule=modal.Period(minutes=1), timeout=120)
def tick() -> dict:
	"""Free leases whose workers died; start workers for jobs waiting in the queue."""
	svc = JobStoreService()
	reaped = svc.reap.remote()
	d = svc.demand.remote()
	for _ in range(min(d["queued"], max(0, MAX_RUNNING - d["running"]))):
		run_worker.spawn()
	if d["oldest_queued_s"] > 600:
		print(f"ALERT backlog: oldest queued job has waited {d['oldest_queued_s']:.0f} s", flush=True)
	return {"reaped": reaped, **d}


# ---------------------------------------------------------------- operations


@app.function(image=probe_image, volumes={WEIGHTS_DIR: weights_vol}, timeout=3600)
def prefetch(models: list[str] | None = None) -> list[str]:
	"""Download weights and tokenizers into the read-only weights volume (zoo models by default)."""
	from huggingface_hub import snapshot_download

	done = []
	for repo in models or ZOO_MODELS:
		snapshot_download(
			repo,
			cache_dir=f"{WEIGHTS_DIR}/hf/hub",
			allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "tokenizer*", "*.tiktoken"],
		)
		done.append(repo)
	weights_vol.commit()
	return done


@app.local_entrypoint()
def smoke(sample: str = "sample://zoo/finetune_lora_hf", gpus: str = "L4"):
	"""Submit one zoo script and follow it to the record: the G1 critical path in one command.

	Probes stay on `gpus` (comma-separated): L4 by default, since team runs use T4 or L4 unless
	Chiya approves a bigger GPU, and without a constraint an OOM would step up the ladder.
	"""
	import json
	import time

	svc = modal.Cls.from_name(APP_NAME, SERVICE_NAME)()
	request = {
		"source": {"kind": "sample", "uri": sample},
		"constraints": {"gpu_types": [g.strip() for g in gpus.split(",") if g.strip()]},
	}
	sub = svc.submit.remote(request, "smoke", uuid.uuid4().hex)
	print("submitted", sub)
	seen = -1
	while True:
		for ev in svc.events.remote(sub["job_id"], seen):
			seen = ev["seq"]
			print(json.dumps(ev))
		info = svc.status.remote(sub["job_id"])
		if info["status"] in ("done", "failed", "cancelled"):
			print(json.dumps(info, indent=2))
			return
		time.sleep(5)
