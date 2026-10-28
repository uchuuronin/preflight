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
import uuid
from pathlib import Path

import modal

from preflight.runner.queue import MAX_RUNNING

APP_NAME = "preflight-runner"
SERVICE_NAME = "JobStoreService"
# Deployed or run from the repo, this file is <repo>/python/preflight/runner/modal_app.py and the
# image definitions below read the repo through ROOT. Inside a container Modal loads this file on
# its own, as /root/modal_app.py (module `modal_app`, not `preflight.runner.modal_app`), so no
# lookup here may assume the repo layout: ROOT is only read while building images, which happens
# on the machine that deploys. Package files at runtime come from the installed `preflight`.
_HERE = Path(__file__).resolve()
ROOT = (
	_HERE.parents[3]
	if len(_HERE.parents) > 3 and (_HERE.parents[3] / "zoo").is_dir()
	else Path("/repo-not-here")
)
PKG = ROOT / "python" / "preflight"
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

# The public demo probes on GPUs up to A100-80GB, never H200 (Chiya, Oct 27); a visitor's OOM may
# still step up within the job's dollar budget (pipeline.JOB_BUDGET_USD).
GPU_CEILING = "A100-SXM-80GB"
# derived: Modal's sandbox rates (3x list: $0.0000394 per core-s, $0.00000667 per GiB-s, modal.com/pricing,
# Oct 27) for ~4 cores and ~16 GiB while a 1.5B LoRA probe loads and trains = $0.95/h. guess on the
# usage figures: the first smoke run's Modal bill replaces them
SANDBOX_OVERHEAD_USD_PER_H = 0.95
_base = modal.Image.debian_slim(python_version="3.11").env(
	{"PYTHONPATH": REMOTE_PKG, "PREFLIGHT_ZOO": "/root/zoo", "PREFLIGHT_GPU_CEILING": GPU_CEILING}
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
	import preflight

	pkg = Path(preflight.__file__).resolve().parent  # the installed package, here or in the container
	h = hashlib.sha256("\n".join(PROBE_PACKAGES).encode())
	files = [p for sub in ("recorder", "probes") for p in sorted((pkg / sub).glob("*.py"))]
	if not files:  # never key the cache on nothing: a wrong path would make every probe a hit
		raise RuntimeError(f"no recorder or probe sources under {pkg}")
	for p in files:
		h.update(p.relative_to(pkg).as_posix().encode())
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
	def fail(self, lease: dict, error: str, retryable: bool, error_code: str | None = None) -> None:
		self.m.fail(lease, error, retryable, error_code)

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

	return ModelClient.meta() if ModelClient.meta_configured() else None


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
	writer = LocalRecordWriter(RECORDS_DIR, on_write=records_vol.commit, on_read=records_vol.reload)
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
				image=tag, blobs=blobs, reader=writer, install_requirements=True,
				overhead_usd_per_h=SANDBOX_OVERHEAD_USD_PER_H)  # fmt: skip
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
def prefetch(models: str = "") -> list[str]:
	"""Download weights and tokenizers into the read-only weights volume (zoo models by default).

	models: comma-separated Hub ids (a plain string: `modal run` can't parse list arguments)."""
	from huggingface_hub import snapshot_download

	done = []
	for repo in [m.strip() for m in models.split(",") if m.strip()] or ZOO_MODELS:
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


@app.function(image=worker_image, timeout=300)
def startup_check_worker() -> dict:
	"""What a job-store or worker container does when it starts, without touching the shared state:
	import this module as Modal loads it, build a job store and round-trip its snapshot, take a
	job, and read, map and skeleton the LoRA zoo sample."""
	import tempfile

	from preflight.harness.skeleton import skeleton_record
	from preflight.harness.static import map_static
	from preflight.runner.queue import InMemoryJobStore
	from preflight.runner.remote_store import DirBlobs, StoreMethods
	from preflight.runner.sources import ZOO, fetch

	saved: list[dict] = []
	store = InMemoryJobStore(on_change=saved.append)
	m = StoreMethods(store, blobs=DirBlobs(tempfile.mkdtemp()))
	sub = m.submit({"source": {"kind": "sample", "uri": "sample://zoo/finetune_lora_hf"}}, "check", "check-1")
	restored = InMemoryJobStore.restore(store.snapshot())
	lease = restored.lease("check-worker")
	f = fetch(lease.request["source"])
	rec = skeleton_record(map_static(f.files, entrypoint=f.entrypoint), f.source)
	return {"module": __name__, "file": __file__, "zoo": str(ZOO), "zoo_samples": sorted(p.name for p in ZOO.iterdir()),
		"submitted": sub, "saves": len(saved), "lens": rec.job.lens, "knobs": len(rec.knobs),
		"image_tag": image_tag()}  # fmt: skip


@app.function(image=probe_image, timeout=600)  # CPU on purpose: proves the probe path without a GPU
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


@app.local_entrypoint()
def startup_check():
	"""Before any deploy: import and start both images' code as Modal runs it, with no GPU.

	modal run --env main python/preflight/runner/modal_app.py::startup_check
	Writes out/startup_check.json; exits non-zero if either side fails."""
	import json
	import sys

	result: dict = {}
	for name, fn in (("worker", startup_check_worker), ("probe", startup_check_probe)):
		try:
			result[name] = fn.remote()
		except Exception as exc:  # the remote traceback is in the log above
			result[name] = {"failed": f"{type(exc).__name__}: {exc}"[:2000]}
	w, p = result["worker"], result["probe"]
	result["ok"] = bool(
		"failed" not in w and w.get("lens") == "finetune" and w.get("knobs", 0) > 0
		and "failed" not in p and p.get("status") == "ok" and p.get("reachable_by_probe_uid")
	)  # fmt: skip
	Path("out").mkdir(exist_ok=True)
	Path("out/startup_check.json").write_text(json.dumps(result, indent=2))
	print(json.dumps(result, indent=2), flush=True)
	if not result["ok"]:
		sys.exit(1)


@app.local_entrypoint()
def smoke(
	sample: str = "sample://zoo/finetune_lora_hf",
	gpus: str = "L4",
	max_minutes: int = 25,
	modal_env: str = "main",
):
	"""Submit one zoo script and follow it to the record: the G1 critical path in one command.

	Probes stay on `gpus` (comma-separated): L4 by default, since team runs use T4 or L4 unless
	Chiya approves a bigger GPU, and without a constraint an OOM would step up the ladder. The job
	is cancelled after `max_minutes`. Writes out/smoke.json (every event and the final status) and
	out/record.json (the flight record), which remote-run.yml brings back with the log.
	"""
	import json
	import time

	svc = modal.Cls.from_name(APP_NAME, SERVICE_NAME, environment_name=modal_env)()
	request = {
		"source": {"kind": "sample", "uri": sample},
		"constraints": {"gpu_types": [g.strip() for g in gpus.split(",") if g.strip()]},
	}
	sub = svc.submit.remote(request, "smoke", uuid.uuid4().hex)
	print("submitted", sub, flush=True)
	out = Path("out")
	out.mkdir(exist_ok=True)
	events: list[dict] = []
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
	print(json.dumps(info, indent=2), flush=True)
	(out / "smoke.json").write_text(json.dumps({"submitted": sub, "info": info, "events": events}, indent=2))
	if info.get("share_token"):
		try:
			data = b"".join(records_vol.read_file(f"{info['share_token']}.json"))
			(out / "record.json").write_bytes(data)
		except Exception as exc:  # the record is on the volume either way; the log says why it's not here
			print(f"could not copy the record: {exc}", flush=True)
