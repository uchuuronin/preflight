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
import os
import uuid
from pathlib import Path

import modal

from preflight.runner.modal_app import (
	APP_NAME,
	SERVICE_NAME,
	WEIGHTS_DIR,
	ZOO_MODELS,
	image_tag,
	probe_image,
	records_vol,
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


@ops.function(image=probe_image, gpu="L4", volumes={WEIGHTS_DIR: weights_vol}, timeout=3600)
def ground_truth_on_l4(script: str, bare_steps: int, recorder_steps: int, warmup: int) -> dict:
	"""recorder/ground_truth.py on an L4: a bare run, then a recorder run alternating clean and
	synced steps, and the paired overhead. The AMD droplet runs the same module."""
	from preflight.probes import remote
	from preflight.recorder.ground_truth import ZOO_SCRIPTS, measure
	from preflight.runner.sources import ZOO

	if script not in ZOO_SCRIPTS:
		raise ValueError(f"unknown zoo script {script!r}")
	remote.link_weights()
	remote.deps_dir()
	return measure(ZOO / script, bare_steps=bare_steps, recorder_steps=recorder_steps, warmup=warmup)


@ops.local_entrypoint()
def ground_truth(script: str, bare_steps: int, recorder_steps: int = 60, warmup: int = 5):
	"""modal run --env main python/preflight/runner/ops_app.py::ground_truth --script finetune_lora_hf --bare-steps 210
	Writes out/ground_truth_<script>.json (step walls, peak memory, how the bare run ended, overhead)."""
	import json

	from preflight.recorder.ground_truth import brief

	res = ground_truth_on_l4.remote(script, bare_steps, recorder_steps, warmup)
	Path("out").mkdir(exist_ok=True)
	Path(f"out/ground_truth_{script}.json").write_text(json.dumps(res, indent=2))
	print(json.dumps(brief(res), indent=2), flush=True)
