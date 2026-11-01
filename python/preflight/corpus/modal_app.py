"""Modal app for the NVIDIA corpus slices.

Dispatches in waves. Before each wave it bills the wave's worst case (every container
starting, running a full batch at the deadline and idling out) to a file on the volume,
and launches only if that fits the money and the time left; after the wave it settles
the bill to the measured container time (sweep.plan_wave, Oct 28). A restarted sweep
resumes from what the volume holds and keeps counting what earlier runs cost (gap 13).
Each config is written once; the volume is committed every COMMIT_EVERY configs.

Run from the repo root (PYTHONPATH=python so the package is importable):

    PYTHONPATH=python modal run python/preflight/corpus/modal_app.py --dry-run
    modal run python/preflight/corpus/modal_app.py::main --kind grouped --n 300 --budgets "L4=5.5"
    modal run python/preflight/corpus/modal_app.py::export --kind kernels

Agents start these through remote-run.yml, only after Chiya's yes; the plan and caps are in corpus/README.md.

Add --detach after `modal run` so a laptop disconnect doesn't cancel the sweep (it resumes either way).
"""

from __future__ import annotations

import concurrent.futures as cf
from typing import Any

import modal

APP_NAME = "preflight-corpus"
VOLUME = "preflight-corpus"
MOUNT = "/corpus"
TEXT = "/assets/tinyshakespeare.txt"
COMMIT_EVERY = (
	4  # guess: volume commits per batch traded against configs lost to a preemption; measured per batch
)

# exact pins, at least two weeks old on Oct 27 2026; they also go into every corpus key
PINS = {
	"torch": "torch==2.8.0",
	"transformers": "transformers==5.19.0",
	"peft": "peft==0.21.2",
	"bitsandbytes": "bitsandbytes==0.50.2",
	"accelerate": "accelerate==1.15.0",
}

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)

image = (
	modal.Image.debian_slim(python_version="3.11")
	.apt_install("curl")
	.pip_install(*PINS.values(), "numpy")
	.run_commands(
		"mkdir -p /assets",
		f"curl -fsSL -o {TEXT} https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
	)
	.add_local_python_source("preflight")
)


@app.cls(
	image=image,
	gpu="L4",
	volumes={MOUNT: vol},
	timeout=3 * 60 * 60,  # a batch of 8 configs, each capped by its own deadline in isolate.py
	# no retries on failure; Modal may still restart a preempted input, which stops at the wave's end time (checked
	# first in run_batch, and a watchdog ends the container there); sweep.worst_wave_seconds bills one restart
	max_containers=4,
	scaledown_window=30,
	startup_timeout=180,  # matches sweep.STARTUP_S, so a slow start can't add to the per-wave timeout unbilled  # set, so the idle tail is known: sweep.SCALEDOWN_S bills it (a test checks they match)
)
class Sweeper:
	@modal.method()
	def check(self) -> dict[str, Any]:
		"""Start-up check in the real image: imports, the GPU, the volume. Runs no config."""
		import importlib.metadata as md
		import os

		import torch

		from preflight.corpus import configs, isolate, store, sweep, worker  # noqa: F401
		from preflight.engine import gpus

		props = torch.cuda.get_device_properties(0)
		vol.reload()
		# one small config through the child process and its deadline, written nowhere
		small = next(c for c in configs.kernel_configs(20) if c.family == "mlp")
		child = isolate.run_isolated(small, device="cuda", timeout_s=120.0)
		return {
			"child_status": str(child.get("status")),
			"child_seconds": round(float(child.get("gpu_seconds") or 0.0), 1),
			"text_asset": bool(os.path.exists(TEXT)),
			"gpu": str(gpus.resolve(props.name, props.total_memory, props.multi_processor_count)),
			"total_memory": int(props.total_memory),
			"torch": str(torch.__version__),  # a TorchVersion would need torch to unpickle on the client
			"transformers": str(md.version("transformers")),
			"volume_ok": True,
		}

	@modal.method()
	def run_batch(
		self,
		requested_gpu: str,
		items: list[dict[str, Any]],
		timeout_s: float | None = None,
		end_at: float | None = None,
	) -> list[dict[str, Any]]:
		import os
		import threading
		import time as _time

		from preflight.corpus import sweep

		# a warm container reused for the next wave must not carry the last wave's watchdog (review, Nov 1)
		old = getattr(self, "_watchdog", None)
		if old is not None:
			old.cancel()
			self._watchdog = None
		# first, before any import or volume work: a restart after preemption that comes up past the end does nothing
		if end_at is not None and _time.time() >= end_at:
			return [{"status": "batch", "wall_s": 0.0, "reload_s": 0.0, "commit_s": 0.0, "late": True}]
		if (
			end_at is not None
		):  # a hang anywhere (reload, a syscall) can't outlast the wave's end and its margin
			left = end_at + sweep.END_MARGIN_S - _time.time()
			dog = threading.Timer(max(1.0, left), lambda: os._exit(70))
			dog.daemon = True  # never keeps an idle container alive at scale-down
			dog.start()
			self._watchdog = dog

		import torch

		from preflight.corpus.configs import CorpusConfig
		from preflight.corpus.isolate import run_isolated
		from preflight.corpus.store import ResultStore
		from preflight.corpus.sweep import make_record
		from preflight.engine import gpus

		t0 = _time.monotonic()
		vol.reload()
		reload_s = _time.monotonic() - t0
		commit_s, uncommitted = 0.0, 0
		store = ResultStore(MOUNT)
		props = torch.cuda.get_device_properties(0)
		name = props.name or getattr(props, "gcnArchName", "")
		actual = gpus.resolve(name, props.total_memory, props.multi_processor_count)
		out = []
		for d in items:
			# a restart after preemption sees the same end time, so the wave never runs past its bound
			if end_at is not None and not sweep.may_start_config(_time.time(), timeout_s or 0.0, end_at):
				break
			cfg = CorpusConfig.from_dict(d)
			prior = store.read(cfg.kind, actual, cfg.key)
			if prior is not None and prior.get("status") not in ("error", "timeout"):
				out.append({"key": cfg.key, "gpu": actual, "status": "skipped", "gpu_seconds": 0.0})
				continue
			# own process and deadline: a hang becomes a timeout row, a kernel fault can't poison the next config
			res = run_isolated(
				cfg, device="cuda", data_path=TEXT if cfg.kind == "divergence" else None, timeout_s=timeout_s
			)
			res["requested_gpu"] = requested_gpu
			store.write_once(cfg.kind, actual, cfg.key, make_record(cfg, actual, res), replace_errors=True)
			uncommitted += 1
			if uncommitted >= COMMIT_EVERY:  # a preempted container loses at most COMMIT_EVERY - 1 configs
				c0 = _time.monotonic()
				vol.commit()
				commit_s += _time.monotonic() - c0
				uncommitted = 0
			out.append(
				{
					"key": cfg.key,
					"family": cfg.family,
					"role": cfg.params.get("role"),
					"gpu": actual,
					"status": res["status"],
					"error_code": res.get("error_code"),
					"error": res.get("error"),
					"gpu_seconds": res.get("gpu_seconds", 0.0),
				}
			)
		if uncommitted:
			c0 = _time.monotonic()
			vol.commit()
			commit_s += _time.monotonic() - c0
		# where the container's time went, so the overhead is measured rather than guessed
		out.append(
			{
				"status": "batch",
				"gpu": actual,
				"wall_s": round(_time.monotonic() - t0, 1),
				"reload_s": round(reload_s, 1),
				"commit_s": round(commit_s, 1),
			}
		)
		if getattr(self, "_watchdog", None) is not None:  # the batch is done: no watchdog past it
			self._watchdog.cancel()
			self._watchdog = None
		return out


@app.function(image=image, volumes={MOUNT: vol}, timeout=5 * 60)
def billing(kind: str, gpu: str, row: dict[str, Any] | None = None) -> dict[str, Any]:
	"""The container-seconds billed on one GPU for one kind across every run, kept on the volume; adds row if given."""
	from preflight.corpus import sweep

	path = f"{MOUNT}/_billing/{kind}-{gpu}.jsonl"
	vol.reload()
	if row is not None:
		sweep.add_billing(path, row)
		vol.commit()
	return {
		"billed_s": sweep.billed_seconds(path),
		"history": sweep.container_history(path),
		"prior_usd": sweep.prior_usd(path),
	}


@app.function(image=image, volumes={MOUNT: vol}, timeout=5 * 60)
def lease(kind: str, gpu: str, run_id: str, until: float | None = None) -> str | None:
	"""Take (until given) or release this GPU's budget for a run; returns the run that owns it after a re-read."""
	import time as _time

	from preflight.corpus import sweep

	d = f"{MOUNT}/_billing/{kind}-{gpu}.leases"
	vol.reload()
	if until is None:
		sweep.release_lease(d, run_id)
	else:
		sweep.write_lease(d, run_id, until, _time.time())
	vol.commit()
	vol.reload()  # see leases other runs committed meanwhile; each run has its own file, so none is lost
	return sweep.lease_winner(d, _time.time())


@app.function(image=image, volumes={MOUNT: vol}, timeout=15 * 60)
def ledger(kind: str, gpu: str, retry_errors: bool = False) -> dict[str, Any]:
	"""For one requested GPU (and the boards it can land on, sweep.LANDS_ON): keys already done, GPU-seconds spent (failed attempts included), and each
	config's latest seconds for the running estimate. Reads the small .meta summaries only.
	Errors and timeouts count as done unless retried.
	"""
	from preflight.corpus.store import ResultStore
	from preflight.corpus.sweep import ledger_from

	vol.reload()
	return ledger_from(ResultStore(MOUNT), kind, gpu, retry_errors)


def runs_on(cfg, gpu: str) -> bool:
	"""bf16 configs skip GPUs with no bf16 path (T4): emulated bf16 would time the emulation, not the job."""
	from preflight.engine import gpus

	dtype = cfg.params.get("dtype") or cfg.params.get("precision") or "fp32"
	return not (dtype == "bf16" and gpus.get_gpu(gpu).bf16 is None)


@app.function(image=image, volumes={MOUNT: vol}, timeout=15 * 60)
def read_rows(kind: str) -> list[dict[str, Any]]:
	"""Every finished row of one kind, for export (meta summaries left out)."""
	import json
	from pathlib import Path

	vol.reload()
	out = []
	for f in sorted(Path(MOUNT, kind).rglob("*.json")):
		if not f.name.endswith(".meta.json"):
			out.append(json.loads(f.read_text()))
	return out


@app.local_entrypoint()
def startup_check(gpu: str = "T4", out: str = "out/startup_check.json") -> None:
	"""Start the app in its real image once, on the cheapest GPU, before any paid sweep (Chiya, Oct 28):

	modal run --env main python/preflight/corpus/modal_app.py::startup_check
	"""
	import json
	from pathlib import Path

	from preflight.corpus import sweep

	result = {
		"billing": billing.remote("kernels", gpu),
		"sweeper": Sweeper.with_options(gpu=sweep.modal_gpu(gpu))().check.remote(),
	}
	Path(out).parent.mkdir(parents=True, exist_ok=True)
	Path(out).write_text(json.dumps(result, indent=1))
	print(json.dumps(result, indent=1))


@app.local_entrypoint()
def export(kind: str = "kernels", out: str = "out/corpus") -> None:
	"""Bring the volume's rows back as one gzipped JSONL per kind and GPU (remote-run returns ./out):

	modal run python/preflight/corpus/modal_app.py::export --kind kernels
	"""
	import gzip
	import json
	from collections import defaultdict
	from pathlib import Path

	by_gpu: dict[str, list[dict[str, Any]]] = defaultdict(list)
	for row in read_rows.remote(kind):
		by_gpu[row.get("gpu", "unknown")].append(row)
	for gpu, rows in sorted(by_gpu.items()):
		path = Path(out, kind, f"{gpu}.jsonl.gz")
		path.parent.mkdir(parents=True, exist_ok=True)
		with gzip.open(path, "wt") as f:
			for r in rows:
				f.write(json.dumps(r, sort_keys=True) + "\n")
		print(f"{path}: {len(rows)} rows")


@app.local_entrypoint()
def main(
	kind: str = "kernels",
	n: int = 1500,
	seed: int = 0,
	budgets: str = "L4=5",  # the $45 plan (corpus/README.md) passes its own per GPU; never more without Chiya's yes
	batch_size: int = 8,
	max_containers: int = 4,
	retry_errors: bool = False,
	dry_run: bool = False,
	prior_usd: float = 0.0,  # money this GPU and kind already cost before container billing was recorded (Modal's bill)
	max_minutes: float = 100.0,  # every wave's worst case must end inside this; set it under the workflow's timeout
	deadline_s: float = 0.0,  # per-config deadline before any config has finished; 0 keeps the kind's default
) -> None:
	from preflight.corpus import configs as cfgs
	from preflight.corpus import sweep
	from preflight.corpus.isolate import DEFAULT_TIMEOUT_S

	if deadline_s < 0:
		raise SystemExit("--deadline-s must be 0 (the kind's default) or a positive number of seconds")
	from preflight.engine import gpus

	plan = sweep.parse_budget_plan(budgets)
	pinned = {"corpus": cfgs.CODE_VERSION, **{k: v.split("==")[1] for k, v in PINS.items()}}
	if kind != "hf":
		pinned = {k: v for k, v in pinned.items() if k in ("corpus", "torch")}
	all_cfgs = [c.with_versions(pinned) for c in cfgs.configs_for(kind, n, seed)]
	per_gpu = {g: [c for c in all_cfgs if runs_on(c, g)] for g in plan}
	store_kind = all_cfgs[0].kind  # grouped configs are kernels rows; the ledger reads that directory
	rates = {}
	for g in plan:
		p = gpus.price(g, "modal")
		if p is None:
			raise SystemExit(f"no Modal price for {g}")
		rates[g] = p.usd_per_hour

	def run_gpu(gpu: str) -> str:
		import time
		import uuid

		# a smoke run that knows its configs are short can set a tighter first deadline, so its worst case fits a
		# small cap; the adaptive deadline takes over after 10 configs either way
		default_deadline = deadline_s or DEFAULT_TIMEOUT_S.get(store_kind, 900.0)
		run_id = uuid.uuid4().hex[:12]
		owner = lease.remote(store_kind, gpu, run_id, time.time() + (max_minutes + 30) * 60)
		if owner != run_id:
			print(f"[{gpu}] run {owner} holds this GPU's budget; stopping")
			lease.remote(store_kind, gpu, run_id)  # withdraw ours
			return gpu
		try:
			return _sweep_gpu(gpu, default_deadline)
		finally:
			lease.remote(store_kind, gpu, run_id)  # release, so a quick rerun isn't refused

	def _sweep_gpu(gpu: str, default_deadline: float) -> str:
		import time

		if prior_usd:
			billing.remote(
				store_kind, gpu, {"kind": "prior", "prior_usd": prior_usd}
			)  # counted once, however often passed
		wave, last_done = 0, -1
		started = time.monotonic()
		while True:
			led = ledger.remote(store_kind, gpu, retry_errors and wave == 0)
			bill = billing.remote(store_kind, gpu)
			# spend: Modal's figure for earlier runs, plus container time, each wave first billed at its worst case
			spent = bill["prior_usd"] + bill["billed_s"] / 3600 * rates[gpu] * sweep.RATE_OVERHEAD
			deadline = sweep.adaptive_deadline(led["history"], default_deadline)
			todo = [c for c in per_gpu[gpu] if c.key not in set(led["done"])]
			minutes_left = max_minutes - (time.monotonic() - started) / 60
			containers, batch = sweep.plan_wave(
				len(todo), plan[gpu] - spent, minutes_left, deadline, rates[gpu], max_containers, batch_size
			)
			print(
				f"[{gpu}] wave {wave}: ${spent:.2f} of ${plan[gpu]:.2f} spent (container time, worst case for unsettled "
				f"waves), {len(led['done'])} done, {len(todo)} to do, deadline {deadline:.0f} s, "
				f"next wave {containers} x {batch}"
			)
			if not todo or not containers or dry_run:
				if todo and not containers:
					print(
						f"[{gpu}] the next wave's worst case doesn't fit the money or the {minutes_left:.0f} minutes left; stopping, a rerun resumes"
					)
				return gpu
			if len(led["done"]) == last_done:
				print(f"[{gpu}] no progress in the last wave (rows filed under another GPU?); stopping")
				return gpu
			last_done = len(led["done"])
			chunk = sweep.batches(todo[: containers * batch], batch)
			worst = sweep.worst_wave_seconds(len(chunk), batch, deadline)
			cap_s = sweep.container_timeout_s(batch, deadline)
			# Modal cuts every container off at cap_s, so a hang anywhere (driver, volume) ends where the bound assumed
			sweeper = Sweeper.with_options(
				gpu=sweep.modal_gpu(gpu), max_containers=max_containers, timeout=cap_s
			)()
			end_at = sweep.wave_end_at(time.time(), batch, deadline)
			args = [(gpu, [c.to_dict() for c in b], deadline, end_at) for b in chunk]
			billing.remote(store_kind, gpu, {"wave": wave, "kind": "pending", "billed_s": round(worst, 1)})
			w0 = time.monotonic()
			# spawn, then wait with a wall-clock guard: starmap alone waits forever on a wave that never gets its GPUs
			calls = [sweeper.run_batch.spawn(*a) for a in args]
			wait_s = sweep.wave_wait_s(cap_s, minutes_left)
			results, late = sweep.collect_wave(calls, wait_s, w0, time.monotonic)
			wall = time.monotonic() - w0
			measured = sweep.wave_billed_seconds(wall, len(args))
			n_ran = sum(
				1 for res in results if not isinstance(res, BaseException)
				for r in res if r.get("status") not in ("skipped", "batch")
			)  # fmt: skip
			billing.remote(
				store_kind, gpu,
				{"wave": wave, "kind": "settle", "billed_s": round(measured - worst, 1), "measured_s": round(measured, 1),
					"wall_s": round(wall, 1), "containers": len(args), "configs": n_ran},
			)  # fmt: skip
			print(
				f"[{gpu}] wave {wave} took {wall:.0f} s on {len(args)} containers: {measured:.0f} container-s "
				f"(${measured / 3600 * rates[gpu] * sweep.RATE_OVERHEAD:.2f}, worst case was ${worst / 3600 * rates[gpu] * sweep.RATE_OVERHEAD:.2f}) for {n_ran} configs"
			)
			rows: list[dict[str, Any]] = []
			for res in results:
				if isinstance(res, BaseException):
					# a container died (preemption, infra); its committed configs are on the volume, the rest rerun
					print(f"[{gpu}] a batch failed: {type(res).__name__}: {res}")
					continue
				for r in res:
					if r.get("status") == "batch" and r.get("late"):
						print(
							f"[{gpu}] a batch started after the wave's end time (a restart after preemption?); it ran nothing"
						)
					elif r.get("status") == "batch":
						print(
							f"[{gpu}] batch: {r['wall_s']} s in the container, reload {r['reload_s']} s, commits {r['commit_s']} s"
						)
				res = [r for r in res if r.get("status") != "batch"]
				rows.extend(res)
				for r in res:  # measured seconds per config, in the log, so pricing needs no export run
					if r.get("status") != "skipped":
						print(
							f"[{gpu}] config {r.get('key', '?')[:12]} {r.get('family', '')} {r.get('role') or ''} {r.get('status')}: "
							f"{float(r.get('gpu_seconds') or 0.0):.1f} s"
						)
				bad = [r for r in res if r["gpu"] != gpu]
				if bad:
					print(f"[{gpu}] warning: ran on {bad[0]['gpu']}; rows filed under the actual GPU")
			if late:
				print(
					f"[{gpu}] {late} of {len(calls)} batches gave nothing back in {wait_s:.0f} s (no {gpu} free, or a "
					"hang past Modal's cut-off); cancelled them and stopping. Committed configs are kept; a rerun resumes."
				)
				return gpu
			ran = [r for r in rows if r["status"] != "skipped"]
			bugs = [r for r in ran if r.get("error_code") == "user_error"]
			if ran and len(bugs) * 2 >= len(ran):
				print(
					f"[{gpu}] {len(bugs)} of {len(ran)} configs failed inside corpus code (e.g. {bugs[0].get('error')}); stopping. Fix it, then rerun with --retry-errors."
				)
				return gpu
			wave += 1

	with cf.ThreadPoolExecutor(max_workers=len(plan)) as pool:
		for done in pool.map(run_gpu, plan):
			print(f"[{done}] finished")
