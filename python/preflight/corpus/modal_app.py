"""Modal app for the NVIDIA corpus slices.

Dispatches in waves: after each wave it re-reads the ledger on the volume and
re-plans against the per-GPU budget, so the sweep halts at its budget even when
real configs run longer than the estimate, and a restarted sweep resumes from
what the volume already holds (gap 13). Each config is written once and the
volume committed after every config, so a preempted container loses one config.

Run from the repo root (PYTHONPATH=python so the package is importable):

    PYTHONPATH=python modal run python/preflight/corpus/modal_app.py --dry-run
    PYTHONPATH=python modal run python/preflight/corpus/modal_app.py \\
        --kind kernels --budgets "L4=25,A100-SXM-80GB=45,H100-SXM-80GB=70"
    PYTHONPATH=python modal run python/preflight/corpus/modal_app.py \\
        --kind divergence --n 800 --budgets "L4=35"
    modal volume get preflight-corpus /kernels ./corpus/kernels

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
	retries=modal.Retries(max_retries=1),  # preemption; finished configs are skipped on the retry
	max_containers=4,
)
class Sweeper:
	@modal.method()
	def run_batch(self, requested_gpu: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
		import torch

		from preflight.corpus.configs import CorpusConfig
		from preflight.corpus.isolate import run_isolated
		from preflight.corpus.store import ResultStore
		from preflight.corpus.sweep import make_record
		from preflight.engine import gpus

		vol.reload()
		store = ResultStore(MOUNT)
		props = torch.cuda.get_device_properties(0)
		name = props.name or getattr(props, "gcnArchName", "")
		actual = gpus.resolve(name, props.total_memory, props.multi_processor_count)
		out = []
		for d in items:
			cfg = CorpusConfig.from_dict(d)
			prior = store.read(cfg.kind, actual, cfg.key)
			if prior is not None and prior.get("status") not in ("error", "timeout"):
				out.append({"key": cfg.key, "gpu": actual, "status": "skipped", "gpu_seconds": 0.0})
				continue
			# own process and deadline: a hang becomes a timeout row, a kernel fault can't poison the next config
			res = run_isolated(cfg, device="cuda", data_path=TEXT if cfg.kind == "divergence" else None)
			res["requested_gpu"] = requested_gpu
			store.write_once(cfg.kind, actual, cfg.key, make_record(cfg, actual, res), replace_errors=True)
			vol.commit()
			out.append(
				{
					"key": cfg.key,
					"gpu": actual,
					"status": res["status"],
					"error_code": res.get("error_code"),
					"error": res.get("error"),
					"gpu_seconds": res.get("gpu_seconds", 0.0),
				}
			)
		return out


@app.function(image=image, volumes={MOUNT: vol}, timeout=15 * 60)
def ledger(kind: str, gpu: str, retry_errors: bool = False) -> dict[str, Any]:
	"""For one GPU: keys already done, GPU-seconds spent (failed attempts included), and each
	config's latest seconds for the running estimate. Reads the small .meta summaries only.
	Errors and timeouts count as done unless retried.
	"""
	from preflight.corpus.store import ResultStore

	vol.reload()
	out: dict[str, Any] = {"done": [], "gpu_seconds": 0.0, "history": []}
	for _, key, meta in ResultStore(MOUNT).iter_meta(kind, gpu):
		if not (retry_errors and meta.get("status") in ("error", "timeout")):
			out["done"].append(key)
		secs = float(meta.get("gpu_seconds") or 0.0)
		out["gpu_seconds"] += secs + float(meta.get("prior_gpu_seconds") or 0.0)
		out["history"].append(secs)
	return out


@app.local_entrypoint()
def main(
	kind: str = "kernels",
	n: int = 1500,
	seed: int = 0,
	budgets: str = "L4=25,A100-SXM-80GB=45,H100-SXM-80GB=70",
	batch_size: int = 8,
	max_containers: int = 4,
	overhead: float = 1.25,
	retry_errors: bool = False,
	dry_run: bool = False,
) -> None:
	from preflight.corpus import configs as cfgs
	from preflight.corpus import sweep
	from preflight.engine import gpus

	plan = sweep.parse_budget_plan(budgets)
	pinned = {"corpus": cfgs.CODE_VERSION, **{k: v.split("==")[1] for k, v in PINS.items()}}
	if kind != "hf":
		pinned = {k: v for k, v in pinned.items() if k in ("corpus", "torch")}
	all_cfgs = [c.with_versions(pinned) for c in cfgs.configs_for(kind, n, seed)]
	store_kind = all_cfgs[0].kind  # grouped configs are kernels rows; the ledger reads that directory
	rates = {}
	for g in plan:
		p = gpus.price(g, "modal")
		if p is None:
			raise SystemExit(f"no Modal price for {g}")
		rates[g] = p.usd_per_hour

	def run_gpu(gpu: str) -> str:
		sweeper = Sweeper.with_options(gpu=sweep.modal_gpu(gpu), max_containers=max_containers)()
		wave_size = batch_size * max_containers
		wave, last_done = 0, -1
		while True:
			led = ledger.remote(store_kind, gpu, retry_errors and wave == 0)
			spent = led["gpu_seconds"] * overhead / 3600 * rates[gpu]
			todo = sweep.plan_from(
				all_cfgs, set(led["done"]), spent, plan[gpu], rates[gpu], overhead=overhead,
				history=led["history"], reserve_configs=wave_size,
			)  # fmt: skip
			print(
				f"[{gpu}] wave {wave}: ${spent:.2f} of ${plan[gpu]:.2f} spent, {len(led['done'])} done, {len(todo)} fit in budget"
			)
			if not todo or dry_run:
				return gpu
			if len(led["done"]) == last_done:
				print(f"[{gpu}] no progress in the last wave (rows filed under another GPU?); stopping")
				return gpu
			last_done = len(led["done"])
			chunk = sweep.batches(todo[:wave_size], batch_size)
			args = [(gpu, [c.to_dict() for c in b]) for b in chunk]
			rows: list[dict[str, Any]] = []
			for res in sweeper.run_batch.starmap(args, return_exceptions=True):
				if isinstance(res, BaseException):
					# a container died (preemption past its retry, infra); its finished configs are on the volume
					print(f"[{gpu}] a batch failed: {type(res).__name__}: {res}")
					continue
				rows.extend(res)
				bad = [r for r in res if r["gpu"] != gpu]
				if bad:
					print(f"[{gpu}] warning: ran on {bad[0]['gpu']}; rows filed under the actual GPU")
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
