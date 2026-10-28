"""Budgeted, resumable sweep driver.

Used directly on a plain GPU box (the AMD MI300X droplet), and by the Modal app
for the NVIDIA slices. Spend is counted from what the store already holds, so a
restarted sweep picks up where it stopped and never exceeds its budget (gap 13).

Budget rule (docs/decisions/engine-settings.md, S5 chosen B5): the cost of the
next config is estimated from the running p90 of real configs on that GPU once
three have finished, times a billing overhead, and one wave is held in reserve,
so a run of slow configs can't push the sweep past its budget.

    python -m preflight.corpus.sweep export-amd --out amd_slice.jsonl
    python -m preflight.corpus.sweep run --kind kernels --configs amd_slice.jsonl \\
        --gpu MI300X --out /data/corpus --budget-usd 16
"""

from __future__ import annotations

import argparse
import math
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

from preflight.corpus import configs as cfgs
from preflight.corpus.configs import SCHEMA, CorpusConfig
from preflight.corpus.store import ResultStore

DEFAULT_EST_SECONDS = {
	"kernels": 45.0,
	"divergence": 180.0,
	"hf": 120.0,
}  # guess: per-config seconds before any finish; replaced by the running p90 after 3 configs (measured)
DEFAULT_OVERHEAD = 1.25  # guess: container start and idle Modal bills; replaced by Modal's billed/elapsed ratio after the first wave
ESTIMATE_QUANTILE = 0.9  # simulated: S5 B5
MIN_HISTORY = 3  # simulated: S5 B5
MAX_CONSECUTIVE_USER_ERRORS = 3  # guess: three corpus-code errors in a row means a bug, not a config
_EPS = 1e-6

Runner = Callable[[CorpusConfig], dict[str, Any]]


def cost_usd(seconds: float, usd_per_hour: float, overhead: float = 1.0) -> float:
	return seconds * overhead / 3600 * usd_per_hour


def quantile(xs: list[float], q: float) -> float:
	s = sorted(xs)
	pos = q * (len(s) - 1)
	lo = math.floor(pos)
	hi = min(lo + 1, len(s) - 1)
	return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def estimate_seconds(
	kind: str, history: list[float] | None, est_seconds: float | None = None
) -> tuple[float, bool]:
	"""(seconds per config, learned). The running p90 once MIN_HISTORY configs have finished."""
	if est_seconds is not None:
		return est_seconds, False
	if history and len(history) >= MIN_HISTORY:
		return quantile(history, ESTIMATE_QUANTILE), True
	return DEFAULT_EST_SECONDS[kind], False


def plan_from(
	configs: list[CorpusConfig],
	done: set[str],
	spent_usd: float,
	budget_usd: float,
	usd_per_hour: float,
	est_seconds: float | None = None,
	overhead: float = DEFAULT_OVERHEAD,
	history: list[float] | None = None,
	reserve_configs: int = 0,
) -> list[CorpusConfig]:
	"""Configs not yet done, cut to what the remaining budget allows after the reserve."""
	if not configs:
		return []
	est, _ = estimate_seconds(configs[0].kind, history, est_seconds)
	per = cost_usd(est, usd_per_hour, overhead)
	reserve = reserve_configs * per if len(history or []) >= MIN_HISTORY else 0.0
	room = max(0, math.floor((budget_usd - spent_usd - reserve) / per + _EPS)) if per > 0 else len(configs)
	return [c for c in configs if c.key not in done][:room]


def plan(
	configs: list[CorpusConfig],
	store: ResultStore,
	gpu: str,
	budget_usd: float,
	usd_per_hour: float,
	est_seconds: float | None = None,
	overhead: float = DEFAULT_OVERHEAD,
	retry_errors: bool = False,
) -> list[CorpusConfig]:
	"""Configs still to run on this GPU, cut to what the remaining budget allows."""
	if not configs:
		return []
	kind = configs[0].kind
	done = store.done_keys(kind, gpu, retry_errors=retry_errors)
	spent = store.spent_usd({gpu: usd_per_hour}, kind=kind, gpu=gpu)
	history = store.gpu_seconds(kind, gpu)
	return plan_from(configs, done, spent, budget_usd, usd_per_hour, est_seconds, overhead, history)


def modal_gpu(gpu: str) -> str:
	"""Modal's GPU string for a table name; H100 is pinned so Modal can't upgrade it to H200."""
	from preflight.engine import gpus

	key = gpus.resolve(gpu)
	if key not in gpus.MODAL_GPU:
		raise ValueError(f"{key} is not offered on Modal")
	return gpus.MODAL_GPU[key]


def parse_budget_plan(spec: str) -> dict[str, float]:
	"""'L4=25,A100-80GB=45,H100=70' -> {table name: dollars}; Modal strings and table names both work."""
	from preflight.engine import gpus

	out: dict[str, float] = {}
	for part in spec.split(","):
		name, _, dollars = part.strip().partition("=")
		out[gpus.resolve(name.strip())] = float(dollars)
	return out


def batches(configs: list[CorpusConfig], size: int) -> list[list[CorpusConfig]]:
	return [configs[i : i + size] for i in range(0, len(configs), size)]


def make_record(cfg: CorpusConfig, gpu: str, result: dict[str, Any]) -> dict[str, Any]:
	return {"schema": SCHEMA, "key": cfg.key, "gpu": gpu, "config": cfg.to_dict(), **result}


def run_local(
	configs: list[CorpusConfig],
	store: ResultStore,
	gpu: str,
	budget_usd: float,
	usd_per_hour: float,
	runner: Runner,
	est_seconds: float | None = None,
	overhead: float = DEFAULT_OVERHEAD,
	retry_errors: bool = False,
	log: Callable[[str], None] = print,
	max_user_errors: int = MAX_CONSECUTIVE_USER_ERRORS,
	ledger=None,
) -> int:
	"""Run configs one at a time, writing each result once; returns how many ran.

	Spend counts the billing overhead. With an uptime ledger (ledger.UptimeLedger), spend is the droplet's
	wall-clock bill instead, idle time included. Several user_error rows in a row mean a bug in corpus code,
	not bad configs: the sweep stops and says so.
	"""
	if not configs:
		return 0
	kind = configs[0].kind
	done = store.done_keys(kind, gpu, retry_errors=retry_errors)
	spent = store.spent_usd({gpu: usd_per_hour}, kind=kind, gpu=gpu) * overhead
	if ledger is not None:
		spent = ledger.spent()
		ledger.record("start", gpu=gpu, kind=kind, budget_usd=budget_usd)
	history = store.gpu_seconds(kind, gpu)
	n, user_errors = 0, 0
	for cfg in configs:
		if cfg.key in done:
			continue
		est, _ = estimate_seconds(kind, history, est_seconds)
		per = cost_usd(est, usd_per_hour, 1.0 if ledger is not None else overhead)
		if ledger is not None:
			spent = ledger.spent()  # idle time between configs is billed too
		reserve = per if len(history) >= MIN_HISTORY else 0.0  # one config is this driver's wave
		if spent + per + reserve > budget_usd + _EPS:
			log(f"budget reached: ${spent:.2f} of ${budget_usd:.2f} spent on {gpu}; halting")
			if ledger is not None:
				ledger.record("stop", reason="budget")
			break
		t0 = time.monotonic()
		try:
			result = runner(cfg)
		except Exception as exc:  # recorded so the sweep keeps going; retried with --retry-errors
			result = {
				"status": "error",
				"error_code": "user_error",
				"error": f"{type(exc).__name__}: {exc}",
				"traceback": traceback.format_exc(limit=5),
				"gpu_seconds": time.monotonic() - t0,
			}
		store.write_once(kind, gpu, cfg.key, make_record(cfg, gpu, result), replace_errors=True)
		secs = float(result.get("gpu_seconds", 0.0))
		if ledger is not None:
			spent = ledger.spent()
			ledger.record("config", key=cfg.key, status=result.get("status"), config_s=round(secs, 1))
		else:
			spent += cost_usd(secs, usd_per_hour, overhead)
		history.append(secs)
		n += 1
		log(f"[{n}] {cfg.family} {cfg.key} {result.get('status')} ${spent:.2f}")
		user_errors = user_errors + 1 if result.get("error_code") == "user_error" else 0
		if user_errors >= max_user_errors:
			log(
				f"{user_errors} configs in a row failed inside corpus code (last: {result.get('error')}); stopping. Fix the bug, then rerun with --retry-errors."
			)
			break
	return n


def _main(argv: list[str] | None = None) -> None:
	ap = argparse.ArgumentParser(prog="preflight.corpus.sweep")
	sub = ap.add_subparsers(dest="cmd", required=True)

	ex = sub.add_parser("export-amd", help="write the AMD slice config list")
	ex.add_argument("--out", required=True)
	ex.add_argument(
		"--kind",
		choices=["kernels", "grouped"],
		default="kernels",
		help="grouped: the same probe-like groups the NVIDIA GPUs run, so every MI300X row has NVIDIA pairs",
	)
	ex.add_argument("--n-total", type=int, default=1500)
	ex.add_argument("--n", type=int, default=500)
	ex.add_argument("--seed", type=int, default=0)

	run = sub.add_parser("run", help="run a sweep on this machine's GPU")
	run.add_argument("--kind", choices=list(cfgs.KINDS), required=True)
	run.add_argument("--configs", help="jsonl list; default: generate")
	run.add_argument("--n", type=int, default=1500)
	run.add_argument("--seed", type=int, default=0)
	run.add_argument("--gpu", required=True, help="table name, e.g. MI300X or L4")
	run.add_argument("--out", required=True)
	run.add_argument("--budget-usd", type=float, required=True)
	run.add_argument("--usd-per-hour", type=float)
	run.add_argument("--device", default="cuda")
	run.add_argument("--data", help="text file for divergence runs; synthetic if absent")
	run.add_argument("--retry-errors", action="store_true")
	run.add_argument(
		"--overhead", type=float, default=DEFAULT_OVERHEAD, help="billing overhead on measured seconds"
	)
	run.add_argument(
		"--billing",
		choices=["per-config", "uptime"],
		default="per-config",
		help="uptime: the machine bills from start to destroy (the AMD droplet); needs --droplet-started",
	)
	run.add_argument("--droplet-started", help="when the droplet was created, ISO 8601 with a time zone")
	run.add_argument(
		"--copy-to",
		help="also copy the results and ledger here when the sweep stops, even on a timeout (amd-run brings back ./out)",
	)
	args = ap.parse_args(argv)

	if args.cmd == "export-amd":
		if args.kind == "grouped":
			out = cfgs.configs_for(
				"grouped", args.n, args.seed
			)  # the first n // 5 groups, as on every NVIDIA GPU
		else:
			out = cfgs.amd_slice(cfgs.kernel_configs(args.n_total, seed=args.seed), args.n)
		cfgs.write_jsonl(out, args.out)
		print(f"wrote {len(out)} configs to {args.out}")
		return

	from preflight.corpus.isolate import run_isolated
	from preflight.engine import gpus

	gpu = gpus.resolve(args.gpu)
	if args.device.startswith("cuda"):
		import torch

		props = torch.cuda.get_device_properties(0)
		name = props.name or getattr(props, "gcnArchName", "")  # some ROCm builds report an empty name
		actual = gpus.resolve(name, props.total_memory, props.multi_processor_count)
		if actual != gpu:
			raise SystemExit(
				f"--gpu says {gpu} but this machine has {actual} ({props.name}); rows would be mislabelled"
			)
	rate = args.usd_per_hour
	if rate is None:
		p = gpus.price(gpu, "modal") or gpus.price(gpu, "amd_devcloud")
		if p is None:
			raise SystemExit(f"no price for {gpu}; pass --usd-per-hour")
		rate = p.usd_per_hour
	todo = cfgs.read_jsonl(args.configs) if args.configs else cfgs.configs_for(args.kind, args.n, args.seed)
	versions = cfgs.local_versions(args.kind)
	todo = [c.with_versions(versions) for c in todo]  # code and library versions in every key

	def runner(cfg: CorpusConfig) -> dict[str, Any]:
		return run_isolated(cfg, device=args.device, data_path=args.data)

	ledger = None
	if args.billing == "uptime":
		from datetime import datetime

		from preflight.corpus.ledger import UptimeLedger

		if not args.droplet_started:
			raise SystemExit("--billing uptime needs --droplet-started (when the droplet was created)")
		ledger = UptimeLedger(
			Path(args.out) / f"ledger-{gpu}.jsonl", datetime.fromisoformat(args.droplet_started), rate
		)
	import signal

	def _stop(signum, frame):  # `timeout` sends SIGTERM; turn it into an exit so the copy below still runs
		raise SystemExit(f"stopped by signal {signum}")

	signal.signal(signal.SIGTERM, _stop)
	try:
		run_local(
			todo,
			ResultStore(args.out),
			gpu,
			args.budget_usd,
			rate,
			runner,
			overhead=args.overhead,
			retry_errors=args.retry_errors,
			ledger=ledger,
		)
	finally:
		if args.copy_to:
			copy_results(args.out, args.copy_to)


def copy_results(src: str | Path, dst: str | Path) -> int:
	"""Copy a result store (rows, meta, ledgers) into dst, keeping its layout; returns the files copied."""
	import shutil

	src, dst = Path(src), Path(dst)
	n = 0
	for f in src.rglob("*"):
		if f.is_file():
			t = dst / f.relative_to(src)
			t.parent.mkdir(parents=True, exist_ok=True)
			shutil.copy2(f, t)
			n += 1
	return n


if __name__ == "__main__":
	_main()
