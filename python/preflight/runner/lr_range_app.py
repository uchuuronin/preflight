"""The learning-rate range test on a zoo script, on a real GPU (D, Nov 2; Chiya's list item 3).

	modal run --env main python/preflight/runner/lr_range_app.py::check --scripts finetune_lora_hf,pretrain_nanogpt
	modal run --env main python/preflight/runner/lr_range_app.py::check --scripts finetune_lora_hf --lr 1e-3

`--lr` sets the script's own `learning_rate=` (exactly one in the script) first: a planted bad rate,
to see whether the reading catches it (Chiya, Nov 2, 17:33). `--confirm-steps` sets the held runs'
length.

Its own app ("preflight-runner-d-lr"), as ops_app.py is, so `modal run` never starts a copy of the
deployed runner; it reuses the runner's probe image with this checkout's code, so what it checks
is what the runner would run once A redeploys. For each script, in one container on an A10 (else
an L4; never a bigger GPU):

  1. the range test: a probe with the rate ramped from 1e-7 to 1 over 100 steps, eval off
     (`ProbeSpec.lr_range`, the same spec the pipeline's `p1-lr` point sends)
  2. 50 steps held at the suggested rate, and 50 at the script's own rate as the baseline, eval off
     (the recorder holds the rate, so the script's warm-up and schedule don't change it): does it
     train (no divergence, the last 10 steps' mean below the first 10's), and how does its loss
     compare with the script's own rate?

Writes out/lr_range_<script>.json: the curve's reading, the script's own rate, both 50-step runs,
each probe's GPU seconds and dollars at list price. Cap: the container's timeout (30 min) at the
A10's US$1.10 an hour is US$0.55 [derived]; two scripts stay under Chiya's US$2.
"""

import json
from pathlib import Path

import modal

from preflight.runner.modal_app import APP_NAME, WEIGHTS_DIR, _read_only, probe_image, weights_vol

app = modal.App(APP_NAME + "-d-lr")
TIMEOUT_S = 30 * 60
RANGE_TIMEOUT_S = 600  # the range test, then two confirmations: 600 + 2 x 420 s < TIMEOUT_S
CONFIRM_TIMEOUT_S = 420
CONFIRM_STEPS = 50
SCRIPTS = ("finetune_lora_hf", "pretrain_nanogpt")
USD_PER_H = {"A10": 1.10, "L4": 0.80}  # cited: probes/gpu_choice.py's list prices


def _gpu() -> str:
	import torch

	name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
	return "A10" if "A10" in name else "L4" if "L4" in name else name


@app.function(image=probe_image, gpu=["A10", "L4"], timeout=TIMEOUT_S,
	volumes={WEIGHTS_DIR: _read_only(weights_vol)})  # fmt: skip
def range_and_confirm(name: str, files: dict, entrypoint: str, confirm_steps: int = CONFIRM_STEPS) -> dict:
	import math
	import time

	from preflight.harness.static import map_static
	from preflight.probes import remote
	from preflight.probes.local import ProbeSpec, run_probe
	from preflight.probes.lr_range import LR_END, LR_START, STEPS, curve
	from preflight.probes.lr_range import losses as stored_losses

	t0 = time.monotonic()
	remote.link_weights()
	remote.deps_dir()
	gpu = _gpu()
	rate = USD_PER_H.get(gpu, 0.0)
	sm = map_static(files, entrypoint=entrypoint)
	lr_knob = next((k for k in sm.knobs if k.id == "lr"), None)
	own = float(lr_knob.current) if lr_knob is not None and isinstance(lr_knob.current, int | float) else None
	out: dict = {"script": name, "gpu": gpu, "lens": sm.lens, "own_lr": own}

	def ramp(pid: str, start: float, end: float, steps: int, timeout_s: float):
		spec = ProbeSpec(probe_id=pid, files=files, entrypoint=entrypoint, warmup=steps, timed=0, eval_batches=0,
			lr_range={"start": start, "end": end, "steps": steps}, timeout_s=timeout_s, gpu_usd_per_hour=rate)  # fmt: skip
		return run_probe(spec).probe

	try:
		p = ramp("p1-lr", LR_START, LR_END, STEPS, RANGE_TIMEOUT_S)
		ex = p.extras
		out["range"] = {"status": p.status, "error": (p.error or "")[:400], "error_code": p.error_code,
			"steps": ex.get("lr_range.steps"), "suggested_lr": ex.get("lr_range.suggested_lr"),
			"min_loss_lr": ex.get("lr_range.min_loss_lr"), "diverged_lr": ex.get("lr_range.diverged_lr"),
			"flat": ex.get("lr_range.flat"), "checked": ex.get("lr_range.checked"),
			"not_set": ex.get("lr_range.not_set"), "unverified": ex.get("lr_range.unverified"),
			"other_optimizers": ex.get("lr_range.other_optimizers"), "stalled": ex.get("lr_range.stalled"),
			"curve": curve(ex), "gpu_s": p.cost.gpu_seconds,
			"usd": p.cost.usd}  # fmt: skip
	except Exception as exc:  # recorded; the JSON still comes back
		out["range"] = {"failed": f"{type(exc).__name__}: {str(exc)[:400]}"}
	print(json.dumps({k: v for k, v in out["range"].items() if k != "curve"}), flush=True)
	sug = out["range"].get("suggested_lr")
	# where the script's own rate sits on the curve: what a guide line would say about it
	r = out["range"]
	if own and sug:
		out["own_on_curve"] = {"times_suggested": own / sug,
			"past_lowest_loss": bool(r.get("min_loss_lr") and own > r["min_loss_lr"]),
			"past_divergence": bool(r.get("diverged_lr") and own >= r["diverged_lr"]),
			"times_below_divergence": (r["diverged_lr"] / own) if r.get("diverged_lr") else None}  # fmt: skip
		print(json.dumps(out["own_on_curve"]), flush=True)
	# 50 steps held at one rate (the recorder holds it, so the script's warm-up and decay don't
	# change it): the suggested rate, and the script's own as the baseline
	for key, lr in (("at_suggested", sug), ("at_own", own)):
		if not lr:
			out[key] = {"skipped": "no rate"}
			continue
		try:
			c = ramp(key, float(lr), float(lr), confirm_steps, CONFIRM_TIMEOUT_S)
			losses = stored_losses(c.extras)  # a run at one rate keeps its losses, not its rate
			first = sum(losses[:10]) / 10 if len(losses) >= 20 else None
			last = sum(losses[-10:]) / 10 if len(losses) >= 20 else None
			out[key] = {"lr": float(lr), "status": c.status, "error": (c.error or "")[:400],
				"steps": len(losses), "diverged": "lr_range.diverged_lr" in c.extras,
				"checked": c.extras.get("lr_range.checked"), "first10_mean": first, "last10_mean": last,
				"trains": bool(first is not None and last is not None and last < first
					and "lr_range.diverged_lr" not in c.extras and all(math.isfinite(v) for v in losses)),
				"losses": losses, "gpu_s": c.cost.gpu_seconds, "usd": c.cost.usd}  # fmt: skip
		except Exception as exc:
			out[key] = {"lr": lr, "failed": f"{type(exc).__name__}: {str(exc)[:400]}"}
		print(json.dumps({k: v for k, v in out[key].items() if k != "losses"}), flush=True)
	out["container_s"] = round(time.monotonic() - t0, 1)
	out["container_usd"] = round(out["container_s"] * rate / 3600, 4)
	return out


@app.local_entrypoint()
def check(scripts: str = ",".join(SCRIPTS), lr: float = 0.0, confirm_steps: int = CONFIRM_STEPS):
	import re

	from preflight.runner.sources import _read_tree

	if not 0 < confirm_steps <= 100:  # guess: 100 held steps fit CONFIRM_TIMEOUT_S on an A10 or L4
		raise SystemExit("--confirm-steps runs from 1 to 100")

	root = Path(__file__).resolve().parents[3]
	Path("out").mkdir(exist_ok=True)
	for name in [s for s in scripts.split(",") if s]:
		if name not in SCRIPTS:
			raise SystemExit(f"{name!r}: this check runs on {SCRIPTS} only")
		files = _read_tree(root / "zoo" / name)
		tag = ""
		if lr:
			files["train.py"], n = re.subn(
				r"\blearning_rate\s*=\s*[0-9.eE+-]+", f"learning_rate={lr!r}", files["train.py"]
			)
			if n != 1:
				raise SystemExit(f"{name}: expected one learning_rate= in train.py, found {n}")
			tag = f"_lr{lr:g}"
		print(f"[lr range] {name}: {len(files)} files, waiting for an A10, else an L4", flush=True)
		try:
			res = range_and_confirm.remote(name, files, "train.py", confirm_steps)
		except Exception as exc:  # recorded; the next script still runs
			res = {"script": name, "failed": f"{type(exc).__name__}: {str(exc)[:500]}"}
		Path(f"out/lr_range_{name}{tag}.json").write_text(json.dumps(res, indent=1, default=str))
		print(json.dumps({k: v for k, v in res.items() if k not in ("range", "at_suggested", "at_own")} | {
			"suggested": (res.get("range") or {}).get("suggested_lr"),
			"trains_at_suggested": (res.get("at_suggested") or {}).get("trains"),
			"trains_at_own": (res.get("at_own") or {}).get("trains")}, default=str), flush=True)  # fmt: skip
