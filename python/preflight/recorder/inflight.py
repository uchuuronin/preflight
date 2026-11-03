"""In-flight mode: a light collector during the real run that streams into the flight record and
says when the run leaves its predicted band.

	preflight watch records/<share>.json train.py -- --epochs 3
	python -m preflight.recorder.inflight records/<share>.json OUT_DIR train.py -- --epochs 3

It runs the script unchanged, to the end, with one optimizer post-step hook: the wall clock at each
step boundary (no device sync, so the run isn't slowed; over a window of steps the mean is the real
step time because the host can't run ahead of the device for long) and, once a window, the device's
peak memory (a counter read, no sync). Every WINDOW steps it appends a line to live.jsonl, rewrites
the live record (a new record, source kind `inflight`, `parent_id` the predicted one, one probe
`live`) and checks the window against the parent's predicted bands:

- time.step: a window's mean outside [lo, hi], two windows in a row (one slow window is noise)
- memory.peak: the peak above hi, at once
- time.total: elapsed plus the remaining steps at the recent pace, above hi (when the step count
  is known: --total-steps, or the probes' planned_steps)

Alerts go to stderr as `[preflight] ...` lines and into the jsonl, once each until the run is back
in its band. Nothing leaves the machine; the guide reads the live record.
"""

import contextlib
import json
import os
import signal
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

WINDOW = 20  # guess: a few seconds to a minute of a typical run between updates
MAX_LOG = 256  # the schema's step_log cap: windows past it are thinned evenly
REPEAT_TO_ALERT = 2  # guess: two windows out of band in a row before saying so


@dataclass
class Bands:
	step_s: tuple[float, float] | None = None
	peak_bytes: tuple[float, float] | None = None
	total_s: tuple[float, float] | None = None
	total_steps: int | None = None


def _band(est) -> tuple[float, float] | None:
	if est is None or est.status == "abstain" or est.lo is None or est.hi is None:
		return None
	return float(est.lo), float(est.hi)


def bands_from(record, total_steps: int | None = None) -> Bands:
	"""The parent record's predicted bands for the knobs it was predicted at."""
	est = record.predictions.estimates
	planned = total_steps
	if planned is None:
		# the current-settings probe's count (p1); other probes ran at other knobs
		for p in sorted(record.probes, key=lambda p: p.id != "p1"):
			v = p.extras.get("planned_steps")
			if v and p.id == "p1":
				planned = int(v)
				break
	return Bands(
		_band(est.get("time.step")), _band(est.get("memory.peak")), _band(est.get("time.total")), planned
	)


@dataclass
class Collector:
	bands: Bands
	window: int = WINDOW
	clock: object = time.monotonic
	peak: object = None  # () -> (peak allocated, peak reserved) bytes so far, or None (no GPU)
	alert: object = None  # (text) -> None
	write: object = None  # (dict) -> None, one jsonl line
	steps: int = 0
	t0: float | None = None
	setup_s: float | None = None  # run start to the first optimizer step: imports, loading, data
	windows: list = field(default_factory=list)
	_w_start: float | None = None
	_last_t: float | None = None  # when the last optimizer step ended
	_base: int = 0  # the step the first window starts after
	_out_of_band: dict = field(default_factory=dict)
	_alerted: set = field(default_factory=set)

	def start(self):
		self.t0 = self.clock()

	def step(self):
		"""Called after every optimizer step. Windows start at the first step, so setup (imports,
		loading the model and data) is kept out of step time."""
		self.steps += 1
		self._last_t = self.clock()
		if self._w_start is None:
			self._w_start = self.clock()
			self.setup_s = self._w_start - (self.t0 if self.t0 is not None else self._w_start)
			self._base = self.steps
			return
		if (self.steps - self._base) % self.window == 0:
			self._flush()

	def _flush(self, partial: bool = False):
		# a partial window ends at its last step, not at the stop: B's A10 run (Nov 2) counted the second between its
		# last step and SIGTERM as step time, and raised a false total-time alert
		now = self._last_t if partial and self._last_t is not None else self.clock()
		n = (self.steps - self._base) % self.window if partial else self.window
		if n <= 0 or self._w_start is None:
			return None
		w = {"step": self.steps, "elapsed_s": now - self.t0, "step_s": (now - self._w_start) / n, "steps": n}
		self._w_start = now
		pk = self.peak() if self.peak else None
		if pk is not None:
			# the band is fitted on reserved memory (what the allocator holds, fragmentation included)
			w["peak_allocated_bytes"], w["peak_bytes"] = int(pk[0]), int(pk[1])
		w["alerts"] = self._check(w)
		self.windows.append(w)
		if self.write:
			self.write(w)
		return w

	def _check(self, w) -> list[str]:
		out = []
		b = self.bands
		if b.step_s is not None:
			lo, hi = b.step_s
			side = "above" if w["step_s"] > hi else "below" if w["step_s"] < lo else None
			self._streak("time.step", side, out,
				f"step {w['step']}: {w['step_s']:.3g} s a step, {side} the predicted {lo:.3g}–{hi:.3g} s")  # fmt: skip
		if b.peak_bytes is not None and "peak_bytes" in w:
			hi = b.peak_bytes[1]
			over = w["peak_bytes"] > hi
			self._once("memory.peak", over, out,
				f"step {w['step']}: peak memory {w['peak_bytes'] / 2**30:.2f} GiB, above the predicted "
				f"{b.peak_bytes[0] / 2**30:.2f}–{hi / 2**30:.2f} GiB")  # fmt: skip
		if b.total_s is not None and b.total_steps:
			recent = [x["step_s"] for x in self.windows[-2:]] + [w["step_s"]]
			pace = sum(recent) / len(recent)
			left = max(0, b.total_steps - w["step"])
			proj = w["elapsed_s"] + left * pace
			lo, hi = b.total_s
			self._once("time.total", proj > hi, out,
				f"step {w['step']}: at this pace the run takes {proj / 60:.1f} min, above the predicted "
				f"{lo / 60:.1f}–{hi / 60:.1f} min")  # fmt: skip
			w["projected_total_s"] = proj
		for text in out:
			if self.alert:
				self.alert(text)
		return out

	def _streak(self, key, side, out, text):
		if side is None:
			self._out_of_band[key] = 0
			self._alerted.discard(key)
			return
		self._out_of_band[key] = self._out_of_band.get(key, 0) + 1
		if self._out_of_band[key] >= REPEAT_TO_ALERT and key not in self._alerted:
			self._alerted.add(key)
			out.append(text)

	def _once(self, key, bad, out, text):
		if not bad:
			self._alerted.discard(key)
		elif key not in self._alerted:
			self._alerted.add(key)
			out.append(text)

	def finish(self):
		"""The last partial window, then the run as a probe's fields."""
		self._flush(partial=True)
		return self.summary()

	def summary(self) -> dict:
		ws = self.windows
		out: dict = {"steps": self.steps, "setup_s": self.setup_s}
		if self.t0 is not None:
			out["duration_s"] = self.clock() - self.t0
		if ws:
			per = sorted(x["step_s"] for x in ws)
			total = sum(x["step_s"] * x["steps"] for x in ws)
			out["step_time"] = {"n_steps": sum(x["steps"] for x in ws), "mean_s": total / sum(x["steps"] for x in ws),
				"p50_s": per[len(per) // 2], "p95_s": per[min(len(per) - 1, int(0.95 * len(per)))],
				"first_step_s": ws[0]["step_s"], "unprofiled": True}  # fmt: skip
			peaks = [x["peak_bytes"] for x in ws if "peak_bytes" in x]
			if peaks:
				out["memory"] = {"peak_reserved_bytes": max(peaks),
					"peak_allocated_bytes": max(x["peak_allocated_bytes"] for x in ws if "peak_allocated_bytes" in x)}  # fmt: skip
			keep = (
				ws
				if len(ws) <= MAX_LOG
				else [ws[round(i * (len(ws) - 1) / (MAX_LOG - 1))] for i in range(MAX_LOG)]
			)
			out["step_log"] = [{"index": x["step"], "kind": "timed", "wall_s": x["step_s"],
				**({"peak_allocated_bytes": x["peak_allocated_bytes"], "peak_reserved_bytes": x["peak_bytes"]}
					if "peak_bytes" in x else {})} for x in keep]  # fmt: skip
		out["alerts"] = [a for x in ws for a in x["alerts"]]
		return out


def _card(name: str) -> str:
	return " ".join(name.lower().replace("nvidia", "").split())


# review, Nov 2: cards PyTorch names apart but the GPU table resolves to one row; each is a different GPU here
_VARIANTS = ("a10g", "4090 d")


def card_change(
	record_gpu: str | None,
	device: str | None,
	memory_bytes: int | None = None,
	compute_units: int | None = None,
) -> str | None:
	"""A sentence when this run's card isn't the one the record's bands were measured on, else None.

	B's A10 check (Nov 2): Modal's "A10" gave an A10G one run in two, which ran about 11% slower than the A10 the
	record was probed on, and the engine resolves both to "A10". So a different table GPU, a MIG slice of a full card,
	or a named variant of the same row (A10G against A10, RTX 4090 D against RTX 4090) is a different GPU: step and
	total time aren't checked against bands from another card. The device's memory and compute units go to the
	table's resolver, which splits names PyTorch reports alike (MI250 and MI250X). The record names the card as the
	runner normalised it, so a probe that itself ran on an A10G reads as "A10" here; until the runner keeps the
	device's own name, an A10G run against such a record says "different GPU" too, which is the safe side."""
	if not record_gpu or not device:
		return None
	here, there = _card(device), _card(record_gpu)
	try:
		from preflight.engine import gpus

		same_table = gpus.resolve(device, memory_bytes, compute_units) == gpus.resolve(record_gpu)
	except Exception:  # an unknown name: compare the names alone
		same_table = here == there
	mig = " mig " in f" {here} " and " mig " not in f" {there} "
	variant = any(v in here and v not in there for v in _VARIANTS)
	if same_table and not mig and not variant:
		return None
	return (
		f"different GPU: this run is on {device.strip()}, the record's bands were measured on {record_gpu}; "
		"step and total time aren't checked against them (memory still is)"
	)


def live_record(parent, summary: dict, *, status: str, error: str | None = None, ids=None):
	"""A new record of the real run: the parent's map, knobs and predictions, plus probe `live`."""
	from preflight.runner.record_jobs import derive
	from preflight.schema import Probe

	rec = derive(parent, ids)
	src = parent.job.source.model_copy(update={"kind": "inflight", "record_ref": parent.share_token})
	probe = Probe.model_validate({"id": "live", "knobs": {}, "status": status, "steps": summary["steps"],
		"duration_s": summary.get("duration_s"), "error": error, "step_time": summary.get("step_time"),
		"memory": summary.get("memory"), "step_log": summary.get("step_log", [])})  # fmt: skip
	kept = [p for p in parent.probes if p.id != "live"]  # watching a live record: one live probe
	return rec.model_copy(
		update={"job": rec.job.model_copy(update={"source": src}), "probes": [*kept, probe]}
	)


def _atomic(path: Path, text: str):
	tmp = path.with_suffix(path.suffix + ".tmp")
	tmp.write_text(text)
	os.replace(tmp, path)


WRITE_EVERY_S = 5.0  # derived: the live record (the parent's probes and all) is rewritten at most this often


def _rank() -> int:
	for k in ("RANK", "LOCAL_RANK"):
		with contextlib.suppress(ValueError):
			if os.environ.get(k):
				return int(os.environ[k])
	return 0


def _run_script(script: Path, args: list[str]) -> None:
	import runpy

	sys.argv = [str(script), *args]
	sys.path.insert(0, str(script.resolve().parent))
	runpy.run_path(str(script), run_name="__main__")


def main(argv: list[str] | None = None) -> int:
	argv = list(sys.argv[1:] if argv is None else argv)
	script_args = argv[argv.index("--") + 1 :] if "--" in argv else []
	head = argv[: argv.index("--")] if "--" in argv else argv
	import argparse

	ap = argparse.ArgumentParser(prog="python -m preflight.recorder.inflight")
	ap.add_argument("record", type=Path)
	ap.add_argument("out", type=Path)
	ap.add_argument("script", type=Path)
	ap.add_argument("--total-steps", type=int)
	ap.add_argument("--window", type=int, default=WINDOW)
	ns = ap.parse_args(head)
	if _rank() != 0:
		# torchrun: rank 0 watches; the others run as they would, writing and saying nothing
		_run_script(ns.script, script_args)
		return 0

	from preflight.schema import load_record
	from preflight.schema.record import new_share_token, new_ulid

	parent = load_record(ns.record.read_text())
	ns.out.mkdir(parents=True, exist_ok=True)
	ids = (new_ulid(), new_share_token())
	rec_path = ns.out / f"{ids[1]}.json"
	log_path = ns.out / f"{ids[1]}.live.jsonl"
	log = log_path.open("a", buffering=1)

	import torch

	cuda = torch.cuda.is_available()

	def say(t):
		print(f"[preflight] {t}", file=sys.stderr, flush=True)

	last_write = [0.0]

	def write(w):
		log.write(json.dumps(w) + "\n")
		now = time.monotonic()
		if now - last_write[0] >= WRITE_EVERY_S:
			last_write[0] = now
			_atomic(rec_path, live_record(parent, col.summary(), status="running", ids=ids).model_dump_json())

	def peak():
		return torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()  # counters, no sync

	bands = bands_from(parent, ns.total_steps)
	record_gpu = parent.env.gpu.name if parent.env and parent.env.gpu else None
	props = torch.cuda.get_device_properties(0) if cuda else None
	changed = card_change(
		record_gpu,
		torch.cuda.get_device_name(0) if cuda else None,
		props.total_memory if props else None,
		props.multi_processor_count if props else None,
	)
	if changed:
		say(changed)
		bands = Bands(None, bands.peak_bytes, None, bands.total_steps)
	col = Collector(bands, window=max(1, ns.window), peak=peak if cuda else None,
		alert=say, write=write)  # fmt: skip
	if script_args != list(parent.job.source.args or []):
		say(f"the record was predicted for arguments {parent.job.source.args or []}, this run has {script_args}: "
			"its bands may not apply")  # fmt: skip
	counts: dict = {}
	main_opt = {"id": None, "others": 0}

	def post(opt, args, kwargs):
		# the optimizer that steps the run: the first one, until another steps 50 times without it
		# (an LR finder's throwaway, or one rebuilt for the next stage); GANs alternate, so both stay
		k = id(opt)
		counts[k] = counts.get(k, 0) + 1
		if main_opt["id"] is None:
			main_opt["id"] = k
		if k == main_opt["id"]:
			main_opt["others"] = 0
			col.step()
			return
		main_opt["others"] += 1
		if main_opt["others"] > 50:
			main_opt["id"], main_opt["others"] = k, 0
			col.step()

	from torch.optim.optimizer import register_optimizer_step_post_hook

	handle = register_optimizer_step_post_hook(post)

	def on_term(signum, frame):
		raise SystemExit(128 + signum)

	if signal.getsignal(signal.SIGTERM) is signal.SIG_DFL:  # Slurm preemption: the record still ends
		signal.signal(signal.SIGTERM, on_term)
	say(f"watching {ns.script} against record {parent.share_token}; live record {rec_path}")
	status, error, code = "ok", None, 0
	col.start()
	try:
		_run_script(ns.script, script_args)
	except SystemExit as exc:
		if isinstance(exc.code, str):
			print(exc.code, file=sys.stderr)
		code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
		if code == 128 + signal.SIGTERM:
			status, error = "cancelled", "stopped by SIGTERM"
		elif code:
			status, error = "error", f"the script exited with {exc.code}"
	except KeyboardInterrupt:
		status, error, code = "cancelled", "interrupted", 130
	except BaseException as exc:  # the run's own failure: its traceback, and the record says so
		traceback.print_exc()
		status = "oom" if "out of memory" in str(exc).lower() else "error"
		error, code = f"{type(exc).__name__}: {str(exc)[:500]}", 1
	finally:
		handle.remove()
		summary = col.finish()
		_atomic(rec_path, live_record(parent, summary, status=status, error=error, ids=ids).model_dump_json())
		log.close()
		say(f"{summary['steps']} steps, {len(summary['alerts'])} alerts; record {rec_path}")
	return code


if __name__ == "__main__":
	raise SystemExit(main())
