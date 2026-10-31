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

import json
import os
import sys
import time
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
		for p in record.probes:
			v = p.extras.get("planned_steps")
			if v:
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
	peak: object = None  # () -> peak allocated bytes so far, or None (no GPU)
	alert: object = None  # (text) -> None
	write: object = None  # (dict) -> None, one jsonl line
	steps: int = 0
	t0: float | None = None
	windows: list = field(default_factory=list)
	_w_start: float | None = None
	_out_of_band: dict = field(default_factory=dict)
	_alerted: set = field(default_factory=set)

	def start(self):
		self.t0 = self._w_start = self.clock()

	def step(self):
		"""Called after every optimizer step."""
		self.steps += 1
		if self.steps % self.window == 0:
			self._flush()

	def _flush(self, partial: bool = False):
		now = self.clock()
		n = self.steps % self.window if partial else self.window
		if n <= 0 or self._w_start is None:
			return None
		w = {"step": self.steps, "elapsed_s": now - self.t0, "step_s": (now - self._w_start) / n, "steps": n}
		self._w_start = now
		pk = self.peak() if self.peak else None
		if pk is not None:
			w["peak_bytes"] = int(pk)
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
		out: dict = {"steps": self.steps}
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
				out["memory"] = {"peak_allocated_bytes": max(peaks)}
			keep = (
				ws
				if len(ws) <= MAX_LOG
				else [ws[round(i * (len(ws) - 1) / (MAX_LOG - 1))] for i in range(MAX_LOG)]
			)
			out["step_log"] = [{"index": x["step"], "kind": "timed", "wall_s": x["step_s"],
				**({"peak_allocated_bytes": x["peak_bytes"]} if "peak_bytes" in x else {})} for x in keep]  # fmt: skip
		out["alerts"] = [a for x in ws for a in x["alerts"]]
		return out


def live_record(parent, summary: dict, *, status: str, error: str | None = None, ids=None):
	"""A new record of the real run: the parent's map, knobs and predictions, plus probe `live`."""
	from preflight.runner.record_jobs import derive
	from preflight.schema import Probe

	rec = derive(parent, ids)
	src = parent.job.source.model_copy(update={"kind": "inflight", "record_ref": parent.share_token})
	probe = Probe.model_validate({"id": "live", "knobs": {}, "status": status, "steps": summary["steps"],
		"duration_s": summary.get("duration_s"), "error": error, "step_time": summary.get("step_time"),
		"memory": summary.get("memory"), "step_log": summary.get("step_log", [])})  # fmt: skip
	return rec.model_copy(
		update={"job": rec.job.model_copy(update={"source": src}), "probes": [*parent.probes, probe]}
	)


def _atomic(path: Path, text: str):
	tmp = path.with_suffix(path.suffix + ".tmp")
	tmp.write_text(text)
	os.replace(tmp, path)


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

	from preflight.schema import load_record

	parent = load_record(ns.record.read_text())
	ns.out.mkdir(parents=True, exist_ok=True)
	ids = None
	from preflight.schema.record import new_share_token, new_ulid

	ids = (new_ulid(), new_share_token())
	rec_path = ns.out / f"{ids[1]}.json"
	log_path = ns.out / f"{ids[1]}.live.jsonl"
	log = log_path.open("a", buffering=1)

	import torch

	cuda = torch.cuda.is_available()

	def write(w):
		log.write(json.dumps(w) + "\n")
		_atomic(rec_path, live_record(parent, col.summary(), status="running", ids=ids).model_dump_json())

	col = Collector(bands_from(parent, ns.total_steps), window=max(1, ns.window),
		peak=(lambda: torch.cuda.max_memory_allocated()) if cuda else None,
		alert=lambda t: print(f"[preflight] {t}", file=sys.stderr, flush=True), write=write)  # fmt: skip
	seen: dict = {}

	def post(opt, args, kwargs):
		first = seen.setdefault("opt", opt)
		if opt is first:  # several optimizers (GANs, multi-head): count the first one's steps
			col.step()

	from torch.optim.optimizer import register_optimizer_step_post_hook

	handle = register_optimizer_step_post_hook(post)
	print(f"[preflight] watching {ns.script} against record {parent.share_token}; live record {rec_path}",
		file=sys.stderr, flush=True)  # fmt: skip
	status, error, code = "ok", None, 0
	sys.argv = [str(ns.script), *script_args]
	sys.path.insert(0, str(ns.script.resolve().parent))
	col.start()
	try:
		import runpy

		runpy.run_path(str(ns.script), run_name="__main__")
	except SystemExit as exc:
		code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
		if code:
			status, error = "error", f"the script exited with {exc.code}"
	except BaseException as exc:  # the run's own failure, recorded; an OOM says so
		status = "oom" if "out of memory" in str(exc).lower() else "error"
		error, code = f"{type(exc).__name__}: {str(exc)[:500]}", 1
	finally:
		handle.remove()
		summary = col.finish()
		_atomic(rec_path, live_record(parent, summary, status=status, error=error, ids=ids).model_dump_json())
		log.close()
		print(f"[preflight] {summary['steps']} steps, {len(summary['alerts'])} alerts; record {rec_path}",
			file=sys.stderr, flush=True)  # fmt: skip
	return code


if __name__ == "__main__":
	raise SystemExit(main())
