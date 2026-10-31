"""Run a script with nothing but a CUDA event at each optimizer step: the ground truth's baseline.

	python -m preflight.recorder.bare SCRIPT OUT.json --steps 1000 --seed 1337 [-- script args]

No recorder hooks, no sync points: one global optimizer-step post hook records an event, and the
run stops after --steps optimizer steps (or when the script ends or fails, an OOM included, which
is itself ground truth). Writes the step walls in seconds (from the events, read once at the end),
peak memory, and how the run ended. Used for our own zoo scripts on our own GPUs, never visitor code.
"""

import argparse
import itertools
import json
import os
import runpy
import sys
import time
import traceback


class _Enough(Exception):
	pass


PROGRESS_EVERY = 50  # optimizer steps between progress lines on stderr


def main(argv: list[str] | None = None) -> int:
	argv = list(sys.argv[1:] if argv is None else argv)
	script_args = argv[argv.index("--") + 1 :] if "--" in argv else []
	ap = argparse.ArgumentParser(prog="python -m preflight.recorder.bare")
	ap.add_argument("script")
	ap.add_argument("out")
	ap.add_argument("--steps", type=int, required=True)
	ap.add_argument("--seed", type=int, default=None)
	ns = ap.parse_args(argv[: argv.index("--")] if "--" in argv else argv)

	import torch
	from torch.optim.optimizer import register_optimizer_step_post_hook

	cuda = torch.cuda.is_available()
	events: list = []
	host: list[float] = []

	def hook(_opt, _args, _kwargs):
		host.append(time.perf_counter())
		if len(host) % PROGRESS_EVERY == 0:  # host clock only: no sync, nothing on the GPU
			print(f"[bare] step {len(host)} at {host[-1] - t0:.0f} s", file=sys.stderr, flush=True)
		if cuda:
			ev = torch.cuda.Event(enable_timing=True)
			ev.record()
			events.append(ev)
		if len(host) >= ns.steps:
			raise _Enough

	if ns.seed is not None:
		torch.manual_seed(ns.seed)
	register_optimizer_step_post_hook(hook)
	workdir = os.path.dirname(os.path.abspath(ns.script))
	os.chdir(workdir)
	sys.path.insert(0, workdir)
	sys.argv = [ns.script, *script_args]
	ended, error = "finished", None
	t0 = time.perf_counter()
	print(f"[bare] running {ns.script} for {ns.steps} optimizer steps", file=sys.stderr, flush=True)
	try:
		runpy.run_path(os.path.abspath(ns.script), run_name="__main__")
	except _Enough:
		ended = "steps"
	except torch.cuda.OutOfMemoryError as exc:
		ended, error = "oom", str(exc)[:500]
	except BaseException as exc:  # the script's own end (SystemExit) or failure, recorded as is
		ended, error = type(exc).__name__, "".join(traceback.format_exception_only(exc))[-500:]
	if cuda:
		torch.cuda.synchronize()
		walls = [None] + [a.elapsed_time(b) / 1000.0 for a, b in itertools.pairwise(events)]
	else:
		walls = [None] + [b - a for a, b in itertools.pairwise(host)]
	out = {
		"steps": len(host),
		"walls_s": walls,  # walls_s[i]: from step i-1's end to step i's end; [0] has no start
		"ended": ended,
		"error": error,
		"seconds": time.perf_counter() - t0,
		"peak_allocated_bytes": torch.cuda.max_memory_allocated() if cuda else None,
		"peak_reserved_bytes": torch.cuda.max_memory_reserved() if cuda else None,
		"device": torch.cuda.get_device_name(0) if cuda else "cpu",
		"timing": "cuda_events" if cuda else "host_clock",
	}
	with open(ns.out, "w") as f:
		json.dump(out, f)
	print(json.dumps({k: v for k, v in out.items() if k != "walls_s"}), flush=True)
	return 0


if __name__ == "__main__":
	sys.exit(main())
