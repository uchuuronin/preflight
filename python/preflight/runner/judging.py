"""The judging window, Nov 5 to 8 (Chiya approved, Nov 1: options (a) and (c) of A's warm-pool ask).

	in_judging(time.time())      # True from PREFLIGHT_JUDGING_FROM to PREFLIGHT_JUDGING_TO, ET dates
	judging_gpu("L4", allowed)   # "A10" in the window when the A10 is allowed: it skips the L4 queue

(c) the GPU ladder starts at the A10 instead of the L4 during judging: every A10 request on Nov 1
scheduled within seconds while the L4 queued 19 to 59 min [measured, about 6 runs]; about $0.05
more a job [derived], nothing billed while idle. (a) one worker container kept warm during judging
(modal_app.py's daily warm-pool check sets it), about $1.51 a day. No held GPU.
Both read the window from the environment, so it can move without a code change.
"""

import os

JUDGING_FROM = "2026-11-05"  # ET calendar dates, inclusive
JUDGING_TO = "2026-11-08"


def window() -> tuple[str, str]:
	return os.environ.get("PREFLIGHT_JUDGING_FROM", JUDGING_FROM), os.environ.get(
		"PREFLIGHT_JUDGING_TO", JUDGING_TO
	)


def in_judging(ts: float) -> bool:
	from preflight.runner.queue import et_day

	start, end = window()
	return start <= et_day(ts) <= end


def judging_gpu(gpu: str, allowed: list[str] | None, ts: float) -> str:
	"""During judging, the A10 instead of the L4, when the visitor allows the A10. Both have 24 GB,
	so a job that fits one fits the other; any other GPU stays as chosen."""
	if gpu == "L4" and in_judging(ts) and (allowed is None or "A10" in allowed):
		return "A10"
	return gpu


def warm_workers(ts: float) -> int:
	"""How many worker containers to keep warm: one during judging, none otherwise."""
	return 1 if in_judging(ts) else 0
