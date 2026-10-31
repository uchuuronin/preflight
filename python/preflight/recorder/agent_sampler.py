"""The agent sampler: run 30 of an agent run's tasks through the recorder's proxy, as an AgentSample.

	sample = run_sample(["python", "run.py"], cwd=work, proxy=proxy, population=500)
	# B's engine.agent_cost.agent_totals(sample) projects the whole run; when the cost range
	# straddles the budget (budget_check), the guide offers 20 more on a click:
	more = run_sample(["python", "run.py"], cwd=work, proxy=proxy, population=500, k=20, skip=30)
	sample = combine_samples(sample, more)

The script runs unchanged, as one process, with its OpenAI-compatible client pointed at the proxy
(OPENAI_BASE_URL and the older OPENAI_API_BASE); it never sees the provider key. The proxy splits
its calls into tasks by conversation and refuses the call that would start task k+1, so a run of
500 tasks costs the 30 it was sampled for and then ends fast. A top-up refuses the first `skip`
tasks at their first call, so they cost nothing the second time.

Which tasks: the first k the script starts, unless it reads PREFLIGHT_TASK_INDICES (a
comma-separated list the sampler draws at random from the population) and runs those instead;
`selection` says the indices were offered (the proxy can't tell whether the script used them).
A script that reads PREFLIGHT_SAMPLE_TASKS can stop by itself.
"""

import os
import random
import signal
import subprocess
import time
from pathlib import Path

from preflight.schema import AgentSample

DEFAULT_SAMPLE = 30  # cited: Chiya, Oct 27 (Contracts tab); B's engine.agent_cost.DEFAULT_SAMPLE
TOP_UP_TASKS = 20  # cited: Chiya, Oct 27: 20 more when the cost range straddles the budget
SAMPLE_SEED = 20261031  # derived: fixed, so a rerun samples the same tasks
TIMEOUT_S = 1800  # guess: 30 coding-agent tasks at the golden record's ~50 s each, sequential, is 25 min
DUMMY_KEY = "sk-preflight-proxy"  # what the script's client sends; the proxy drops it


def draw_indices(population: int | None, k: int, *, seed: int = SAMPLE_SEED, exclude=()) -> list[int] | None:
	"""k task indices drawn without replacement from the population, minus those already run."""
	if not population or population <= 0:
		return None
	pool = sorted(set(range(population)) - set(exclude))
	return sorted(random.Random(seed).sample(pool, min(k, len(pool))))


def run_sample(cmd: list[str], *, cwd: Path, proxy, population: int | None, k: int = DEFAULT_SAMPLE,
	skip: int = 0, run_id: str = "s1", timeout_s: float = TIMEOUT_S, env: dict | None = None,
	exclude: tuple[int, ...] = ()) -> AgentSample:  # fmt: skip
	"""Run the script once through the proxy; the tasks it ran, as the record's agent_sample."""
	url = proxy.open_run(run_id, max_tasks=k, skip=skip)
	indices = draw_indices(population, k, exclude=exclude)
	child_env = {
		**(env if env is not None else os.environ),
		"OPENAI_BASE_URL": url,
		"OPENAI_API_BASE": url,
		"OPENAI_API_KEY": DUMMY_KEY,
		"PREFLIGHT_SAMPLE_TASKS": str(k),
	}
	if indices is not None:
		child_env["PREFLIGHT_TASK_INDICES"] = ",".join(map(str, indices))
	t0 = time.monotonic()
	proc = subprocess.Popen(cmd, cwd=cwd, env=child_env, start_new_session=True,
		stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # fmt: skip
	timed_out = False
	try:
		proc.communicate(timeout=timeout_s)
	except subprocess.TimeoutExpired:
		timed_out = True
		with _quiet():
			os.killpg(proc.pid, signal.SIGKILL)
		proc.communicate()
	wall = time.monotonic() - t0
	ids = proxy.run_tasks(run_id)
	tasks = [proxy.task(t, timed_out=timed_out and i == len(ids) - 1) for i, t in enumerate(ids)]
	how = f"the first {len(ids)} tasks the script started"
	if indices is not None:
		how += " (random indices offered in PREFLIGHT_TASK_INDICES)"
	if skip:
		how += f", after skipping the {skip} already sampled"
	selection = f"{how}; calls split into tasks by conversation; script wall {wall:.0f} s"
	if timed_out:
		selection += f", cut at {timeout_s:.0f} s"
	return AgentSample(tasks=tasks, population=population, selection=selection[:200])


class _quiet:
	def __enter__(self):
		return self

	def __exit__(self, *exc):
		return exc[0] is not None and issubclass(exc[0], ProcessLookupError | PermissionError)
