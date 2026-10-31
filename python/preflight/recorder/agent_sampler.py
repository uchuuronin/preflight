"""The agent sampler: run 30 of an agent run's tasks through the recorder's proxy, as an AgentSample.

	sample = run_sample(["python", "run.py"], cwd=work, proxy=proxy, population=500)
	# B's engine.agent_cost.agent_totals(sample) projects the whole run; when the cost range
	# straddles the budget (budget_check), the guide offers 20 more on a click:
	more = run_sample(["python", "run.py"], cwd=work, proxy=proxy, population=500, k=20, run_id="s2",
		skip_keys=[task_key(t.task_id) for t in sample.tasks], offered_first=draw_indices(500, 30))
	sample = combine_samples(sample, more)

The script runs unchanged, as one process, with its OpenAI-compatible client pointed at the proxy
(OPENAI_BASE_URL and the older OPENAI_API_BASE); it never sees the provider key. The proxy splits
its calls into tasks by conversation and refuses the call that would start task k+1, so a run of
500 tasks costs the 30 it was sampled for and then ends fast (the refusal tells the SDK not to
retry). A top-up refuses the tasks already sampled at their first call, matched by content (a hash
of the first call), so they cost nothing the second time, in whatever order they start.

Other provider keys (*_API_KEY, *_API_TOKEN) are taken out of the script's environment. Grouping
is by conversation, so an agent's side calls (a judge, a sub-agent) and a history trimmed mid-task
each start a task of their own; `selection` says so when most tasks are a single call.

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


def _scrubbed(env: dict, upstream_key: str | None) -> dict:
	"""The script's environment without other provider keys, so a client with a hard-coded base
	URL can't reach a provider unmetered: *_API_KEY and *_API_TOKEN variables, and anything that
	holds the proxy's own key, are dropped."""
	out = {}
	for k, v in env.items():
		if k.endswith(("_API_KEY", "_API_TOKEN")) or (upstream_key and v == upstream_key):
			continue
		out[k] = v
	return out


def run_sample(cmd: list[str], *, cwd: Path, proxy, population: int | None, k: int = DEFAULT_SAMPLE,
	skip_keys=(), offered_first=(), run_id: str = "s1", timeout_s: float = TIMEOUT_S,
	env: dict | None = None, should_stop=None) -> AgentSample:  # fmt: skip
	"""Run the script once through the proxy; the tasks it ran, as the record's agent_sample.

	skip_keys: opening keys of tasks already sampled (agent.task_key of each first-sample task
	id); the proxy refuses them at their first call. offered_first: the first sample's indices,
	offered again ahead of the new ones, so a script that reads PREFLIGHT_TASK_INDICES starts with
	the tasks that will be refused, as a script that can't be told where to start does.
	should_stop: polled every second (the job was cancelled); the script is stopped."""
	url = proxy.open_run(run_id, max_tasks=k, skip_keys=skip_keys)
	new = draw_indices(population, k, exclude=offered_first)
	indices = None if new is None else [*offered_first, *new]
	child_env = {
		**_scrubbed(env if env is not None else dict(os.environ), getattr(proxy, "_key", None)),
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
	cut = None
	while proc.poll() is None:
		if time.monotonic() - t0 > timeout_s:
			cut = "timeout"
		elif should_stop is not None and should_stop():
			cut = "cancelled"
		if cut:
			break
		with _quiet():
			proc.wait(timeout=1.0)
	in_flight = set(proxy.run_stats(run_id).get("in_flight", [])) if cut else set()
	with _quiet():
		os.killpg(proc.pid, signal.SIGKILL)  # the script, and anything it left running
	with _quiet():
		proc.wait(timeout=10)
	wall = time.monotonic() - t0
	ids = proxy.run_tasks(run_id)
	tasks = [proxy.task(t, timed_out=t in in_flight) for t in ids]
	st = proxy.run_stats(run_id)
	parts = [f"the first {len(ids)} tasks the script started"]
	if st.get("skipped"):
		parts.append(f"{st['skipped']} already sampled skipped")
	if st.get("refused"):
		parts.append(f"{st['refused']} calls refused past the sample")
	if st.get("single_call") and len(ids) and st["single_call"] * 2 > len(ids):
		parts.append(f"{st['single_call']} one-call tasks (side calls may split tasks)")
	parts.append(f"script wall {wall:.0f} s" + (f", {cut} at {timeout_s:.0f} s" if cut == "timeout" else
		", cancelled" if cut else ""))  # fmt: skip
	selection = "; ".join(parts)
	if len(selection) > 200:
		selection = selection[:197].rsplit(" ", 1)[0] + "..."
	return AgentSample(tasks=tasks, population=population, selection=selection)


class _quiet:
	def __enter__(self):
		return self

	def __exit__(self, *exc):
		quiet = ProcessLookupError | PermissionError | subprocess.TimeoutExpired
		return exc[0] is not None and issubclass(exc[0], quiet)
