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

Other model providers' keys (PROVIDER_KEYS) are taken out of the script's environment; tool keys
stay. A task's key is a hash of its whole first call, so a timestamp or id in the first prompt
gives it a new key each run and a top-up can't recognise it. Grouping
is by conversation, so an agent's side calls (a judge, a sub-agent) and a history trimmed mid-task
each start a task of their own; `selection` says so when most tasks are a single call.

Which tasks: the first k the script starts, unless it reads PREFLIGHT_TASK_INDICES (a
comma-separated list the sampler draws at random from the population) and runs those instead;
`selection` says the indices were offered (the proxy can't tell whether the script used them).
A script that reads PREFLIGHT_SAMPLE_TASKS can stop by itself.
"""

import os
import random
import re
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


# model providers' own key variables: a client that reads one of these with a hard-coded base URL
# would reach that provider unmetered. Tool keys (search, sandboxes) stay, or the agent's tools fail.
PROVIDER_KEYS = frozenset({
	"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "AZURE_OPENAI_API_KEY", "BASETEN_API_KEY", "CEREBRAS_API_KEY",
	"COHERE_API_KEY", "CO_API_KEY", "DEEPINFRA_API_KEY", "DEEPSEEK_API_KEY", "FIREWORKS_API_KEY",
	"GEMINI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "HUGGINGFACEHUB_API_TOKEN", "LLAMA_API_KEY",
	"META_API_KEY", "MISTRAL_API_KEY", "MOONSHOT_API_KEY", "NVIDIA_API_KEY", "OPENROUTER_API_KEY",
	"PERPLEXITY_API_KEY", "PPLX_API_KEY", "REPLICATE_API_TOKEN", "TOGETHER_API_KEY", "XAI_API_KEY",
})  # fmt: skip


def _scrubbed(env: dict, upstream_key: str | None) -> tuple[dict, list[str]]:
	"""The script's environment without other model providers' keys, and the names taken out."""
	out, gone = {}, []
	for k, v in env.items():
		if k in PROVIDER_KEYS or (upstream_key and v == upstream_key and k != "OPENAI_API_KEY"):
			gone.append(k)
			continue
		out[k] = v
	return out, sorted(gone)


# where agent scripts read their base URL and key (OpenAI's SDK names, and the Meta Model API's
# names the zoo's agent script and docs/setup/cloud.md use); all point at the proxy, all get the
# dummy key
BASE_URL_VARS = ("OPENAI_BASE_URL", "OPENAI_API_BASE", "META_API_BASE", "META_BASE_URL")
KEY_VARS = ("OPENAI_API_KEY", "META_API_KEY")
# Chiya, Oct 31: 2 min with no model call through the proxy, and none in flight, is a stalled run
# (D's plant 3, a tool retried forever, then shows as a stalled task, not a long wait)
STALL_S = 120.0


class _LocalProc:
	"""The script as a process here, in its own session (so its children die with it)."""

	def __init__(self, cmd, cwd, env):
		self.p = subprocess.Popen(cmd, cwd=cwd, env=env, start_new_session=True,
			stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # fmt: skip

	def poll(self):
		return self.p.poll()

	def wait(self, timeout):
		with _quiet():
			self.p.wait(timeout=timeout)

	def kill(self):
		with _quiet():
			os.killpg(self.p.pid, signal.SIGKILL)  # the script, and anything it left running
		self.wait(10)


def _own_reply(status: int, code: str, model: str | None, n: int) -> str:
	"""400 unpriced_model (muse-spark-1.3) x2; a model name is the script's, so only its plain characters"""
	name = re.sub(r"[^\w./:-]", "", model or "")[:24]
	code = re.sub(r"[^a-z0-9_.]", "", code or "")[:32] or "error"
	return f"{status} {code}" + (f" ({name})" if name else "") + (f" x{n}" if n > 1 else "")


def run_sample(cmd: list[str] | None, *, cwd: Path | None, proxy, population: int | None,
	k: int = DEFAULT_SAMPLE, skip_keys=(), offered_first=(), run_id: str = "s1", timeout_s: float = TIMEOUT_S,
	env: dict | None = None, should_stop=None, launch=None, base_url_var: str | None = None,
	stall_s: float = STALL_S, inspect=None, single: bool = False, note_replies: bool = False) -> AgentSample:  # fmt: skip
	"""Run the script once through the proxy; the tasks it ran, as the record's agent_sample.

	skip_keys: opening keys of tasks already sampled (agent.task_key of each first-sample task
	id); the proxy refuses them at their first call. offered_first: the first sample's indices,
	offered again ahead of the new ones, so a script that reads PREFLIGHT_TASK_INDICES starts with
	the tasks that will be refused, as a script that can't be told where to start does.
	should_stop: polled every second (the job was cancelled); the script is stopped.
	launch: (env) -> a process with poll(), wait(timeout) and kill(); the default runs cmd here.
	A sandbox launch gets only the variables set here (pass env={}): never the caller's own.
	base_url_var: the variable the script map says the client reads its base URL from.
	stall_s: the script is stopped when no call has come through the proxy for this long.
	single (probes.agent_single_task): the script runs one task, not a benchmark: every call is task 1.
	note_replies (probes.agent_proxy_replies): the selection names the answers the proxy gave itself
	(a refusal's status and code), which an SDK may swallow."""
	url = proxy.open_run(run_id, max_tasks=k, skip_keys=skip_keys, **({"single": True} if single else {}))
	new = draw_indices(population, k, exclude=offered_first)
	indices = None if new is None else [*offered_first, *new]
	kept, withheld = _scrubbed(env if env is not None else dict(os.environ), getattr(proxy, "_key", None))
	urls = {v: url for v in (*BASE_URL_VARS, *([base_url_var] if base_url_var else []))}
	child_env = {**kept, **urls, **dict.fromkeys(KEY_VARS, DUMMY_KEY), "PREFLIGHT_SAMPLE_TASKS": str(k)}
	if indices is not None:
		child_env["PREFLIGHT_TASK_INDICES"] = ",".join(map(str, indices))
	t0 = time.monotonic()
	proc = launch(child_env) if launch is not None else _LocalProc(cmd, cwd, child_env)
	cut = None
	while True:
		try:
			if proc.poll() is not None:
				break
		except Exception:  # the sandbox went away under us: what was measured still counts
			cut = "lost"
			break
		idle = proxy.run_stats(run_id).get("idle_s")
		if time.monotonic() - t0 > timeout_s:
			cut = "timeout"
		elif should_stop is not None and should_stop():
			cut = "cancelled"
		elif idle is not None and idle > stall_s:
			cut = "stalled"  # tasks open, nothing calling the model: a tool stuck in a loop
		if cut:
			break
		proc.wait(1.0)
	if cut is None and time.monotonic() - t0 >= timeout_s - 5:
		cut = "timeout"  # ended by its sandbox's own limit, at about the same time
	at_cut = proxy.run_stats(run_id) if cut else {}
	# the tasks open when the script was stopped: those with a call in flight
	in_flight = set(at_cut.get("in_flight", []))
	# At the proxy a task stuck in its tool looks exactly like one that finished on a tool call (a
	# submit, a passing test run), so no task is singled out: the run's own stall is the signal,
	# and selection says how many tasks ended waiting on a tool, of which up to the script's
	# workers may be stuck
	waiting = set(at_cut.get("tool_wait_s") or {}) if cut == "stalled" else set()
	proc.kill()
	wall = time.monotonic() - t0
	ids = proxy.run_tasks(run_id)
	ended_at = time.monotonic()
	spans = proxy.run_stats(run_id).get("span") or {}

	def task_wall(t):
		"""B (Oct 31): the task's own wall time, first call to last, tool time between turns
		included; a task still in a call when the run was stopped runs to the stop. Tool time after a
		task's last answer can't be seen at the proxy."""
		opened, end = spans.get(t, (None, None))
		if opened is None:
			return None
		return round(max(0.0, (end if end is not None and t not in in_flight else ended_at) - opened), 3)

	# a task stopped with the whole run by a cancel or the run's time limit is "cut", not a timeout; on a
	# stall, the tasks still waiting on a tool get no stop reason (possibly stuck: the planted runaways)
	cut_tasks = in_flight if cut in ("cancelled", "timeout") else set()
	open_tasks = (waiting | in_flight) if cut == "stalled" else set()
	tasks = [proxy.task(t, wall_s=task_wall(t), cut=t in cut_tasks, open_end=t in open_tasks) for t in ids]
	st = proxy.run_stats(run_id)
	parts = [f"the first {len(ids)} tasks the script started"]
	if st.get("skipped"):
		parts.append(f"{st['skipped']} already sampled skipped")
	if st.get("refused"):
		parts.append(f"{st['refused']} calls refused past the sample")
	if withheld:
		parts.append(f"{len(withheld)} other provider keys withheld")
	if st.get("single_call") and len(ids) and st["single_call"] * 2 > len(ids):
		parts.append(f"{st['single_call']} one-call tasks (side calls may split tasks)")
	if st.get("budget_hit"):
		parts.append("dollar cap reached")
	ended = (getattr(proc, "report", None) or {}).get("ended")
	if ended and ended not in ("exited", "timeout"):
		parts.append(f"sandbox: {ended}")  # e.g. install_failed: not the same as making no calls
	why = {"timeout": f", cut at {timeout_s:.0f} s", "cancelled": ", cancelled", "lost": ", sandbox lost",
		"stalled": f", stopped after {stall_s:.0f} s with no model calls"}.get(cut, "")  # fmt: skip
	parts.insert(1, f"script wall {wall:.0f} s{why}")  # how it ended survives the 200-character cut
	failed = (getattr(proc, "report", None) or {}).get("error") if ended == "exited" else None
	own = proxy.own_replies() if note_replies and hasattr(proxy, "own_replies") else []
	own = [r for r in own if r[0] != 404]  # a stray request at the tunnel's address, not the script's
	if own:  # what the script heard from the proxy itself, which its SDK may not say
		parts.insert(2, "proxy answered " + ", ".join(_own_reply(*r) for r in own[:2]))
	if failed:  # probes.agent_error_line: why a script that made no calls stopped, ahead of the rest
		parts.insert(2, f"script failed: {failed}")
	if cut == "stalled" and waiting:
		w = at_cut.get("peak_concurrent") or 1
		parts.insert(2, f"{len(waiting)} ended on a tool call, up to {min(w, len(waiting))} may be stuck")
	selection = "; ".join(parts)
	if len(selection) > 200:
		selection = selection[:197].rsplit(" ", 1)[0] + "..."
	if inspect is not None:
		# how each task ended, for an ops run's own scoring (counts and order only, never content): its
		# opening rank, the last call's finish reason, and whether it ended waiting on a tool
		order = sorted(ids, key=lambda t: (spans.get(t, (None,))[0] is None, spans.get(t, (0.0,))[0] or 0.0))
		inspect({t: {"opened_rank": order.index(t), "last_finish": proxy.last_finish(t), "waiting_on_tool": t in waiting,
			"in_flight_at_stop": t in in_flight} for t in ids})  # fmt: skip
	# 0.6.0: how the sampled run ended (a lost sandbox is none of these)
	how = {"stalled": "stalled", "cancelled": "cancelled", "timeout": "timeout", None: "finished"}.get(cut)
	# 0.6.1: the script's workers, so the engine caps how many tasks of a stalled run can be stuck.
	# The proxy's peak of calls in flight misses workers busy in a tool between calls, so also the
	# most task spans (first call to last call's end; one still in flight, to the end) that overlap.
	# A task that ended on a tool call ends at that call: most of them finished, and which are
	# stuck can't be told, so their tool time after it isn't counted (a lower bound still)
	workers = max((at_cut or st).get("peak_concurrent") or 0, _max_overlap(spans)) or None
	return AgentSample(tasks=tasks, population=population, selection=selection, ended=how,
		concurrency=int(workers) if workers else None)  # fmt: skip


def _max_overlap(spans: dict) -> int:
	"""The most task spans open at once; a span with no end (a call in flight) stays open to the end."""
	edges = []
	for opened, end in spans.values():
		if opened is None:
			continue
		edges.append((opened, 1))
		if end is not None:
			edges.append((end, -1))
	most = cur = 0
	for _, d in sorted(edges, key=lambda e: (e[0], e[1])):  # an end before a start at the same instant
		cur += d
		most = max(most, cur)
	return most


class _quiet:
	def __enter__(self):
		return self

	def __exit__(self, *exc):
		quiet = ProcessLookupError | PermissionError | subprocess.TimeoutExpired
		return exc[0] is not None and issubclass(exc[0], quiet)
