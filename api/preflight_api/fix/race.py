"""Race candidate fixes in parallel worktrees, Muse Code editing each, and hand back their diffs.

One race is one sandbox holding a git copy of the visitor's files. Each candidate gets its own
worktree, its own Muse Code config (the edits-only hook, muse.settings) and its own `muse exec`,
all at once; a candidate that runs past its deadline is stopped and loses. What comes back per
candidate is a diff of tracked files only, the final message, token usage and what the hook
blocked, with any candidate whose output carries the key discarded (muse.leaks, a backstop: the
gate is what keeps the key out of reach).

The visitor's files are data that git reads, so nothing in them may configure git or Muse Code:
a path through `.git` is refused (a planted `.git/config` could run a command on every git call),
Muse Code's own config files are dropped, and every git call runs with system and global config
off, no fsmonitor and no hooks.

The winner isn't decided here: each patched candidate goes to a verify job (A's job store, mode
"verify", with the patch), and verify.py's both-sides checks pick it. A race only proposes.

`Box` is the sandbox: Modal's in api/fix_app.py, a local one in the tests.
"""

from __future__ import annotations

import json
import math
import shlex
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Protocol

from . import muse
from .templates import TEMPLATES, task_for

ROOT = "/fix"
REPO = f"{ROOT}/repo"
# Source: guess, a template fix is a few minutes of reading and one edit; the contract's Muse
# Code timeout is 5 min per fix (Failure and degradation table).
CANDIDATE_TIMEOUT_S = 300
# Source: derived, the plan's race shows three fixes; more would multiply token spend.
MAX_CANDIDATES = 3
# The PATH a candidate's Muse Code sees (the sandbox's own environment is dropped).
SYSTEM_PATH = "/usr/local/bin:/usr/bin:/bin"
# Exit codes: coreutils `timeout` gives 124 when it stops a command and 137 when it has to kill
# it; a box that stops the run at the cost cap gives STOPPED, which no command or `timeout` uses.
TIMED_OUT = (124, 137)
STOPPED = 199

# git, reading nothing but the repo's own minimal config: no system or global files, no
# fsmonitor command, no hooks, and paths printed raw so names with spaces stay whole.
GIT = [
	"env",
	"GIT_CONFIG_NOSYSTEM=1",
	"GIT_CONFIG_GLOBAL=/dev/null",
	"git",
	"-c",
	"core.fsmonitor=false",
	"-c",
	"core.hooksPath=/dev/null",
	"-c",
	"core.quotePath=false",
]


class Box(Protocol):
	def write(self, path: str, text: str) -> None: ...

	def run(
		self, argv: list[str], *, cwd: str | None = None, stdin: str | None = None, timeout: float = 60
	) -> tuple[int, str, str]:
		"""Exit code (124 or 137 on timeout, STOPPED at the cost cap), stdout, stderr."""
		...


@dataclass
class Candidate:
	id: str
	template_id: str
	value: object | None
	status: str = "pending"  # patched | no_change | refused | timeout | over_budget | failed | leaked
	diff: str = ""
	files: list[str] = field(default_factory=list)
	final: str | None = None
	usage: dict = field(default_factory=dict)
	blocked: list[str] = field(default_factory=list)
	seconds: float = 0.0
	exit_code: int | None = None
	problem: str | None = None
	#: The last lines of Muse Code's JSONL, for the smoke test only (they can quote visitor code).
	log_tail: str = ""
	#: Every tool call the gate saw: tool, input key names, paths, decision (never contents).
	tool_calls: list[dict] = field(default_factory=list)

	def public(self, *, with_log: bool = False) -> dict:
		"""What may leave the sandbox and be shown: never stderr, which can quote the environment."""
		d = asdict(self)
		if not with_log or self.status == "leaked":
			d.pop("log_tail")
		if self.status == "leaked":
			d.update(diff="", final=None, files=[], tool_calls=[])
		return d


class BadUpload(ValueError):
	pass


def clean_files(files: dict[str, str]) -> tuple[dict[str, str], list[str]]:
	"""The files a race may use, and the ones dropped. Refuses absolute paths, `..` and anything
	inside a `.git` directory; drops Muse Code and agent config (`muse.NO_WRITE`) wherever it sits,
	since Muse Code may read project settings from its working directory."""
	kept: dict[str, str] = {}
	dropped: list[str] = []
	for rel, text in files.items():
		parts = rel.split("/")
		if not rel or rel.startswith("/") or "\\" in rel or any(p in ("", ".", "..") for p in parts):
			raise BadUpload(f"bad path {rel!r}")
		if ".git" in parts:
			raise BadUpload(f"a git directory can't be part of the upload: {rel!r}")
		if any(p in muse.NO_WRITE for p in parts):
			dropped.append(rel)
			continue
		kept[rel] = text
	if not kept:
		raise BadUpload("nothing to fix")
	return kept, dropped


def prepare(box: Box, files: dict[str, str], python: str = muse.HOOK_PYTHON) -> None:
	"""Write the cleaned files as one commit, so each worktree starts from the same tree, then prove
	the gate denies a command."""
	for rel, text in files.items():
		box.write(f"{REPO}/{rel}", text)
	box.write(f"{ROOT}/hook.py", muse.HOOK_SOURCE)
	ident = ["-c", "user.name=preflight", "-c", "user.email=fix@preflight.invalid"]
	for argv in (
		[*GIT, "init", "-q", REPO],
		[*GIT, "-C", REPO, "add", "-A"],
		[*GIT, "-C", REPO, *ident, "commit", "-qm", "visitor files"],
	):
		code, _, err = box.run(argv)
		if code != 0:
			raise RuntimeError(f"git {argv[-2:]} failed: {err[-300:]}")
	gate_check(box, REPO, python)


class GateBroken(RuntimeError):
	"""The edits-only hook didn't deny a command. Muse Code runs a tool whose hook fails, so no
	candidate may start."""


def gate_check(box: Box, root: str, python: str = muse.HOOK_PYTHON) -> None:
	"""Run the hook exactly as settings.json will, with an empty environment, on a shell command and
	on a read outside the worktree, and require a deny for both."""
	argv = ["env", "-i", *muse.hook_command(f"{ROOT}/hook.py", root, python).split()]
	for tool, args in (("bash", {"command": "env"}), ("read_file", {"path": "/proc/self/environ"})):
		payload = json.dumps(
			{"hook_event_name": "PreToolUse", "tool_name": tool, "tool_input": args, "cwd": root}
		)
		code, out, _ = box.run(argv, stdin=payload, timeout=20)
		try:
			decision = json.loads(out)["hookSpecificOutput"]["permissionDecision"]
		except (ValueError, KeyError, TypeError):
			decision = None
		if code != 0 or decision != "deny":
			raise GateBroken(f"the edits-only gate didn't deny {tool}")


def _one(
	box: Box,
	cand: Candidate,
	key: str,
	*,
	steps: int,
	timeout: float,
	measured: str | None,
	secrets: list[str],
	files: set[str],
	python: str,
	path: str,
	settings_extra: dict | None,
	extra_flags: tuple[str, ...],
	log_lines: int,
) -> Candidate:
	wt = f"{ROOT}/wt/{cand.id}"
	cfg = f"{ROOT}/cfg/{cand.id}"
	home = f"{ROOT}/home/{cand.id}"
	prompt = f"{ROOT}/prompt-{cand.id}.txt"
	hooklog = f"{ROOT}/hooklog/{cand.id}.jsonl"
	try:
		code, _, _ = box.run([*GIT, "-C", REPO, "worktree", "add", "-q", "--detach", wt])
		if code != 0:
			cand.status, cand.problem = "failed", "worktree"
			return cand
		box.write(
			f"{cfg}/muse/settings.json",
			json.dumps(muse.settings(f"{ROOT}/hook.py", wt, python, hooklog, settings_extra)),
		)
		box.write(f"{home}/.keep", "")
		box.write(hooklog, "")
		box.write(prompt, task_for(cand.template_id, cand.value, measured))
	except Exception:  # the sandbox ended (cost cap) or a write failed
		cand.status, cand.problem = "failed", "setup"
		return cand
	# The key travels on stdin only: never argv, never the environment (`--api-key-stdin`).
	argv = [
		"env",
		"-i",
		f"PATH={path}",
		f"XDG_CONFIG_HOME={cfg}",
		f"HOME={home}",
		*muse.command(prompt, steps=steps, extra_flags=extra_flags),
	]
	t0 = time.monotonic()
	code, out, _err = box.run(argv, cwd=wt, stdin=key + "\n", timeout=timeout)
	cand.seconds = round(time.monotonic() - t0, 1)
	cand.exit_code = code
	ev = muse.parse_events(out.splitlines())
	cand.usage = {**asdict(ev.usage), "summed": asdict(ev.usage_sum)}
	cand.blocked = ev.blocked
	cand.final = ev.final
	cand.log_tail = "\n".join(out.splitlines()[-log_lines:])
	try:
		_, calls, _ = box.run(["cat", hooklog])
		if not muse.leaks(calls, secrets):
			for line in calls.splitlines()[:200]:
				try:
					cand.tool_calls.append(json.loads(line))
				except ValueError:
					continue
	except Exception:  # the log is a report, never part of the decision
		cand.tool_calls = [{"tool": None, "decision": "log_unreadable"}]
	if muse.leaks(out, secrets) or muse.leaks(ev.final or "", secrets):
		cand.status, cand.problem, cand.log_tail, cand.final = "leaked", "secret_in_output", "", None
		return cand
	if code == STOPPED:
		cand.status = "over_budget"
		return cand
	try:
		c1, diff, _ = box.run([*GIT, "-C", wt, "diff", "--no-color", "--no-ext-diff", "--no-textconv"])
		c2, names, _ = box.run([*GIT, "-C", wt, "diff", "--name-only", "-z", "--no-ext-diff"])
	except Exception:
		cand.status, cand.problem = "failed", "diff"
		return cand
	if STOPPED in (c1, c2):  # another candidate hit the cost cap and the sandbox ended
		cand.status = "over_budget"
		return cand
	if c1 != 0 or c2 != 0:
		cand.status, cand.problem = "failed", "diff"
		return cand
	touched = [n for n in names.split("\0") if n]
	if muse.leaks(diff, secrets):
		cand.status, cand.problem, cand.log_tail, cand.final = "leaked", "secret_in_output", "", None
		return cand
	cand.diff, cand.files = diff, touched
	if code in TIMED_OUT:
		cand.status = "timeout"
		return cand
	problem = muse.diff_problem(diff, touched, files)
	if problem == "no_change":
		cand.status = "no_change"
	elif problem:
		cand.status, cand.problem = "refused", problem
	else:
		cand.status = "patched"
	return cand


def race(
	box: Box,
	files: dict[str, str],
	candidates: list[tuple[str, object | None]],
	key: str,
	*,
	steps: int = muse.MAX_MODEL_STEPS,
	timeout: float = CANDIDATE_TIMEOUT_S,
	measured: str | None = None,
	extra_secrets: list[str] | None = None,
	python: str = muse.HOOK_PYTHON,
	path: str = SYSTEM_PATH,
	settings_extra: dict | None = None,
	extra_flags: tuple[str, ...] = (),
	log_lines: int = 200,
) -> list[Candidate]:
	"""Run every candidate at once; each is (template id, the knob value it sets or None).
	`measured` is Preflight's own sentence about the problem (A's rule finding), never visitor text."""
	if not 1 <= len(candidates) <= MAX_CANDIDATES:
		raise ValueError(f"between 1 and {MAX_CANDIDATES} candidates")
	for tid, v in candidates:
		if tid not in TEMPLATES:
			raise KeyError(tid)
		task_for(tid, v)  # refuses a missing or unsafe value before anything runs
	kept, _dropped = clean_files(files)
	prepare(box, kept, python)
	cands = [Candidate(id=f"c{i + 1}", template_id=tid, value=v) for i, (tid, v) in enumerate(candidates)]
	secrets = [key, *(extra_secrets or [])]
	limit = max(1, math.ceil(timeout))
	with ThreadPoolExecutor(max_workers=len(cands)) as pool:
		futs = [
			pool.submit(
				_one,
				box,
				c,
				key,
				steps=steps,
				timeout=limit,
				measured=measured,
				secrets=secrets,
				files=set(kept),
				python=python,
				path=path,
				settings_extra=settings_extra,
				extra_flags=extra_flags,
				log_lines=log_lines,
			)
			for c in cands
		]
		return [f.result() for f in futs]


def shell_line(argv: list[str]) -> str:
	"""For logs only: the command with the key never in it (it travels on stdin)."""
	return " ".join(shlex.quote(a) for a in argv)
