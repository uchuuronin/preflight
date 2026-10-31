"""Race candidate fixes in parallel worktrees, Muse Code editing each, and hand back their diffs.

One race is one sandbox holding a git copy of the visitor's files. Each candidate gets its own
worktree, its own Muse Code config (the edits-only hook, muse.settings) and its own `muse exec`,
all at once; a candidate that runs past its deadline is stopped and loses. What comes back per
candidate is a diff of tracked files only, the final message, token usage and what the hook
blocked, with any candidate whose output carries the key discarded (muse.leaks).

The winner isn't decided here: each patched candidate goes to a verify job (A's job store,
mode "verify", with the patch), and verify.py's both-sides checks pick it. A race only proposes.

`Box` is the sandbox: Modal's in api/fix_app.py, a local one in the tests.
"""

from __future__ import annotations

import json
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


class Box(Protocol):
	def write(self, path: str, text: str) -> None: ...

	def run(
		self, argv: list[str], *, cwd: str | None = None, stdin: str | None = None, timeout: float = 60
	) -> tuple[int, str, str]:
		"""Exit code (124 on timeout), stdout, stderr."""
		...


@dataclass
class Candidate:
	id: str
	template_id: str
	value: object | None
	status: str = "pending"  # patched | no_change | refused | timeout | over_budget | failed | leaked
	diff: str = ""
	final: str | None = None
	usage: dict = field(default_factory=dict)
	blocked: list[str] = field(default_factory=list)
	seconds: float = 0.0
	exit_code: int | None = None
	problem: str | None = None
	#: The last lines of Muse Code's JSONL, for the smoke test only (they can quote visitor code).
	log_tail: str = ""

	def public(self, *, with_log: bool = False) -> dict:
		"""What may leave the sandbox and be shown: never stderr, which can quote the environment."""
		d = asdict(self)
		if not with_log or self.status == "leaked":
			d.pop("log_tail")
		if self.status == "leaked":
			d.update(diff="", final=None)
		return d


def prepare(box: Box, files: dict[str, str], python: str = muse.HOOK_PYTHON) -> None:
	"""Write the visitor's files as one commit, so each worktree starts from the same tree."""
	for rel, text in files.items():
		if rel.startswith("/") or ".." in rel.split("/"):
			raise ValueError(f"bad path {rel!r}")
		box.write(f"{REPO}/{rel}", text)
	box.write(f"{ROOT}/hook.py", muse.HOOK_SOURCE)
	for argv in (
		["git", "init", "-q", REPO],
		["git", "-C", REPO, "add", "-A"],
		[
			"git",
			"-C",
			REPO,
			"-c",
			"user.name=preflight",
			"-c",
			"user.email=fix@preflight.invalid",
			"commit",
			"-qm",
			"visitor files",
		],
	):
		code, _, err = box.run(argv)
		if code != 0:
			raise RuntimeError(f"{argv[1:3]} failed: {err[-300:]}")
	gate_check(box, REPO, python)


class GateBroken(RuntimeError):
	"""The edits-only hook didn't deny a command. Muse Code runs a tool whose hook fails, so no
	candidate may start."""


def gate_check(box: Box, root: str, python: str = muse.HOOK_PYTHON) -> None:
	"""Run the hook exactly as settings.json will, with an empty environment, on a shell command,
	and require a deny."""
	argv = ["env", "-i", *muse.hook_command(f"{ROOT}/hook.py", root, python).split()]
	payload = json.dumps(
		{"hook_event_name": "PreToolUse", "tool_name": "bash", "tool_input": {"command": "env"}, "cwd": root}
	)
	code, out, _ = box.run(argv, stdin=payload, timeout=20)
	try:
		decision = json.loads(out)["hookSpecificOutput"]["permissionDecision"]
	except (ValueError, KeyError, TypeError):
		decision = None
	if code != 0 or decision != "deny":
		raise GateBroken("the edits-only gate didn't deny a command")


def _one(
	box: Box,
	cand: Candidate,
	key: str,
	*,
	key_on_stdin: bool,
	steps: int,
	timeout: float,
	finding: str | None,
	secrets: list[str],
	files: set[str],
	python: str,
	path: str,
) -> Candidate:
	wt = f"{ROOT}/wt/{cand.id}"
	cfg = f"{ROOT}/cfg/{cand.id}"
	prompt = f"{ROOT}/prompt-{cand.id}.txt"
	code, _, _ = box.run(["git", "-C", REPO, "worktree", "add", "-q", "--detach", wt])
	if code != 0:
		cand.status, cand.problem = "failed", "worktree"
		return cand
	box.write(f"{cfg}/muse/settings.json", json.dumps(muse.settings(f"{ROOT}/hook.py", wt, python)))
	box.write(prompt, task_for(cand.template_id, cand.value, finding))
	env = [f"XDG_CONFIG_HOME={cfg}", f"HOME={cfg}"]
	if not key_on_stdin:
		env.append(f"META_API_KEY={key}")
	argv = [
		"env",
		"-i",
		f"PATH={path}",
		*env,
		*muse.command(prompt, steps=steps, key_on_stdin=key_on_stdin),
	]
	t0 = time.monotonic()
	code, out, _err = box.run(argv, cwd=wt, stdin=key + "\n" if key_on_stdin else None, timeout=timeout)
	cand.seconds = round(time.monotonic() - t0, 1)
	cand.exit_code = code
	ev = muse.parse_events(out.splitlines())
	cand.usage = asdict(ev.usage)
	cand.blocked = ev.blocked
	cand.final = ev.final
	cand.log_tail = "\n".join(out.splitlines()[-200:])
	_, diff, _ = box.run(["git", "-C", wt, "diff", "--no-color", "--no-ext-diff"])
	cand.diff = diff
	if any(muse.leaks(x or "", secrets) for x in (diff, ev.final, out)):
		cand.status, cand.problem, cand.log_tail = "leaked", "secret_in_output", ""
		return cand
	if code == 124:
		cand.status = "timeout"
		return cand
	if code == 125:
		cand.status = "over_budget"
		return cand
	problem = muse.diff_problem(diff, files)
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
	key_on_stdin: bool = True,
	steps: int = muse.MAX_MODEL_STEPS,
	timeout: float = CANDIDATE_TIMEOUT_S,
	finding: str | None = None,
	extra_secrets: list[str] | None = None,
	python: str = muse.HOOK_PYTHON,
	path: str = SYSTEM_PATH,
) -> list[Candidate]:
	"""Run every candidate at once; each is (template id, the knob value it sets or None)."""
	if not 1 <= len(candidates) <= MAX_CANDIDATES:
		raise ValueError(f"between 1 and {MAX_CANDIDATES} candidates")
	for tid, _ in candidates:
		if tid not in TEMPLATES:
			raise KeyError(tid)
	prepare(box, files, python)
	cands = [Candidate(id=f"c{i + 1}", template_id=tid, value=v) for i, (tid, v) in enumerate(candidates)]
	secrets = [key, *(extra_secrets or [])]
	with ThreadPoolExecutor(max_workers=len(cands)) as pool:
		futs = [
			pool.submit(
				_one,
				box,
				c,
				key,
				key_on_stdin=key_on_stdin,
				steps=steps,
				timeout=timeout,
				finding=finding,
				secrets=secrets,
				files=set(files),
				python=python,
				path=path,
			)
			for c in cands
		]
		return [f.result() for f in futs]


def shell_line(argv: list[str]) -> str:
	"""For logs only: the command with the key never in it (it travels on stdin)."""
	return " ".join(shlex.quote(a) for a in argv)
