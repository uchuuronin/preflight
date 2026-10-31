"""Muse Code headless, edits only (Chiya's option 1, docs/decisions/muse-code-sandbox.md).

How one candidate runs, inside a sandbox that holds a copy of the visitor's files and no secrets:

- A PreToolUse hook (HOOK_SOURCE, written to the sandbox) allows read_file, search, edit_file and
  write_file on paths inside the candidate's worktree, and denies every other tool: bash, shell,
  web, subagents, memory, cron and MCP tools included. Its deny output is the shape Muse Code
  honours (`hookSpecificOutput.permissionDecision: "deny"`); any error inside the hook also
  denies, because a hook that fails lets the tool run (observed on Muse Code 1.4.10:
  github.com/Falconiere/toolu/issues/540).
- `--approval-mode never`: a headless run can't answer a prompt, and "ask" under the default
  mode hangs until killed (same source). The hook is the gate; Muse Code's own OS sandbox stays on
  with network `proxy-only`, and the sandbox's own egress is Meta's API host only.
- The key reaches Muse Code on stdin (`--api-key-stdin`) when the installed version has the flag,
  so it is in no process's environment; otherwise only in Muse Code's own environment, which the
  hook keeps every tool from reading (`/proc` is outside the worktree).
- Every output (diff, final message, session log) is scanned for the key, plain and encoded,
  before it leaves the sandbox; a hit discards the candidate.

Muse Code's exit code means the turn finished, not that the fix is right: every candidate is
gated on Preflight's re-probe and verify (preflight_api/verify.py).

Stdlib only.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
import urllib.parse
from dataclasses import dataclass, field

# Source: derived, the tools Muse Code 1.4.10 exposes (toolu issue 540's inventory); these four
# are the read and edit tools option 1 keeps. Everything else is denied by name, so a tool added
# in a later version is denied until it is listed here on purpose.
ALLOWED_TOOLS = ("read_file", "search", "edit_file", "write_file")
# Paths inside the worktree a candidate may not write: git's own state and Muse Code's config,
# which could otherwise loosen the hook.
NO_WRITE = (".git", ".muse", ".agents", "AGENTS.md", "CLAUDE.md", ".claude")

# Source: guess, a template fix is a few lines in one or two files; 15 model steps leaves room
# to read, edit and summarise. Replaced by the smoke test's measured step count.
MAX_MODEL_STEPS = 15
# Source: guess, a patch over 64 KiB is not one of these templates' small changes.
MAX_DIFF_BYTES = 64 * 1024

HOOK_SOURCE = r'''#!/usr/bin/env python3
"""Preflight's PreToolUse gate for Muse Code: edits only, inside one worktree. Any error denies."""
import json, os, sys

ROOT = os.path.realpath(sys.argv[1])
ALLOWED = set(sys.argv[2].split(","))
NO_WRITE = set(sys.argv[3].split(","))
PATH_KEYS = ("path", "file_path", "file", "paths", "target", "directory", "dir", "cwd")


def out(decision, reason):
	print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse",
		"permissionDecision": decision, "permissionDecisionReason": reason}}))
	sys.exit(0)


def inside(p, base):
	full = os.path.realpath(p if os.path.isabs(p) else os.path.join(base, p))
	return full == ROOT or full.startswith(ROOT + os.sep), full


def paths(value):
	if isinstance(value, str):
		return [value]
	if isinstance(value, list):
		return [v for v in value if isinstance(v, str)]
	return []


try:
	ev = json.loads(sys.stdin.read() or "{}")
	tool = ev.get("tool_name")
	if tool not in ALLOWED:
		out("deny", "Preflight: only reading and editing files is allowed here; no commands.")
	base = ev.get("cwd") or ROOT
	ok, _ = inside(base, ROOT)
	if not ok:
		out("deny", "Preflight: outside the repository.")
	args = ev.get("tool_input") or {}
	if not isinstance(args, dict):
		out("deny", "Preflight: unreadable tool input.")
	for key in PATH_KEYS:
		for p in paths(args.get(key)):
			ok, full = inside(p, base)
			if not ok:
				out("deny", "Preflight: outside the repository.")
			if tool in ("edit_file", "write_file"):
				rel = os.path.relpath(full, ROOT).split(os.sep)
				if rel and rel[0] in NO_WRITE:
					out("deny", "Preflight: that file isn't part of the fix.")
	sys.exit(0)  # no decision: the tool runs
except SystemExit:
	raise
except BaseException:
	out("deny", "Preflight: the gate failed, so the tool is denied.")
'''


# Source: derived, where Modal's debian_slim image puts Python. Muse Code runs hooks with a cleared
# environment, so the interpreter is named by absolute path: a bare `python3` with no PATH would
# fail to start, and a hook that fails lets the tool run.
HOOK_PYTHON = "/usr/local/bin/python3"


def hook_command(hook_path: str, root: str, python: str = HOOK_PYTHON) -> str:
	return f"{python} {hook_path} {root} {','.join(ALLOWED_TOOLS)} {','.join(NO_WRITE)}"


def settings(hook_path: str, root: str, python: str = HOOK_PYTHON) -> dict:
	"""$XDG_CONFIG_HOME/muse/settings.json for one candidate: Claude-format hooks, matching every
	tool. `schema_version` 1 is required or Muse Code won't start (kenn-io/kit PR 194)."""
	return {
		"schema_version": 1,
		"hooks": {
			"PreToolUse": [
				{
					"matcher": "*",
					"hooks": [{"type": "command", "command": hook_command(hook_path, root, python)}],
				}
			]
		},
	}


def command(
	prompt_file: str, *, steps: int = MAX_MODEL_STEPS, key_on_stdin: bool = True, model: str | None = None
) -> list[str]:
	"""The `muse exec` line for one candidate. Root options go after `exec` (kenn-io/kit PR 194)."""
	argv = [
		"muse",
		"exec",
		"--json",
		"--prompt-file",
		prompt_file,
		"--max-model-steps",
		str(steps),
		"--approval-mode",
		"never",
		"--contributor-mode",
		"off",
		"--sandbox-network",
		"proxy-only",
	]
	if key_on_stdin:
		argv.append("--api-key-stdin")
	if model:
		argv += ["--model", model]
	return argv


@dataclass
class Usage:
	input_tokens: int = 0
	cached_input_tokens: int = 0
	output_tokens: int = 0

	def usd(self, price_in: float, price_cached: float, price_out: float) -> float:
		"""Cached input is counted inside input_tokens, as the guide's Responses usage does."""
		fresh = max(0, self.input_tokens - self.cached_input_tokens)
		return (
			fresh * price_in + self.cached_input_tokens * price_cached + self.output_tokens * price_out
		) / 1e6


@dataclass
class Events:
	usage: Usage = field(default_factory=Usage)
	#: Tools the hook blocked, by name, in order.
	blocked: list[str] = field(default_factory=list)
	tools: list[str] = field(default_factory=list)
	final: str | None = None
	session_id: str | None = None
	unknown: int = 0


_IN = re.compile(r"(^|_)(input|prompt)_tokens$")
_CACHED = re.compile(r"cache(d|_read)(_input)?_tokens$")
_OUT = re.compile(r"(^|_)(output|completion)_tokens$")


def _walk(obj, fn, depth=0):
	if depth > 8:
		return
	if isinstance(obj, dict):
		fn(obj)
		for v in obj.values():
			_walk(v, fn, depth + 1)
	elif isinstance(obj, list):
		for v in obj:
			_walk(v, fn, depth + 1)


def parse_events(lines) -> Events:
	"""Read `muse exec --json` output. Muse Code's event names aren't documented, so this reads by
	field: usage counters wherever they appear (the largest seen per counter, since some events
	repeat running totals), tool names, the hook's deny text, the session id and the last agent
	message. Lines that aren't JSON are counted and kept out."""
	ev = Events()
	best = {"in": 0, "cached": 0, "out": 0}
	for raw in lines:
		raw = raw.strip()
		if not raw:
			continue
		try:
			obj = json.loads(raw)
		except ValueError:
			ev.unknown += 1
			continue

		def look(d: dict) -> None:
			for k, v in d.items():
				if not isinstance(v, int) or isinstance(v, bool):
					continue
				if _CACHED.search(k):
					best["cached"] = max(best["cached"], v)
				elif _IN.search(k):
					best["in"] = max(best["in"], v)
				elif _OUT.search(k):
					best["out"] = max(best["out"], v)
			name = d.get("tool_name") or d.get("tool")
			if isinstance(name, str):
				ev.tools.append(name)
			for k in ("result", "output", "error", "message"):
				v = d.get(k)
				if isinstance(v, str) and ("blocked by hook" in v or "denied by hook" in v):
					ev.blocked.append(name if isinstance(name, str) else "?")
			sid = d.get("session_id")
			if isinstance(sid, str):
				ev.session_id = sid

		_walk(obj, look)
		if isinstance(obj, dict) and obj.get("role") == "assistant" and isinstance(obj.get("text"), str):
			ev.final = obj["text"]
		for k in ("final_message", "last_message", "final"):
			if isinstance(obj, dict) and isinstance(obj.get(k), str):
				ev.final = obj[k]
	ev.usage = Usage(best["in"], best["cached"], best["out"])
	return ev


def _encodings(secret: str) -> list[str]:
	b = secret.encode()
	forms = {secret, secret[::-1], b.hex(), b.hex().upper(), urllib.parse.quote(secret, safe="")}
	forms.add(base64.urlsafe_b64encode(b).decode().rstrip("="))
	for shift in range(3):
		# Base64 of the secret at each alignment inside a longer string: drop the edge characters
		# that depend on its neighbours.
		enc = base64.b64encode(b"\0" * shift + b).decode()
		core = enc[(shift * 4 + 2) // 3 : len(enc) - 4]
		if len(core) >= 12:
			forms.add(core)
	return [f for f in forms if len(f) >= 8]


def leaks(text: str, secrets: list[str]) -> bool:
	"""True when any secret appears in the text plain, reversed, hex, URL-encoded or base64 at any
	alignment. Short secrets (under 8 characters) aren't checked: they'd match by chance."""
	if not text:
		return False
	keep = [s for s in secrets if s and len(s) >= 8]
	if any(f in text for s in keep for f in _encodings(s)):
		return True
	# Long base64 runs decoded, in case the key was encoded with something around it.
	for run in re.findall(r"[A-Za-z0-9+/=_-]{24,}", text):
		try:
			raw = base64.b64decode(run + "=" * (-len(run) % 4), altchars=b"-_", validate=False)
		except (binascii.Error, ValueError):
			continue
		dec = raw.decode("utf-8", "ignore")
		if any(s in dec for s in keep):
			return True
	return False


def diff_problem(diff: str, files: set[str]) -> str | None:
	"""Why a candidate's diff can't be offered, or None. It must change only files that were in the
	visitor's upload or repo, stay small, and change something."""
	if not diff.strip():
		return "no_change"
	if len(diff.encode()) > MAX_DIFF_BYTES:
		return "too_big"
	touched = set(re.findall(r"^\+\+\+ b/(.+)$", diff, flags=re.M)) | set(
		re.findall(r"^--- a/(.+)$", diff, flags=re.M)
	)
	if re.search(r"^(new|deleted) file mode|^rename (from|to) |^Binary files", diff, flags=re.M):
		return "not_an_edit"
	if not touched or not touched <= files:
		return "outside_files"
	return None
