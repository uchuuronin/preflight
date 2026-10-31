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
- The key reaches Muse Code on stdin (`--api-key-stdin`) only, so it is in no argv, environment
  or file; a Muse Code without that flag isn't run at all (the startup check refuses it).
- Every output (diff, final message, session log) is scanned for the key, plain and in common
  encodings, before it leaves the sandbox; a hit discards the candidate. The scan is a backstop
  that a determined encoding beats: the gate keeping the key out of reach is the control.

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

_ROT13 = str.maketrans(
	"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
	"NOPQRSTUVWXYZABCDEFGHIJKLMnopqrstuvwxyzabcdefghijklm",
)

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
"""Preflight's PreToolUse gate for Muse Code: edits only, inside one worktree.

Deny by default: a tool not listed, an input key not listed for that tool, a nested value, a path
outside the worktree or into its git or Muse Code state, or any error in here."""
import json, os, sys

ROOT = os.path.realpath(sys.argv[1])
NO_WRITE = set(sys.argv[2].split(","))
# Input keys each allowed tool may carry. Path keys hold a path (or a list of paths for "paths");
# pattern keys hold a glob that must stay relative; text keys hold scalars only.
PATH_KEYS = {"path", "file_path", "paths", "directory", "dir"}
PATTERN_KEYS = {"glob", "include", "exclude"}
TEXT_KEYS = {
	"read_file": {"offset", "limit", "start_line", "end_line", "line"},
	"search": {"query", "pattern", "regex", "case_sensitive", "max_results", "literal"},
	"edit_file": {"old_string", "new_string", "old_str", "new_str", "replace_all", "old_text", "new_text"},
	"write_file": {"content", "text"},
}
WRITES = {"edit_file", "write_file"}


def out(reason):
	print(json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse",
		"permissionDecision": "deny", "permissionDecisionReason": "Preflight: " + reason}}))
	sys.exit(0)


def inside(p):
	if not isinstance(p, str) or not p or "\0" in p or p[0] in "~$" or ":" in p.split("/")[0]:
		return None
	full = os.path.realpath(p if os.path.isabs(p) else os.path.join(ROOT, p))
	return full if full == ROOT or full.startswith(ROOT + os.sep) else None


try:
	ev = json.loads(sys.stdin.read() or "{}")
	tool = ev.get("tool_name")
	if tool not in TEXT_KEYS:
		out("only reading and editing files is allowed here; no commands.")
	cwd = ev.get("cwd")
	if cwd is not None and inside(cwd) is None:
		out("outside the repository.")
	args = ev.get("tool_input")
	if not isinstance(args, dict):
		out("unreadable tool input.")
	seen_path = False
	for key, value in args.items():
		if key in PATH_KEYS:
			values = value if key == "paths" and isinstance(value, list) else [value]
			for v in values:
				full = inside(v)
				if full is None:
					out("outside the repository.")
				first = os.path.relpath(full, ROOT).split(os.sep)[0]
				if tool in WRITES and first in NO_WRITE:
					out("that file isn't part of the fix.")
				seen_path = True
		elif key in PATTERN_KEYS:
			if not isinstance(value, str) or value.startswith(("/", "~", "$")) or ".." in value or ":" in value:
				out("patterns must stay inside the repository.")
		elif key in TEXT_KEYS[tool]:
			if isinstance(value, (dict, list)):
				out("unexpected tool input.")
		else:
			out("unexpected tool input: " + str(key)[:40])
	if tool in WRITES and not seen_path:
		out("an edit must name its file.")
	sys.exit(0)  # no decision: the tool runs
except SystemExit:
	raise
except BaseException:
	out("the gate failed, so the tool is denied.")
'''


# Source: derived, where Modal's debian_slim image puts Python. Muse Code runs hooks with a cleared
# environment, so the interpreter is named by absolute path: a bare `python3` with no PATH would
# fail to start, and a hook that fails lets the tool run.
HOOK_PYTHON = "/usr/local/bin/python3"


def hook_command(hook_path: str, root: str, python: str = HOOK_PYTHON) -> str:
	return f"{python} {hook_path} {root} {','.join(NO_WRITE)}"


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


#: Every flag `command` passes; the startup check requires all of them in `muse exec --help`.
REQUIRED_FLAGS = (
	"--json",
	"--prompt-file",
	"--max-model-steps",
	"--approval-mode",
	"--contributor-mode",
	"--sandbox-network",
	"--api-key-stdin",
)


def command(prompt_file: str, *, steps: int = MAX_MODEL_STEPS, model: str | None = None) -> list[str]:
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
		"--api-key-stdin",
	]
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
	#: The largest value seen per counter (right if events repeat running totals).
	usage: Usage = field(default_factory=Usage)
	#: Every value added up (right if events report per turn). The cost cap uses this, the larger.
	usage_sum: Usage = field(default_factory=Usage)
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
	total = {"in": 0, "cached": 0, "out": 0}
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
				kind = (
					"cached"
					if _CACHED.search(k)
					else "in"
					if _IN.search(k)
					else "out"
					if _OUT.search(k)
					else None
				)
				if kind:
					best[kind] = max(best[kind], v)
					total[kind] += v
			name = d.get("tool_name") or d.get("tool")
			if isinstance(name, str):
				ev.tools.append(name)
			for k in ("result", "output", "error", "message"):
				v = d.get(k)
				if isinstance(v, str) and (
					"by hook" in v or v.startswith("Preflight: ") or "\nPreflight: " in v
				):
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
	ev.usage_sum = Usage(total["in"], total["cached"], total["out"])
	return ev


def _encodings(secret: str) -> list[str]:
	b = secret.encode()
	forms = {secret, secret[::-1], b.hex(), b.hex().upper(), urllib.parse.quote(secret, safe="")}
	forms.add(base64.urlsafe_b64encode(b).decode().rstrip("="))
	forms.add(base64.b32encode(b).decode().rstrip("="))
	forms.add(secret.translate(_ROT13))
	for shift in range(3):
		# Base64 of the secret at each alignment inside a longer string: drop the edge characters
		# that depend on its neighbours.
		enc = base64.b64encode(b"\0" * shift + b).decode()
		core = enc[(shift * 4 + 2) // 3 : len(enc) - 4]
		if len(core) >= 12:
			forms.add(core)
	return [f for f in forms if len(f) >= 8]


def _squash(text: str) -> str:
	"""Letters and digits only, so a key split by spaces, newlines, quotes or commas still shows."""
	return re.sub(r"[^A-Za-z0-9]", "", text)


def leaks(text: str, secrets: list[str]) -> bool:
	"""True when any secret appears in the text plain, split up by punctuation or whitespace,
	reversed, rot13, hex, URL-encoded, base32 or base64 at any alignment, as \\u escapes or as a list
	of character codes. Short secrets (under 8 characters) aren't checked: they'd match by chance."""
	if not text:
		return False
	keep = [s for s in secrets if s and len(s) >= 8]
	if any(f in text for s in keep for f in _encodings(s)):
		return True
	squashed = _squash(text)
	if any(len(_squash(s)) >= 8 and (_squash(s) in squashed or _squash(s)[::-1] in squashed) for s in keep):
		return True
	codes = "".join(chr(int(n)) for n in re.findall(r"\b\d{2,3}\b", text) if 32 <= int(n) < 127)
	escaped = re.sub(
		r"\\u([0-9a-fA-F]{4})|\\x([0-9a-fA-F]{2})", lambda m: chr(int(m.group(1) or m.group(2), 16)), text
	)
	if any(s in codes or s in escaped for s in keep):
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


def diff_problem(diff: str, touched: list[str], files: set[str]) -> str | None:
	"""Why a candidate's diff can't be offered, or None. It must change only files that were in the
	visitor's upload or repo, stay small, and change something. `touched` comes from
	`git diff --name-only -z`, so names with spaces or other characters arrive whole."""
	if not diff.strip() and not touched:
		return "no_change"
	if len(diff.encode()) > MAX_DIFF_BYTES:
		return "too_big"
	if re.search(r"^(new|deleted) file mode|^rename (from|to) |^Binary files", diff, flags=re.M):
		return "not_an_edit"
	if not touched or not set(touched) <= files:
		return "outside_files"
	return None
