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
NO_WRITE = (
	".git",
	".muse",
	".agents",
	"AGENTS.md",
	"CLAUDE.md",
	"MUSE.md",
	".claude",
	".mcp.json",
	".cursor",
	"copilot-instructions.md",
)

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
# Optional: where to note each call (tool, input key names, paths, decision), never contents.
LOG = sys.argv[3] if len(sys.argv) > 3 else None
SEEN = {"tool": None, "keys": [], "paths": []}


def note(decision, reason=""):
	if not LOG:
		return
	try:
		with open(LOG, "a") as f:
			f.write(json.dumps({**SEEN, "decision": decision, "reason": reason}) + "\n")
	except BaseException:
		pass
# Input keys each allowed tool may carry. Path keys hold a path (or a list of paths for "paths");
# pattern keys hold a glob that must stay relative; text keys hold scalars only.
PATH_KEYS = {"path", "file_path", "paths", "directory", "dir"}
PATTERN_KEYS = {"glob", "include", "exclude"}
TEXT_KEYS = {
	"read_file": {"offset", "limit", "start_line", "end_line", "line"},
	"search": {"query", "pattern", "regex", "case_sensitive", "max_results", "literal", "output_mode"},
	"edit_file": {"old_string", "new_string", "old_str", "new_str", "replace_all", "old_text", "new_text", "find", "replace"},
	"write_file": {"content", "text"},
}
WRITES = {"edit_file", "write_file"}


def out(reason):
	note("deny", reason)
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
	args = ev.get("tool_input")
	SEEN["tool"] = str(tool)[:60]
	if isinstance(args, dict):
		SEEN["keys"] = sorted(str(k)[:40] for k in args)[:20]
		SEEN["paths"] = [str(args[k])[:120] for k in sorted(args) if k in PATH_KEYS | PATTERN_KEYS][:5]
	if tool not in TEXT_KEYS:
		out("only reading and editing files is allowed here; no commands.")
	cwd = ev.get("cwd")
	if cwd is not None and inside(cwd) is None:
		out("outside the repository.")
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
				parts = os.path.relpath(full, ROOT).split(os.sep)
				if tool in WRITES and any(p in NO_WRITE for p in parts):
					out("that file isn't part of the fix.")
				seen_path = True
		elif key in PATTERN_KEYS:
			# One glob or a list of them (Muse Code 1.4.14 sends a list). No absolute or parent
			# paths, and none of the glob syntax that can spell one indirectly ({/proc,x}, [/]proc,
			# an escaped slash).
			globs = value if isinstance(value, list) else [value]
			for g in globs:
				if not isinstance(g, str) or g.startswith(("/", "~", "$")) or ".." in g or any(c in g for c in ":{}[]\\"):
					out("patterns must stay inside the repository.")
		elif key in TEXT_KEYS[tool]:
			if isinstance(value, (dict, list)):
				out("unexpected tool input.")
		else:
			out("unexpected tool input: " + str(key)[:40])
	if tool in WRITES and not seen_path:
		out("an edit must name its file.")
	note("allow")
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


def hook_command(hook_path: str, root: str, python: str = HOOK_PYTHON, log: str | None = None) -> str:
	return f"{python} {hook_path} {root} {','.join(NO_WRITE)}" + (f" {log}" if log else "")


# Flags some Muse Code versions might take to turn the background observers off; the startup check
# looks for each in `muse exec --help` and the fix loop passes the ones it finds. Source: guess,
# Meta's docs name the observers (memory recall, skill recommendations, goal tracking,
# verification) but no switch for them; replaced by what the installed help text says.
OBSERVER_FLAGS = ("--disable-reminders", "--no-observers", "--disable-observers", "--observers=off")


def settings(
	hook_path: str, root: str, python: str = HOOK_PYTHON, log: str | None = None, extra: dict | None = None
) -> dict:
	"""$XDG_CONFIG_HOME/muse/settings.json for one candidate: Claude-format hooks, matching every
	tool. `schema_version` 1 is required or Muse Code won't start (kenn-io/kit PR 194)."""
	out = {
		**(extra or {}),
		"schema_version": 1,
		"hooks": {
			"PreToolUse": [
				{
					"matcher": "*",
					"hooks": [{"type": "command", "command": hook_command(hook_path, root, python, log)}],
				}
			]
		},
	}
	return out


#: Every flag `command` passes; the startup check requires all of them in `muse exec --help`.
REQUIRED_FLAGS = (
	"--json",
	"--prompt-file",
	"--max-model-steps",
	"--approval-mode",
	"--contributor-mode",
	"--sandbox-network",
	"--api-key-stdin",
	"--disable-web-tools",
	"--disable-shell",
	"--no-foreign-personal-context",
)


# Source: measured, the model id the guide runs on Meta's API (docs/status/c.md, Oct 27). Named on
# every run: left to its default, Muse Code 1.4.14 ran muse-spark-1.3-contributor even with
# --contributor-mode off (smoke run 41246585559), whose usage Meta may use to improve its products.
MODEL = "muse-spark-1.3"


def is_contributor(model: str | None) -> bool:
	return bool(model) and "contributor" in str(model).lower()


def command(
	prompt_file: str,
	*,
	steps: int = MAX_MODEL_STEPS,
	model: str | None = MODEL,
	extra_flags: tuple[str, ...] = (),
	allow_shell: bool = False,
	key_on_stdin: bool = True,
) -> list[str]:
	"""The `muse exec` line for one candidate. Root options go after `exec` (kenn-io/kit PR 194).
	Defence in depth around the gate (Muse Code 1.4.14 help): no web tools, no personal rules or
	skills from elsewhere, and no shell at all unless `allow_shell` is set, which only the smoke test does,
	to prove the gate blocks a command on its own."""
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
		"--disable-web-tools",
		"--no-foreign-personal-context",
	]
	# Always for real runs; only the startup check's no-model `echo` runs leave it out, since Muse
	# Code refuses it with any provider but meta or local.
	if key_on_stdin:
		argv.append("--api-key-stdin")
	if not allow_shell:
		argv.append("--disable-shell")
	if model:
		argv += ["--model", model]
	argv += list(extra_flags)
	return argv


@dataclass
class Usage:
	input_tokens: int = 0
	cached_input_tokens: int = 0
	output_tokens: int = 0
	#: Muse Code 1.4.14 reports these beside output_tokens without saying whether they're inside it;
	#: priced as output on top, so a cost reading can be high but never low.
	reasoning_tokens: int = 0

	def usd(self, price_in: float, price_cached: float, price_out: float) -> float:
		"""Cached input is counted inside input_tokens, as the guide's Responses usage does."""
		fresh = max(0, self.input_tokens - self.cached_input_tokens)
		out = self.output_tokens + self.reasoning_tokens
		return (fresh * price_in + self.cached_input_tokens * price_cached + out * price_out) / 1e6


@dataclass
class Events:
	#: The largest value seen per counter (right if events repeat running totals).
	usage: Usage = field(default_factory=Usage)
	#: Every value added up (right if events report per turn).
	usage_sum: Usage = field(default_factory=Usage)
	#: What the cost cap reads: per counter, the largest value for running totals, the sum otherwise.
	usage_cap: Usage = field(default_factory=Usage)
	#: Tools the hook blocked, by name, in order.
	blocked: list[str] = field(default_factory=list)
	tools: list[str] = field(default_factory=list)
	final: str | None = None
	session_id: str | None = None
	#: The model Muse Code says it ran (run.model.configured).
	model: str | None = None
	unknown: int = 0


_IN = re.compile(r"(^|_)(input|prompt)_tokens$")
_CACHED = re.compile(r"cache(d|_read)(_input)?_tokens$")
_REASON = re.compile(r"(^|_)reasoning_tokens$")
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
	best = {"in": 0, "cached": 0, "out": 0, "reason": 0}
	total = dict(best)
	last = dict(best)
	dropped = {k: False for k in best}
	kinds: dict[str, str] = {}  # task id -> tool, from Muse Code 1.4.14's task events
	said: list[str] = []
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
					else "reason"
					if _REASON.search(k)
					else "in"
					if _IN.search(k)
					else "out"
					if _OUT.search(k)
					else None
				)
				if kind:
					if v < last[kind]:
						dropped[kind] = True
					last[kind] = v
					best[kind] = max(best[kind], v)
					total[kind] += v
			name = d.get("tool_name") or d.get("tool")
			if isinstance(name, str):
				ev.tools.append(name)
			for k in ("result", "output", "error", "message"):
				v = d.get(k)
				# Muse Code's own wording when a hook denies; text a tool read back can't start a result.
				if isinstance(v, str) and v.lstrip().startswith(
					("tool blocked by hook", "tool denied by hook", "Preflight: ")
				):
					ev.blocked.append(name if isinstance(name, str) else "?")
			sid = d.get("session_id")
			if isinstance(sid, str):
				ev.session_id = sid

		_walk(obj, look)
		# Muse Code 1.4.14 envelopes: {"payload_type": ..., "payload": {...}} (smoke run 41246585559).
		pt = obj.get("payload_type") if isinstance(obj, dict) else None
		pl = obj.get("payload") if isinstance(obj, dict) and isinstance(obj.get("payload"), dict) else {}
		event = pl.get("event") if isinstance(pl.get("event"), dict) else {}
		if pt == "run.output.delta" and isinstance(pl.get("text"), str):
			said.append(pl["text"])
		elif pt == "task.lifecycle.proposed" and isinstance(event.get("task_kind"), str):
			kinds[str(event.get("task_id"))] = event["task_kind"].removeprefix("tool.")
		elif pt == "task.lifecycle.rejected" and str(event.get("reason", "")).startswith("Preflight: "):
			ev.blocked.append(kinds.get(str(event.get("task_id")), "?"))
		elif pt == "run.model.configured" and isinstance(pl.get("model_id"), str):
			ev.model = pl["model_id"]
		if isinstance(obj, dict) and obj.get("role") == "assistant" and isinstance(obj.get("text"), str):
			ev.final = obj["text"]
		for k in ("final_message", "last_message", "final"):
			if isinstance(obj, dict) and isinstance(obj.get(k), str):
				ev.final = obj[k]
	if said:
		ev.final = "".join(said)
	ev.usage = Usage(best["in"], best["cached"], best["out"], best["reason"])
	ev.usage_sum = Usage(total["in"], total["cached"], total["out"], total["reason"])
	# For the cost cap: a counter that never goes down is a running total (its largest value is
	# the spend); one that goes down is per turn (its sum is).
	pick = {k: total[k] if dropped[k] else best[k] for k in best}
	ev.usage_cap = Usage(pick["in"], pick["cached"], pick["out"], pick["reason"])
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
