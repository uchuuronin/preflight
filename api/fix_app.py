"""The fix loop's Modal app: Muse Code in a sandbox, edits only (preflight_api/fix/).

Two entry points, both run with `modal run --env main` (an ephemeral app: nothing is deployed and
nothing bills once the call returns), each only after Chiya's yes:

	modal run --env main api/fix_app.py::startup_check   # no model call
	modal run --env main api/fix_app.py::smoke            # one Muse Code fix, capped

How a fix runs: a Function holding `preflight-meta` creates a Sandbox from the fix image with no
secrets, outbound traffic only to Meta's API host, writes the visitor's files in, and runs one
`muse exec` per candidate in its own git worktree. The key goes to each Muse Code process on
stdin, never into the sandbox's environment or a file. The Sandbox is terminated when the race
ends, at the token-cost cap, or at its own timeout.

Secrets (never in the repo): "preflight-meta" with META_API_KEY and META_API_BASE.
"""

from __future__ import annotations

import contextlib
import json
import os
import shlex
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import modal

_HERE = Path(__file__).resolve()
ROOT = _HERE.parent.parent if (_HERE.parent.parent / "flags.json").is_file() else Path("/repo-not-here")

# Source: derived, where Meta's installer puts the binary is not documented, so both likely
# places are on PATH and the startup check records which one `muse` resolved to.
MUSE_PATH = "/root/.local/bin:/root/.muse/bin:/usr/local/bin:/usr/bin:/bin"
# Source: derived, the guide's measured price table (guide/muse_spark.py; $3.01 measured on
# Meta's Usage page against $2.91 from the table, docs/decisions/guide-limits.md).
PRICE_IN, PRICE_CACHED, PRICE_OUT = 1.25, 0.15, 4.25
# Source: guess, the smoke test's approved cap (Oct 31); a public race's cap comes from the
# smoke test's measured cost per candidate.
SMOKE_STOP_USD = 1.00
# Source: derived, CANDIDATE_TIMEOUT_S (5 min, the contract's Muse Code timeout) plus image
# start and git; the Sandbox ends on its own after this however the Function fares.
SANDBOX_TIMEOUT_S = 8 * 60

fix_image = (
	modal.Image.debian_slim(python_version="3.11")
	.apt_install("git", "curl", "ca-certificates", "coreutils")
	# Meta's documented install (dev.meta.ai/docs/muse-code); no version pin exists, so the
	# version is printed here and recorded by the startup check and every fix.
	# The installer uses bash (`set -o pipefail`), so it is piped to bash, not Debian's sh (dash).
	.run_commands("curl -fsSL https://dev.meta.ai/install.sh | bash", f"PATH={MUSE_PATH} muse --version")
	.env({"PATH": MUSE_PATH})
)

runner_image = modal.Image.debian_slim(python_version="3.11").add_local_dir(
	ROOT / "api" / "preflight_api" / "fix", "/root/fixpkg/preflight_api/fix"
)

app = modal.App("preflight-fix", image=runner_image)
META = modal.Secret.from_name("preflight-meta")


def _pkg():
	import sys

	if "/root/fixpkg" not in sys.path:
		sys.path.insert(0, "/root/fixpkg")
	(Path("/root/fixpkg/preflight_api/__init__.py")).touch()


class ModalBox:
	"""race.Box over a Modal Sandbox. `stop_usd` ends the whole sandbox once Muse Code's reported
	token cost across every candidate passes it."""

	def __init__(self, sb, stop_usd: float | None = None):
		self.sb = sb
		self.stop_usd = stop_usd
		self.spent = 0.0
		self.stopped = False
		self._lock = threading.Lock()
		self._seen: dict[int, float] = {}

	def write(self, path: str, text: str) -> None:
		d = os.path.dirname(path)
		p = self.sb.exec("sh", "-c", f"mkdir -p {shlex.quote(d)} && cat > {shlex.quote(path)}")
		p.stdin.write(text)
		p.stdin.write_eof()
		p.stdin.drain()
		if p.wait() != 0:
			raise RuntimeError(f"write {path} failed")

	def run(self, argv, *, cwd=None, stdin=None, timeout=60):
		import math

		from preflight_api.fix import muse, race

		if self.stopped:
			return race.STOPPED, "", "stopped at the cost cap"
		argv = ["timeout", "--kill-after=5", str(max(1, math.ceil(timeout))), *argv]
		p = self.sb.exec(*argv, workdir=cwd)
		if stdin is not None:
			p.stdin.write(stdin)
			p.stdin.write_eof()
			p.stdin.drain()
		err: list[str] = []
		# stderr drains on its own thread, so a chatty stderr can't stall stdout.
		drain = threading.Thread(target=lambda: err.append(p.stderr.read()), daemon=True)
		drain.start()
		watch = self.stop_usd is not None and "muse" in argv
		lines: list[str] = []
		buf = ""
		try:
			for chunk in p.stdout:
				if not watch:
					buf += chunk
					continue
				# The stream may hand over chunks, not lines: split them here.
				buf += chunk
				*whole, buf = buf.split("\n")
				lines.extend(whole)
				if not whole:
					continue
				ev = muse.parse_events(lines)
				u = ev.usage_cap.usd(PRICE_IN, PRICE_CACHED, PRICE_OUT)
				with self._lock:
					self._seen[id(p)] = u
					self.spent = sum(self._seen.values())
					over = self.spent > self.stop_usd
				if over:
					with self._lock:
						first = not self.stopped
						self.stopped = True
					if first:
						self.sb.terminate()
					return race.STOPPED, "\n".join(lines), "stopped at the cost cap"
		except Exception:
			if self.stopped:
				return race.STOPPED, "\n".join(lines) + buf, "stopped at the cost cap"
			raise
		out = "\n".join([*lines, buf]) if watch else buf
		try:
			code = p.wait()
		except Exception:
			if self.stopped:
				return race.STOPPED, out, "stopped at the cost cap"
			raise
		drain.join(timeout=5)
		return code, out, "".join(err)


def _sandbox(app_ref, *, network: bool):
	host = urlparse(os.environ.get("META_API_BASE", "https://api.meta.ai/v1")).hostname or "api.meta.ai"
	kw = {"outbound_domain_allowlist": [host]} if network else {"block_network": True}
	return modal.Sandbox.create(
		app=app_ref, image=fix_image, timeout=SANDBOX_TIMEOUT_S, cpu=1.0, memory=2048, **kw
	)


@app.function(timeout=10 * 60, retries=0)
def startup_check_fix() -> dict:
	"""No model call: the fix image starts, `muse` is there and takes every flag the fix loop passes,
	the hook's interpreter exists, the gate denies a command run exactly as Muse Code will run it,
	and the exact command line starts and ends with Muse Code's no-model `echo` provider. It also
	finds what turns the background observers off, the same way."""
	_pkg()
	from preflight_api.fix import muse, race

	sb = _sandbox(app, network=False)
	box = ModalBox(sb)
	try:
		out: dict = {}
		help_text = ""
		for name, argv in {
			"which": ["sh", "-c", "command -v muse"],
			"version": ["env", "-i", f"PATH={MUSE_PATH}", "muse", "--version"],
			"exec_help": ["env", "-i", f"PATH={MUSE_PATH}", "muse", "exec", "--help"],
			"python": ["sh", "-c", f"test -x {muse.HOOK_PYTHON} && {muse.HOOK_PYTHON} --version"],
		}.items():
			code, so, se = box.run(argv, timeout=60)
			if name == "exec_help":
				help_text = so + se  # judged whole; only the report is shortened
			out[name] = {"code": code, "out": so[:12000], "err": se[-1000:]}
		# Every flag the fix loop passes must exist; without --api-key-stdin the key would have to
		# travel in the environment, so nothing runs.
		out["flags"] = {f: f in help_text for f in muse.REQUIRED_FLAGS}
		try:
			race.prepare(box, {"train.py": "print('hello')\n"})  # ends with gate_check
			out["gate"] = "denies a command and a read outside the worktree"
		except Exception as e:  # GateBroken or git failing in the image
			out["gate"] = f"FAILED: {type(e).__name__}: {e}"
		echo = _echo_runner(box)
		out["echo"] = echo(())
		out["observers"] = _observer_switch(echo, help_text)
		_, so, se = box.run(
			[
				"env",
				"-i",
				f"PATH={MUSE_PATH}",
				"HOME=/fix/home/echo",
				"XDG_CONFIG_HOME=/fix/cfg/echo",
				"muse",
				"config",
				"status",
			],
			timeout=60,
		)
		out["config_status"] = (so + se)[:6000]
		out["ok"] = (
			out["version"]["code"] == 0
			and out["python"]["code"] == 0
			and all(out["flags"].values())
			and not out["gate"].startswith("FAILED")
			and out["echo"]["code"] == 0
		)
		return out
	finally:
		sb.terminate()


# Settings that might turn Muse Code's background observers off, tried as `-c KEY=VALUE`, which
# refuses the run on an unknown key (Muse Code 1.4.14 help). Source: guess, from the observer
# names in Meta's docs; one is used only when the echo provider accepts it and refuses a made-up key.
OBSERVER_CONFIG = (
	"runtime_capabilities.memory_recall=false",
	"runtime_capabilities.skill_recommendations=false",
	"runtime_capabilities.goal_tracking=false",
	"runtime_capabilities.verification=false",
	"runtime_capabilities.observers=false",
)


def _bare_flag(help_text: str, flag: str) -> bool:
	import re

	if "=" in flag:
		return flag.split("=")[0] in help_text and bool(
			re.search(re.escape(flag.split("=")[0]) + r"[ =]<?(on|off)", help_text)
		)
	return bool(re.search(re.escape(flag) + r"(?![\w-])(?!\s*[=<]|\s+[A-Z]{2,})", help_text))


def _echo_runner(box: ModalBox):
	"""Run the fix loop's exact `muse exec` line in a worktree with the hook configured, on the `echo`
	provider (no model, no key needed), adding `extra` flags. Returns its exit code and output."""
	from preflight_api.fix import muse, race

	wt, cfg, home = "/fix/wt/echo", "/fix/cfg/echo", "/fix/home/echo"
	box.run([*race.GIT, "-C", race.REPO, "worktree", "add", "-q", "--detach", wt])
	box.write(
		f"{cfg}/muse/settings.json",
		json.dumps(muse.settings("/fix/hook.py", wt, log="/fix/hooklog/echo.jsonl")),
	)
	box.write(f"{home}/.keep", "")
	box.write("/fix/prompt-echo.txt", "Say hello.")

	def run(extra: tuple[str, ...]) -> dict:
		argv = [
			"env",
			"-i",
			f"PATH={MUSE_PATH}",
			f"XDG_CONFIG_HOME={cfg}",
			f"HOME={home}",
			*muse.command(
				"/fix/prompt-echo.txt",
				steps=2,
				model=None,  # "Model id for non-echo providers"
				extra_flags=("--provider", "echo", *extra),
				key_on_stdin=False,
			),
		]
		code, so, se = box.run(argv, cwd=wt, timeout=90)
		return {
			"code": code,
			"out_head": "\n".join(so.splitlines()[:30]),
			"err": se[-800:],
			"extra": list(extra),
		}

	return run


def _observer_switch(echo, help_text: str) -> dict:
	"""Whether this Muse Code can run without its background observers, and how: a bare flag from
	`muse exec --help`, or `-c` settings, each kept only if the echo provider runs with it (and, for
	settings, refuses a made-up key, so acceptance means something). Nothing here calls a model."""
	from preflight_api.fix import muse

	found: dict = {"flags": [], "config": [], "tried": []}
	for f in muse.OBSERVER_FLAGS:
		if _bare_flag(help_text, f):
			r = echo((f,))
			found["tried"].append(r)
			if r["code"] == 0:
				found["flags"].append(f)
	# A made-up key at the top and inside the map: a capability name counts only if Muse Code would
	# have refused a made-up one in the same place.
	bogus = echo(("-c", "zz_not_a_setting=false"))
	inner = echo(("-c", "runtime_capabilities.zz_not_a_capability=false"))
	found["tried"] += [bogus, inner]
	if bogus["code"] != 0 and inner["code"] != 0:
		for kv in OBSERVER_CONFIG:
			r = echo(("-c", kv))
			found["tried"].append(r)
			if r["code"] == 0:
				found["config"].append(kv)
	return found


SMOKE_EXTRA = (
	"\nBefore editing, run `python train.py --help` with your shell tool to see the script's options. "
	"If the tool is refused, say so in your reply and carry on with the edit.\n"
)
# The preamble tells real candidates not to try commands, so the first smoke run's model never
# asked for one (run 41246585559). The smoke test drops that one sentence, so the model tries
# and the gate is seen refusing it; everything else in the task is the real one.
NO_COMMANDS = "You can't run commands, tests or the\n  script; don't try. "


@app.function(secrets=[META], timeout=12 * 60, retries=0)
def smoke_fix(files: dict[str, str], observers: dict | None = None) -> dict:
	"""One Muse Code fix on our own zoo script: eval.wrap_no_grad, one candidate, capped at
	SMOKE_STOP_USD. The task also asks for a command, to prove the gate blocks it headless."""
	_pkg()
	from preflight_api.fix import race, templates

	key = os.environ["META_API_KEY"]
	# The smoke test's own task text: the template plus the command request.
	orig = templates.PREAMBLE
	if NO_COMMANDS not in orig:
		raise RuntimeError("the preamble changed; update the smoke test's prompt")
	templates.PREAMBLE = orig.replace(NO_COMMANDS, "") + SMOKE_EXTRA
	sb = _sandbox(app, network=True)
	box = ModalBox(sb, stop_usd=SMOKE_STOP_USD)
	t0 = time.monotonic()
	try:
		obs = observers or {}
		extra = tuple(obs.get("flags", [])) + tuple(x for kv in obs.get("config", []) for x in ("-c", kv))
		cands = race.race(
			box,
			files,
			[("eval.wrap_no_grad", None)],
			key,
			path=MUSE_PATH,
			extra_flags=extra,
			# Shell stays available here only, so the smoke test proves the gate blocks a command.
			allow_shell=True,
			log_lines=3000,
		)
	finally:
		templates.PREAMBLE = orig
		with contextlib.suppress(Exception):  # it may have ended already (cost cap, timeout)
			sb.terminate()
	return {
		"seconds": round(time.monotonic() - t0, 1),
		"usd_from_reported_tokens": round(box.spent, 4),
		"stopped_at_cap": box.stopped,
		"candidates": [c.public(with_log=True) for c in cands],
	}


@app.local_entrypoint()
def startup_check() -> None:
	import json

	res = startup_check_fix.remote()
	print(json.dumps(res, indent=1))
	if not res.get("ok"):
		raise SystemExit(1)


@app.local_entrypoint()
def smoke() -> None:
	import json

	zoo = ROOT / "zoo" / "finetune_lora_hf"
	files = {p.name: p.read_text() for p in sorted(zoo.iterdir()) if p.suffix in (".py", ".txt")}
	check = startup_check_fix.remote()
	if not check.get("ok"):
		print(json.dumps(check, indent=1))
		raise SystemExit("startup check failed; no model call made")
	res = smoke_fix.remote(files, check.get("observers"))
	print(
		json.dumps(
			{"startup": {k: check[k] for k in ("flags", "gate", "version", "observers")}, **res}, indent=1
		)
	)
	c = res["candidates"][0]
	# Passes when the fix landed and the command it was asked to run was blocked, under the cap.
	ok = (
		c["status"] == "patched"
		and any(b in ("bash", "shell", "bash_input") for b in c["blocked"])
		and not res["stopped_at_cap"]
	)
	raise SystemExit(0 if ok else 1)
