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

# The Function's own image: the fix package, and A's package and zoo so a race reads the record's
# code exactly as A's runner does (record_jobs.resolve, sources.fetch: git for repo records).
runner_image = (
	modal.Image.debian_slim(python_version="3.11")
	.apt_install("git", "ca-certificates")
	.pip_install("pydantic>=2.8,<3", "numpy>=1.26", "httpx>=0.27")
	.env({"PREFLIGHT_ZOO": "/root/zoo"})
	.add_local_dir(ROOT / "api" / "preflight_api" / "fix", "/root/fixpkg/preflight_api/fix")
	.add_local_dir(ROOT / "python" / "preflight", "/root/fixpkg/preflight")
	.add_local_dir(ROOT / "zoo", "/root/zoo")
)

app = modal.App("preflight-fix", image=runner_image)
META = modal.Secret.from_name("preflight-meta")
# A's volumes (python/preflight/runner/modal_app.py creates them); read-only here, never created.
RECORDS_DIR, UPLOADS_DIR = "/records", "/uploads"
records_vol = modal.Volume.from_name("preflight-records")
uploads_vol = modal.Volume.from_name("preflight-uploads")
# Source: derived, SANDBOX_TIMEOUT_S plus reading the code (a repo clone is capped by A's
# CLONE_TIMEOUT_S) and the sandbox's start; the API treats a race older than this as lost.
RACE_TIMEOUT_S = 12 * 60


def _read_only(vol):
	try:
		return vol.with_mount_options(read_only=True)
	except AttributeError:  # older SDKs
		return vol.read_only()


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
		out["reads_code"] = _reads_code()
		out["ok"] = (
			out["reads_code"].startswith("ok")
			and out["version"]["code"] == 0
			and out["python"]["code"] == 0
			and all(out["flags"].values())
			and not out["gate"].startswith("FAILED")
			and out["echo"]["code"] == 0
		)
		return out
	finally:
		sb.terminate()


def _reads_code() -> str:
	"""race_fix's half, in this Function's own image: A's package imports, git is there, and a
	sample's code reads the way a race reads it (no volume needed for a sample)."""
	import subprocess

	try:
		from preflight_api.fix import run

		git = subprocess.run(
			["/usr/bin/git", "--version"], capture_output=True, text=True, check=True
		).stdout.strip()
		files = run.source_files("sample:lora-7b", reader=None, blobs=None)
		return f"ok: {git}; sample lora-7b reads as {len(files)} files"
	except Exception as e:
		return f"FAILED: {type(e).__name__}: {e}"


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


@app.function(
	secrets=[META],
	timeout=RACE_TIMEOUT_S,
	retries=0,
	volumes={RECORDS_DIR: _read_only(records_vol), UPLOADS_DIR: _read_only(uploads_vol)},
)
def race_fix(record_ref: str, candidates: list[dict], measured: str | None, stop_usd: float) -> dict:
	"""One public race, spawned by the API (preflight_api/fix/service.py ModalFixer). The record's
	code is read here, never by the API; the sandbox opens only once it is in hand."""
	_pkg()
	from preflight.runner.records import LocalRecordWriter
	from preflight.runner.remote_store import DirBlobs

	from preflight_api.fix import run

	reader = LocalRecordWriter(RECORDS_DIR, on_read=records_vol.reload)
	blobs = DirBlobs(UPLOADS_DIR, reload=uploads_vol.reload)
	boxes: list[ModalBox] = []

	def open_box():
		sb = _sandbox(app, network=True)
		box = ModalBox(sb, stop_usd=stop_usd)
		boxes.append(box)

		def close():
			with contextlib.suppress(Exception):  # it may have ended already (cost cap, timeout)
				sb.terminate()

		return box, close

	return run.run_race(
		record_ref,
		candidates,
		measured,
		reader=reader,
		blobs=blobs,
		open_box=open_box,
		key=os.environ["META_API_KEY"],
		path=MUSE_PATH,
	)


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


@app.local_entrypoint()
def race_sample() -> None:
	"""The first real race, as a visitor would start it, on the lora-7b sample's eval
	out-of-memory finding (three fixes), run once with Chiya's yes:

		modal run --env main api/fix_app.py::race_sample

	The startup check runs first (no model call); the race is capped at three candidates' worth."""
	import json
	import sys

	sys.path.insert(0, str(ROOT / "api"))  # this runs on the caller's machine, from the repo
	from preflight_api.fix import run

	check = startup_check_fix.remote()
	if not check.get("ok"):
		print(json.dumps(check, indent=1)[:6000])
		raise SystemExit("startup check failed; no model call made")
	cands = [
		{"template_id": "eval.reduce_batch", "value": 8},
		{"template_id": "eval.wrap_no_grad"},
		{"template_id": "memory.gradient_checkpointing", "value": True},
	]
	res = race_fix.remote(
		"sample:lora-7b", cands, "First eval will run out of memory", run.stop_usd(len(cands))
	)
	print(json.dumps(res, indent=1))
	patched = [c for c in res.get("candidates", []) if c["status"] == "patched"]
	raise SystemExit(0 if patched and not res.get("error") else 1)


@app.local_entrypoint()
def race_stack(attempt: str = "a", gpu: str = "A10") -> None:
	"""The Nov 2 race (PM's order: one race, US$1 cap): the fix loop as the web runs it, on the LoRA
	zoo script's eval finding (races/nov2_stack.json). Round 1 races the single fix candidatesFor
	offers and verifies it; only when none passes does round 2 race the stacked eval fixes, the
	eval batch from the guide's combined move (web/src/fix/candidates.ts stackedCandidate), and
	verify that. The winner is the passing candidate with the lowest eval peak.

		modal run --env main api/fix_app.py::race_stack

	Spend: before each step, what's spent so far plus that step's worst case must stay under the
	cap. Muse Code's worst case is its stop_usd plus one usage event past it (the box stops only
	once the reported spend is over); it counts at what it reported plus the price table's gap
	to the bill, or at that worst case when it reported nothing. A verify job counts at its
	record's costs (A's US$0.60 job cap when it has none). Verify jobs go to the
	deployed runner's job store; nothing that reaches a record is printed."""
	import json
	import sys

	sys.path.insert(0, str(ROOT / "api"))  # this runs on the caller's machine, from the repo
	import verify_race_app as vr
	from preflight_api.fix import run

	if gpu not in ("L4", "A10"):
		raise SystemExit("gpu: L4 or A10")
	cap = 1.0  # Source: the PM's order, Nov 2 11:16 ("One race, cap $1").
	# Source: measured, Oct 31 (this module's smoke test): the table read US$2.91 for a US$3.01 bill,
	# 3.4% under; 4% covers it.
	bill_gap = 1.04
	# Source: guess, one Muse Code usage event's worth past the stop (the box checks after each).
	overshoot = 0.02
	spec = json.loads((ROOT / "api" / "preflight_api" / "fix" / "races" / "nov2_stack.json").read_text())
	check = startup_check_fix.remote()
	if not check.get("ok"):
		print(json.dumps(check, indent=1)[:6000])
		raise SystemExit("startup check failed; no model call made")
	svc = modal.Cls.from_name("preflight-runner", "JobStoreService", environment_name="main")()
	vol = modal.Volume.from_name("preflight-records", environment_name="main")
	spent, rounds = 0.0, []

	def one_round(n: int, cands: list[dict]) -> list[dict]:
		nonlocal spent
		stop = run.stop_usd(len(cands))
		if spent + stop + overshoot + vr.JOB_CAP_USD * len(cands) > cap:
			rounds.append({"round": n, "status": "not_run", "why": f"cap: US${spent:.4f} spent"})
			return []
		res = race_fix.remote(spec["record_ref"], cands, spec["finding"], stop)
		muse = res.get("usd_from_reported_tokens")
		spent += muse * bill_gap if muse else stop + overshoot
		out = {"round": n, "muse_usd": muse, "seconds": res.get("seconds"), "error": res.get("error"),
			"stopped_at_cap": res.get("stopped_at_cap"), "verified": []}  # fmt: skip
		rounds.append(out)
		for i, c in enumerate(res.get("candidates") or []):
			info = {
				"id": f"r{n}c{i + 1}",
				"template_id": c["template_id"],
				"value": c.get("value"),
				"diff": c.get("diff") or "",
			}
			print(f"round {n} {info['id']} ({c['template_id']}): {c['status']}", flush=True)
			if c["status"] != "patched" or not info["diff"]:
				out["verified"].append(
					{"id": info["id"], "template_id": c["template_id"], "status": c["status"]}
				)
				continue
			if spent + vr.JOB_CAP_USD > cap:
				out["verified"].append(
					{"id": info["id"], "status": "not_run", "why": f"cap: US${spent:.4f} spent"}
				)
				continue
			v, cost = vr.verify_one(svc, vol, info, gpu=gpu, visitor=f"c.fix_verify.nov2.{attempt}",
				name=f"c-fix-stack-{info['id']}-nov2-{attempt}", record_ref=spec["record_ref"])  # fmt: skip
			spent += cost
			v["value"] = c.get("value")
			v["diff_lines"] = info["diff"].count("\n")
			out["verified"].append(v)
			print(json.dumps(v, indent=1), flush=True)
		return [v for v in out["verified"] if (v.get("verify") or {}).get("status") == "passed"]

	passed = one_round(1, spec["round1"])
	if not passed:
		passed = one_round(2, [spec["stack"]])
	winner = min(passed, key=lambda r: r["verify"]["after"] or float("inf")) if passed else None
	print(json.dumps({"spent_usd": round(spent, 4), "cap_usd": cap,
		"winner": winner and {k: winner.get(k) for k in ("id", "template_id", "value", "verify")},
		"rounds": [{**r, "verified": [{k: v.get(k) for k in ("id", "template_id", "value", "status", "why", "job_status", "usd", "record_id")}
			| {"verify": (v.get("verify") or {}).get("status")} for v in r.get("verified", [])]} for r in rounds]}, indent=1))  # fmt: skip
	raise SystemExit(0 if winner else 1)
