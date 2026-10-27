"""Run one probe in a subprocess and turn its measurements into a contract Probe.

The same function runs locally, in CI on CPU and inside a Modal GPU container.
User code is treated as hostile (gap 6):

  - submissions are validated: relative paths inside the workdir, a safe
    character set, caps on file count and size;
  - the child gets a scrubbed environment and a throwaway HOME, so no API key,
    token or credential file is in reach;
  - the parent marks itself non-dumpable, so the child cannot read the
    parent's /proc/<pid>/environ; when the worker runs as root the child runs
    as a fresh unprivileged uid that owns only its workdir;
  - the child runs in its own session; when it ends, for any reason, its whole
    process group and every process of its uid are killed, so nothing it
    started outlives the probe;
  - output is capped; a hard wall-clock cap and a cancellation callback can
    stop it at any time; results from an earlier attempt are wiped first.

Network egress control is the container's job (Modal sandbox allowlist);
the probe container itself must hold no secrets (see runner/modal_app.py).
"""

import collections
import contextlib
import ctypes
import itertools
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from pydantic import ValidationError

from preflight.recorder.errors import classify_exit
from preflight.schema import (
	GPU,
	CheckpointStats,
	DataStats,
	Env,
	EvalStats,
	GenerationStats,
	KernelTable,
	LengthStats,
	Memory,
	Power,
	Probe,
	ProbeCost,
	RLSignals,
	Stability,
	StepTime,
	Throughput,
)

from . import untrusted
from .plan import TIMED_STEPS, WARMUP_STEPS

log = logging.getLogger(__name__)

# variables a probe may see; everything else (keys, tokens, cloud creds) is dropped
SAFE_ENV = (
	"PATH",
	"LANG",
	"LC_ALL",
	"PYTHONPATH",
	"VIRTUAL_ENV",
	"CUDA_VISIBLE_DEVICES",
	"HIP_VISIBLE_DEVICES",
	"ROCR_VISIBLE_DEVICES",
	"NVIDIA_VISIBLE_DEVICES",
	"NVIDIA_DRIVER_CAPABILITIES",
	"LD_LIBRARY_PATH",
	"HF_HOME",
	"HF_HUB_OFFLINE",
	"HF_DATASETS_OFFLINE",
	"TRANSFORMERS_OFFLINE",
	"TORCH_HOME",
	"OMP_NUM_THREADS",
)
MAX_FILES = 2_000
MAX_FILE_BYTES = 8 * 1024 * 1024
MAX_TOTAL_BYTES = 25 * 1024 * 1024
LOG_CAP_BYTES = 5 * 1024 * 1024
LOG_TAIL_BYTES = 16 * 1024
_PATH_OK = re.compile(r"^[A-Za-z0-9._\-/ ]+$")
_UIDS = itertools.count()
_PROBE_LOCK = threading.Lock()
VALID_STATUS = {"ok", "oom", "error", "timeout", "cancelled"}


# Probe.extras keys B reads by name (docs/requests.md, B -> A); keep them stable
EXTRAS_KEYS = (
	"params_total",
	"params_trainable",
	"tokens_per_step",
	"samples_per_step",
	"micro_steps_per_step",
	"host_rss_peak_bytes",
)


class InvalidSubmission(ValueError):
	"""The submitted files can't be probed as given. Not retryable."""


@dataclass
class ProbeSpec:
	probe_id: str
	files: dict[str, str]
	entrypoint: str
	args: list[str] = field(default_factory=list)
	knobs: dict = field(default_factory=dict)
	warmup: int = WARMUP_STEPS
	timed: int = TIMED_STEPS
	eval_batches: int = 2
	profile: bool = False
	timeout_s: float = 600.0
	seed: int | None = None
	out_dir: Path | None = None
	gpu_usd_per_hour: float = 0.0


@dataclass
class ProbeResult:
	probe: Probe
	env: Env
	summary: dict
	# Kernel table from the profiled window, checked; also set on probe.kernels
	kernels: KernelTable | None
	# Model shape read from the running model (recorder/arch.py), checked; untrusted.with_shape()
	# folds it into the script map's ModelInfo when the record is assembled
	arch: dict | None
	trace_file: Path | None
	log_tail: str
	workdir: Path
	kernels_error: str | None = None  # why a profiled probe has no table (shown, never fatal)
	gpu_unresolved: str | None = None  # device name B's table didn't know, for B to add


# ---------------------------------------------------------------- submission checks


def validate_files(files: dict[str, str], *, entrypoint: str) -> None:
	if entrypoint not in files:
		raise InvalidSubmission(f"the entrypoint {entrypoint!r} is not among the submitted files")
	if len(files) > MAX_FILES:
		raise InvalidSubmission(f"{len(files)} files submitted; the limit is {MAX_FILES} files")
	total = 0
	for rel, text in files.items():
		p = PurePosixPath(rel)
		if not rel or not _PATH_OK.match(rel) or p.is_absolute() or ".." in p.parts or rel.startswith("."):
			raise InvalidSubmission(f"file path {rel!r} is not a plain relative path")
		size = len(text.encode("utf-8"))
		if size > MAX_FILE_BYTES:
			raise InvalidSubmission(
				f"{rel} is {size / 2**20:.1f} MB; files are capped at {MAX_FILE_BYTES // 2**20} MB"
			)
		total += size
	if total > MAX_TOTAL_BYTES:
		raise InvalidSubmission(
			f"submission is {total / 2**20:.1f} MB; the cap is {MAX_TOTAL_BYTES // 2**20} MB"
		)


def _write_files(root: Path, files: dict[str, str]) -> None:
	base = root.resolve()
	for rel, text in files.items():
		p = (base / rel).resolve()
		if not p.is_relative_to(base):
			raise InvalidSubmission(f"refusing to write outside the workdir: {rel}")
		p.parent.mkdir(parents=True, exist_ok=True)
		p.write_text(text)


# ---------------------------------------------------------------- isolation


def scrubbed_env(home: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
	env = {k: v for k, v in os.environ.items() if k in SAFE_ENV}
	env.update(
		{
			"HOME": str(home),
			"TMPDIR": str(home / "tmp"),
			"XDG_CACHE_HOME": str(home / ".cache"),
			"PYTHONUNBUFFERED": "1",
			"PYTHONDONTWRITEBYTECODE": "1",
			"TOKENIZERS_PARALLELISM": "false",
			"WANDB_MODE": "disabled",
			"HF_HUB_DISABLE_TELEMETRY": "1",
			"DO_NOT_TRACK": "1",
		}
	)
	if extra:
		env.update(extra)
	return env


def _no_dumps() -> None:
	"""Make this process non-dumpable: other processes of the same uid can't read its /proc/<pid>/environ."""
	if sys.platform.startswith("linux"):
		with contextlib.suppress(Exception):
			ctypes.CDLL(None, use_errno=True).prctl(4, 0, 0, 0, 0)  # PR_SET_DUMPABLE = 4


def _become_subreaper() -> bool:
	"""Orphans of anything this process starts reparent to it, not to init, so they can be found and killed.

	Without this, a probe that detaches a process with setsid escapes the process-group kill
	when the worker isn't root (no per-probe uid to sweep).
	"""
	if not sys.platform.startswith("linux"):
		return False
	try:
		return ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0  # PR_SET_CHILD_SUBREAPER = 36
	except Exception:
		return False


def _parent_map() -> dict[int, int]:
	"""pid -> ppid for every visible process."""
	out = {}
	for d in os.listdir("/proc"):
		if not d.isdigit():
			continue
		with contextlib.suppress(OSError, ValueError, IndexError), open(f"/proc/{d}/stat") as f:
			# "pid (comm) state ppid ...": comm may hold spaces and parentheses, so split after the last ')'
			out[int(d)] = int(f.read().rsplit(")", 1)[1].split()[1])
	return out


def _child_pids(parent: int) -> set[int]:
	return {pid for pid, ppid in _parent_map().items() if ppid == parent}


def _kill_descendants(root: int, keep: set[int]) -> None:
	"""SIGKILL every descendant of root except the subtrees under keep, then reap our direct children."""
	for _ in range(5):
		parents = _parent_map()
		victims, frontier = set(), [pid for pid, ppid in parents.items() if ppid == root and pid not in keep]
		while frontier:
			pid = frontier.pop()
			if pid in victims:
				continue
			victims.add(pid)
			frontier.extend(c for c, pp in parents.items() if pp == pid)
		if not victims:
			return
		for pid in victims:
			with contextlib.suppress(ProcessLookupError, PermissionError):
				os.kill(pid, signal.SIGKILL)
		time.sleep(0.05)
		for pid in victims:
			with contextlib.suppress(ChildProcessError, OSError):
				os.waitpid(
					pid, os.WNOHANG
				)  # only our own (reparented) children; others raise and are skipped


def _probe_uid() -> int | None:
	if os.geteuid() != 0:
		return None
	return 61_000 + (os.getpid() * 7 + next(_UIDS)) % 4_000


def _extra_groups() -> list[int]:
	"""GPU device groups (video, render) so an unprivileged uid can still open ROCm devices."""
	import grp

	out = []
	for name in ("video", "render"):
		with contextlib.suppress(KeyError):
			out.append(grp.getgrnam(name).gr_gid)
	return out


def _chown_tree(path: Path, uid: int) -> None:
	for dirpath, _dirnames, filenames in os.walk(path):
		os.chown(dirpath, uid, uid)
		for n in filenames:
			os.chown(os.path.join(dirpath, n), uid, uid)


def _kill_uid(uid: int) -> None:
	for _ in range(3):
		victims = []
		for d in os.listdir("/proc"):
			if not d.isdigit():
				continue
			with contextlib.suppress(OSError), open(f"/proc/{d}/status") as f:
				for line in f:
					if line.startswith("Uid:") and int(line.split()[1]) == uid:
						victims.append(int(d))
						break
		if not victims:
			return
		for pid in victims:
			with contextlib.suppress(ProcessLookupError, PermissionError):
				os.kill(pid, signal.SIGKILL)
		time.sleep(0.05)


class _CappedLog:
	"""Drain the child's output: keep the first LOG_CAP_BYTES on disk and the last bytes in memory."""

	def __init__(self, pipe, path: Path):
		self.tail = collections.deque(maxlen=64)
		self.written = 0
		self._f = path.open("wb")
		self._t = threading.Thread(target=self._run, args=(pipe,), daemon=True)
		self._t.start()

	def _run(self, pipe):
		for chunk in iter(lambda: pipe.read1(65536) if hasattr(pipe, "read1") else pipe.read(65536), b""):
			self.tail.append(chunk)
			room = LOG_CAP_BYTES - self.written
			if room > 0:
				self._f.write(chunk[:room])
				self.written += min(room, len(chunk))
				if self.written >= LOG_CAP_BYTES:
					self._f.write(b"\n[preflight: log truncated]\n")
		self._f.close()

	def text(self) -> str:
		self._t.join(timeout=5)
		return b"".join(self.tail)[-LOG_TAIL_BYTES:].decode("utf-8", errors="replace")


# ---------------------------------------------------------------- result parsing


def parse_result(raw: str) -> tuple[str, str | None, str | None, dict, dict]:
	"""(status, error_code, error, summary, env). The child shares a process with user code,
	so its output is untrusted: anything malformed becomes a recorder error, never a crash."""
	try:
		res = json.loads(raw)
		status, code, error = res["status"], res.get("error_code"), res.get("error")
		summary, env = res["summary"], res.get("env") or {}
		if status not in VALID_STATUS or not isinstance(summary, dict) or not isinstance(env, dict):
			raise ValueError(f"bad status or shape: {status!r}")
		mfu = summary.get("mfu")
		if mfu is not None and not (0 <= float(mfu) <= 1.0):
			raise ValueError(f"mfu {mfu} is out of range")
		if summary.get("step_time"):
			StepTime(**summary["step_time"])
		return status, code, error, summary, env
	except (ValueError, KeyError, TypeError, ValidationError) as exc:
		return (
			"error",
			"recorder",
			f"The probe's measurements could not be read ({type(exc).__name__})",
			{},
			{},
		)


def _safe_env(raw) -> Env:
	"""The child's env report, or a bare CPU env when it is malformed (it shares a process with user code)."""
	if raw:
		try:
			return _env_model(raw)
		except (ValidationError, TypeError, ValueError, AttributeError):
			log.warning("probe env report could not be read")
	return Env(backend="cpu", provider="local")


def _env_model(raw: dict) -> Env:
	gpu = raw.get("gpu")
	return Env(
		backend=raw.get("backend", "cpu"),
		provider=os.environ.get("PREFLIGHT_PROVIDER", "local"),
		gpu=GPU(**{k: v for k, v in gpu.items() if v is not None}) if gpu else None,
		libraries=raw.get("libraries", {}),
		python=raw.get("python"),
	)


def _opt(model, data):
	return model(**data) if data else None


def _rl(raw: dict | None) -> RLSignals | None:
	if not raw:
		return None
	raw = dict(raw)
	if raw.get("response_length"):
		raw["response_length"] = LengthStats(**raw["response_length"])
	return RLSignals(**raw)


def _probe_model(
	spec: ProbeSpec,
	status: str,
	code: str | None,
	error: str | None,
	s: dict,
	started: datetime,
	duration: float,
) -> Probe:
	st = s.get("step_time")
	mem = s.get("memory") or {}
	by_term = {
		k: int(v) for k, v in mem.items() if k in ("weights", "gradients", "optimizer", "logits") and v
	}
	if mem.get("peak"):
		by_term["activations"] = max(0, int(mem["peak"]) - sum(by_term.values()))
	series = s.get("series") or {}
	thr = s.get("throughput") or {}
	return Probe(
		id=spec.probe_id,
		knobs=spec.knobs,
		status=status,
		steps=int(s.get("steps", 0)),
		started_at=started,
		duration_s=round(duration, 3),
		error=error,
		error_code=code,
		memory=Memory(
			peak_allocated_bytes=mem.get("peak"),
			peak_reserved_bytes=mem.get("reserved"),
			by_term=by_term,
			peak_step=mem.get("peak_step"),
		)
		if mem
		else None,
		step_time=StepTime(**st) if st else None,
		throughput=Throughput(
			tokens_per_s=thr.get("tokens_per_s"),
			samples_per_s=thr.get("samples_per_s"),
			flops_per_step=s.get("flops_per_step"),
			mfu=s.get("mfu"),
		)
		if st
		else None,
		stability=Stability(
			series=series,
			nan_count=int(s.get("nan_count", 0)),
			loss_scale_events=int(s.get("loss_scale_events", 0)),
		)
		if series
		else None,
		data=DataStats(
			padding_fraction=s.get("padding_fraction"),
			tokens_per_sample_mean=s.get("tokens_per_sample_mean"),
			loader_wait_fraction=s.get("loader_wait_fraction"),
			num_workers=s.get("num_workers"),
		)
		if s.get("steps")
		else None,
		rl=_rl(s.get("rl")),
		power=_opt(Power, s.get("power")),
		eval=_opt(EvalStats, s.get("eval")),
		checkpoint=_opt(CheckpointStats, s.get("checkpoint")),
		generation=_opt(GenerationStats, s.get("generation")),
		extras={k: float(s[k]) for k in EXTRAS_KEYS if s.get(k) is not None},
		cost=ProbeCost(gpu_seconds=round(duration, 3), usd=round(duration * spec.gpu_usd_per_hour / 3600, 6)),
	)


# ---------------------------------------------------------------- run


def run_probe(spec: ProbeSpec, *, should_stop: Callable[[], bool] | None = None) -> ProbeResult:
	validate_files(spec.files, entrypoint=spec.entrypoint)
	# where the caller keeps artifacts (log, trace); wiped of any earlier attempt first
	root = Path(spec.out_dir) if spec.out_dir else Path(tempfile.mkdtemp(prefix="preflight-probe-"))
	root.mkdir(parents=True, exist_ok=True)
	for name in ("result.json", "progress.json", "trace.json", "log"):
		(root / f"{spec.probe_id}.{name}").unlink(missing_ok=True)
	# the child runs in a fresh sandbox dir every time, so nothing from an earlier attempt is in reach
	sandbox = Path(tempfile.mkdtemp(prefix="pf-", dir=os.environ.get("PREFLIGHT_SANDBOX_DIR", "/tmp")))
	work = sandbox / "work"
	box = sandbox / "io"  # the only other place the child may write
	work.mkdir()
	box.mkdir()
	home = box / "home"
	(home / "tmp").mkdir(parents=True)
	_write_files(work, spec.files)
	out, progress, trace = box / "result.json", box / "progress.json", box / "trace.json"
	spec_path = box / "spec.json"
	spec_path.write_text(
		json.dumps(
			{
				"workdir": str(work),
				"entrypoint": spec.entrypoint,
				"args": spec.args,
				"warmup": spec.warmup,
				"timed": spec.timed,
				"profile": spec.profile,
				"eval_batches": spec.eval_batches,
				"trace": str(trace),
				"progress": str(progress),
				"out": str(out),
				"seed": spec.seed,
			}
		)
	)
	uid = _probe_uid()
	if uid is not None:
		os.chmod(sandbox, 0o711)  # the probe uid may pass through to its own dirs, not list the parent
		_chown_tree(work, uid)
		_chown_tree(box, uid)
	_no_dumps()
	subreaper = _become_subreaper()
	before = _child_pids(os.getpid()) if subreaper else set()
	pkg_root = str(Path(__file__).resolve().parents[2])
	env = scrubbed_env(home, {"PYTHONPATH": pkg_root + os.pathsep + os.environ.get("PYTHONPATH", "")})
	popen_kw = {"user": uid, "group": uid, "extra_groups": _extra_groups()} if uid is not None else {}

	started = datetime.now(UTC).replace(microsecond=0)
	t0 = time.monotonic()
	_PROBE_LOCK.acquire()  # one probe at a time per process: the descendant sweep below must not touch another probe
	try:
		proc = subprocess.Popen(
			[sys.executable, "-m", "preflight.recorder.run", str(spec_path)],
			cwd=work,
			env=env,
			stdin=subprocess.DEVNULL,
			stdout=subprocess.PIPE,
			stderr=subprocess.STDOUT,
			start_new_session=True,
			**popen_kw,
		)
		log = _CappedLog(proc.stdout, root / f"{spec.probe_id}.log")
	except BaseException:
		_PROBE_LOCK.release()
		raise
	timed_out = cancelled = False
	while True:
		try:
			proc.wait(timeout=0.5)
			break
		except subprocess.TimeoutExpired:
			if time.monotonic() - t0 > spec.timeout_s:
				timed_out = True
			elif should_stop is not None and should_stop():
				cancelled = True
			else:
				continue
			with contextlib.suppress(ProcessLookupError):
				os.killpg(proc.pid, signal.SIGKILL)
			proc.wait()
			break
	# nothing the probe started survives it: its process group, every descendant that detached
	# (reparented to us as subreaper), then anything else running as its uid
	try:
		with contextlib.suppress(ProcessLookupError, PermissionError):
			os.killpg(proc.pid, signal.SIGKILL)
		if subreaper:
			_kill_descendants(os.getpid(), keep=before)
		if uid is not None:
			_kill_uid(uid)
	finally:
		_PROBE_LOCK.release()
	duration = time.monotonic() - t0
	log_tail = log.text()

	prog = {"stage": "starting", "steps": 0}
	with contextlib.suppress(OSError, ValueError):
		prog = json.loads(progress.read_text())
	if cancelled:
		status, code = "cancelled", "cancelled"
		error = "Cancelled while running"
		summary, raw_env = {"steps": int(prog.get("steps", 0))}, {}
	elif timed_out or not out.exists():
		steps = int(prog.get("steps", 0))
		c = classify_exit(
			returncode=None if timed_out else proc.returncode,
			timed_out=timed_out,
			steps=steps,
			stage=str(prog.get("stage", "starting"))[:80],
			timeout_s=spec.timeout_s,
		)
		status, code, error = c.status, c.code, c.message
		if prog.get("in_eval"):
			error += " (during the eval pass)"
		summary, raw_env = {"steps": steps}, {}
	else:
		status, code, error, summary, raw_env = parse_result(out.read_text(errors="replace")[:5_000_000])
	try:
		probe = _probe_model(spec, status, code, error, summary, started, duration)
	except (ValidationError, TypeError, ValueError) as exc:
		probe = _probe_model(
			spec,
			"error",
			"recorder",
			f"The probe's measurements could not be read ({type(exc).__name__})",
			{"steps": int(summary.get("steps", 0)) if isinstance(summary.get("steps"), int) else 0},
			started,
			duration,
		)
	final = None
	if trace.exists() and trace.stat().st_size < 512 * 1024 * 1024:
		final = root / f"{spec.probe_id}.trace.json"
		shutil.move(str(trace), final)
	# uploaded code is not kept on the box after the probe (24 h retention lives in the object store)
	shutil.rmtree(sandbox, ignore_errors=True)
	kernels, kernels_error = untrusted.kernel_table(summary.get("kernels"))
	if kernels is not None and probe.error_code != "recorder":  # not on a probe whose numbers were unreadable
		probe = probe.model_copy(update={"kernels": kernels})
	unresolved = raw_env.get("gpu_unresolved") if isinstance(raw_env, dict) else None
	if isinstance(unresolved, str):
		log.warning("GPU %r is not in the spec table; kept as reported", unresolved[:80])
	return ProbeResult(
		probe=probe,
		env=_safe_env(raw_env),
		summary=summary,
		kernels=kernels,
		kernels_error=kernels_error,
		arch=untrusted.model_shape(summary.get("arch")),
		gpu_unresolved=unresolved[:80] if isinstance(unresolved, str) else None,
		trace_file=final,
		log_tail=log_tail,
		workdir=root,
	)
