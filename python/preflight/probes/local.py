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
import math
import os
import re
import shutil
import signal
import stat
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
	"PREFLIGHT_PIP_CACHE",
	"PREFLIGHT_DEPS_DIR",
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
	# the real run's length, for time.total (A -> B, Oct 28): batches per epoch of the training
	# loader, and from an HF Trainer its epochs and planned optimizer steps
	"train_loader_batches",
	"epochs",
	"planned_steps",
	"train_dataset_len",  # rows the training loader draws per epoch (examples, pairs, images, input nodes)
	# the training set's row lengths in tokens, as the script tokenised them (B, Oct 30), over
	# dataset_rows_scanned rows (the first 200,000 of dataset_rows_total for very large sets)
	"dataset_max_tokens",
	"dataset_p999_tokens",
	"dataset_p50_tokens",  # step and total time follow the median and mean; memory the max (B, Nov 1)
	"dataset_mean_tokens",
	"dataset_rows_scanned",
	"dataset_rows_total",
	# vision: the batch's images (mean over the first training batches)
	"images_per_batch",
	"image_height",
	"image_width",
	"channels_last",
	# graphs: a sampled batch's size, per hop (PyG NeighborLoader), and the largest batch seen
	"graph_nodes_per_batch",
	"graph_edges_per_batch",
	"graph_seed_nodes",
	"graph_nodes_hop0",
	"graph_nodes_hop1",
	"graph_nodes_hop2",
	"graph_nodes_hop3",
	"graph_nodes_max",
	# retrieval: each stage the probe reached (sentence-transformers encode, faiss add, train and
	# search): items (sentences or vectors), seconds, and items a second
	*(
		f"{k}_{m}"
		for k in ("encode", "index_add", "index_train", "search")
		for m in ("items", "seconds", "items_per_s")
	),
	# torchrun (nproc > 1): ranks on the node, how many wrote a result, the largest rank peak and
	# the slowest rank's median step
	# hygiene: generations by different weights on the same inputs, and how many came out identical
	"outputs_compared",
	"outputs_identical",
	"generation_ending_known",  # the sequences generation.truncation_rate is over
	"rl_zero_spread_groups",  # GRPO groups whose summed rewards were all equal, of rl_groups
	"rl_groups",
	"rl_reward_functions",
	"rl_entropy_estimated",  # 1: rl.entropy is the mean -log p of the sampled completion tokens (TRL logged none; not TRL 0.14)
	# hygiene: what the script's own checkpoints hold, for a resume that matches
	"checkpoint_saves_seen",
	"checkpoint_has_optimizer",
	"checkpoint_has_scheduler",
	"checkpoint_has_rng",
	"checkpoint_has_step",
	"world_size",
	"ranks_reported",
	"rank_peak_allocated_max_bytes",
	"rank_step_p50_max_s",
)


class InvalidSubmission(ValueError):
	"""The submitted files can't be probed as given. Not retryable.

	code: the job's error code (`submission.<why>`), shown with t('error.<code>') and its hint.
	"""

	def __init__(self, message: str = "", *, code: str = "submission"):
		super().__init__(message)
		self.code = code


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
	seed_force: bool = False  # override the script's own seeding (verify's seed-to-seed spread)
	alternate_sync: bool = False  # ground truth only: alternate clean and synced steps (recorder/overhead.py)
	install_requirements: bool = False  # pip-install requirements.txt in the probe (hosted probes only)
	# GPUs the hosted runner may use instead when this one won't schedule (none when pinned or ruled out)
	gpu_fallback: tuple[str, ...] = ()
	out_dir: Path | None = None
	gpu_usd_per_hour: float = 0.0
	nproc: int = 1  # processes on this node, one per GPU, under torchrun (DDP, FSDP)
	image: str = "probe"  # which sandbox image runs it: "probe", or "serving" (vLLM, its own torch)
	# a serving probe: the prompts one vLLM generate call keeps (0: all of them), and the probe ends
	# once that call is measured; a load test's whole burst would outrun the probe's time limit
	generate_sample: int = 0
	# an eval probe: the generate calls it measures; the probe ends as the next one starts, so what the
	# script does with each batch (decode, score, dump activations) runs and is measured too
	generate_calls: int = 0


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
		if not isinstance(text, str):
			raise InvalidSubmission(f"{rel} is not text", code="submission.invalid")
		try:
			size = len(text.encode("utf-8"))
		except UnicodeEncodeError as exc:  # an unpaired surrogate ('\ud800'), which no file can hold
			line = text.count("\n", 0, exc.start) + 1
			raise InvalidSubmission(
				f"{rel} has text that isn't valid Unicode (an unpaired surrogate, line {line}); "
				"save it as UTF-8 and check in again",
				code="submission.invalid",
			) from None
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


def uid_can_run_this_python() -> bool:
	"""Whether a fresh unprivileged uid can run this interpreter with its packages. Not when it sits
	somewhere that uid can't enter (a venv in a Docker image's /root, a 0700 home): the uid would
	run the base Python without the venv's torch, so probes there run as the invoking user."""
	return _others_can_reach(Path(sys.executable).parent)


def _probe_uid() -> int | None:
	"""A fresh unprivileged uid per probe when running as root. PREFLIGHT_PROBE_UID pins one uid for
	every probe in this process: in a per-job sandbox the sandbox is the isolation boundary, and
	one uid lets a job's probes share the Hugging Face cache (locks, downloads) between points."""
	if os.geteuid() != 0:
		return None
	pinned = os.environ.get("PREFLIGHT_PROBE_UID")
	if pinned and pinned.isdigit() and int(pinned) >= 1000:
		return int(pinned)
	if not uid_can_run_this_python():
		log.warning(
			"running probes as this user: a fresh uid can't reach %s and its packages", sys.executable
		)
		return None
	return 61_000 + (os.getpid() * 7 + next(_UIDS)) % 4_000


PROBE_USER = "preflight-probe"  # the images' /etc/passwd name for PREFLIGHT_PROBE_UID (runner/modal_app.py)


def _probe_user_env(uid: int | None) -> dict[str, str]:
	"""A name for the probe's uid: getpass.getuser() (torch.compile's cache, torchtitan, many loggers)
	reads USER/LOGNAME before /etc/passwd, where a fresh uid has no entry, and raised KeyError
	without them (torchtitan on an A10, Nov 1)."""
	return {} if uid is None else {"USER": PROBE_USER, "LOGNAME": PROBE_USER}


def _launch_as(uid: int | None, home: Path) -> tuple[str, dict[str, str], dict]:
	"""How a probe starts: the interpreter, the scrubbed env (with the uid's user name) and the
	Popen uid and groups. One place, so the startup check launches exactly as a probe does."""
	pkg_root = str(Path(__file__).resolve().parents[2])
	python = sys.executable
	if uid is not None:
		pkg_root = str(_package_for_probe(Path(pkg_root)))
		python = _python_for_probe()
	env = scrubbed_env(home, {"PYTHONPATH": pkg_root + os.pathsep + os.environ.get("PYTHONPATH", ""),
		**_probe_user_env(uid)})  # fmt: skip
	popen_kw = {"user": uid, "group": uid, "extra_groups": _extra_groups()} if uid is not None else {}
	return python, env, popen_kw


# what the startup check runs as the probe uid in each image (Chiya, Nov 1): the user and group
# lookups visitor code makes, and the imports that read them, so an image bug costs cents on a CPU
PROBE_USER_SCRIPT = """
import getpass, grp, json, os, pwd, sys
def name(lookup, i):  # None when the uid has no entry (a fresh uid on a plain host, not in an image)
	try:
		return lookup(i)[0]
	except KeyError:
		return None


out = {"uid": os.getuid(), "gid": os.getgid(), "user": getpass.getuser(), "groups": sorted(os.getgroups()),
	"passwd": name(pwd.getpwuid, os.getuid()), "group": name(grp.getgrgid, os.getgid())}
for mod in sys.argv[1:]:
	if mod == "+compile":  # a tiny torch.compile on the CPU: inductor's C++ build and its cache dir
		import torch

		f = torch.compile(lambda x: (x * 2).sin().sum())
		out["compiled"] = float(f(torch.ones(8)))
		continue
	if mod == "+gpu":  # the GPU as the probe uid: device nodes (ROCm's /dev/kfd), a matmul, a GPU compile
		import torch

		if os.path.exists("/dev/kfd"):
			os.close(os.open("/dev/kfd", os.O_RDWR))
			out["kfd"] = True
		x = torch.ones(64, 64, device="cuda")
		out["gpu"] = {"name": torch.cuda.get_device_name(0), "hip": torch.version.hip,
			"cuda": torch.version.cuda, "matmul": float((x @ x).sum())}
		g = torch.compile(lambda x: (x * 2).sin().sum())
		out["gpu_compiled"] = float(g(x))
		continue
	if mod.startswith("?"):  # optional: the probe runs without it, but records less (psutil: no RSS)
		try:
			__import__(mod[1:])
		except ImportError:
			out.setdefault("missing", []).append(mod[1:])
		continue
	__import__(mod)
	out.setdefault("imported", []).append(mod)
print("PROBE_USER " + json.dumps(out))
"""


def probe_user_check(imports: tuple[str, ...] = ("+compile",), timeout_s: int = 420) -> dict:
	"""Run PROBE_USER_SCRIPT as a probe would run (same uid, env and interpreter) and report what it
	saw: {"exit", "uid", "user", "passwd", "group", "imported", "compiled"}, or the stderr tail on
	failure. "+compile" in imports runs a tiny torch.compile instead of an import."""
	uid = _probe_uid()
	box = Path(tempfile.mkdtemp(prefix="preflight-usercheck-"))
	home = box / "home"
	(home / "tmp").mkdir(parents=True)
	try:
		if uid is not None:
			os.chmod(box, 0o711)
			_chown_tree(home, uid)
		python, env, popen_kw = _launch_as(uid, home)
		p = subprocess.run([python, "-c", PROBE_USER_SCRIPT, *imports], cwd=home, env=env, stdin=subprocess.DEVNULL, capture_output=True,
			text=True, timeout=timeout_s, **popen_kw)  # fmt: skip
		out: dict = {"exit": p.returncode, "probe_uid": uid}
		line = next((ln for ln in p.stdout.splitlines() if ln.startswith("PROBE_USER ")), None)
		if p.returncode == 0 and line:
			out.update(json.loads(line.removeprefix("PROBE_USER ")))
		else:
			out["stderr"] = p.stderr[-1500:]
		return out
	finally:
		shutil.rmtree(box, ignore_errors=True)


def _extra_groups() -> list[int]:
	"""GPU device groups (video, render) so an unprivileged uid can still open ROCm devices."""
	import grp

	out = []
	for name in ("video", "render"):
		with contextlib.suppress(KeyError):
			out.append(grp.getgrnam(name).gr_gid)
	return out


def _others_can_reach(d: Path) -> bool:
	"""Whether a uid with no special rights can get to directory d (search permission all the way)."""
	for x in (d, *d.parents):
		try:
			if not x.stat().st_mode & 0o001:
				return False
		except OSError:
			return False
	return True


_PKG_COPY: Path | None = None


def _package_for_probe(pkg_root: Path) -> Path:
	"""Our package where the probe uid can import it. Usually where it is; when it sits under a
	directory others can't enter (/root is 0700 in Debian images, where the AMD checkout lives),
	a read-only copy in a fresh world-readable temp dir, made once per process. Nothing on the host
	has its permissions changed."""
	global _PKG_COPY
	if _others_can_reach(pkg_root):
		return pkg_root
	if _PKG_COPY is not None and (_PKG_COPY / "preflight").is_dir():
		return _PKG_COPY
	import atexit

	dst = Path(tempfile.mkdtemp(prefix="preflight-pkg-"))
	shutil.copytree(pkg_root / "preflight", dst / "preflight", symlinks=False,
		ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))  # fmt: skip
	for dirpath, _dirs, files in os.walk(dst):  # root made every entry just now: nothing to follow
		os.chmod(dirpath, 0o755)
		for f in files:
			os.chmod(os.path.join(dirpath, f), 0o644)
	atexit.register(shutil.rmtree, dst, True)
	log.info("the probe user can't reach %s; probes import a copy from %s", pkg_root, dst)
	_PKG_COPY = dst
	return dst


def _python_for_probe() -> str:
	"""The interpreter the probe uid can run: this one, or the base it links to when this one
	sits somewhere the probe can't enter (a venv under /root: its own packages drop out, the
	system's, ROCm torch included, stay)."""
	exe = sys.executable
	if _others_can_reach(Path(exe).parent):
		return exe
	real = os.path.realpath(exe)
	if _others_can_reach(Path(real).parent):
		log.warning("the probe user can't reach %s; probes run %s instead", exe, real)
		return real
	return exe


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
	except (ValueError, KeyError, TypeError, ArithmeticError, ValidationError) as exc:
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
		extras=_extras(s),
		step_log=untrusted.step_log(s.get("step_log")),
		cost=ProbeCost(gpu_seconds=round(duration, 3), usd=round(duration * spec.gpu_usd_per_hour / 3600, 6)),
	)


def _extras(summary: dict) -> dict[str, float]:
	"""The counts B reads by name, from the probe's own (untrusted) summary: a value that isn't a
	finite, non-negative number is left out, never stored as NaN or allowed to crash the parent."""
	out = {}
	for k in EXTRAS_KEYS:
		v = summary.get(k)
		if isinstance(v, bool) or not isinstance(v, int | float):
			continue
		try:
			f = float(v)
		except (OverflowError, ValueError):
			continue
		if math.isfinite(f) and f >= 0:
			out[k] = f
	return out


# ---------------------------------------------------------------- run


def _read_untrusted(path: Path, cap: int = 5_000_000) -> str:
	"""A file the probe's uid could write: never through a link, only a regular file, at most cap
	bytes (a planted link to /dev/zero or a FIFO can't stall or flood the parent)."""
	fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
	try:
		if not stat.S_ISREG(os.fstat(fd).st_mode):
			raise OSError(f"{path.name} is not a regular file")
		return os.read(fd, cap).decode(errors="replace")
	finally:
		os.close(fd)


def _rank_extras(out: Path, nproc: int, summary: dict) -> dict:
	"""What the other ranks' results add to rank 0's: the world size, the largest peak and the
	slowest median step of any rank (the rank that OOMs or straggles sets the run)."""

	def peak_p50(s):
		return (s.get("memory") or {}).get("peak"), (s.get("step_time") or {}).get("p50_s")

	vals, seen = [peak_p50(summary)], 1
	for r in range(1, nproc):
		with contextlib.suppress(OSError, ValueError):
			_, _, _, s, _ = parse_result(_read_untrusted(Path(f"{out}.rank{r}")))
			vals.append(peak_p50(s))
			seen += 1

	def num(x):
		return isinstance(x, int | float) and not isinstance(x, bool)

	res = {"world_size": nproc, "ranks_reported": seen}
	if peaks := [v[0] for v in vals if num(v[0])]:
		res["rank_peak_allocated_max_bytes"] = max(peaks)
	if times := [v[1] for v in vals if num(v[1])]:
		res["rank_step_p50_max_s"] = max(times)
	return res


# A sandbox entry (probes/remote.py) sets this, prints its result, then calls finish_cleanup():
# a probe that wrote many GB (an activation dump) is not lost to a slow delete before the result.
DEFER_CLEANUP = False
_DISCARDED: list[Path] = []


def _discard(path: Path) -> None:
	if not DEFER_CLEANUP:
		shutil.rmtree(path, ignore_errors=True)
		return
	gone = path.with_name(path.name + ".discard")
	try:
		path.rename(gone)  # out of the way at once; deleted after the result is out
	except OSError:
		gone = path
	_DISCARDED.append(gone)


def finish_cleanup(*, background: bool = False) -> None:
	"""Delete what _discard set aside; background=True hands it to a detached `rm -rf`, so the
	sandbox entry can exit at once (the sandbox ends with the job, and the delete with it)."""
	while _DISCARDED:
		path = _DISCARDED.pop()
		if background:
			with contextlib.suppress(OSError):
				subprocess.Popen(["rm", "-rf", "--", str(path)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
					stderr=subprocess.DEVNULL, start_new_session=True)  # fmt: skip
				continue
		shutil.rmtree(path, ignore_errors=True)


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
				"seed_force": spec.seed_force,
				"alternate_sync": spec.alternate_sync,
				"generate_sample": spec.generate_sample,
				"generate_calls": spec.generate_calls,
				"install_requirements": spec.install_requirements,
				"deps": str(box / "deps"),
				"nproc": spec.nproc,
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
	python, env, popen_kw = _launch_as(uid, home)

	started = datetime.now(UTC).replace(microsecond=0)
	t0 = time.monotonic()
	_PROBE_LOCK.acquire()  # one probe at a time per process: the descendant sweep below must not touch another probe
	try:
		launch = [python, "-m", "preflight.recorder.run", str(spec_path)]
		if spec.nproc > 1:
			# one recorder per rank; --standalone: a rendezvous on this node only
			launch = [python, "-m", "torch.distributed.run", "--standalone", f"--nproc_per_node={spec.nproc}",
				"-m", "preflight.recorder.run", str(spec_path)]  # fmt: skip
		proc = subprocess.Popen(
			launch,
			cwd=work,
			env=env,
			stdin=subprocess.DEVNULL,
			stdout=subprocess.PIPE,
			stderr=subprocess.STDOUT,
			start_new_session=True,
			**popen_kw,
		)
		plog = _CappedLog(proc.stdout, root / f"{spec.probe_id}.log")
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
	log_tail = plog.text()

	prog = {"stage": "starting", "steps": 0}
	with contextlib.suppress(OSError, ValueError):
		prog = json.loads(_read_untrusted(progress, 64_000))
	raw = None
	with contextlib.suppress(OSError):
		raw = _read_untrusted(out)
	if cancelled:
		status, code = "cancelled", "cancelled"
		error = "Cancelled while running"
		summary, raw_env = {"steps": int(prog.get("steps", 0))}, {}
	elif raw is None:
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
		# a result written before the time limit stands (a rank hung in teardown after writing it)
		if timed_out:
			log.warning("probe %s hit its time limit after writing its result", spec.probe_id)
		status, code, error, summary, raw_env = parse_result(raw)
		if spec.nproc > 1:
			summary = {**summary, **_rank_extras(out, spec.nproc, summary)}
	try:
		probe = _probe_model(spec, status, code, error, summary, started, duration)
	except (ValidationError, TypeError, ValueError, ArithmeticError) as exc:
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
	_discard(sandbox)
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
