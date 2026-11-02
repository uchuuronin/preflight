"""Install a submission's requirements.txt into the probe before its script runs.

Runs in the probe child (the visitor's uid, the sandbox's egress allowlist: PyPI
and Hugging Face only), so it can do nothing the script itself couldn't. The
probe image's pinned stack (torch, transformers, peft, trl, ...) always wins:
a line naming a package the image already has is skipped and logged, and every
installed package is passed to pip as a constraint so nothing the script needs
can upgrade it underneath the recorder. Only plain `name[extras] <specifiers>`
lines from PyPI are installed: options, URLs, VCS and local paths are skipped
with a note, and only wheels are accepted (no build step, no compiler wait).

pip runs with --prefix, not --target: --target makes pip ignore everything
already installed, so a requirement that depends on torch would pull torch and
its CUDA wheels in again (and an image torch with a local label, +cu121 or
+rocm6.2, couldn't satisfy the constraint at all). With --prefix pip sees the
image's packages and installs only what is missing. The constraints carry
public versions only (`torch==2.5.1` matches the image's `2.5.1+cu121`).

A job's probes share one install when PREFLIGHT_DEPS_DIR is set (the CLI and
the Modal sandbox set it): the first probe installs into a directory keyed by
the requirements and the image's packages, later ones reuse it.
"""

import contextlib
import hashlib
import importlib.metadata as md
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

MAX_REQUIREMENTS = 50  # guess: well above the zoo's and the notebooks we've seen; the bug zoo checks it
INSTALL_TIMEOUT_S = 240  # guess: inside the probe's own 6 min cap; zoo runs time a real install (Oct 28-29)
# a repo's own declared stack (probes.repo_install) is bigger: LitGPT's (lightning, litdata, jsonargparse)
# took over 240 s on Modal (Nov 2, a-after-2-gt-2; its resolve alone is about 40 s). guess: still inside
# the probe's 600 s limit with time to run
REPO_INSTALL_TIMEOUT_S = 420
_NAME = r"[A-Za-z0-9](?:[A-Za-z0-9._-]{0,99})"
_SPEC = r"(?:===|==|~=|!=|<=|>=|<|>)\s*[A-Za-z0-9.*+!_-]{1,64}"
_LINE = re.compile(
	rf"(?P<name>{_NAME})(?P<extras>\[[A-Za-z0-9,._ -]{{1,200}}\])?\s*(?P<spec>{_SPEC}(?:\s*,\s*{_SPEC}){{0,8}})?"
	r"\s*(?P<marker>;[^#]{0,200})?"
)


def _norm(name: str) -> str:
	return re.sub(r"[-_.]+", "-", name).lower()


def installed() -> dict[str, str]:
	out = {}
	for d in md.distributions():
		name = d.metadata["Name"] if d.metadata else None
		if name:
			out.setdefault(_norm(name), d.version)
	return out


def plan(text: str, have: dict[str, str]) -> tuple[list[str], list[str]]:
	"""(lines to install, notes on the lines skipped and why)."""
	keep, notes = [], []
	for raw in text.splitlines():
		line = raw.split(" #", 1)[0].strip()
		if not line or line.startswith("#"):
			continue
		m = _LINE.fullmatch(line)
		if m is None:
			notes.append(f"skipped {line[:80]!r}: only plain PyPI requirements are installed")
			continue
		name = _norm(m["name"])
		if name in have:
			notes.append(f"kept the image's {m['name']} {have[name]} (asked for {line[:80]!r})")
			continue
		keep.append(line)
	if len(keep) > MAX_REQUIREMENTS:
		notes.append(f"installed the first {MAX_REQUIREMENTS} of {len(keep)} requirements")
		keep = keep[:MAX_REQUIREMENTS]
	return keep, notes


# ---------------------------------------------------------------- probes.repo_install (off)

# A module a script imports, and the PyPI distribution that provides it, for the imports the image
# lacks: only these names are installed from an import, never a name guessed from the module (a
# guessed name can be someone else's package). Unseen runs, Nov 2: torchvision, torch_geometric,
# sentence_transformers and lightning stopped four of nine public scripts before their first step.
KNOWN_MODULES = {
	"torchvision": "torchvision", "torchaudio": "torchaudio", "torch_geometric": "torch-geometric",
	"sentence_transformers": "sentence-transformers", "lightning": "lightning",
	"pytorch_lightning": "pytorch-lightning", "lightning_utilities": "lightning-utilities",
	"timm": "timm", "einops": "einops", "omegaconf": "omegaconf", "hydra": "hydra-core",
	# torchao 0.8 on needs torch 2.6 (torch.int1), which pip can't tell from its metadata
	"torchao": "torchao<0.8",
	"torchdata": "torchdata", "tiktoken": "tiktoken", "sentencepiece": "sentencepiece",
	"tokenizers": "tokenizers", "jsonargparse": "jsonargparse", "litdata": "litdata", "evaluate": "evaluate",
	"ogb": "ogb", "beir": "beir", "faiss": "faiss-cpu", "sklearn": "scikit-learn", "scipy": "scipy",
	"pandas": "pandas", "yaml": "pyyaml", "PIL": "pillow", "cv2": "opencv-python-headless",
	"openai": "openai", "tqdm": "tqdm", "rich": "rich", "jinja2": "jinja2", "requests": "requests",
	"huggingface_hub": "huggingface-hub", "dotenv": "python-dotenv", "blobfile": "blobfile",
	"wandb": "wandb", "tensorboard": "tensorboard", "pytrec_eval": "pytrec-eval-terrier",
	"lm_eval": "lm-eval", "smolagents": "smolagents", "networkx": "networkx", "matplotlib": "matplotlib",
	"latex2sympy2_extended": "latex2sympy2-extended", "math_verify": "math-verify",
}  # fmt: skip
# on PyPI only as source (built against one torch and CUDA): never installable with --only-binary
NO_WHEELS = {"torch_sparse", "torch_scatter", "torch_cluster", "torch_spline_conv", "pyg_lib"}
MAX_SCANNED_FILES = 400  # guess: a repo's import closure from its entrypoint; bounds the scan

# ---------------------------------------------------------------- JobRequest.extra_packages (0.6.3)

MAX_EXTRA_PACKAGES = 10


def known_packages() -> dict[str, str]:
	"""{PyPI name, normalized: the line the probe installs} for every package on KNOWN_MODULES' list:
	the only names a visitor can add as extra packages (C's check-in help; GET /v1/packages)."""
	return {_norm(_LINE.match(line)["name"]): line for line in KNOWN_MODULES.values()}


def package_for_module(module: str | None) -> str | None:
	"""The PyPI name that provides an import the probe couldn't find, when it's on the list; None for a
	submodule (`a.b` missing means `a` is there, at a version without `b`) or a module off the list."""
	if not module or "." in module or module not in KNOWN_MODULES:
		return None
	return _LINE.match(KNOWN_MODULES[module])["name"]


class PackageRefused(ValueError):
	"""Extra packages the probe won't install: `code` is missing_package.not_allowed or .no_wheels."""

	def __init__(self, code: str, names: list[str]):
		self.code, self.names = code, names
		listed = ", ".join(names)
		if code == "missing_package.no_wheels":
			msg = f"{listed}: PyPI has no ready-built version for this machine, so Preflight can't install it"
		elif code == "submission.invalid":
			msg = f"a job installs at most {MAX_EXTRA_PACKAGES} extra packages ({listed}, with the record's own)"
		else:
			msg = f"Preflight installs only packages from its list; {listed} isn't on it"
		super().__init__(msg)


def extra_requirements(names) -> list[str]:
	"""The lines to install for a job's extra packages, in order, deduplicated. A name is a PyPI name
	on the list (or the module it provides: `sklearn` for scikit-learn), without a version: the list's
	own line carries any pin. Raises PackageRefused naming every name refused (no-wheel ones first)."""
	if not names:
		return []
	if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
		raise PackageRefused("missing_package.not_allowed", ["the extra packages (a list of names)"])
	if (
		len(names) > 2 * MAX_EXTRA_PACKAGES
	):  # a record's own and a request's at most; the count is checked below
		raise PackageRefused("submission.invalid", [f"{len(names)} asked for"])
	known = known_packages()
	no_wheels = {_norm(m) for m in NO_WHEELS}
	out, refused, unbuilt = [], [], []
	for raw in names:
		name = raw.strip()
		shown = name[:64] if re.fullmatch(_NAME, name) else repr(name[:64])
		line = None
		if re.fullmatch(_NAME, name):
			line = known.get(_norm(name)) or (KNOWN_MODULES.get(name) if name in KNOWN_MODULES else None)
		if line is None:
			(unbuilt if re.fullmatch(_NAME, name) and _norm(name) in no_wheels else refused).append(shown)
		elif line not in out:
			out.append(line)
	if unbuilt:
		raise PackageRefused("missing_package.no_wheels", unbuilt)
	if refused:
		raise PackageRefused("missing_package.not_allowed", refused)
	if len(out) > MAX_EXTRA_PACKAGES:  # counted once each, `sklearn` and `scikit-learn` as one
		raise PackageRefused("submission.invalid", [f"{len(out)} asked for"])
	return out


def _imports(text: str) -> tuple[set[str], set[tuple[int, str]]]:
	"""(absolute imports as dotted names, relative ones as (level, dotted name)) in a file, outside try
	blocks (an import that's allowed to fail is optional). `from a import b` gives a and a.b, since b
	may be a submodule."""
	import ast

	try:
		tree = ast.parse(text)
	except (SyntaxError, ValueError, RecursionError, MemoryError):
		return set(), set()
	guarded: set[int] = set()
	for node in ast.walk(tree):
		# a try whose handler raises again (smolagents: "Please install 'openai' extra") makes the
		# import required, not optional
		if (
			isinstance(node, ast.Try)
			and node.handlers  # a try with only a finally lets the error through: required
			and not any(isinstance(x, ast.Raise) for h in node.handlers for x in ast.walk(h))
		):
			for n in (*node.body, *node.handlers):
				guarded.update(id(x) for x in ast.walk(n))
	absolute, relative = set(), set()
	for node in ast.walk(tree):
		if id(node) in guarded:
			continue
		if isinstance(node, ast.Import):
			absolute.update(a.name for a in node.names)
		elif isinstance(node, ast.ImportFrom):
			base = node.module or ""
			names = [base, *(f"{base}.{a.name}" if base else a.name for a in node.names if a.name != "*")]
			if node.level == 0:
				absolute.update(n for n in names if n)
			else:
				relative.update((node.level, n) for n in names)
	return absolute, relative


def _top_imports(text: str) -> set[str]:
	"""Top-level names of a file's absolute imports, outside try blocks."""
	return {n.split(".")[0] for n in _imports(text)[0]}


def _roots(workdir: Path) -> list[Path]:
	"""Where the script's own packages are importable from: the checkout, and src/ in a src layout."""
	src = workdir / "src"
	return [workdir, *([src] if src.is_dir() and any(src.glob("*/__init__.py")) else [])]


def _module_files(base: Path, dotted: str) -> list[Path]:
	"""The files importing `dotted` from base runs: each package's __init__.py on the way, and the
	module itself (a .py, or a package's __init__.py)."""
	out, cur = [], base
	for part in [p for p in dotted.split(".") if p]:
		if (cur / part / "__init__.py").is_file():
			cur = cur / part
			out.append(cur / "__init__.py")
		elif (cur / f"{part}.py").is_file():
			out.append(cur / f"{part}.py")
			break
		elif (cur / part).is_dir():  # a namespace package
			cur = cur / part
		else:
			break
	return out


def _local_root(top: str, roots: list[Path]) -> Path | None:
	"""The root a top-level name is the repo's own module or package under, if it is."""
	try:
		for r in roots:
			if (r / top / "__init__.py").is_file() or (r / f"{top}.py").is_file():
				return r
			if (r / top).is_dir() and any((r / top).glob("*.py")):  # a namespace package
				return r
	except OSError:  # a name the filesystem can't hold (over 255 bytes): not the repo's own module
		return None
	return None


def missing_imports(workdir: Path, entrypoint: str) -> set[str]:
	"""The third-party modules that the entrypoint, and the repo's own modules it imports, import and
	this Python can't. Follows the imports module by module (not whole packages, whose other modules
	may import things this script never runs), bounded."""
	import importlib.util

	entry = workdir / entrypoint
	roots = [*({entry.parent} - {workdir}), *_roots(workdir)]  # a script's own folder is on its path too
	todo, seen, need = [entry], set(), set()
	while todo and len(seen) < MAX_SCANNED_FILES:
		f = todo.pop()
		if f in seen or not f.is_file():
			continue
		seen.add(f)
		try:
			text = f.read_text(encoding="utf-8", errors="replace")
		except OSError:
			continue
		absolute, relative = _imports(text)
		for level, dotted in relative:
			base = f.parent
			for _ in range(level - 1):
				base = base.parent
			if base != workdir and workdir not in base.parents:  # never above the checkout
				continue
			try:
				todo.extend(_module_files(base, dotted))
			except OSError:
				continue
		for dotted in absolute:
			top = dotted.split(".")[0]
			root = _local_root(top, roots)
			if root is not None:
				with contextlib.suppress(OSError):
					todo.extend(_module_files(root, dotted))
				continue
			if top in sys.stdlib_module_names or top == "__future__":
				continue
			try:
				found = importlib.util.find_spec(top) is not None
			except (ImportError, ValueError):
				found = False
			if not found:
				need.add(top)
	return need


def declared(workdir: Path) -> list[str]:
	"""The dependencies the repo declares: pyproject.toml's [project] dependencies, else setup.cfg's
	[options] install_requires, else setup.py's install_requires when it's a literal list (or a name
	bound to one)."""
	import ast
	import configparser
	import tomllib

	py = workdir / "pyproject.toml"
	if py.is_file():
		try:
			deps = (
				tomllib.loads(py.read_text(encoding="utf-8", errors="replace"))
				.get("project", {})
				.get("dependencies")
			)
		except (tomllib.TOMLDecodeError, AttributeError, RecursionError, MemoryError, OSError):
			deps = None
		if isinstance(deps, list) and all(isinstance(d, str) for d in deps):
			return deps
	cfg = workdir / "setup.cfg"
	if cfg.is_file():
		cp = configparser.ConfigParser(interpolation=None)
		try:
			cp.read_string(cfg.read_text(encoding="utf-8", errors="replace"))
			raw = cp.get("options", "install_requires", fallback="")
		except (configparser.Error, RecursionError, MemoryError, OSError):
			raw = ""
		lines = [x.strip() for x in raw.splitlines() if x.strip() and not x.strip().startswith("#")]
		if lines:
			return lines
	setup = workdir / "setup.py"
	if setup.is_file():
		try:
			tree = ast.parse(setup.read_text(encoding="utf-8", errors="replace"))
		except (SyntaxError, ValueError, RecursionError, MemoryError, OSError):
			return []
		lists = {t.id: n.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
			for t in n.targets if isinstance(t, ast.Name)}  # fmt: skip
		for n in ast.walk(tree):
			if isinstance(n, ast.keyword) and n.arg == "install_requires":
				v = lists.get(n.value.id) if isinstance(n.value, ast.Name) else n.value
				if isinstance(v, ast.List) and all(
					isinstance(e, ast.Constant) and isinstance(e.value, str) for e in v.elts
				):
					return [e.value for e in v.elts]
	return []


def repo_requirement_sets(
	workdir: Path, entrypoint: str, *, log=print, pyg: bool = False
) -> tuple[list[str], list[str]]:
	"""(what the repo declares, what its code imports that the image lacks, by KNOWN_MODULES only) for a
	repo that ships no requirements.txt (probes.repo_install). Noted, never guessed: a module Preflight
	doesn't know, and one with no wheels on PyPI."""
	declared_lines = list(declared(workdir))
	if declared_lines:
		log(
			f"[preflight] requirements: the repo declares {len(declared_lines)} dependencies (pyproject.toml, setup.cfg or setup.py)"
		)
	# a URL or VCS line is skipped by plan(), so its name doesn't stop a known import of it
	# (and plan() installs only plain lines; a marker may not hold here)
	plain = [m for x in declared_lines
		if "@" not in x and ";" not in x and (m := _LINE.fullmatch(x.split(" #", 1)[0].strip()))]  # fmt: skip
	names = {_norm(m["name"]) for m in plain}
	every = {_norm(m["name"]) for x in declared_lines if (m := _LINE.match(x.strip()))}
	imported: list[str] = []
	for mod in sorted(missing_imports(workdir, entrypoint)):
		if mod in NO_WHEELS and pyg:  # PyG's own index has its wheels
			if _norm(mod) not in names:
				imported.append(_norm(mod))
				names.add(_norm(mod))
				log(
					f"[preflight] requirements: the script imports {mod}: installing {_norm(mod)} from PyG's wheels"
				)
		elif mod in NO_WHEELS:
			log(f"[preflight] requirements: {mod} has no wheels on PyPI (only source); not installed")
		elif mod in KNOWN_MODULES:
			dist = _norm(_LINE.match(KNOWN_MODULES[mod])["name"])  # a known line may carry a pin
			if dist not in names:
				imported.append(KNOWN_MODULES[mod])
				names.add(dist)
				log(f"[preflight] requirements: the script imports {mod}: installing {KNOWN_MODULES[mod]}")
		elif _norm(mod) not in every:  # a declared one is being installed already
			log(f"[preflight] requirements: the script imports {mod}, which isn't installed and isn't a "
				"package Preflight installs by name: add it to requirements.txt")  # fmt: skip
	return declared_lines, imported


def repo_requirements(workdir: Path, entrypoint: str, *, log=print) -> str:
	"""Both sets as requirements.txt's text."""
	d, i = repo_requirement_sets(workdir, entrypoint, log=log)
	lines = [*d, *i]
	return "\n".join(lines) + ("\n" if lines else "")


def site_dirs(prefix: Path) -> list[Path]:
	"""Where pip --prefix put packages (lib/pythonX.Y/site-packages, or dist-packages on Debian)."""
	return sorted(
		{
			p
			for p in prefix.glob("**/*-packages")
			if p.is_dir() and p.name in ("site-packages", "dist-packages")
		}
	)


# probes.extra_hosts (off): PyG's own wheel index for the probe image's torch (the packages PyPI has only as
# source, NO_WHEELS); the image names it (runner/modal_app.py), and the probe's egress then allows data.pyg.org
PYG_INDEX_ENV = "PREFLIGHT_PYG_INDEX"


def _uv() -> list[str] | None:
	"""How to run uv here, when the image has it (probes.uv_install)."""
	exe = shutil.which("uv")
	if exe:
		return [exe]
	with contextlib.suppress(Exception):
		import importlib.util

		if importlib.util.find_spec("uv") is not None:
			return [sys.executable, "-I", "-m", "uv"]
	return None


def _pip(keep: list[str], target: Path, constraints: Path, timeout_s: float, *, uv: bool = False,
	find_links: str | None = None, log=None) -> str | None:  # fmt: skip
	"""One install of keep into target with the image pinned; the installer's error tail, or None. uv
	(probes.uv_install): `uv pip install` first, with the same constraints, wheels only and prefix (its
	resolver is much faster on a big stack: LitGPT's ran past 420 s in pip, Nov 2); pip after it if it
	fails, inside the same time limit. find_links: an extra wheel index page (PyG's)."""
	import time

	deadline = time.monotonic() + timeout_s
	links = ["--find-links", find_links] if find_links else []
	cache = os.environ.get("PREFLIGHT_PIP_CACHE")
	cache = cache if cache and os.access(cache, os.W_OK) else None
	runner = _uv() if uv else None
	if runner is not None:
		cmd = [*runner, "pip", "install", "--python", sys.executable, "--prefix", str(target), "--only-binary",
			":all:", "-c", str(constraints), "--no-progress", *links, *(["--cache-dir", cache] if cache else []),
			*keep]  # fmt: skip
		# without the probe's PYTHONPATH: it can name a directory the probe's uid can't list (as pip's -I)
		env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
		try:
			r = subprocess.run(
				cmd, capture_output=True, text=True, timeout=max(1.0, timeout_s), cwd=target, env=env
			)
			if r.returncode == 0:
				return None
			why = ((r.stderr or r.stdout or "").strip().splitlines() or [f"uv exited with {r.returncode}"])[
				-1
			]
		except subprocess.TimeoutExpired:
			return f"uv took longer than {timeout_s:.0f} s"
		except OSError as exc:
			why = f"uv didn't start ({type(exc).__name__})"
		if log is not None:
			log(f"[preflight] requirements: uv didn't install them ({why[:200]}); trying pip")
		timeout_s = deadline - time.monotonic()
		if timeout_s < 5:
			return f"uv failed and no time was left for pip: {why[:500]}"
	# -I: pip walks every sys.path entry, and the probe's PYTHONPATH can name a directory its
	# unprivileged uid can't list (Modal's /root, Nov 1: PermissionError, nothing installed)
	cmd = [sys.executable, "-I", "-m", "pip", "install", "--disable-pip-version-check", "--no-input",
		"--no-warn-script-location", "--only-binary", ":all:", "--prefix", str(target),
		"-c", str(constraints), *links, *keep]  # fmt: skip
	if cache:  # shared across a run's probes
		cmd[5:5] = ["--cache-dir", cache]
	try:
		r = subprocess.run(cmd, capture_output=True, text=True, timeout=max(1.0, timeout_s), cwd=target)
	except subprocess.TimeoutExpired:
		return f"pip took longer than {timeout_s:.0f} s"
	if r.returncode != 0:
		tail = (r.stderr or r.stdout or "").strip().splitlines()[-6:]
		return "\n".join(tail)[-1500:] or f"pip exited with {r.returncode}"
	return None


def install(workdir: Path, target: Path, *, log=print, repo: bool = False, entrypoint: str | None = None,
	extra: list[str] | None = None, uv: bool = False, pyg: bool = False) -> tuple[str | None, list[Path]]:  # fmt: skip
	"""Install workdir/requirements.txt. Returns (pip's error tail or None, the dirs to import from).

	target: this probe's own prefix; a shared one under PREFLIGHT_DEPS_DIR is used instead when set.
	repo (probes.repo_install): with no requirements.txt, install what the repo declares and the known
	packages its code imports (repo_requirement_sets), and put a src/ layout's packages on the path.
	The imported ones are optional: if everything together won't install, the declared ones go alone,
	then each imported package on its own, all inside the one time limit.
	extra: the job's extra packages (extra_requirements' lines), installed with the rest under the same
	rules (the image's stack wins, wheels only), with or without a requirements.txt.
	uv (probes.uv_install): uv before pip. pyg (probes.extra_hosts): PyG's wheel index too, so torch-sparse
	and the other NO_WHEELS packages install, and a repo's imports of them count as known."""
	import time

	req = workdir / "requirements.txt"
	roots = [r for r in _roots(workdir) if r != workdir] if repo else []
	optional: list[str] = []
	if req.is_file():
		text = req.read_text(encoding="utf-8", errors="replace")
	elif repo and entrypoint:
		try:
			decl, optional = repo_requirement_sets(workdir, entrypoint, log=log, pyg=pyg)
		except Exception as exc:  # a repo that can't be read for requirements runs as it would without them
			log(
				f"[preflight] requirements: couldn't read the repo's requirements ({type(exc).__name__}); none installed"
			)
			decl, optional = [], []
		text = "\n".join([*decl, *optional])
	elif extra:
		text = ""
	else:
		return None, roots
	if extra:  # first, so the requirements' own cap (MAX_REQUIREMENTS) never drops them
		text = "\n".join([*extra, text])
	have = installed()
	keep, notes = plan(text, have)
	for n in notes:
		log(f"[preflight] requirements: {n}")
	if not keep:
		return None, roots
	pins = "".join(f"{n}=={v.split('+', 1)[0]}\n" for n, v in sorted(have.items()))
	shared = os.environ.get("PREFLIGHT_DEPS_DIR")
	if shared and os.access(shared, os.W_OK):
		extra_key = f"\0{os.environ.get(PYG_INDEX_ENV)}" if pyg else ""  # what PyG's index adds
		key = hashlib.sha256(("\n".join(keep) + "\0" + pins + extra_key).encode()).hexdigest()[:24]
		target = Path(shared) / key
		if (target / ".ok").is_file():
			log("[preflight] requirements: already installed for this job")
			return None, [*site_dirs(target), *roots]
		shutil.rmtree(target, ignore_errors=True)  # a half-finished install from a killed probe
	target.mkdir(parents=True, exist_ok=True)
	constraints = target / "constraints.txt"
	constraints.write_text(pins)
	limit = REPO_INSTALL_TIMEOUT_S if repo and not req.is_file() else INSTALL_TIMEOUT_S
	deadline = time.monotonic() + limit
	log(f"[preflight] requirements: installing {', '.join(keep)}")
	links = os.environ.get(PYG_INDEX_ENV) if pyg else None
	how = {"uv": uv, "find_links": links, "log": log}
	err = _pip(keep, target, constraints, limit, **how)
	opt = [k for k in keep if k in optional]
	if err and opt:  # one optional package can't sink the ones the repo declares
		log(
			f"[preflight] requirements: together they didn't install ({err.splitlines()[-1][:200]}); trying them apart"
		)
		base = [k for k in keep if k not in optional]
		left = deadline - time.monotonic()
		if left < 5:  # the first try used the time: its error stands
			return err, []
		err = _pip(base, target, constraints, left, **how) if base else None
		if err is None:
			for k in opt:
				left = deadline - time.monotonic()
				if left < 5:
					log(f"[preflight] requirements: out of install time; {k} not installed")
					continue
				e = _pip([k], target, constraints, left, **how)
				log(
					f"[preflight] requirements: {k} "
					+ ("installed" if e is None else f"didn't install: {e.splitlines()[-1][:200]}")
				)
	if err:
		return err, []
	(target / ".ok").write_text("ok")
	return None, [*site_dirs(target), *roots]
