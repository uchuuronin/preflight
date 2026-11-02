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
	"torchao": "torchao", "torchdata": "torchdata", "tiktoken": "tiktoken", "sentencepiece": "sentencepiece",
	"tokenizers": "tokenizers", "jsonargparse": "jsonargparse", "litdata": "litdata", "evaluate": "evaluate",
	"ogb": "ogb", "beir": "beir", "faiss": "faiss-cpu", "sklearn": "scikit-learn", "scipy": "scipy",
	"pandas": "pandas", "yaml": "pyyaml", "PIL": "pillow", "cv2": "opencv-python-headless",
	"openai": "openai", "tqdm": "tqdm", "rich": "rich", "jinja2": "jinja2", "requests": "requests",
	"huggingface_hub": "huggingface-hub", "dotenv": "python-dotenv", "blobfile": "blobfile",
	"wandb": "wandb", "tensorboard": "tensorboard", "pytrec_eval": "pytrec-eval-terrier",
	"lm_eval": "lm-eval", "smolagents": "smolagents", "networkx": "networkx", "matplotlib": "matplotlib",
}  # fmt: skip
# on PyPI only as source (built against one torch and CUDA): never installable with --only-binary
NO_WHEELS = {"torch_sparse", "torch_scatter", "torch_cluster", "torch_spline_conv", "pyg_lib"}
MAX_SCANNED_FILES = 400  # guess: a repo's import closure from its entrypoint; bounds the scan


def _top_imports(text: str) -> set[str]:
	"""Top-level names of the absolute imports in a file, outside try blocks (an import that's
	allowed to fail is optional)."""
	import ast

	try:
		tree = ast.parse(text)
	except (SyntaxError, ValueError, RecursionError):
		return set()
	guarded: set[int] = set()
	for node in ast.walk(tree):
		if isinstance(node, ast.Try):
			for n in (*node.body, *node.handlers):
				guarded.update(id(x) for x in ast.walk(n))
	out = set()
	for node in ast.walk(tree):
		if id(node) in guarded:
			continue
		if isinstance(node, ast.Import):
			out.update(a.name.split(".")[0] for a in node.names)
		elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
			out.add(node.module.split(".")[0])
	return out


def _roots(workdir: Path) -> list[Path]:
	"""Where the script's own packages are importable from: the checkout, and src/ in a src layout."""
	src = workdir / "src"
	return [workdir, *([src] if src.is_dir() and any(src.glob("*/__init__.py")) else [])]


def _local(name: str, roots: list[Path]) -> Path | None:
	for r in roots:
		if (r / name / "__init__.py").is_file() or (r / f"{name}.py").is_file():
			return (r / name) if (r / name).is_dir() else r / f"{name}.py"
		if (r / name).is_dir() and any((r / name).glob("*.py")):  # a namespace package
			return r / name
	return None


def missing_imports(workdir: Path, entrypoint: str) -> set[str]:
	"""The third-party modules the entrypoint and the repo's own modules it imports need, that this
	Python can't import. Scans the import closure from the entrypoint, bounded."""
	import importlib.util

	roots = _roots(workdir)
	todo, seen, need = [workdir / entrypoint], set(), set()
	while todo and len(seen) < MAX_SCANNED_FILES:
		f = todo.pop()
		if f in seen or not f.is_file():
			continue
		seen.add(f)
		for name in _top_imports(f.read_text(encoding="utf-8", errors="replace")):
			loc = _local(name, roots)
			if loc is not None:
				todo.extend([loc] if loc.suffix == ".py" else sorted(loc.rglob("*.py"))[:MAX_SCANNED_FILES])
				continue
			if name in sys.stdlib_module_names or name == "__future__":
				continue
			try:
				found = importlib.util.find_spec(name) is not None
			except (ImportError, ValueError):
				found = False
			if not found:
				need.add(name)
	return need


def declared(workdir: Path) -> list[str]:
	"""The dependencies the repo declares: pyproject.toml's [project] dependencies, else setup.py's
	install_requires when it's a literal list (or a name bound to one)."""
	import ast
	import tomllib

	py = workdir / "pyproject.toml"
	if py.is_file():
		try:
			deps = (
				tomllib.loads(py.read_text(encoding="utf-8", errors="replace"))
				.get("project", {})
				.get("dependencies")
			)
		except (tomllib.TOMLDecodeError, AttributeError):
			deps = None
		if isinstance(deps, list) and all(isinstance(d, str) for d in deps):
			return deps
	setup = workdir / "setup.py"
	if setup.is_file():
		try:
			tree = ast.parse(setup.read_text(encoding="utf-8", errors="replace"))
		except (SyntaxError, ValueError, RecursionError):
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


def repo_requirements(workdir: Path, entrypoint: str, *, log=print) -> str:
	"""requirements.txt's text for a repo that ships none (probes.repo_install): what the repo declares,
	then what its code imports that the image lacks, by KNOWN_MODULES only. Noted, never guessed: a
	module Preflight doesn't know, and one with no wheels on PyPI."""
	lines = list(declared(workdir))
	if lines:
		log(
			f"[preflight] requirements: the repo declares {len(lines)} dependencies (pyproject.toml or setup.py)"
		)
	names = {_norm(_LINE.match(x.strip())["name"]) for x in lines if _LINE.match(x.strip())}
	for mod in sorted(missing_imports(workdir, entrypoint)):
		if mod in NO_WHEELS:
			log(f"[preflight] requirements: {mod} has no wheels on PyPI (only source); not installed")
		elif mod in KNOWN_MODULES:
			if _norm(KNOWN_MODULES[mod]) not in names:
				lines.append(KNOWN_MODULES[mod])
				names.add(_norm(KNOWN_MODULES[mod]))
				log(f"[preflight] requirements: the script imports {mod}: installing {KNOWN_MODULES[mod]}")
		else:
			log(f"[preflight] requirements: the script imports {mod}, which isn't installed and isn't a "
				"package Preflight installs by name: add it to requirements.txt")  # fmt: skip
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


def install(workdir: Path, target: Path, *, log=print, repo: bool = False, entrypoint: str | None = None
) -> tuple[str | None, list[Path]]:  # fmt: skip
	"""Install workdir/requirements.txt. Returns (pip's error tail or None, the dirs to import from).

	target: this probe's own prefix; a shared one under PREFLIGHT_DEPS_DIR is used instead when set.
	repo (probes.repo_install): with no requirements.txt, install what the repo declares and the known
	packages its code imports (repo_requirements), and put a src/ layout's packages on the path."""
	req = workdir / "requirements.txt"
	extra = [r for r in _roots(workdir) if r != workdir] if repo else []
	if req.is_file():
		text = req.read_text(encoding="utf-8", errors="replace")
	elif repo and entrypoint:
		text = repo_requirements(workdir, entrypoint, log=log)
	else:
		return None, extra
	have = installed()
	keep, notes = plan(text, have)
	for n in notes:
		log(f"[preflight] requirements: {n}")
	if not keep:
		return None, extra
	pins = "".join(f"{n}=={v.split('+', 1)[0]}\n" for n, v in sorted(have.items()))
	shared = os.environ.get("PREFLIGHT_DEPS_DIR")
	if shared and os.access(shared, os.W_OK):
		key = hashlib.sha256(("\n".join(keep) + "\0" + pins).encode()).hexdigest()[:24]
		target = Path(shared) / key
		if (target / ".ok").is_file():
			log("[preflight] requirements: already installed for this job")
			return None, [*site_dirs(target), *extra]
		shutil.rmtree(target, ignore_errors=True)  # a half-finished install from a killed probe
	target.mkdir(parents=True, exist_ok=True)
	constraints = target / "constraints.txt"
	constraints.write_text(pins)
	log(f"[preflight] requirements: installing {', '.join(keep)}")
	# -I: pip walks every sys.path entry, and the probe's PYTHONPATH can name a directory its
	# unprivileged uid can't list (Modal's /root, Nov 1: PermissionError, nothing installed)
	cmd = [sys.executable, "-I", "-m", "pip", "install", "--disable-pip-version-check", "--no-input",
		"--no-warn-script-location", "--only-binary", ":all:", "--prefix", str(target),
		"-c", str(constraints), *keep]  # fmt: skip
	cache = os.environ.get("PREFLIGHT_PIP_CACHE")
	if cache and os.access(cache, os.W_OK):  # shared across a run's probes
		cmd[5:5] = ["--cache-dir", cache]
	try:
		r = subprocess.run(cmd, capture_output=True, text=True, timeout=INSTALL_TIMEOUT_S, cwd=target)
	except subprocess.TimeoutExpired:
		return f"pip took longer than {INSTALL_TIMEOUT_S} s", []
	if r.returncode != 0:
		tail = (r.stderr or r.stdout or "").strip().splitlines()[-6:]
		return "\n".join(tail)[-1500:] or f"pip exited with {r.returncode}", []
	(target / ".ok").write_text("ok")
	return None, [*site_dirs(target), *extra]
