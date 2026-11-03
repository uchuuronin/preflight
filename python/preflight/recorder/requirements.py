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


def pin_conflicts(text: str, have: dict[str, str]) -> list[str]:
	"""Requirement lines whose versions exclude the image's own copy ("torch>=2.7 (the image has 2.5.1)"):
	plan() keeps the image's, so what needs the newer one can't resolve (LitGPT, Nov 2)."""
	try:
		from packaging.specifiers import InvalidSpecifier, SpecifierSet
		from packaging.version import InvalidVersion, Version
	except ImportError:
		return []
	out = []
	for raw in text.splitlines():
		m = _LINE.fullmatch(raw.split(" #", 1)[0].strip())
		# a marker may not apply here, and a local label (+cu121) can't be compared with the image's public version
		if m is None or not m["spec"] or m["marker"] or "+" in m["spec"] or _norm(m["name"]) not in have:
			continue
		got = have[_norm(m["name"])].split("+", 1)[0]
		try:
			if not SpecifierSet(m["spec"].replace(" ", "")).contains(Version(got), prereleases=True):
				out.append(f"{m['name']}{m['spec'].replace(' ', '')} (the image has {got})")
		except (InvalidSpecifier, InvalidVersion):
			continue
	return out[:4]


# ---------------------------------------------------------------- probes.torch_new (off)

# the probe images' torch: the default image's, and the newer image's (runner/modal_app.py PROBE_NEW_PACKAGES)
IMAGE_TORCH = "2.5.1"
NEW_IMAGE_TORCH = "2.7.1"


def entry_requirements(names, entrypoint: str | None) -> str:
	"""probes.entry_requirements (F1, Nov 3: word_language_model/requirements.txt says torch>=2.6, and only the repo
	top's was read): the requirements file nearest the entry point, from its own folder up to the top; the top's
	"requirements.txt" when there's none nearer (or no entry point)."""
	import posixpath

	names = set(names)
	d = posixpath.dirname(posixpath.normpath(entrypoint or "")) if entrypoint else ""
	if d.startswith(("..", "/")):
		return "requirements.txt"
	while d:
		cand = f"{d}/requirements.txt"
		if cand in names:
			return cand
		d = posixpath.dirname(d)
	return "requirements.txt"


def _code_lines(files: dict[str, str], req: str = "requirements.txt") -> list[str]:
	"""The requirement lines the code's own files state (req: requirements.txt, or the entry point's own,
	entry_requirements; pyproject.toml, setup.cfg, a literal setup.py list)."""
	import posixpath
	import tempfile

	lines = []
	if req in files:
		for raw in files[req].splitlines():
			m = re.fullmatch(r"\s*(?:-r|--requirement)[\s=]+(\S+)\s*", raw.split(" #", 1)[0])
			inc = posixpath.normpath(posixpath.join(posixpath.dirname(req), m[1])) if m else None
			if m and inc in files and inc != req:  # one level: base.txt beside it (B's review)
				lines += files[inc].splitlines()
			else:
				lines.append(raw)
	with tempfile.TemporaryDirectory() as tmp:
		for name in ("pyproject.toml", "setup.cfg", "setup.py"):
			if name in files:
				(Path(tmp) / name).write_text(files[name], encoding="utf-8", errors="replace")
		lines += declared(Path(tmp))
	return lines


def torch_asks(files: dict[str, str], req: str = "requirements.txt") -> list[str]:
	"""The torch requirements the code's own files state (requirements.txt or `req`, pyproject.toml, setup.cfg, a
	literal setup.py list), without a marker: "torch>=2.7" from LitGPT's pyproject.toml, Nov 2."""
	out = []
	for raw in _code_lines(files, req):
		m = _LINE.fullmatch(raw.split(" #", 1)[0].strip())
		if m and _norm(m["name"]) == "torch" and m["spec"] and not m["marker"]:
			out.append(f"torch{m['spec'].replace(' ', '')}")
	return out


def image_overrides(
	files: dict[str, str], packages: list[str], req: str = "requirements.txt"
) -> list[tuple[str, str, str]]:
	"""(name, the image's version, the code's ask) for each of the code's own pins an image's packages
	override (packages: "name==version" lines, PROBE_NEW_PACKAGES): torch 2.6.0 with transformers 4.46.3 and
	TRL 0.14.0, probed on the torch 2.7.1 image, runs on transformers 4.52.3 and TRL 0.18.0 (B's review, Nov 3).
	plan() keeps the image's copy of every package it has, so each is measured on the image's version."""
	try:
		from packaging.specifiers import InvalidSpecifier, SpecifierSet
		from packaging.version import InvalidVersion, Version
	except ImportError:
		return []
	have = {}
	for p in packages:
		name, sep, ver = p.partition("==")
		if sep:
			have[_norm(name)] = ver.strip()
	out, seen = [], set()
	for raw in _code_lines(files, req):
		m = _LINE.fullmatch(raw.split(" #", 1)[0].strip())
		# a marker may not apply here; a local label (+cu124) isn't a version of the package
		if m is None or not m["spec"] or m["marker"] or _norm(m["name"]) not in have:
			continue
		name, spec = _norm(m["name"]), re.sub(r"\+[0-9A-Za-z.]+", "", m["spec"].replace(" ", ""))
		if name in seen:
			continue
		try:
			if not SpecifierSet(spec).contains(Version(have[name]), prereleases=True):
				out.append((m["name"], have[name], spec))
				seen.add(name)
		except (InvalidSpecifier, InvalidVersion):
			continue
	return out


def overrides_note(overrides: list[tuple[str, str, str]]) -> str:
	""" "measured on transformers 4.52.3; your code pins 4.46.3" (the PM, Nov 3), every override in one line."""

	def ask(name: str, spec: str) -> str:
		exact = spec.startswith("==") and "," not in spec and "*" not in spec
		if exact or (spec.startswith("===") and "," not in spec):
			return f"pins {spec.lstrip('=')}"
		return f"asks for {name}{spec}"

	return "; ".join(f"measured on {n} {v}; your code {ask(n, s)}" for n, v, s in overrides)


def _torch_specs(asks: list[str]):
	"""Each ask's clauses, local labels (+cu124) dropped: a CUDA build label is not a torch version."""
	from packaging.specifiers import InvalidSpecifier, SpecifierSet

	for a in asks:
		with contextlib.suppress(InvalidSpecifier):
			yield from SpecifierSet(re.sub(r"\+[0-9A-Za-z.]+", "", a[len("torch") :]))


def torch_fits(asks: list[str], version: str) -> bool:
	"""Whether every torch requirement admits this version (no asks: it does)."""
	try:
		from packaging.version import Version
	except ImportError:
		return True
	return all(s.contains(Version(version), prereleases=True) for s in _torch_specs(asks))


def torch_too_old(asks: list[str], version: str) -> str | None:
	"""Said plainly when the code's own torch requirement asks for a newer torch than the image's: "asks for
	torch>=2.7; Preflight's probe image has torch 2.5.1" (the PM, Nov 3), else None; a note, never a stop. Only an ask that rules
	the image's version out from below (>=, >, ~=, or == a newer one): an older pin, a != or a < runs as it
	did, with the image's torch (review, Nov 3)."""
	try:
		from packaging.version import InvalidVersion, Version
	except ImportError:
		return None
	have = Version(version)
	newer = False
	for s in _torch_specs(asks):
		if s.contains(have, prereleases=True):
			continue
		try:
			bound = Version(s.version.removesuffix(".*"))
		except InvalidVersion:
			return None
		if s.operator in (">=", ">", "~=") or (s.operator in ("==", "===") and bound > have):
			newer = True
		else:
			return None  # an older pin, a cap or an exclusion: not a newer torch
	return f"asks for {', '.join(asks[:2])}; Preflight's probe image has torch {version}" if newer else None


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


def built_in_code(workdir: Path) -> bool:
	"""Whether the repo declares dependencies that only running its build can list: a pyproject.toml whose
	[project] dependencies are `dynamic`, or, with no [project] table, a setup.py that passes install_requires
	something other than a literal (open-r1's `deps_list(...)`, Nov 2). A list or tuple of strings (empty too), None,
	or a name bound to one is a literal (B's review, Nov 3); with a [project] table, setuptools ignores setup.py's
	list, so only `dynamic` counts."""
	import ast
	import tomllib

	py = workdir / "pyproject.toml"
	if py.is_file():
		proj = None
		with contextlib.suppress(
			tomllib.TOMLDecodeError, AttributeError, RecursionError, MemoryError, OSError
		):
			proj = tomllib.loads(py.read_text(encoding="utf-8", errors="replace")).get("project")
		if isinstance(proj, dict):
			return "dependencies" in (proj.get("dynamic") or [])
	setup = workdir / "setup.py"
	if not setup.is_file():
		return False
	try:
		tree = ast.parse(setup.read_text(encoding="utf-8", errors="replace"))
	except (SyntaxError, ValueError, RecursionError, MemoryError, OSError):
		return False

	def literal(v) -> bool:
		if isinstance(v, ast.Constant) and v.value is None:
			return True
		return isinstance(v, ast.List | ast.Tuple) and all(
			isinstance(e, ast.Constant) and isinstance(e.value, str) for e in v.elts
		)

	bound = {}
	for n in ast.walk(tree):
		if isinstance(n, ast.Assign):
			for t in n.targets:
				if isinstance(t, ast.Name):
					bound.setdefault(t.id, []).append(n.value)
	for n in ast.walk(tree):
		if isinstance(n, ast.keyword) and n.arg == "install_requires":
			v = n.value
			if literal(v) or (
				isinstance(v, ast.Name) and v.id in bound and all(literal(x) for x in bound[v.id])
			):
				continue
			return True
	return False


def needs_build(workdir: Path) -> bool:
	"""Only running the build can list the dependencies (the PM, Nov 3): no literal list is declared, and the
	setup.py computes install_requires or the pyproject's dependencies are dynamic (built_in_code)."""
	return not declared(workdir) and built_in_code(workdir)


BUILD_TIMEOUT_S = (
	120.0  # guess: pip's metadata build of one project (open-r1's took 6 s with setuptools fetched)
)
MAX_BUILD_REPORT = 4 * 2**20  # guess: bytes of pip's report read back; one project's metadata is a few kB


# a build that tries to compile gets a compiler that fails (the PM, Nov 3: never compile an extension); pip's
# metadata step needs none, so only a setup.py that compiles as it's read is stopped
_NO_COMPILER = {"CC": "false", "CXX": "false", "LDSHARED": "false", "NVCC": "false", "CUDA_HOME": "/nonexistent",
	"CUDA_PATH": "/nonexistent", "MAX_JOBS": "1"}  # fmt: skip
# and on PATH, ahead of the real ones, tools that fail: a setup.py or Makefile that calls gcc, nvcc, make or cmake
# by name gets an error, not a build (B's review, Nov 3); one that calls a compiler by its full path still can
_SHIMS = ("gcc", "g++", "cc", "c++", "clang", "clang++", "nvcc", "ld", "make", "cmake", "ninja", "gfortran")


MAX_COPY_BYTES = (
	64 * 2**20
)  # guess: the repo's text a build copy carries; a fetched repo is capped below this


def _shim_path(tmp: Path) -> str:
	shims = tmp / "no-compilers"
	shims.mkdir()
	for name in _SHIMS:
		(shims / name).write_text(
			"#!/bin/sh\necho 'preflight: no compiling in a metadata build' >&2\nexit 1\n"
		)
		(shims / name).chmod(0o755)
	return f"{shims}{os.pathsep}{os.environ.get('PATH', '')}"


def _build_copy(workdir: Path, dest: Path) -> bool:
	"""The working copy's files, copied for a build to run in (the PM, Nov 3: never write into the visitor's
	copy): pip's metadata step writes an .egg-info, and a build may write anything, beside its setup.py."""
	total = 0
	for src in sorted(workdir.rglob("*")):
		if src.is_symlink() or not src.is_file() or ".git" in src.relative_to(workdir).parts:
			continue
		total += src.stat().st_size
		if total > MAX_COPY_BYTES:
			return False
		out = dest / src.relative_to(workdir)
		out.parent.mkdir(parents=True, exist_ok=True)
		shutil.copyfile(src, out)
	return True


def build_metadata(workdir: Path, *, log=print, timeout_s: float = BUILD_TIMEOUT_S) -> dict | None:
	"""probes.repo_install_exec (off): the repo's own package metadata, by pip's metadata-only build
	(`pip install --dry-run --no-deps --report`), only when the dependency list is built in code (built_in_code;
	the PM, Nov 3). That runs the visitor's setup.py, so it is only called inside the probe's sandbox, where their
	training code runs too, and on a copy of the repo in a scratch folder, with compilers that fail: nothing is
	written into the visitor's copy, nothing is installed, and no extension is compiled. None, saying why, when
	it isn't built in code or the build fails."""
	if not needs_build(workdir):
		return None
	key = _tree_digest(workdir)
	if key is None or key not in _BUILT:  # one build for the same files: the list's read and the metadata's
		meta = _build_metadata(workdir, log=log, timeout_s=timeout_s)
		if key is None:
			return meta
		_BUILT[key] = meta
	return _BUILT[key]


_BUILT: dict[str, dict | None] = {}  # by the files' digest, never by path: a worker reuses its folders


def _tree_digest(workdir: Path) -> str | None:
	"""The working copy's files (paths and bytes), as one digest; None past MAX_COPY_BYTES."""
	h, total = hashlib.sha256(), 0
	for src in sorted(workdir.rglob("*")):
		if src.is_symlink() or not src.is_file() or ".git" in src.relative_to(workdir).parts:
			continue
		data = src.read_bytes()
		total += len(data)
		if total > MAX_COPY_BYTES:
			return None
		h.update(str(src.relative_to(workdir)).encode() + b"\0" + hashlib.sha256(data).digest())
	return h.hexdigest()


def _build_metadata(workdir: Path, *, log, timeout_s: float) -> dict | None:
	import json
	import tempfile

	with tempfile.TemporaryDirectory() as tmp:
		copy = Path(tmp) / "repo"
		if not _build_copy(workdir, copy):
			log("[preflight] requirements: the repo is too big to copy for its build; its build wasn't run")
			return None
		report = Path(tmp) / "report.json"
		cmd = [
			sys.executable,
			"-I",
			"-m",
			"pip",
			"install",
			"--dry-run",
			"--no-deps",
			"--ignore-installed",
			"--disable-pip-version-check",
			"--no-input",
			"--quiet",
			"--only-binary",
			":all:",
			"--report",
			str(report),
			str(copy),
		]  # fmt: skip  (--only-binary: the build's own build dependencies come as wheels, never built)
		env = {
			**{k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
			**_NO_COMPILER,
			"PATH": _shim_path(Path(tmp)),
		}
		try:
			r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, cwd=tmp, env=env)
		except subprocess.TimeoutExpired:
			log(
				f"[preflight] requirements: the repo's build didn't list its dependencies in {timeout_s:.0f} s"
			)
			return None
		except OSError as exc:
			log(f"[preflight] requirements: pip didn't start to read the repo's build ({type(exc).__name__})")
			return None
		if r.returncode != 0 or not report.is_file() or report.stat().st_size > MAX_BUILD_REPORT:
			tail = ((r.stderr or r.stdout or "").strip().splitlines() or [f"pip exited with {r.returncode}"])[
				-1
			]
			log(f"[preflight] requirements: the repo's build didn't list its dependencies: {tail[:300]}")
			return None
		try:
			doc = json.loads(report.read_text(encoding="utf-8", errors="replace"))
		except (ValueError, RecursionError):
			doc = None
	# the report is the visitor's build's output: only a list of one item whose metadata is a dict
	items = doc.get("install") if isinstance(doc, dict) else None
	meta = (
		items[0].get("metadata")
		if isinstance(items, list) and len(items) == 1 and isinstance(items[0], dict)
		else None
	)
	return meta if isinstance(meta, dict) else None


def requires_from_build(workdir: Path, *, log=print, timeout_s: float = BUILD_TIMEOUT_S) -> list[str]:
	"""probes.repo_install_exec (off): the requirements the repo's own build lists (build_metadata). The lines
	go through the usual install (the image's stack pinned, wheels only). Extras are left out. [] when the
	list isn't built in code or the build fails, saying why."""
	meta = build_metadata(workdir, log=log, timeout_s=timeout_s)
	if meta is None:
		return []
	reqs = meta.get("requires_dist")
	if not isinstance(reqs, list):
		log("[preflight] requirements: the repo's build gave no dependency list Preflight could read")
		return []
	out = [
		x.strip()
		for x in reqs[: MAX_REQUIREMENTS * 4]
		if isinstance(x, str) and "extra" not in x.partition(";")[2]
	]
	log(f"[preflight] requirements: the repo's setup.py builds its list in code; its build lists {len(out)} "
		"dependencies (read by running it on a copy in this sandbox)")  # fmt: skip
	return out


def _version_in(path: Path) -> str | None:
	"""A file's own `__version__ = "1.2.3"` (or `version = ...`), read by its syntax tree, never run."""
	import ast

	try:
		tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
	except (SyntaxError, ValueError, RecursionError, MemoryError, OSError):
		return None
	for n in tree.body:
		if isinstance(n, ast.Assign | ast.AnnAssign):
			targets = n.targets if isinstance(n, ast.Assign) else [n.target]
			v = n.value
			if any(isinstance(t, ast.Name) and t.id in ("__version__", "version") for t in targets) and (
				isinstance(v, ast.Constant) and isinstance(v.value, str)
			):
				return v.value
	return None


def _attr_version(workdir: Path, attr: str) -> str | None:
	"""setuptools' `attr: pkg.__version__` (setup.cfg, or pyproject's [tool.setuptools.dynamic]), read from the
	module's file, flat or under src/, without importing it."""
	mod, _, name = attr.strip().rpartition(".")
	if not mod or name not in ("__version__", "version") or not re.fullmatch(r"[A-Za-z_][\w.]*", mod):
		return None
	rel = Path(*mod.split("."))
	for base in (workdir, workdir / "src"):
		for f in (base / rel / "__init__.py", base.joinpath(*rel.parts[:-1], f"{rel.parts[-1]}.py")):
			if f.is_file() and f.resolve().is_relative_to(workdir.resolve()):
				return _version_in(f)
	return None


def static_name_version(workdir: Path) -> tuple[str, str] | None:
	"""The project's name and version as its files state them, read without running anything: pyproject's
	[project] (a dynamic version from [tool.setuptools.dynamic]'s attr or [tool.hatch.version]'s path), setup.cfg's
	[metadata] (its `attr:` too), or literal name= and version= strings in setup.py (B's review, Nov 3: a version
	kept in the package's own __version__ is common)."""
	import ast
	import configparser
	import tomllib

	py = workdir / "pyproject.toml"
	if py.is_file():
		with contextlib.suppress(
			tomllib.TOMLDecodeError, AttributeError, RecursionError, MemoryError, OSError, TypeError
		):
			doc = tomllib.loads(py.read_text(encoding="utf-8", errors="replace"))
			proj = doc.get("project", {})
			if isinstance(proj, dict) and isinstance(proj.get("name"), str):
				v = proj.get("version")
				if not isinstance(v, str) and "version" in (proj.get("dynamic") or []):
					tool = doc.get("tool", {})
					attr = ((tool.get("setuptools") or {}).get("dynamic") or {}).get("version") or {}
					path = (tool.get("hatch") or {}).get("version") or {}
					if isinstance(attr, dict) and isinstance(attr.get("attr"), str):
						v = _attr_version(workdir, attr["attr"])
					elif isinstance(path, dict) and isinstance(path.get("path"), str):
						f = workdir / path["path"]
						v = (
							_version_in(f)
							if f.resolve().is_relative_to(workdir.resolve()) and f.is_file()
							else None
						)
				if isinstance(v, str) and v:
					return proj["name"], v
	cfg = workdir / "setup.cfg"
	if cfg.is_file():
		cp = configparser.ConfigParser(interpolation=None)
		with contextlib.suppress(configparser.Error, RecursionError, MemoryError, OSError):
			cp.read_string(cfg.read_text(encoding="utf-8", errors="replace"))
			n, v = (
				cp.get("metadata", "name", fallback="").strip(),
				cp.get("metadata", "version", fallback="").strip(),
			)
			if v.startswith("attr:"):
				v = _attr_version(workdir, v[len("attr:") :]) or ""
			if n and v and not v.startswith("file:"):
				return n, v
	setup = workdir / "setup.py"
	if setup.is_file():
		try:
			tree = ast.parse(setup.read_text(encoding="utf-8", errors="replace"))
		except (SyntaxError, ValueError, RecursionError, MemoryError, OSError):
			return None
		kw = {n.arg: n.value.value for n in ast.walk(tree) if isinstance(n, ast.keyword)
			and n.arg in ("name", "version") and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str)}  # fmt: skip
		if "name" in kw and "version" in kw:
			return kw["name"], kw["version"]
	return None


def own_metadata(workdir: Path, target: Path, *, log=print, timeout_s: float = BUILD_TIMEOUT_S) -> bool:
	"""probes.repo_install_exec: the repo's own package metadata in the probe's own prefix, for code that reads its
	own version (lm-evaluation-harness's lm_eval/__init__.py calls importlib.metadata.version("lm_eval"):
	a-all4-5s, Nov 3). A dist-info folder with the project's name and version, read from its files; the build runs
	(build_metadata) only when the dependency list is built in code and the files don't state both (the PM,
	Nov 3). Nothing of the repo is installed or written into its copy, and nothing compiled: the working copy stays
	what the probe imports. A failure is said and the probe runs on."""
	nv = static_name_version(workdir)
	if nv is None and needs_build(workdir):
		meta = build_metadata(workdir, log=log, timeout_s=timeout_s) or {}
		if isinstance(meta.get("name"), str) and isinstance(meta.get("version"), str):
			nv = meta["name"], meta["version"]
	if nv is None:
		log(
			"[preflight] requirements: no metadata written for the repo itself: its files don't state its name and "
			"version, and its build, run only when its list is built in code, didn't give them"
		)
		return False
	name, version = (re.sub(r"[^A-Za-z0-9._+!-]", "", x)[:100] for x in nv)
	if not re.fullmatch(_NAME, name) or not version:
		log(
			"[preflight] requirements: the repo's own name or version isn't one Preflight can write; no metadata written"
		)
		return False
	site = target / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
	info = site / f"{re.sub(r'[-.]+', '_', name)}-{version}.dist-info"
	try:
		info.mkdir(parents=True, exist_ok=True)
		(info / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
		(info / "INSTALLER").write_text("preflight\n")
	except OSError as exc:
		log(f"[preflight] requirements: the repo's own metadata wasn't written ({type(exc).__name__})")
		return False
	log(
		f"[preflight] requirements: wrote the repo's own package metadata ({name} {version}; nothing installed)"
	)
	return True


def repo_requirement_sets(
	workdir: Path, entrypoint: str, *, log=print, pyg: bool = False, build: bool = False
) -> tuple[list[str], list[str]]:
	"""(what the repo declares, what its code imports that the image lacks, by KNOWN_MODULES only) for a
	repo that ships no requirements.txt (probes.repo_install). Noted, never guessed: a module Preflight
	doesn't know, and one with no wheels on PyPI. build (probes.repo_install_exec): when the declared
	list is built in code (built_in_code), the list the repo's build gives (requires_from_build)."""
	declared_lines = list(declared(workdir))
	if not declared_lines and build and built_in_code(workdir):
		declared_lines = requires_from_build(workdir, log=log)
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
	missing = missing_imports(workdir, entrypoint)
	if pyg and "torch_geometric" in missing:
		# PyG imports its compiled extensions at run time (T.ToSparseTensor needs torch_sparse; OGB arxiv, Nov 2), so
		# the static scan never sees them: with PyG's index at hand they come with torch_geometric
		missing = missing | {"torch_scatter", "torch_sparse"}
	for mod in sorted(missing):
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


UV_SHARE = 0.6  # guess: of the install's time limit, what uv's resolve may use before pip gets the rest
UV_STACK_BYTES = 64 * 2**20  # guess: well past what LitGPT's backtrack needed
NO_SOLUTION = "no version set installs under the image's pinned stack: "


def _no_solution(text: str) -> str:
	"""uv's reason, cut to the lines that say which requirement conflicts with which pin."""
	why = [
		ln.strip(" ╰─▶")
		for ln in text.splitlines()
		if "requires" in ln or "depends on" in ln or "unsatisfiable" in ln
	]
	return " ".join(why)[:600] or text.splitlines()[-1][:600]


def _uv_then_pip(
	runner, keep, target, constraints, links, cache, resolve_s, deadline, sdists=()
) -> str | None:
	"""uv pip compile under the image's pins, then `pip install --no-deps` of the pins the image doesn't have;
	None when it installed, else why not (and pip resolves as usual after it)."""
	import time

	env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}  # as pip's -I: the probe's path
	# a deep backtrack overflowed uv 0.5's default resolver stack on LitGPT (Nov 2, a-trial-litgpt-install-uv)
	env.update({"UV_STACK_SIZE": str(UV_STACK_BYTES), "RUST_MIN_STACK": str(UV_STACK_BYTES)})
	cmd = [*runner, "pip", "compile", "-", "--python", sys.executable, "--only-binary", ":all:", "-c",
		str(constraints), "--no-header", "--no-annotate", "--quiet", *links, *(["--cache-dir", cache] if cache else []),
		*(a for n in sdists for a in ("--no-binary-package", n))]  # fmt: skip
	try:
		r = subprocess.run(cmd, input="\n".join(keep) + "\n", capture_output=True, text=True,
			timeout=max(1.0, resolve_s), cwd=target, env=env)  # fmt: skip
	except subprocess.TimeoutExpired:
		return f"uv's resolve took longer than {resolve_s:.0f} s"
	except OSError as exc:
		return f"uv didn't start ({type(exc).__name__})"
	if r.returncode != 0:
		text = (r.stderr or r.stdout or "").strip()
		if "No solution found" in text:  # unsatisfiable under the image's pins: pip would only spend the time
			return NO_SOLUTION + _no_solution(text)
		return (text.splitlines() or [f"uv exited with {r.returncode}"])[-1]
	have = installed()
	pins = [ln.strip() for ln in r.stdout.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
	todo = [p for p in pins if (m := _LINE.match(p)) and _norm(m["name"]) not in have]
	if any(not re.fullmatch(rf"{_NAME}==[A-Za-z0-9.+!_-]{{1,64}}", p) for p in todo):
		return "uv's resolve wasn't plain pins"
	if not todo:
		return None
	left = deadline - time.monotonic()
	cmd = [sys.executable, "-I", "-m", "pip", "install", "--disable-pip-version-check", "--no-input",
		"--no-warn-script-location", "--only-binary", ":all:", *(a for n in sdists for a in ("--no-binary", n)),
		"--no-deps", "--prefix", str(target), *links, *todo]  # fmt: skip
	try:
		r = subprocess.run(cmd, capture_output=True, text=True, timeout=max(1.0, left), cwd=target)
	except subprocess.TimeoutExpired:
		return f"pip took longer than {left:.0f} s installing uv's pins"
	if r.returncode != 0:
		return ((r.stderr or r.stdout or "").strip().splitlines() or [f"pip exited with {r.returncode}"])[-1]
	return None


def _pip(keep: list[str], target: Path, constraints: Path, timeout_s: float, *, uv: bool = False,
	find_links: str | None = None, log=None, sdists: tuple[str, ...] = ()) -> str | None:  # fmt: skip
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
		# uv resolves (its resolver is the fast part); pip installs exactly what it chose that the image lacks,
		# without resolving again. Not `uv pip install --prefix`: uv doesn't see the image's packages there and
		# would put a second torch in the prefix (review, Nov 2)
		why = _uv_then_pip(
			runner, keep, target, constraints, links, cache, timeout_s * UV_SHARE, deadline, sdists
		)
		if why is None or why.startswith(NO_SOLUTION):
			return why
		if log is not None:
			log(f"[preflight] requirements: uv's route didn't install them ({why[:200]}); trying pip")
		timeout_s = deadline - time.monotonic()
		if timeout_s < 5:
			return f"no time was left for pip after uv's route: {why[:500]}"
	# -I: pip walks every sys.path entry, and the probe's PYTHONPATH can name a directory its
	# unprivileged uid can't list (Modal's /root, Nov 1: PermissionError, nothing installed)
	cmd = [sys.executable, "-I", "-m", "pip", "install", "--disable-pip-version-check", "--no-input",
		"--no-warn-script-location", "--only-binary", ":all:", "--prefix", str(target),
		"-c", str(constraints), *(a for n in sdists for a in ("--no-binary", n)), *links, *keep]  # fmt: skip
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


def _project(workdir: Path) -> bool:
	return (workdir / "pyproject.toml").is_file() or (workdir / "setup.py").is_file()


def install(workdir: Path, target: Path, *, log=print, repo: bool = False, entrypoint: str | None = None,
	extra: list[str] | None = None, uv: bool = False, pyg: bool = False, sdists: tuple[str, ...] = (),
	build: bool = False, torch_note: bool = False, requirements: str = "requirements.txt",
) -> tuple[str | None, list[Path]]:  # fmt: skip
	"""Install workdir/requirements.txt (or `requirements`, the entry point's own: probes.entry_requirements). Returns (pip's error tail or None, the dirs to import from).

	target: this probe's own prefix; a shared one under PREFLIGHT_DEPS_DIR is used instead when set.
	repo (probes.repo_install): with no requirements.txt, install what the repo declares and the known
	packages its code imports (repo_requirement_sets), and put a src/ layout's packages on the path.
	The imported ones are optional: if everything together won't install, the declared ones go alone,
	then each imported package on its own, all inside the one time limit.
	extra: the job's extra packages (extra_requirements' lines), installed with the rest under the same
	rules (the image's stack wins, wheels only), with or without a requirements.txt.
	uv (probes.uv_install): uv before pip. pyg (probes.extra_hosts): PyG's wheel index too, so torch-sparse
	and the other NO_WHEELS packages install, and a repo's imports of them count as known. build
	(probes.repo_install_exec): a setup.py that builds its list in code is run to read it (requires_from_build).
	torch_note (probes.torch_new): when what's installed asks for a newer torch than the image's, say so plainly
	(torch_too_old) and run on with the image's (the PM, Nov 3; never a stop: review)."""
	import time

	started = time.monotonic() if build else None  # the build's own read counts against the install's limit
	own = (
		target.parent / f"{target.name}-own"
	)  # the repo's own metadata: this probe's prefix, never the shared one keyed by its deps
	rel = Path(requirements or "requirements.txt")
	if rel.is_absolute() or ".." in rel.parts:
		rel = Path("requirements.txt")
	req = workdir / rel
	roots = [r for r in _roots(workdir) if r != workdir] if repo else []
	optional: list[str] = []
	if req.is_file():
		text = req.read_text(encoding="utf-8", errors="replace")
	elif repo and entrypoint:
		try:
			decl, optional = repo_requirement_sets(workdir, entrypoint, log=log, pyg=pyg, build=build)
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
	if torch_note and "torch" in have:
		asks = [f"torch{m['spec'].replace(' ', '')}" for raw in text.splitlines()
			if (m := _LINE.fullmatch(raw.split(' #', 1)[0].strip())) and _norm(m['name']) == 'torch' and m['spec'] and not m['marker']]  # fmt: skip
		newer = torch_too_old(asks, have["torch"].split("+", 1)[0])
		if newer:  # said plainly and run on: a pin pip freeze wrote (torch==2.6.0) often runs on a near torch (review)
			log(f"[preflight] requirements: the code {newer}, and runs on it")
	keep, notes = plan(text, have)
	for n in notes:
		log(f"[preflight] requirements: {n}")
	if not keep:
		project = (workdir / "pyproject.toml").is_file() or (workdir / "setup.py").is_file()
		# every dependency is the image's already: the repo's own metadata still (review)
		if build and repo and project and own_metadata(workdir, own, log=log, timeout_s=BUILD_TIMEOUT_S):
			return None, [*site_dirs(own), *roots]
		return None, roots
	pins = "".join(f"{n}=={v.split('+', 1)[0]}\n" for n, v in sorted(have.items()))
	shared = os.environ.get("PREFLIGHT_DEPS_DIR")
	if shared and os.access(shared, os.W_OK):
		extra_key = f"\0{os.environ.get(PYG_INDEX_ENV)}" if pyg else ""  # what PyG's index adds
		extra_key += f"\0{','.join(sdists)}" if sdists else ""  # what building from source allows
		key = hashlib.sha256(("\n".join(keep) + "\0" + pins + extra_key).encode()).hexdigest()[:24]
		target = Path(shared) / key
		if (target / ".ok").is_file():
			log("[preflight] requirements: already installed for this job")
			mine = build and repo and _project(workdir) and own_metadata(workdir, own, log=log)
			return None, [*site_dirs(target), *(site_dirs(own) if mine else []), *roots]
		shutil.rmtree(target, ignore_errors=True)  # a half-finished install from a killed probe
	target.mkdir(parents=True, exist_ok=True)
	constraints = target / "constraints.txt"
	constraints.write_text(pins)
	limit = REPO_INSTALL_TIMEOUT_S if repo and not req.is_file() else INSTALL_TIMEOUT_S
	deadline = (started if started is not None else time.monotonic()) + limit
	limit = deadline - time.monotonic()
	log(f"[preflight] requirements: installing {', '.join(keep)}")
	links = os.environ.get(PYG_INDEX_ENV) if pyg else None
	# a check-in by command's own pure-Python dependencies PyPI has only as source (sources.COMMANDS)
	src_ok = tuple(n for n in sdists if re.fullmatch(_NAME, n))
	how = {"uv": uv, "find_links": links, "log": log, "sdists": src_ok}
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
		clash = pin_conflicts(text, have)
		if clash:  # the likeliest reason, ahead of pip's or uv's tail: no installer fixes the image's own pin
			err = f"the code asks for {'; '.join(clash)}, which the probe image can't change. {err}"
		return err, []
	(target / ".ok").write_text("ok")
	mine = (build and repo and _project(workdir) and own_metadata(workdir, own, log=log,
		timeout_s=min(BUILD_TIMEOUT_S, max(5.0, deadline - time.monotonic()))))  # fmt: skip
	return None, [*site_dirs(target), *(site_dirs(own) if mine else []), *roots]
