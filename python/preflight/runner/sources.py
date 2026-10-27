"""Fetch a job's source into {path: text}: a bug-zoo sample, inline files, a notebook, or a GitHub repo.

Everything that comes back is visitor data. Paths and sizes are checked by
probes.local.validate_files before anything runs. A repo is cloned shallow and
without a checkout (no hooks, submodules, symlinks or working tree), blobs over
the per-file cap are never downloaded, and only text files are read, straight
from git's object store; the clone is deleted before fetch returns.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from preflight.probes.local import (
	MAX_FILE_BYTES,
	MAX_FILES,
	MAX_TOTAL_BYTES,
	InvalidSubmission,
	validate_files,
)
from preflight.schema import Source

ROOT = Path(__file__).resolve().parents[3]
ZOO = Path(os.environ.get("PREFLIGHT_ZOO", ROOT / "zoo"))  # set in images, where the package sits elsewhere
TEXT_SUFFIXES = (".py", ".sh", ".sbatch", ".slurm", ".yaml", ".yml", ".json", ".toml", ".cfg", ".txt", ".md")
ENTRYPOINT_GUESSES = ("train.py", "main.py", "run.py", "finetune.py", "pretrain.py")
_GITHUB = re.compile(r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+?(\.git)?/?")
# a full hash: GitHub serves `git fetch origin <sha>` only for the whole 40 characters
_COMMIT = re.compile(r"[0-9a-f]{40}")
MAX_ARGS = 64
MAX_ARG_LEN = 4096
CLONE_TIMEOUT_S = 60


@dataclass
class Fetched:
	files: dict[str, str]
	entrypoint: str
	args: list[str] = field(default_factory=list)
	source: Source | None = None


def content_hash(files: dict[str, str]) -> str:
	h = hashlib.sha256()
	for path in sorted(files):
		h.update(path.encode())
		h.update(b"\0")
		h.update(files[path].encode("utf-8", "surrogatepass"))
		h.update(b"\0")
	return "sha256:" + h.hexdigest()


def _wanted(rel: str) -> bool:
	return not (rel.startswith(".") or "/." in rel) and Path(rel).suffix in TEXT_SUFFIXES


def _too_big() -> InvalidSubmission:
	return InvalidSubmission("the source is larger than Preflight reads; point it at a smaller directory")


def _read_tree(root: Path) -> dict[str, str]:
	files, total = {}, 0
	for p in sorted(root.rglob("*")):
		rel = p.relative_to(root).as_posix()
		if not p.is_file() or p.is_symlink() or not _wanted(rel):
			continue
		size = p.stat().st_size
		if size > MAX_FILE_BYTES:
			continue  # big data or weights checked into a repo: not code
		total += size
		if len(files) >= MAX_FILES or total > MAX_TOTAL_BYTES:
			raise _too_big()
		try:
			files[rel] = p.read_text(encoding="utf-8")
		except UnicodeDecodeError:
			continue
	return files


def _notebook_to_script(text: str) -> str:
	"""Code cells in order; shell and magic lines (!pip, %time) become comments."""
	if not isinstance(text, str):
		raise InvalidSubmission("the notebook is not text")
	try:
		nb = json.loads(text)
	except (json.JSONDecodeError, RecursionError) as exc:
		raise InvalidSubmission("the notebook is not valid JSON") from exc
	cells = nb.get("cells") if isinstance(nb, dict) else None
	if not isinstance(cells, list):
		raise InvalidSubmission("the notebook has no list of cells")
	out = []
	for cell in cells:
		if not isinstance(cell, dict) or cell.get("cell_type") != "code":
			continue
		src = cell.get("source", "")
		if isinstance(src, str):
			lines = src.splitlines(keepends=True)
		elif isinstance(src, list) and all(isinstance(ln, str) for ln in src):
			lines = src
		else:
			raise InvalidSubmission("a notebook cell's source is not text")
		out.append("".join(("# " + ln) if ln.lstrip().startswith(("!", "%")) else ln for ln in lines))
	return "\n\n".join(out) + "\n"


def _pick_entrypoint(files: dict[str, str], wanted) -> str:
	if wanted is not None and not isinstance(wanted, str):
		raise InvalidSubmission("the entrypoint is a file path")
	if wanted:
		if wanted not in files:
			raise InvalidSubmission(f"the entrypoint {wanted!r} is not among the files")
		return wanted
	for name in ENTRYPOINT_GUESSES:
		if name in files:
			return name
	py = [p for p in files if p.endswith(".py")]
	if len(py) == 1:
		return py[0]
	raise InvalidSubmission("say which file to run: more than one Python file and none is called train.py")


def _sample(uri: str) -> Path:
	name = uri.removeprefix("sample://").removeprefix("zoo/").strip("/")
	if not re.fullmatch(r"[a-z0-9_\-]+", name):
		raise InvalidSubmission(f"unknown sample {uri!r}")
	path = ZOO / name
	if not path.is_dir():
		raise InvalidSubmission(f"unknown sample {uri!r}")
	return path


def _git(base: list[str], *args: str, env: dict, stdin: bytes | None = None) -> bytes:
	return subprocess.run([*base, *args], input=stdin, check=True, capture_output=True,
		timeout=CLONE_TIMEOUT_S, env=env).stdout  # fmt: skip


def _git_files(repo: Path, treeish: str, *, base: list[str], env: dict) -> dict[str, str]:
	"""Text files of a tree, read from the object store without a checkout or a lazy fetch.

	Blobs the partial clone left out (over the per-file cap) are listed as missing and skipped
	before anything asks for them, so nothing big is ever downloaded; symlinks and submodules
	(modes 120000 and 160000) are not regular blobs and are skipped too.
	"""
	g = [*base, "-C", str(repo)]
	missing = {
		ln[1:]
		for ln in _git(g, "rev-list", "--objects", "--missing=print", treeish, env=env).decode().splitlines()
		if ln.startswith("?")
	}
	entries = []
	for rec in _git(g, "ls-tree", "-r", "-z", "--full-tree", treeish, env=env).split(b"\0"):
		if not rec:
			continue
		meta, _, raw = rec.partition(b"\t")
		mode, kind, oid = meta.decode().split()
		try:
			rel = raw.decode("utf-8")
		except UnicodeDecodeError:
			continue
		if kind == "blob" and mode in ("100644", "100755") and oid not in missing and _wanted(rel):
			entries.append((rel, oid))
	entries.sort()
	if not entries:
		return {}
	sizes = _git(g, "cat-file", "--batch-check=%(objectname) %(objectsize)", env=env,
		stdin="".join(f"{oid}\n" for _, oid in entries).encode()).decode().split("\n")  # fmt: skip
	size_of = dict(ln.split() for ln in sizes if ln.count(" ") == 1)
	keep, total = [], 0
	for rel, oid in entries:
		size = int(size_of.get(oid, MAX_FILE_BYTES + 1))
		if size > MAX_FILE_BYTES:
			continue
		total += size
		if len(keep) >= MAX_FILES or total > MAX_TOTAL_BYTES:
			raise _too_big()
		keep.append((rel, oid))
	if not keep:
		return {}
	out = _git(g, "cat-file", "--batch", env=env, stdin="".join(f"{oid}\n" for _, oid in keep).encode())
	files, pos = {}, 0
	for rel, _oid in keep:  # each answer: "<oid> blob <size>\n<bytes>\n", in request order
		nl = out.find(b"\n", pos)
		head = out[pos:nl].split() if nl >= 0 else []
		if len(head) != 3 or head[1] != b"blob":
			raise InvalidSubmission("the repository could not be read")
		size = int(head[2])
		body = out[nl + 1 : nl + 1 + size]
		pos = nl + 1 + size + 1
		try:
			files[rel] = body.decode("utf-8")
		except UnicodeDecodeError:
			continue
	return files


def _clone(uri, commit) -> dict[str, str]:
	"""The text files of a public GitHub repo at its default branch or a full commit hash."""
	if not isinstance(uri, str) or not _GITHUB.fullmatch(uri):
		raise InvalidSubmission("repositories are read from https://github.com/<owner>/<repo>")
	if commit is not None and (not isinstance(commit, str) or not _COMMIT.fullmatch(commit)):
		raise InvalidSubmission("a commit is its full 40-character hash (git rev-parse HEAD)")
	tmp = Path(tempfile.mkdtemp(prefix="pf-src-"))
	dest = tmp / "repo"
	env = {"GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_NOSYSTEM": "1", "HOME": str(tmp), "PATH": "/usr/bin:/bin"}
	base = ["git", "-c", "core.hooksPath=/dev/null", "-c", "protocol.file.allow=never"]
	blob_cap = f"--filter=blob:limit={MAX_FILE_BYTES}"
	try:
		_git(
			base,
			"clone",
			"--depth",
			"1",
			"--no-tags",
			"--no-checkout",
			blob_cap,
			"--",
			uri,
			str(dest),
			env=env,
		)
		treeish = "HEAD"
		if commit:
			_git(
				base,
				"-C",
				str(dest),
				"fetch",
				"--depth",
				"1",
				"--no-tags",
				blob_cap,
				"origin",
				commit,
				env=env,
			)
			treeish = commit
		return _git_files(dest, treeish, base=base, env=env)
	except subprocess.TimeoutExpired as exc:
		raise InvalidSubmission("the repository took too long to clone") from exc
	except subprocess.CalledProcessError as exc:
		raise InvalidSubmission(
			"the repository could not be cloned (is it public, and is the commit right?)"
		) from exc
	finally:
		shutil.rmtree(tmp, ignore_errors=True)


def _args(raw) -> list[str]:
	if raw is None:
		return []
	if not isinstance(raw, list) or not all(isinstance(a, str) for a in raw):
		raise InvalidSubmission("the script's arguments are a list of strings")
	if len(raw) > MAX_ARGS or any(len(a) > MAX_ARG_LEN or "\0" in a for a in raw):
		raise InvalidSubmission(f"up to {MAX_ARGS} arguments of up to {MAX_ARG_LEN} characters each")
	return list(raw)


def fetch(source: dict, *, blobs=None) -> Fetched:
	"""source: the JobRequest's source dict (a Source, plus `files` for uploads and notebooks, or
	`files_ref` once the job store has moved them to its blob store)."""
	if not isinstance(source, dict):
		raise InvalidSubmission("the job has no source")
	if "files_ref" in source and "files" not in source:
		if blobs is None:
			raise InvalidSubmission("the uploaded files are no longer available; submit again")
		source = {**source, "files": blobs.get(source["files_ref"])}
	kind = source.get("kind")
	uri = source.get("uri") or ""
	if not isinstance(uri, str):
		raise InvalidSubmission("the source uri is text")
	commit = source.get("commit")
	if commit is not None and not isinstance(commit, str):
		raise InvalidSubmission("a commit is its full 40-character hash (git rev-parse HEAD)")
	args = _args(source.get("args"))
	if kind in ("sample", "template"):
		files = _read_tree(_sample(uri))
	elif kind in ("upload", "cli", "inflight"):
		files = source.get("files")
		if not isinstance(files, dict) or not all(
			isinstance(k, str) and isinstance(v, str) for k, v in files.items()
		):
			raise InvalidSubmission("an upload carries its files as {path: text}")
	elif kind == "notebook":
		files = source.get("files")
		if not isinstance(files, dict) or len(files) != 1:
			raise InvalidSubmission("a notebook upload is one .ipynb file")
		((name, text),) = files.items()
		if not isinstance(name, str) or not name.endswith(".ipynb"):
			raise InvalidSubmission("a notebook upload is one .ipynb file")
		stem = Path(name).with_suffix(".py").name
		if stem in (".py", "..py") or not re.fullmatch(r"[^/\\\0]{1,200}", stem):
			raise InvalidSubmission("the notebook's file name is not usable")
		files = {stem: _notebook_to_script(text)}
	elif kind == "repo":
		files = _clone(uri, commit)
	else:
		raise InvalidSubmission(f"unknown source kind {kind!r}")
	entry = _pick_entrypoint(
		files, source.get("entrypoint") or (next(iter(files)) if kind == "notebook" else None)
	)
	validate_files(files, entrypoint=entry)
	src = Source(
		kind=kind,
		uri=uri or None,
		commit=commit,
		content_hash=content_hash(files),
		entrypoint=entry,
		args=args,
	)
	return Fetched(files=files, entrypoint=entry, args=args, source=src)
