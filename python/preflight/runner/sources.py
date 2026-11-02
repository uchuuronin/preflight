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
_FILES_REF = re.compile(r"sha256:[0-9a-f]{64}")
MAX_ARGS = 64
MAX_ARG_LEN = 4096
CLONE_TIMEOUT_S = 60


@dataclass
class Fetched:
	files: dict[str, str]
	entrypoint: str
	args: list[str] = field(default_factory=list)
	source: Source | None = None
	# a repository too big to read whole: its entry point and the files it imports, bounded
	# (harness/closure.py): files and bytes read, imported files left out, the repository's files
	partial: dict | None = None


def content_hash(files: dict[str, str]) -> str:
	h = hashlib.sha256()
	for path in sorted(files):
		h.update(path.encode())
		h.update(b"\0")
		h.update(files[path].encode("utf-8", "surrogatepass"))
		h.update(b"\0")
	return "sha256:" + h.hexdigest()


# small text data a package ships beside its code (ogb's master.csv, Nov 2); read from a repo only
# with probes.repo_install on, under the same size caps
DATA_SUFFIXES = (".csv", ".tsv")


def _wanted(rel: str, data: bool = False) -> bool:
	suffixes = (*TEXT_SUFFIXES, *DATA_SUFFIXES) if data else TEXT_SUFFIXES
	return not (rel.startswith(".") or "/." in rel) and Path(rel).suffix in suffixes


# a repository over MAX_FILES / MAX_TOTAL_BYTES is read as its entry point and what it imports, up to
# these: the static mapper reads up to 2 MB of Python (harness/static.py MAX_SOURCE_BYTES), so a
# partial read never fails there
CLOSURE_MAX_FILES = 600
CLOSURE_MAX_BYTES = 2 * 1024 * 1024


def _too_big(
	n_files: int | None = None, n_bytes: int | None = None, *, repo: bool = False
) -> InvalidSubmission:
	limits = f"{MAX_FILES:,} files and {MAX_TOTAL_BYTES // 2**20} MB of text"
	size = (
		f" ({n_files:,} text files, {n_bytes / 2**20:.0f} MB)"
		if n_files is not None and n_bytes is not None
		else ""
	)
	if repo:
		return InvalidSubmission(
			f"the repository{size} is more than Preflight reads whole ({limits}); name its entry point, "
			"the script you run, and Preflight reads that file and the files it imports",
			code="submission.too_large",
		)
	return InvalidSubmission(f"the source{size} is more than Preflight reads ({limits}); point it at a smaller "
		"directory, or probe it on your machine", code="submission.too_large")  # fmt: skip


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


_PIP = re.compile(r"\s*[!%]\s*(?:python3?\s+-m\s+)?pip3?\s+install\s+(?P<args>.*)")
_PIP_TAKES_VALUE = {"-r", "--requirement", "-e", "--editable", "-c", "--constraint", "-i", "--index-url",
	"--extra-index-url", "-f", "--find-links", "--target", "-t", "--prefix", "--root"}  # fmt: skip


def notebook_requirements(script: str) -> list[str]:
	"""The packages a notebook's `!pip install` / `%pip install` lines ask for, in order, deduplicated.

	Options are dropped (and the values of those that take one); what's left is what pip would
	install from PyPI. The probe installs them from requirements.txt like any upload's."""
	reqs: list[str] = []
	for line in script.splitlines():
		m = _PIP.fullmatch(line.removeprefix("# "))
		if m is None:
			continue
		words = re.split(r"[#;|&`$]", m["args"], maxsplit=1)[0].split()  # this command only
		skip = False
		for w in words:
			if skip:
				skip = False
				continue
			if w in _PIP_TAKES_VALUE:
				skip = True
				continue
			w = w.strip("'\"")
			if not w or w.startswith("-"):
				continue
			if w and w not in reqs:
				reqs.append(w)
	return reqs


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


# zoo scripts for ops checks only: they need an image the runner doesn't have (torchtitan: torch 2.7)
OPS_ONLY_SAMPLES = frozenset({"torchtitan_debug"})


# what we measured from a sample, kept beside it (a record, a benchmark, a sample's tasks): not part
# of the script, and with runner.hide_sample_outputs on (off), never sent with it, so a map or a model
# reading the files can't read the answers (B, Nov 2: zoo/grpo_trl's record_nov1_a10.json)
_SAMPLE_OUTPUT = re.compile(r"(record|bench|sample|gt)_[^/]*\.json")


def _without_outputs(files: dict[str, str]) -> dict[str, str]:
	from .flags import flag

	if not flag("runner.hide_sample_outputs"):
		return files
	return {k: v for k, v in files.items() if not _SAMPLE_OUTPUT.fullmatch(k)}


def _repo_data() -> bool:
	from .flags import flag

	return flag("probes.repo_install")


def _sample(uri: str) -> Path:
	name = uri.removeprefix("sample://").removeprefix("zoo/").strip("/")
	if not re.fullmatch(r"[a-z0-9_\-]+", name):
		raise InvalidSubmission(f"unknown sample {uri!r}")
	path = ZOO / name
	if not path.is_dir() or name in OPS_ONLY_SAMPLES:
		raise InvalidSubmission(f"unknown sample {uri!r}")
	return path


def _git(base: list[str], *args: str, env: dict, stdin: bytes | None = None) -> bytes:
	return subprocess.run([*base, *args], input=stdin, check=True, capture_output=True,
		timeout=CLONE_TIMEOUT_S, env=env).stdout  # fmt: skip


def _git_files(repo: Path, treeish: str, *, base: list[str], env: dict, data: bool = False) -> dict[str, str]:
	"""The whole tree's text files (see _git_read), as before."""
	return _git_read(repo, treeish, base=base, env=env, data=data)[0]


def _git_read(
	repo: Path,
	treeish: str,
	*,
	base: list[str],
	env: dict,
	data: bool = False,
	entry: str | None = None,
	only_entry: bool = False,
) -> tuple[dict[str, str], dict | None]:
	"""Text files of a tree, read from the object store without a checkout or a lazy fetch.
	only_entry (Source.scope "entrypoint"): the entry point and what it imports, at any size.

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
		if kind == "blob" and mode in ("100644", "100755") and oid not in missing and _wanted(rel, data):
			entries.append((rel, oid))
	entries.sort()
	if not entries:
		return {}, None
	sizes = _git(g, "cat-file", "--batch-check=%(objectname) %(objectsize)", env=env,
		stdin="".join(f"{oid}\n" for _, oid in entries).encode()).decode().split("\n")  # fmt: skip
	size_of = dict(ln.split() for ln in sizes if ln.count(" ") == 1)
	keep = [(rel, oid, int(size_of.get(oid, MAX_FILE_BYTES + 1))) for rel, oid in entries]
	keep = [(rel, oid, n) for rel, oid, n in keep if n <= MAX_FILE_BYTES]
	total = sum(n for _, _, n in keep)
	if not only_entry and len(keep) <= MAX_FILES and total <= MAX_TOTAL_BYTES:
		return _cat(g, env, [(rel, oid) for rel, oid, _ in keep]), None
	# too big to read whole: the entry point and the files it imports (harness/closure.py), with
	# runner.read_closure on (off: refused, saying how big it is), or always when the visitor asked
	# for only the entry point
	from .flags import flag

	if not only_entry and not flag("runner.read_closure"):
		raise InvalidSubmission(
			f"the repository ({len(keep):,} text files, {total / 2**20:.0f} MB) is more than Preflight reads "
			f"({MAX_FILES:,} files and {MAX_TOTAL_BYTES // 2**20} MB of text); probe it on your machine",
			code="submission.too_large",
		)
	if not entry:
		raise _too_big(len(keep), total, repo=True)
	oid_of = {rel: oid for rel, oid, _ in keep}
	if entry not in oid_of:
		raise InvalidSubmission(f"the entrypoint {entry!r} is not among the repository's text files")
	if not entry.endswith(".py"):
		if only_entry:
			raise InvalidSubmission(f"reading only the entry point needs a Python file whose imports Preflight "
				f"can follow; {entry!r} isn't one, so name the Python script it runs", code="submission.invalid")  # fmt: skip
		raise InvalidSubmission(
			f"the repository ({len(keep):,} text files, {total / 2**20:.0f} MB) is more than Preflight reads whole, "
			f"and {entry!r} isn't a Python file whose imports it can follow; name the Python script it runs",
			code="submission.too_large",
		)
	from preflight.harness.closure import import_closure

	files, info = import_closure(oid_of, lambda paths: _cat(g, env, [(p, oid_of[p]) for p in paths]), entry,
		max_files=CLOSURE_MAX_FILES, max_bytes=CLOSURE_MAX_BYTES, sizes={rel: n for rel, _, n in keep})  # fmt: skip
	if entry not in files:  # it's in the tree, so it wasn't UTF-8 text
		raise InvalidSubmission(f"the entrypoint {entry!r} isn't UTF-8 text")
	info["repo_bytes"] = total
	if only_entry:
		info["asked"] = True
	return files, info


def _cat(g: list[str], env: dict, keep: list[tuple[str, str]]) -> dict[str, str]:
	"""{path: text} for (path, blob id) pairs, read from the object store in one batch."""
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


def _clone(uri, commit, *, data: bool = False) -> tuple[dict[str, str], str]:
	"""The whole repository's text files and the commit (see _clone_read)."""
	files, sha, _ = _clone_read(uri, commit, data=data)
	return files, sha


def _clone_read(
	uri, commit, entry: str | None = None, *, data: bool = False, only_entry: bool = False
) -> tuple[dict[str, str], str, dict | None]:
	"""The text files of a public GitHub repo at its default branch or a full commit hash, and the
	commit they were read at, so a later probe of the same record reads the same code. data: small
	.csv and .tsv files too (probes.repo_install)."""
	if not isinstance(uri, str) or not _GITHUB.fullmatch(uri):
		raise InvalidSubmission("repositories are read from https://github.com/<owner>/<repo>")
	if commit is not None and (not isinstance(commit, str) or not _COMMIT.fullmatch(commit)):
		raise InvalidSubmission("a commit is its full 40-character hash (git rev-parse HEAD)")
	tmp = Path(tempfile.mkdtemp(prefix="pf-src-"))
	dest = tmp / "repo"
	env = {"GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_NOSYSTEM": "1", "HOME": str(tmp), "PATH": "/usr/bin:/bin"}
	base = ["git", "-c", "core.hooksPath=/dev/null", "-c", "protocol.file.allow=never"]
	shallow = ["--depth", "1", "--no-tags", f"--filter=blob:limit={MAX_FILE_BYTES}"]
	try:
		_git(base, "clone", *shallow, "--no-checkout", "--", uri, str(dest), env=env)
		treeish = "HEAD"
		if commit:
			_git(base, "-C", str(dest), "fetch", *shallow, "origin", commit, env=env)
			treeish = commit
		sha = _git(base, "-C", str(dest), "rev-parse", f"{treeish}^{{commit}}", env=env).decode().strip()
		files, partial = _git_read(dest, treeish, base=base, env=env, data=data, entry=entry,
			only_entry=only_entry)  # fmt: skip
		return files, sha, partial
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
	for a in raw:
		try:
			a.encode("utf-8")
		except UnicodeEncodeError:  # an unpaired surrogate: no record or command line can hold it
			raise InvalidSubmission(
				"a script argument has text that isn't valid Unicode (an unpaired surrogate)",
				code="submission.invalid",
			) from None
	return list(raw)


def fetch(source: dict, *, blobs=None) -> Fetched:
	"""source: the JobRequest's source dict (a Source, plus `files` for uploads and notebooks, or
	`files_ref` once the job store has moved them to its blob store)."""
	if not isinstance(source, dict):
		raise InvalidSubmission("the job has no source")
	upload_ref = None
	if "files_ref" in source and "files" not in source:
		upload_ref = source["files_ref"]
		if blobs is None or not isinstance(upload_ref, str) or not _FILES_REF.fullmatch(upload_ref):
			raise InvalidSubmission("the uploaded files are no longer available; submit again",
				code="submission.files_gone")  # fmt: skip
		try:
			files = blobs.get(upload_ref)
		except (OSError, ValueError) as exc:
			raise InvalidSubmission("the uploaded files are no longer available; submit again",
				code="submission.files_gone") from exc  # fmt: skip
		source = {**source, "files": files}
	kind = source.get("kind")
	uri = source.get("uri") or ""
	if not isinstance(uri, str):
		raise InvalidSubmission("the source uri is text")
	commit = source.get("commit")
	if commit is not None and not isinstance(commit, str):
		raise InvalidSubmission("a commit is its full 40-character hash (git rev-parse HEAD)")
	args = _args(source.get("args"))
	scope = source.get("scope")
	if scope not in (None, "repo", "entrypoint"):
		raise InvalidSubmission("source.scope is repo or entrypoint", code="submission.invalid")
	only_entry = scope == "entrypoint"
	want = source.get("entrypoint")
	if only_entry and not (isinstance(want, str) and want):
		raise InvalidSubmission("reading only the entry point needs source.entrypoint, the script you run",
			code="submission.invalid")  # fmt: skip
	extra = _extra_packages(source.get("extra_packages"))
	partial = None
	if kind in ("sample", "template"):
		files = _without_outputs(_read_tree(_sample(uri)))
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
		script = _notebook_to_script(text)
		files = {stem: script}
		reqs = notebook_requirements(script)
		if reqs:  # the notebook's !pip install lines, installed like any upload's requirements.txt
			files["requirements.txt"] = "\n".join(reqs) + "\n"
	elif kind == "repo":
		files, commit, partial = _clone_read(uri, commit, want if isinstance(want, str) and want else None,
			data=_repo_data(), only_entry=only_entry)  # fmt: skip
	else:
		raise InvalidSubmission(f"unknown source kind {kind!r}")
	if only_entry and kind != "repo":  # files already in hand: the same walk over them
		files, partial = _only_entry(files, want)
	entry = _pick_entrypoint(
		files, source.get("entrypoint") or (next(iter(files)) if kind == "notebook" else None)
	)
	validate_files(files, entrypoint=entry)
	if kind in ("upload", "cli", "inflight", "notebook"):
		# never the visitor's own uri: it says where the files are kept, so a later probe of this
		# record reads exactly them (record_jobs.refetch), and only the job store sets that
		uri = f"upload://{upload_ref}" if upload_ref is not None else ""
	src = Source(
		kind=kind,
		uri=uri or None,
		commit=commit,
		content_hash=content_hash(files),
		entrypoint=entry,
		args=args,
		scope="entrypoint" if only_entry else None,
		extra_packages=extra or None,
	)
	return Fetched(files=files, entrypoint=entry, args=args, source=src, partial=partial)


def _extra_packages(raw) -> list[str]:
	"""The job's extra packages, as the PyPI names on the list (JobRequest.extra_packages, 0.6.3); a name
	off the list, or one with no wheels, refuses the job before anything is read or run, naming it."""
	from preflight.recorder.requirements import PackageRefused, extra_requirements

	if not raw:
		return []
	try:
		lines = extra_requirements(raw)
	except PackageRefused as exc:
		if exc.code == "missing_package.no_wheels":
			raise InvalidSubmission(str(exc), code="missing_package.no_wheels") from None
		raise InvalidSubmission(str(exc), code="missing_package.not_allowed") from None
	return [_REQ_NAME.match(line)[0] for line in lines]


_REQ_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def extra_lines(src: Source | None) -> tuple[str, ...]:
	"""What the probe installs for a fetched source's extra packages (the list's lines, with any pin)."""
	from preflight.recorder.requirements import extra_requirements

	names = (src.extra_packages if src is not None else None) or []
	return tuple(extra_requirements(list(names)))


def _only_entry(files: dict[str, str], entry) -> tuple[dict[str, str], dict]:
	"""An upload's or sample's files cut to the entry point and what it imports (Source.scope)."""
	from preflight.harness.closure import import_closure

	if not isinstance(entry, str) or entry not in files:
		raise InvalidSubmission(f"the entrypoint {entry!r} is not among the files")
	if not entry.endswith(".py"):
		raise InvalidSubmission(f"reading only the entry point needs a Python file whose imports Preflight "
			f"can follow; {entry!r} isn't one, so name the Python script it runs", code="submission.invalid")  # fmt: skip
	sizes = {p: len(t.encode("utf-8", "surrogatepass")) for p, t in files.items()}
	got, info = import_closure(files, lambda paths: {p: files[p] for p in paths}, entry,
		max_files=CLOSURE_MAX_FILES, max_bytes=CLOSURE_MAX_BYTES, sizes=sizes)  # fmt: skip
	info["repo_bytes"] = sum(sizes.values())
	info["asked"] = True
	return got, info
