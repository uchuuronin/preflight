"""The files an entry point imports, followed through the repository, bounded (D, Nov 2).

	files, info = import_closure(paths, read, "recipes/lora_finetune_single_device.py")

A repository too big to read whole (lm-evaluation-harness, vLLM) or to map whole (torchtune: over
2 MB of Python) is read as its entry point plus the files that entry point imports, transitively,
nearest first, until a file, byte or lookup budget is spent. The cut is deterministic: the same
repository and entry point always read the same files (no clock), so a record's content hash doesn't
change between reads. Nothing outside the repository is
followed (an import of torch resolves to no file here), nothing is executed, and a file that
doesn't parse is kept but not followed. Resolution is the static, ordinary case: `import a.b` and
`from a.b import c` to `a/b.py` or `a/b/__init__.py` (and `a/b/c.py` for the `from` form), each
package's `__init__.py` on the way, relative imports from the importing file's package, under the
first of the roots Python would search that has the module: the entry point's own folder, the
repository's top, `src/`. Imports under `if TYPE_CHECKING:` aren't followed, except in a package's `__init__.py`
(a lazy package's real modules, B's must-fix 4). Imports made by string
(importlib, registries, entry points in pyproject) aren't seen.

The repository is untrusted: a dotted name, an import's names and a file's imports are capped, and
a path is only built under folders the repository has, so a hostile file can't make the work grow
faster than its own size (paths over MAX_PATH aren't followed).

`read(paths)` returns {path: text} for the paths asked (one batch per breadth-first layer, so a git
object store is read once per layer, never whole).
"""

import ast
from collections.abc import Callable, Iterable

# read beside the code when they're at the repository's top (a probe installs from them); never followed
PROJECT_FILES = ("requirements.txt", "pyproject.toml", "setup.py", "setup.cfg")
MAX_PARTS = 32  # guess: dotted parts of one module name (real packages nest a handful)
MAX_NAMES = 64  # names in one `from x import ...` followed as possible submodules
MAX_IMPORTS = 2_000  # import statements followed in one file
MAX_LINE = 4_000  # guess: characters in a line before a file's imports aren't followed
# guess: import resolutions in the whole walk, each a few dict lookups (MAX_PARTS deep at most), so the
# walk's work is bounded by the bytes read and this, not a clock
MAX_LOOKUPS = 200_000
# guess: characters in a file's path; longer ones aren't followed (each lookup builds and hashes paths, so a
# 1 MB path made 127,000 lookups cost 105 s of CPU, review Nov 3); real repositories' paths are under 300
MAX_PATH = 1_024


def _type_checking(node: ast.AST) -> bool:
	t = node.test if isinstance(node, ast.If) else None
	return (isinstance(t, ast.Name) and t.id == "TYPE_CHECKING") or (
		isinstance(t, ast.Attribute) and t.attr == "TYPE_CHECKING"
	)


def _imports(path: str, text: str) -> list[tuple[list[str], int, list[str]]]:
	"""(module parts, level, names) per import statement, capped; [] when the file doesn't parse, or
	has a line over MAX_LINE (CPython's parser takes seconds on one very long dotted name)."""
	# backslash continuations joined first, so a dotted name can't hide across short lines
	if any(len(line) > MAX_LINE for line in text.replace("\\\r\n", "").replace("\\\n", "").splitlines()):
		return []
	try:
		tree = ast.parse(text, filename=path)
	except (SyntaxError, ValueError, RecursionError, MemoryError):
		return []
	out: list[tuple[list[str], int, list[str]]] = []
	stack: list[ast.AST] = [tree]
	while stack and len(out) < MAX_IMPORTS:
		node = stack.pop()
		if isinstance(node, ast.Import):
			out += [(a.name.split(".")[:MAX_PARTS], 0, []) for a in node.names[:MAX_NAMES]]
		elif isinstance(node, ast.ImportFrom):
			parts = node.module.split(".")[:MAX_PARTS] if node.module else []
			names = [a.name for a in node.names if a.name != "*"][:MAX_NAMES]
			out.append((parts, min(node.level or 0, MAX_PARTS), names))
		elif _type_checking(node):
			stack.extend(node.orelse)  # the `if TYPE_CHECKING:` body only runs under a type checker,
			if path.endswith(
				"__init__.py"
			):  # except a lazy package's: its body names the real modules (B, Nov 3)
				stack.extend(node.body)
		else:
			stack.extend(ast.iter_child_nodes(node))
	return out[:MAX_IMPORTS]


DOCS = (".md", ".rst", ".html", ".ipynb")  # beside the code but not what it opens: never read as data
MAX_STRINGS = 500  # guess: string literals a file is read for paths it may open


def _path_strings(path: str, text: str) -> list[str]:
	"""String literals in a file that could name a file it opens (a slash or a dot, no NUL, under 300 characters),
	capped; [] when the file doesn't parse."""
	if any(len(line) > MAX_LINE for line in text.splitlines()):
		return []
	try:
		tree = ast.parse(text, filename=path)
	except (SyntaxError, ValueError, RecursionError, MemoryError):
		return []
	out = []
	for node in ast.walk(tree):
		if isinstance(node, ast.Constant) and isinstance(node.value, str):
			v = node.value.strip()
			if 0 < len(v) < 300 and "\0" not in v and ("/" in v or "." in v) and "\n" not in v:
				out.append(v)
				if len(out) >= MAX_STRINGS:
					break
	return out


def _as_repo_path(name: str, base: str) -> str | None:
	"""A string as a path inside the repository, relative to `base` (a folder), or None if it leaves it."""
	parts: list[str] = [p for p in base.split("/") if p] if not name.startswith("/") else []
	for part in name.split("/"):
		if part in ("", "."):
			continue
		if part == "..":
			if not parts:
				return None
			parts.pop()
		else:
			parts.append(part)
	return "/".join(parts) or None


def _resolve(parts: list[str], names: list[str], prefix: str, have: set[str], dirs: set[str]) -> list[str]:
	"""Files under one root that an import loads; [] when this root doesn't have the module. Paths
	are built only down folders the repository has."""
	out: list[str] = []
	path = prefix
	for i, part in enumerate(parts):
		path = f"{path}/{part}" if path else part
		last = i == len(parts) - 1
		if f"{path}/__init__.py" in have:  # a package: its __init__.py runs
			out.append(f"{path}/__init__.py")
			continue
		if last and f"{path}.py" in have:  # a module beats a folder of the same name with no __init__
			return [*out, f"{path}.py"]
		if path in dirs:  # a namespace folder
			continue
		return []
	head = f"{path}/" if path else ""
	for n in names:  # `from a.b import c`, `from . import c`: c may be a module of its own
		for c in (f"{head}{n}.py", f"{head}{n}/__init__.py"):
			if c in have:
				out.append(c)
	return out


def import_closure(
	paths: Iterable[str],
	read: Callable[[list[str]], dict[str, str]],
	entry: str,
	*,
	max_files: int,
	max_bytes: int,
	sizes: dict[str, int] | None = None,
	max_lookups: int = MAX_LOOKUPS,
	args: Iterable[str] = (),
) -> tuple[dict[str, str], dict]:
	"""({path: text} of the entry point and what it imports, nearest first; what happened).

	`paths`: every file in the repository; `sizes`: their sizes when known before reading (a git
	tree's), so a file over the budget is never read (the entry point is always read). info: files,
	Python files imported (`imported`, the entry point and project files not counted), bytes, how
	many imported files were found but left out for the budget (`skipped`, a floor: what they
	import is never seen), and `cut`: "lookups" when the lookup budget ran out (the rest of the walk
	is skipped and counted), else None.

	Files the code may open (B, Nov 3: `--config configs/a.yaml`, a `data/vocab.json` beside a module) are read
	too, after the imports and inside the same budget, but never followed: repo paths named in `args`, string
	literals in the Python read that name a repo file (from the repo's top, the file's folder or the entry
	point's), and the other text files in the folders already read. `data_read` counts them, `data_skipped` the
	ones the budget left out."""
	have = {p for p in paths if len(p) <= MAX_PATH}
	if len(entry) > MAX_PATH:  # read, never followed
		text = read([entry]).get(entry)
		got = {entry: text} if text is not None else {}
		n = len(text.encode("utf-8", "surrogatepass")) if text is not None else 0
		return got, {
			"files": len(got),
			"imported": 0,
			"bytes": n,
			"skipped": 0,
			"cut": None,
			"repo_files": len(have),
		}
	dirs: set[str] = set()
	for p in have:
		parts = p.split("/")
		dirs.update("/".join(parts[:k]) for k in range(1, len(parts)))
	entry_dir = entry.rsplit("/", 1)[0] if "/" in entry else ""
	roots = list(dict.fromkeys([entry_dir, "", *(["src"] if "src" in dirs else [])]))
	out: dict[str, str] = {}
	total, skipped, lookups, cut = 0, 0, 0, None
	named: list[
		str
	] = []  # files the code may open: arguments first, then string literals (read after the imports)
	for a in args:
		for v in (a, a.partition("=")[2]):
			for base in dict.fromkeys([entry_dir, ""]):
				c = _as_repo_path(v, base) if v and len(v) < 300 else None
				if c and c in have and c != entry:
					named.append(c)
	extra = [p for p in PROJECT_FILES if p in have and p != entry]
	seen = {entry, *extra}
	layer = [entry, *extra]
	while layer:
		if cut:
			skipped += sum(p not in PROJECT_FILES for p in layer)
			break
		want = []
		for p in layer:
			size = (sizes or {}).get(p)
			over = size is not None and total + size > max_bytes
			if p != entry and (len(out) + len(want) >= max_files or over):
				skipped += p not in PROJECT_FILES  # "imported files left out": project files aren't
				continue
			want.append(p)
		got = read(want) if want else {}
		nxt: list[str] = []
		for p in want:
			text = got.get(p)
			if text is None:
				continue
			n = len(text.encode("utf-8", "surrogatepass"))
			if p != entry and total + n > max_bytes:
				skipped += p not in PROJECT_FILES
				continue
			out[p] = text
			total += n
			if cut or not p.endswith(".py") or p in PROJECT_FILES:  # after a cut, read but not parsed
				continue
			pkg = p.rsplit("/", 1)[0] if "/" in p else ""
			for lit in _path_strings(p, text):
				for base in dict.fromkeys(["", pkg, entry_dir]):
					c = _as_repo_path(lit, base)
					if c and c in have and c not in seen and not c.endswith(".py"):
						named.append(c)
			for parts, level, names in _imports(p, text):
				lookups += (1 + len(names)) * (1 if level else len(roots))  # _resolve's calls and names tried
				if lookups > max_lookups:
					cut = "lookups"  # files later in this layer are still read; their imports aren't parsed
					break
				if level:
					up = pkg.split("/") if pkg else []
					if level - 1 > len(up):
						continue
					found = _resolve(parts, names, "/".join(up[: len(up) - (level - 1)]), have, dirs)
				else:
					found = next(
						(r for r in (_resolve(parts, names, root, have, dirs) for root in roots) if r), []
					)
				for c in found:
					if c not in seen:
						seen.add(c)
						nxt.append(c)
		layer = nxt
	# the files the code may open, then the text files beside the Python read, inside what's left of the budget
	folders = sorted({p.rsplit("/", 1)[0] if "/" in p else "" for p in out if p.endswith(".py")})
	beside = sorted(
		c
		for c in have
		if not c.endswith((".py", *DOCS)) and (c.rsplit("/", 1)[0] if "/" in c else "") in folders
	)
	# counted before the data pass: an argument naming a .py file is read as data, not as an import (review)
	imported = sum(1 for p in out if p.endswith(".py") and p != entry and p not in PROJECT_FILES)
	data_files: list[str] = []
	left_out: list[str] = []
	want, known = [], {}
	for c in dict.fromkeys([*named, *beside]):
		if c in out:
			continue
		size = (sizes or {}).get(c)
		if len(out) + len(want) >= max_files or (size is not None and total + size > max_bytes):
			left_out.append(c)
			continue
		want.append(c)
		known[c] = size
		total += size or 0
	got = read(want) if want else {}
	for c in want:
		text = got.get(c)
		if text is None:
			continue
		n = len(text.encode("utf-8", "surrogatepass"))
		if known[c] is None:  # no size before the read: checked now, as the import walk does (review)
			if total + n > max_bytes:
				left_out.append(c)
				continue
			total += n
		out[c] = text
		data_files.append(c)
	return out, {"files": len(out), "imported": imported, "bytes": total, "skipped": skipped,
		"cut": cut, "repo_files": len(have), "data_read": len(data_files), "data_skipped": len(left_out),
		"data_files": data_files, "data_left_out": left_out[:5]}  # fmt: skip
