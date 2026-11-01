"""JobRequest.patch (contract 0.5.0): a unified diff from C's fix loop, applied to the record's code
before a probe or verify job.

Only edits to files the record already has: a diff that creates, deletes or renames a file, changes
a mode, or touches a binary file is refused (`patch.not_allowed`); one that doesn't apply cleanly is
`patch.does_not_apply`. `git apply` does the work in a scratch directory holding only the record's
files, with no git configuration read from anywhere (the record's files may not hold any), so a
diff can't reach anything else. Probes run on patched code carry extras["patched"] = 1.
"""

import hashlib
import subprocess
import tempfile
from pathlib import Path

from preflight.probes.local import InvalidSubmission

GIT_TIMEOUT_S = 30


def patch_id(diff: str) -> str:
	"""How the verify finding names the patch it ran."""
	return "sha256:" + hashlib.sha256(diff.encode("utf-8", "surrogatepass")).hexdigest()[:16]


def _git(args, cwd: Path, home: Path, diff: str) -> subprocess.CompletedProcess:
	"""git apply with no configuration from anywhere: not the system's, not a user's, not the files'
	(no repository above the work dir is found, no attributes are read)."""
	env = {"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(home), "GIT_CONFIG_NOSYSTEM": "1",
		"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_ATTR_NOSYSTEM": "1", "GIT_CEILING_DIRECTORIES": str(cwd.parent),
		"LC_ALL": "C.UTF-8"}  # fmt: skip
	try:
		return subprocess.run(["git", "apply", *args, "-"], cwd=cwd, input=diff, capture_output=True, text=True,
			encoding="utf-8", errors="replace", timeout=GIT_TIMEOUT_S, env=env)  # fmt: skip
	except subprocess.TimeoutExpired:
		raise InvalidSubmission("the patch took too long to apply", code="patch.does_not_apply") from None


def apply_patch(files: dict[str, str], diff: str) -> dict[str, str]:
	"""The files with the diff applied; only the files it edits change."""
	git_files = {".git", ".gitattributes", ".gitmodules", ".gitconfig"}  # .github/ and .gitignore are fine
	if any(part in git_files for rel in files for part in Path(rel).parts):
		raise InvalidSubmission("the record's files hold git settings, which a patch can't run beside",
			code="patch.not_allowed")  # fmt: skip
	with tempfile.TemporaryDirectory(prefix="preflight-patch-") as d:
		root, home = Path(d) / "work", Path(d) / "home"
		root.mkdir()
		home.mkdir()
		try:
			for rel, text in files.items():
				p = (root / rel).resolve()
				if root.resolve() not in p.parents:
					continue
				p.parent.mkdir(parents=True, exist_ok=True)
				p.write_text(text, encoding="utf-8", newline="")
		except OSError as exc:  # a file and a directory of the same name, say
			raise InvalidSubmission(
				f"the record's files can't be laid out to patch: {exc}", code="patch.not_allowed"
			) from None
		summary = _git(["--summary", "--numstat"], root, home, diff)
		if summary.returncode != 0:
			raise InvalidSubmission(f"the patch doesn't apply: {summary.stderr.strip()[:300]}",
				code="patch.does_not_apply")  # fmt: skip
		touched = []
		for line in summary.stdout.splitlines():
			line = line.strip()
			if line.startswith(("create mode", "delete mode", "rename ", "copy ", "mode change")):
				raise InvalidSubmission(f"the patch may only edit existing files ({line[:120]})",
					code="patch.not_allowed")  # fmt: skip
			parts = line.split("\t")
			if len(parts) == 3:
				if parts[0] == "-":  # binary
					raise InvalidSubmission("the patch may only edit text files", code="patch.not_allowed")
				touched.append(parts[2])
		if not touched:
			raise InvalidSubmission("the patch changes nothing", code="patch.does_not_apply")
		unknown = [t for t in touched if t not in files]
		if unknown:
			raise InvalidSubmission(f"the patch edits files the record doesn't have: {unknown[:3]}",
				code="patch.not_allowed")  # fmt: skip
		check = _git(["--check"], root, home, diff)
		if check.returncode != 0:
			raise InvalidSubmission(f"the patch doesn't apply: {check.stderr.strip()[:300]}",
				code="patch.does_not_apply")  # fmt: skip
		done = _git([], root, home, diff)
		if done.returncode != 0:
			raise InvalidSubmission(f"the patch doesn't apply: {done.stderr.strip()[:300]}",
				code="patch.does_not_apply")  # fmt: skip
		out = dict(files)
		for rel in touched:
			with open(root / rel, encoding="utf-8", newline="") as f:  # line endings as they are (CRLF too)
				out[rel] = f.read()
		return out


def relocate_knobs(sm, before: dict[str, str], after: dict[str, str]):
	"""The record's own script map (static and Muse Spark knobs alike) with each knob's line moved to
	where it sits in the patched files: an unchanged line by the diff's alignment; a line the patch
	edited, to the one new line in that hunk that still holds the knob's whole expression. A knob that
	can't be placed that way loses its line, so its override says it can't set it rather than
	landing on another literal."""
	import difflib
	import re

	def holds(line: str, expr: str) -> bool:  # the whole expression: "default=3" isn't in "default=32"
		return re.search(r"(?<![\w.])" + re.escape(expr) + r"(?![\w.])", line) is not None

	maps: dict[str, dict[int, int | None]] = {}
	for f in {k.location.file for k in sm.knobs if k.location is not None and k.location.line}:
		if f not in after or f not in before or after[f] == before[f]:
			continue
		a, b = before[f].splitlines(), after[f].splitlines()
		m: dict[int, int | None] = {}
		for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
			for k in range(i1, i2):
				m[k + 1] = (j1 + (k - i1) + 1) if tag == "equal" else None
			if tag == "replace":
				m.update({("hunk", k + 1): (j1, j2) for k in range(i1, i2)})
		maps[f] = m
	knobs = []
	for k in sm.knobs:
		loc = k.location
		if loc is not None and loc.line and loc.file in maps:
			m = maps[loc.file]
			new = m.get(loc.line)
			lines = after[loc.file].splitlines()
			if new is None and ("hunk", loc.line) in m and loc.expr:
				# an edited line: the one new line in its hunk that still holds the whole expression
				# ("batch_size=32", "default=16"), not just the name, which a sibling may share
				j1, j2 = m[("hunk", loc.line)]
				hits = [j + 1 for j in range(j1, j2) if holds(lines[j], loc.expr)]
				new = hits[0] if len(hits) == 1 else None
			if new is not None and loc.expr and not holds(lines[new - 1], loc.expr):
				new = None  # the line moved but no longer holds the knob
			k = k.model_copy(update={"location": loc.model_copy(update={"line": new})})
		knobs.append(k)
	return sm.model_copy(update={"knobs": knobs})
