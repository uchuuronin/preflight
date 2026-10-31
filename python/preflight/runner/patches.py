"""JobRequest.patch (contract 0.5.0): a unified diff from C's fix loop, applied to the record's code
before a probe or verify job.

Only edits to files the record already has: a diff that creates, deletes or renames a file, changes
a mode, or touches a binary file is refused (`patch.not_allowed`); one that doesn't apply cleanly is
`patch.does_not_apply`. `git apply` does the work in a scratch directory holding only the record's
files, so a diff can't reach anything else.
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


def _git(args, cwd: Path, diff: str) -> subprocess.CompletedProcess:
	return subprocess.run(["git", "apply", *args, "-"], cwd=cwd, input=diff, capture_output=True, text=True,
		timeout=GIT_TIMEOUT_S, env={"PATH": "/usr/bin:/bin:/usr/local/bin", "GIT_CONFIG_NOSYSTEM": "1",
		"HOME": str(cwd)})  # fmt: skip


def apply_patch(files: dict[str, str], diff: str) -> dict[str, str]:
	"""The files with the diff applied; only the files it edits change."""
	with tempfile.TemporaryDirectory(prefix="preflight-patch-") as d:
		root = Path(d)
		for rel, text in files.items():
			p = (root / rel).resolve()
			if root.resolve() not in p.parents:
				continue
			p.parent.mkdir(parents=True, exist_ok=True)
			p.write_text(text, encoding="utf-8", newline="")
		summary = _git(["--summary", "--numstat"], root, diff)
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
		check = _git(["--check"], root, diff)
		if check.returncode != 0:
			raise InvalidSubmission(f"the patch doesn't apply: {check.stderr.strip()[:300]}",
				code="patch.does_not_apply")  # fmt: skip
		done = _git([], root, diff)
		if done.returncode != 0:
			raise InvalidSubmission(f"the patch doesn't apply: {done.stderr.strip()[:300]}",
				code="patch.does_not_apply")  # fmt: skip
		out = dict(files)
		for rel in touched:
			out[rel] = (root / rel).read_text(encoding="utf-8")
		return out
