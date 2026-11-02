"""Check the published unseen-scripts list against the hash committed before the validation runs.

	uv run python scripts/check_unseen_hash.py unseen-scripts.md

Prints the file's SHA-256 and the committed one (docs/validation/unseen-hash.txt, committed in 4ca3704, Nov 1 2026,
01:46 EDT), and exits 0 when they match, 1 when they don't, 2 when the file isn't there. The hash is of the file's
exact bytes, as `sha256sum` computes it, so the list must be checked as published, without re-saving or changing line
endings.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

HASH_FILE = Path(__file__).resolve().parents[1] / "docs" / "validation" / "unseen-hash.txt"
HASH_COMMIT = "4ca3704"  # cited: git log -- docs/validation/unseen-hash.txt


def committed_hash(text: str) -> str:
	"""The one 64-hex-digit value in the hash file."""
	found = set(re.findall(r"\b[0-9a-f]{64}\b", text))
	if len(found) != 1:
		raise ValueError(f"expected one SHA-256 in the hash file, found {len(found)}")
	return found.pop()


def main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument("list_file", type=Path, help="the published unseen-scripts list")
	ap.add_argument("--hash-file", type=Path, default=HASH_FILE)
	args = ap.parse_args(argv)
	if not args.list_file.is_file():
		print(f"{args.list_file}: no such file", file=sys.stderr)
		return 2  # review, Nov 2: a missing file isn't a mismatch
	got = hashlib.sha256(args.list_file.read_bytes()).hexdigest()
	want = committed_hash(args.hash_file.read_text())
	print(f"{args.list_file}: {got}")
	print(f"committed ({HASH_COMMIT}): {want}")
	if got == want:
		print("match: this is the list fixed before the validation runs")
		return 0
	print("no match: this file is not the list whose hash was committed (or its bytes changed)")
	return 1


if __name__ == "__main__":
	sys.exit(main())
