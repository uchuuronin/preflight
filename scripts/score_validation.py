"""Fill the README's validation table from full-run records (Chiya, Oct 31: built now so Nov 2 runs without setup).

	uv run --extra probe python scripts/score_validation.py <manifest.json> [--divergence report.json]
		[--out table.md] [--readme README.md]

The manifest format is in python/preflight/corpus/validate.py. Paths inside it are relative to the manifest.
With --readme, the table replaces the block between <!-- validation:start --> and <!-- validation:end -->.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from preflight.corpus import validate


def main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument("manifest", type=Path)
	ap.add_argument("--divergence", type=Path, help="the divergence classifier's held-out report (JSON)")
	ap.add_argument("--out", type=Path)
	ap.add_argument("--readme", type=Path)
	args = ap.parse_args(argv)
	manifest = json.loads(args.manifest.read_text())
	div = json.loads(args.divergence.read_text()) if args.divergence else None
	rows, runs = validate.build(manifest, args.manifest.parent, div)
	stamp = f"Scored {datetime.now(UTC):%Y-%m-%d %H:%M} UTC from {args.manifest.name}: {len(runs)} runs."
	table = validate.markdown(rows, stamp)
	if args.out:
		args.out.write_text(table)
	if args.readme and not validate.into_readme(args.readme, table):
		print(
			f"{args.readme} has no {validate.START} ... {validate.END} block; nothing written there",
			file=sys.stderr,
		)
		return 1
	print(table)
	return 0


if __name__ == "__main__":
	sys.exit(main())
