"""Fill the README's validation table from full-run records (Chiya, Oct 31: built now so Nov 2 runs without setup).

	uv run --extra probe python scripts/score_validation.py <manifest.json> [--divergence report.json]
		[--corpus store ...] [--out table.md] [--readme README.md]

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
from preflight.engine import api


def main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument("manifest", type=Path)
	ap.add_argument(
		"--divergence",
		type=Path,
		default=Path(__file__).resolve().parents[1]
		/ "python"
		/ "preflight"
		/ "corpus"
		/ "reports"
		/ "divergence_nov1.json",
		help="the divergence classifier's held-out report (JSON); default: the final Nov 1 fit",
	)
	ap.add_argument(
		"--corpus",
		type=Path,
		nargs="*",
		default=[],
		help="kernel stores for the cross-GPU row, added to the manifest's own (an export lives outside the repo)",
	)
	ap.add_argument("--out", type=Path)
	ap.add_argument("--readme", type=Path)
	args = ap.parse_args(argv)
	manifest = json.loads(args.manifest.read_text())
	if args.corpus:
		manifest["corpus"] = [*manifest.get("corpus", []), *(str(p.resolve()) for p in args.corpus)]
	div = json.loads(args.divergence.read_text()) if args.divergence else None
	# Nov 1: Monday's runs are predicted under production's switches (engine.rl goes on there in round five)
	flags = validate.production_flags(Path(__file__).resolve().parents[1])
	rows, runs = validate.build(manifest, args.manifest.parent, div, flags)
	# some rows still score; a run with no record yet is neither held out nor used
	have = [
		r
		for r in manifest.get("runs", [])
		if r.get("record") and (args.manifest.parent / r["record"]).exists()
	]
	held = sum(1 for r in have if "*" not in (r.get("calibrates") or ()))
	models = validate.models_used(manifest.get("runs", []), args.manifest.parent, flags)
	# the engine features these flags switch (engine.memory gates the runner's fill, not the scoring)
	on = sorted(k for k in api.FLAG_OF.values() if flags.get(k)) or ["none"]
	waiting = [r.script for r in runs if any(": not measured, no " in n for n in r.notes)]
	scored = f"{len(runs) - len(waiting)} runs scored" + (
		f"; not measured yet: {', '.join(waiting)} ({len(waiting)})" if waiting else ""
	)
	stamp = (
		f"Scored {datetime.now(UTC):%Y-%m-%d %H:%M} UTC from {args.manifest.name}: {scored}; {held} of the {len(have)} with records held out of at least one row. "
		f"{validate.calibration_note(manifest)} Engine flags on (flags.json with flags.production.json): "
		f"{', '.join(on)}. Models: {', '.join(models)}."
	)
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
