"""Write web/src/workspace/record/gpus.json from B's GPU table (python/preflight/engine/gpus.py).

The browser needs each GPU's memory for the capacity ring and the fits line, and must resolve
the same names the engine does (canonical names, aliases, Modal names). Generated so there is
one table, not two. Run from the repo root:

	uv run python web/scripts/make_gpu_table.py          # write
	uv run python web/scripts/make_gpu_table.py --check  # fail if the file is stale
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from preflight.engine import gpus

OUT = Path(__file__).resolve().parents[1] / "src" / "workspace" / "record" / "gpus.json"
GIB = 2**30


def build() -> dict:
	table: dict[str, dict] = {}
	aliases: dict[str, str] = {}
	for name, spec in sorted(gpus.GPUS.items()):
		usable = spec.usable_gib if spec.usable_gib is not None else spec.memory_gib
		table[name] = {
			"vendor": spec.vendor,
			"arch": spec.arch,
			"memory_bytes": round(usable * GIB),
			"confidence": spec.confidence,
		}
		aliases[name.lower()] = name
		for a in spec.aliases:
			aliases[a.lower()] = name
	for canonical, modal in gpus.MODAL_GPU.items():
		if canonical in table:
			aliases[modal.lower().rstrip("!")] = canonical
	return {
		"_generated_by": "web/scripts/make_gpu_table.py from python/preflight/engine/gpus.py",
		"gpus": table,
		"aliases": dict(sorted(aliases.items())),
	}


def main() -> int:
	text = json.dumps(build(), indent="\t") + "\n"
	if "--check" in sys.argv:
		if not OUT.exists() or OUT.read_text() != text:
			print(f"{OUT} is stale: run uv run python web/scripts/make_gpu_table.py", file=sys.stderr)
			return 1
		return 0
	OUT.write_text(text)
	print(f"wrote {OUT} ({len(text)} bytes)")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
