"""Write web/src/workspace/record/prices.json from B's price table (python/preflight/engine/prices).

The cost readout's rate box shows the table's rate, source and date, and rescales every dollar
readout when the visitor types their own rate; it needs the same rows the engine priced with.
Generated so there is one table, not two. Run from the repo root:

	uv run python web/scripts/make_price_table.py          # write
	uv run python web/scripts/make_price_table.py --check  # fail if the file is stale
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from preflight.engine import prices

OUT = Path(__file__).resolve().parents[1] / "src" / "workspace" / "record" / "prices.json"


def build() -> dict:
	return {
		"version": prices.PRICE_TABLE_VERSION,
		"gpu": [
			{
				"id": r.id,
				"gpu": r.gpu,
				"provider": r.provider,
				"region": r.region,
				"kind": r.kind,
				"usd_per_hour": r.usd_per_hour,
				"as_of": r.as_of,
				"source": r.source,
				"confidence": r.confidence,
			}
			for r in prices.GPU_RATES
		],
		"tokens": [
			{
				"id": r.id,
				"model": r.model,
				"provider": r.provider,
				"region": r.region,
				"input_per_m": r.input_per_m,
				"output_per_m": r.output_per_m,
				"cached_input_per_m": r.cached_input_per_m,
				"as_of": r.as_of,
				"source": r.source,
				"confidence": r.confidence,
			}
			for r in prices.TOKEN_RATES
		],
	}


def main(argv: list[str]) -> int:
	text = json.dumps(build(), indent="\t", sort_keys=True) + "\n"
	if "--check" in argv:
		if not OUT.exists() or OUT.read_text() != text:
			print(f"{OUT} is stale: run uv run python web/scripts/make_price_table.py", file=sys.stderr)
			return 1
		return 0
	OUT.write_text(text)
	print(f"wrote {OUT}")
	return 0


if __name__ == "__main__":
	sys.exit(main(sys.argv[1:]))
