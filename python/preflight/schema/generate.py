"""Generate JSON Schema (schema/*.schema.json) and TypeScript types (schema/ts/contract.ts).

	uv run python -m preflight.schema.generate          # JSON Schema + TS
	uv run python -m preflight.schema.generate --check  # exit 1 if anything on disk is stale

TypeScript comes from schema/contract.schema.json through json-schema-to-typescript
(scripts/gen-ts.mjs), so the two can never drift from the Pydantic models.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel
from pydantic.json_schema import models_json_schema

from .events import JobEvent, JobRequest
from .prediction import Estimate, Grid, KnobImpact, Objective, Prediction, Target
from .record import SCHEMA_VERSION, FlightRecord
from .script_map import ScriptMap

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "schema"

TOP: dict[str, type[BaseModel]] = {
	"flight-record": FlightRecord,
	"job-event": JobEvent,
	"job-request": JobRequest,
	"script-map": ScriptMap,
}

# everything the TypeScript side imports by name
CONTRACT: list[type[BaseModel]] = [
	FlightRecord,
	JobEvent,
	JobRequest,
	ScriptMap,
	Prediction,
	Estimate,
	Grid,
	Target,
	Objective,
	KnobImpact,
]


def _schema(model: type[BaseModel]) -> dict:
	s = model.model_json_schema(mode="serialization")
	s["$schema"] = "https://json-schema.org/draft/2020-12/schema"
	s["$id"] = f"https://preflight.dev/schema/{SCHEMA_VERSION}/{model.__name__}"
	return s


def contract_schema() -> dict:
	_, defs = models_json_schema([(m, "serialization") for m in CONTRACT], ref_template="#/$defs/{model}")
	pool = defs["$defs"]
	return {
		"$schema": "https://json-schema.org/draft/2020-12/schema",
		"$id": f"https://preflight.dev/schema/{SCHEMA_VERSION}/contract",
		"title": "PreflightContract",
		"description": f"Preflight contract v{SCHEMA_VERSION}. Generated; do not edit.",
		"type": "object",
		"properties": {m.__name__: {"$ref": f"#/$defs/{m.__name__}"} for m in CONTRACT},
		"$defs": pool,
	}


def json_schemas() -> dict[str, dict]:
	out = {name: _schema(m) for name, m in TOP.items()}
	out["contract"] = contract_schema()
	return out


def _dump(obj: dict) -> str:
	return json.dumps(obj, indent="\t", ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument("--check", action="store_true")
	ap.add_argument("--no-ts", action="store_true")
	args = ap.parse_args(argv)
	stale = []
	for name, schema in json_schemas().items():
		path = OUT / f"{name}.schema.json"
		text = _dump(schema)
		if args.check:
			if not path.exists() or path.read_text() != text:
				stale.append(str(path.relative_to(ROOT)))
		else:
			path.write_text(text)
	if not args.no_ts:
		cmd = ["node", str(ROOT / "scripts" / "gen-ts.mjs")] + (["--check"] if args.check else [])
		res = subprocess.run(cmd, cwd=ROOT)
		if res.returncode != 0:
			stale.append("schema/ts/contract.ts")
	if stale:
		print("stale generated files: " + ", ".join(stale), file=sys.stderr)
		print("run: uv run python -m preflight.schema.generate", file=sys.stderr)
		return 1
	return 0


if __name__ == "__main__":
	sys.exit(main())
