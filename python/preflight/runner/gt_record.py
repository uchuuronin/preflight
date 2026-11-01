"""A flight record from a ground-truth run's recorder probe, built here with no GPU.

	python -m preflight.runner.gt_record out/ground_truth_grpo_trl.json zoo/grpo_trl > record.json

The record is what the runner would write for that script with this one long probe: the static
reader's map and skeleton, the probe as the recorder measured it, and the measured findings over
it (run hygiene, GRPO's zero reward spread with its Wilson interval). Nothing is predicted.
"""

import json
import sys
from pathlib import Path


def build(gt: dict, files: dict[str, str], sample: str):
	from preflight.harness.skeleton import skeleton_record
	from preflight.harness.static import map_static
	from preflight.runner.measured_rules import hygiene_findings
	from preflight.schema import Probe, Source, load_record

	if not gt.get("recorder", {}).get("probe"):
		raise SystemExit("the ground truth has no recorder probe (an older run, or the recorder half failed)")
	sm = map_static(files, entrypoint="train.py")
	rec = skeleton_record(sm, Source(kind="sample", uri=f"sample://{sample}"))
	probe = Probe.model_validate({**gt["recorder"]["probe"], "id": "p1", "knobs": {}})
	rec = rec.model_copy(update={"probes": [probe]})
	rec = rec.model_copy(update={"findings": [*rec.findings, *hygiene_findings(rec, files)]})
	return load_record(rec.model_dump_json())  # validated as any stored record is


def main(argv: list[str] | None = None) -> int:
	from preflight.runner.sources import _read_tree

	gt_path, script_dir = (argv or sys.argv[1:])[:2]
	gt = json.loads(Path(gt_path).read_text())
	files = _read_tree(Path(script_dir))
	rec = build(gt, files, Path(script_dir).name)
	sys.stdout.write(rec.model_dump_json(indent=1, exclude_none=True) + "\n")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
