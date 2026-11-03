"""Gather the final validation table's inputs into docs/validation/final/ (Chiya, Nov 3): every record, ground truth
and manifest the frozen scorer reads, committed, not in a scratch folder.

	uv run python scripts/build_final_validation.py [--fresh <manifest of the fresh scripts>]

The frozen scorer (56c2141) reads records up to contract 0.6.2. A record of a later contract is copied with its
`schema_version` set to "0.6.2" (a relabelled copy): its fields from 0.6.3 to 0.6.5 are additive (A's check-in help
fields, `Grid.independent_of`, `resend_of`), and the frozen schema ignores fields it doesn't know, so nothing the
scorer reads changes. Every relabelled copy is listed in docs/validation/final/RELABELLED.md with what it drops.

Writes four manifests beside the files: main.json, every row but the unseen-script row (the zoo's after-freeze full
runs with the samples' own L4 records and BEIR, the AMD pairs, the two production check-ins of the LoRA sample for the
cost of one run, C's knob timing); unseen_first.json, the nine unseen scripts' first attempts (final: 0 of 9);
unseen_after.json, the nine after the freeze (A's ops runs with the install flags, fixes designed on these nine); and
fresh.json, the fresh scripts run once through production (when --fresh is given).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "docs" / "validation" / "final"
FROZEN_VERSION = "0.6.2"
VAL = ROOT / "tests" / "fixtures" / "validation"
SOURCES = {
	"main": [VAL / "zoo_gt" / "manifest_zoo_after_freeze_nov2.json", VAL / "amd" / "manifest_amd_nov2.json",
		VAL / "cost" / "manifest_cost_nov2.json"],
	"unseen_first": [VAL / "manifest_unseen_nov2.json"],
}  # fmt: skip
# C's live-site timing (remote run 41655597938): 3 samples x 20 moves at 1366x768 and 390x844 (PM, Nov 3: use C's)
KNOB_MS = VAL / "knob" / "knob_ms_live_nov3.json"
# A's from-a.md (Nov 3, item 8): the nine rerun with every bring-your-own-script fix on, ops full runs on the A10, the
# list's own arguments, no hand edits: 4 of 9 reach their first step (fixes designed on these nine)
AFTER_REACHED = {"unseen-4", "unseen-7", "unseen-8", "unseen-9"}
AFTER_RUNS = {"unseen-4": "a-all4-4a 41649825142", "unseen-7": "a-all4-7 41647588867",
	"unseen-8": "a-all4-8-r 41652843880", "unseen-9": "a-all4-9 41653607701"}  # fmt: skip


def _version(v: str) -> tuple[int, ...]:
	return tuple(int(x) for x in v.split("."))


def copy(src: Path, relabelled: list[str]) -> str:
	"""Copy one input under DEST/files at its repo path; relabel a record newer than the frozen contract."""
	rel = src.resolve().relative_to(ROOT)
	out = DEST / "files" / rel
	out.parent.mkdir(parents=True, exist_ok=True)
	data = json.loads(src.read_text())
	v = data.get("schema_version") if isinstance(data, dict) else None
	if isinstance(v, str) and _version(v) > _version(FROZEN_VERSION):
		data["schema_version"] = FROZEN_VERSION
		out.write_text(json.dumps(data, indent=1) + "\n")
		relabelled.append(f"| `{rel}` | {v} | `files/{rel}` |")
	else:
		shutil.copyfile(src, out)
	return f"files/{rel}"


def gather(manifests: list[Path], relabelled: list[str]) -> list[dict]:
	runs = []
	for m in manifests:
		for run in json.loads(m.read_text())["runs"]:
			run = dict(run)
			for key in ("record", "ground_truth"):
				if run.get(key) and (m.parent / run[key]).exists():
					run[key] = copy(m.parent / run[key], relabelled)
				elif run.get(key):
					run[key] = None  # not there: the scorer says "not measured" for it
			runs.append(run)
	return runs


def main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument("--fresh", type=Path, help="manifest of the fresh scripts (A's runs through production)")
	a = ap.parse_args(argv)
	if DEST.exists():
		shutil.rmtree(DEST / "files", ignore_errors=True)
	DEST.mkdir(parents=True, exist_ok=True)
	relabelled: list[str] = []
	main_runs = gather(SOURCES["main"], relabelled)
	knob = copy(KNOB_MS, relabelled)
	(DEST / "main.json").write_text(json.dumps({"$comment": "final table, every row but unseen", "knob_ms": knob,
		"runs": main_runs}, indent=1) + "\n")  # fmt: skip
	first = [r for r in gather(SOURCES["unseen_first"], relabelled) if r.get("set") == "unseen"]
	(DEST / "unseen_first.json").write_text(json.dumps({"$comment": "the nine, first attempts", "runs": first},
		indent=1) + "\n")  # fmt: skip
	after = [{"script": r["script"], "set": "unseen", "harnessed_without_edits": r["script"] in AFTER_REACHED,
		**({"$run": AFTER_RUNS[r["script"]]} if r["script"] in AFTER_RUNS else {})} for r in first]  # fmt: skip
	(DEST / "unseen_after.json").write_text(json.dumps({"$comment": "the nine after the freeze: A's item-8 ops rerun, "
		"every fix on, no hand edits (from-a.md, Nov 3); reaching the first step counts", "runs": after},
		indent=1) + "\n")  # fmt: skip
	if a.fresh:
		fresh = gather([a.fresh], relabelled)
		(DEST / "fresh.json").write_text(json.dumps({"$comment": "fresh scripts, once", "runs": fresh}, indent=1)
			+ "\n")  # fmt: skip
	rows = "\n".join(relabelled) or "| (none) | | |"
	(DEST / "RELABELLED.md").write_text(
		"# Relabelled copies\n\nThe frozen scorer (56c2141) reads records up to contract 0.6.2. These inputs are of a "
		"later contract, so they are copied with `schema_version` set to 0.6.2. Contracts 0.6.3 to 0.6.5 are additive "
		"(check-in help's job fields, `Grid.independent_of`, `resend_of`), and the frozen schema ignores fields it "
		"doesn't know, so what the scorer reads is unchanged; the original is at its repo path.\n\n"
		f"| Original | Its version | Copy |\n| --- | --- | --- |\n{rows}\n"
	)
	print(json.dumps({"main": len(main_runs), "unseen_first": len(first), "relabelled": len(relabelled)}))
	return 0


if __name__ == "__main__":
	sys.exit(main())
