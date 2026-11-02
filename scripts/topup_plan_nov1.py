"""The Nov 1 cross-GPU step-time top-up: which configs each GPU runs, from the published corpus (Chiya, 6 PM ET).

    python scripts/topup_plan_nov1.py --corpus <preflight-corpus-0.3.0.jsonl.gz>

Writes python/preflight/corpus/data/topup_nov1_exclude.txt (every config key with an A100 row: the cross-GPU
holdout, which is never run again or calibrated on) and python/preflight/corpus/data/topup_nov1_plan.json (per GPU:
the grouped configs it will run, by seed, that it hasn't run yet and that aren't excluded). The sweep itself skips
what each GPU's store already holds; this file is the list the PM checks against the held-out set.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
DATA = ROOT / "python/preflight/corpus/data"
# GPU -> seeds of the 300-config grouped list it runs (Nov 1 plan in docs/requests/from-b.md)
RUNS = {
	"H100-SXM-80GB": (0, 1),
	"L40S": (0, 1),
	"A10": (1,),
	"L4": (1,),
	"T4": (1,),
}
N = 300  # the shared grouped list's size (corpus/README.md: every GPU runs the first n / 5 groups of a seed)


def plan(corpus: str) -> dict:
	from preflight.corpus import configs as C
	from preflight.corpus import modal_app as M

	pinned = {"corpus": C.CODE_VERSION, "torch": M.PINS["torch"].split("==")[1]}
	done: dict[str, set] = collections.defaultdict(set)
	with gzip.open(corpus, "rt") as fh:
		for line in fh:
			r = json.loads(line)
			if r["kind"] == "kernels":
				done[r["gpu"]].add(r["key"])
	from preflight.corpus.calibrate import held_out  # the calibration's own rule, MIG slices included

	holdout = sorted(set().union(*(keys for g, keys in done.items() if held_out(g))))
	out: dict = {"n": N, "excluded_holdout_keys": len(holdout), "gpus": {}}
	for gpu, seeds in RUNS.items():
		rows = []
		for seed in seeds:
			for c in (c.with_versions(pinned) for c in C.configs_for("grouped", N, seed)):
				if c.key in holdout or c.key in done[gpu] or not M.runs_on(c, gpu):
					continue
				rows.append({"seed": seed, "key": c.key, "family": c.family, "dtype": c.params.get("dtype")})
		uniq = list({r["key"]: r for r in rows}.values())
		out["gpus"][gpu] = {
			"seeds": list(seeds),
			"configs": len(uniq),
			"by_family": dict(collections.Counter(r["family"] for r in uniq)),
			"keys": [r["key"] for r in uniq],
		}
	(DATA / "topup_nov1_exclude.txt").write_text("".join(k + "\n" for k in holdout))
	(DATA / "topup_nov1_plan.json").write_text(json.dumps(out, indent=1) + "\n")
	return out


def main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument("--corpus", required=True)
	a = ap.parse_args(argv)
	out = plan(a.corpus)
	print(
		json.dumps({g: {k: v for k, v in d.items() if k != "keys"} for g, d in out["gpus"].items()}, indent=1)
	)
	print("excluded holdout keys:", out["excluded_holdout_keys"])
	return 0


if __name__ == "__main__":
	sys.exit(main())
