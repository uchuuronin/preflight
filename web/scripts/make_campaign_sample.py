"""The campaign sample (PF420): the pretrain golden record with a campaign block from B's planner.

Every number in the block is B's: this script calls `preflight.campaign.planner.plan` on a research
grid and writes `to_campaign()` into a copy of the record, with the planner's model card. B's planner
lives on ws-b until round four merges, so run it from a ws-b checkout:

	git worktree add /tmp/wsb origin/ws-b
	PYTHONPATH=/tmp/wsb/python .venv/bin/python web/scripts/make_campaign_sample.py \\
		/tmp/wsb/schema/examples/pretrain.json schema/examples/pretrain.json \\
		web/src/workspace/fixtures/campaign-grid.record.json

The first path is the record B's engine reads (ws-b's copy); the second is this branch's copy of the
same golden record, which the sample is written from so it validates against this branch's contract.

Generated Nov 1 from ws-b 6ce7730. Illustrative: the golden record's numbers, not a measured run.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime

from preflight.campaign import planner
from preflight.schema.record import FlightRecord

# The research grid: the pretrain record's batch size and sequence length, with two seeds each.
GRID = {"batch_size": [8, 12, 16], "seq_len": [512, 1024], "seed": [0, 1]}
# Three people's lanes on a shared cluster (one with two GPUs), a 4 h job limit (Slurm --time) and
# an interruption about once a day. Guesses for an illustrative lab; the budget comes from the record.
CLUSTER = planner.Cluster(lanes={"chiya": 2, "ana": 1, "ben": 1}, wall_time_cap_s=4 * 3600, mtti_s=24 * 3600)
START = datetime(2026, 11, 2, 9, 0, tzinfo=UTC)


def main(engine_record: str, branch_record: str, out_path: str) -> None:
	rec = FlightRecord.model_validate(json.loads(open(engine_record).read()))
	plan = planner.plan(rec, GRID, CLUSTER, start=START)
	out = json.loads(open(branch_record).read())
	out["campaign"] = plan.to_campaign().model_dump(mode="json", exclude_none=True)
	card = plan.card.model_dump(mode="json", exclude_none=True)
	out["models"] = [m for m in out.get("models", []) if m.get("id") != card["id"]] + [card]
	out["record_id"] = "01JBGOLDENCAMPGRD000000000"
	out["job"]["title"] = "Pretraining research grid: batch size, sequence length and seed"
	out.pop("share_token", None)  # share tokens never go in the repo
	with open(out_path, "w") as f:
		json.dump(out, f, indent="\t")
		f.write("\n")
	print(
		f"wrote {out_path}: {len(plan.cells)} runs, {len(plan.cut_list)} cut, finish {plan.finish_s.status}"
	)


if __name__ == "__main__":
	main(*sys.argv[1:4])
